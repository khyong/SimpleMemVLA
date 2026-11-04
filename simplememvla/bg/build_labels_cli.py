"""Build BG label sidecars from replay snapshots (oracle) and validate them.

    python -m simplememvla.bg.build_labels_cli oracle --replay <replay_dir> --npz_dir <episodes_npz> \
        --out <label_root> [--tasks ...] [--max_interactions 4] [--workers 16]

Writes <label_root>/oracle_<hash12>/labels.parquet + manifest.json (immutable; a new content hash gives a new
dir) and <label_root>/oracle_<hash12>/validation.jsonl (per-episode task check, see bg/oracle_checks.py) and
causality.jsonl (per-episode: every fact evidence_t <= t, and BG_t recomputed from the stream truncated at t
equals BG_t for sampled t).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _episode(job):
    ep, task, replay_dir, npz_dir, K, n_trunc = job
    from simplememvla.bg.oracle_checks import SUITE, check
    from simplememvla.bg.oracle_robomme import build_episode

    path = os.path.join(replay_dir, task, f"ep{ep:04d}.pkl.gz")
    if not os.path.exists(path):
        return {"ep": ep, "task": task, "missing": True}
    rec = pickle.load(gzip.open(path))
    fid = rec.get("fidelity") or {}
    bad = (not fid.get("all_subgoals_done") or (fid.get("qpos_maxdiff_demo") or 0.0) > 1e-2
           or (fid.get("qpos_maxdiff") or 0.0) > 0.1 or fid.get("error") or not fid.get("demo_equal", True))
    if bad:
        return {"ep": ep, "task": task, "excluded": True, "fidelity": fid}
    d = np.load(os.path.join(npz_dir, f"ep{ep:04d}.npz"), allow_pickle=True)
    instr = str(d["instruction"])
    rec["instruction"] = instr
    n = len(d["state"])
    rows, b = build_episode(rec, task, n, K)
    ok, detail = check(task, b, instr, b.rel_at_pick)
    # causality: rebuild with the replay cut right after the snapshot of frame t; BG_t must not change
    rng = random.Random(ep)
    cut_ok, cuts = True, []
    d2s = rec.get("data_to_snap") or list(range(n))
    for t in sorted(rng.sample(range(n), min(n_trunc, n))):
        cut = dict(rec)
        cut["snaps"] = rec["snaps"][: d2s[t] + 1]
        cut["data_to_snap"] = d2s[: t + 1]
        rows_t, _ = build_episode(cut, task, t + 1, K)
        same = rows_t[t]["bg_json"] == rows[t]["bg_json"]
        cut_ok &= same
        cuts.append((t, same))
    for r in rows:
        r.update(episode_index=ep, task=task)
    return {"ep": ep, "task": task, "suite": SUITE.get(task), "rows": rows, "check": ok, "detail": detail,
            "causal_ok": cut_ok, "cuts": cuts, "fidelity": rec.get("fidelity")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["oracle"])
    ap.add_argument("--replay", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--max_interactions", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_trunc", type=int, default=3, help="truncation re-builds per episode (causality test)")
    ap.add_argument("--allow_missing", action="store_true")
    a = ap.parse_args()
    try:
        from setproctitle import setproctitle

        setproctitle("hoyong-bglabels")
    except Exception:
        pass
    index = {int(k): v for k, v in json.load(open(os.path.join(a.npz_dir, "index.json"))).items()}
    jobs = [(ep, v["task"], a.replay, a.npz_dir, a.max_interactions, a.n_trunc) for ep, v in sorted(index.items())
            if not a.tasks or v["task"] in a.tasks]
    rows, val, missing, excluded = [], [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(_episode, jobs, chunksize=1):
            if r.get("missing"):
                missing.append(r["ep"])
                continue
            if r.get("excluded"):
                excluded.append({"ep": r["ep"], "task": r["task"], "fidelity": r["fidelity"]})
                continue
            rows.extend(r.pop("rows"))
            val.append(r)
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} episodes have no replay yet (e.g. {missing[:5]}); use --allow_missing")
    import pandas as pd

    from simplememvla.bg.labels import write_labels

    df = pd.DataFrame(rows)
    from simplememvla.bg.serialize import FORMAT_VERSION

    h = hashlib.sha256(pd.util.hash_pandas_object(
        df[["episode_index", "frame_index", "bg_json", "bg_mnemonic", "bg_full"]], index=False).values.tobytes()
        + FORMAT_VERSION.encode()).hexdigest()[:12]
    out = os.path.join(a.out, f"oracle_{h}")
    if os.path.exists(os.path.join(out, "labels.parquet")):
        print(f"labels already exist: {out}")
    else:
        digest = write_labels(out, rows, {"source": "oracle", "max_interactions": a.max_interactions,
                                          "builder": "simplememvla.bg.oracle_robomme v1", "content_hash12": h,
                                          "format_version": FORMAT_VERSION,
                                          "missing_episodes": missing,
                                          "excluded_episodes": sorted(e["ep"] for e in excluded),
                                          "tasks": sorted({v["task"] for v in val})})
        print(f"wrote {out} label_hash={digest}")
    with open(os.path.join(out, "excluded.json"), "w") as f:
        json.dump(excluded, f, indent=1, default=str)
    with open(os.path.join(out, "excluded_ids.json"), "w") as f:  # -> --exclude_episodes_file (R2/R3/R4 alike)
        json.dump(sorted(e["ep"] for e in excluded), f)
    with open(os.path.join(out, "validation.jsonl"), "w") as f:
        for v in val:
            f.write(json.dumps({k: v[k] for k in ("ep", "task", "suite", "check", "detail", "causal_ok", "cuts",
                                                 "fidelity")}, default=str) + "\n")
    by_task: dict[str, list] = {}
    for v in val:
        by_task.setdefault(v["task"], []).append(v)
    for t, vs in sorted(by_task.items()):
        okc = [v["check"] for v in vs if v["check"] is not None]
        print(f"{t:18s} eps={len(vs):4d} check_pass={sum(okc)}/{len(okc)} causal_ok={sum(v['causal_ok'] for v in vs)}/{len(vs)}")
    print(out)


if __name__ == "__main__":
    main()
