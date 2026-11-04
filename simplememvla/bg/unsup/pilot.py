"""UnsupBG pilot / full generation driver (per episode, cached, resumable).

    python -m simplememvla.bg.unsup.pilot --root <robomme_lerobot> --npz_dir <episodes_npz> --replay <replay_dir> \
        --owl <owlv2 dir> --sam2 <sam2 dir> --out <dir> [--per_suite 5] [--episodes ...] [--device cuda]
        [--spec llm --llm Qwen/Qwen2.5-7B-Instruct | --spec rule] --cam_dir <h5 cams dir (h5cam extract)>

The front-camera calibration comes from the dataset H5 (``--cam_dir``, freeze v2). ``--cam_from_replay`` reproduces
the pilot runs up to freeze v1, which read it from the oracle simulator replay. ``--replay`` is otherwise used only
for the oracle side of the metric.

Per episode: decode the front video, proprio events, frame schedule, OWLv2 + SAM2 tracks (cached in
<out>/tracks/ep%04d.pkl), instruction spec (cached), UnsupBuilder rows (<out>/rows/ep%04d.pkl) and, when the oracle
replay exists, metrics vs. the oracle (<out>/metrics.jsonl). ``--labels_out`` writes a label sidecar (source unsup)
from all rows present. ``--spec rule`` is only for CPU smoke tests; labels for training use the LLM spec.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time

import numpy as np

CAMS = None


def decode_front(root: str, ep: int) -> np.ndarray:
    import glob

    import av
    import pandas as pd

    global CAMS
    if CAMS is None:
        CAMS = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))])
    r = CAMS[CAMS.episode_index == ep].iloc[0]
    key = "videos/observation.images.front"
    f = f"{root}/videos/observation.images.front/chunk-{int(r[key + '/chunk_index']):03d}/file-{int(r[key + '/file_index']):03d}.mp4"
    t0, t1 = float(r[key + "/from_timestamp"]), float(r[key + "/to_timestamp"])
    n = int(r["length"])
    out = []
    with av.open(f) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, t0 - 0.5) / s.time_base), stream=s)
        for fr in c.decode(s):
            if fr.time < t0 - 1e-3:
                continue
            if fr.time >= t1 - 1e-3 or len(out) >= n:
                break
            out.append(fr.to_ndarray(format="rgb24"))
    if len(out) != n:
        raise RuntimeError(f"ep {ep}: decoded {len(out)} of {n} frames")
    return np.stack(out)


def episode_cam_replay(replay_dir: str, task: str, ep: int, cam_cache: dict) -> dict:
    """Legacy (pilot runs up to freeze v1): front-camera calibration read from the oracle simulator replay."""
    if task in cam_cache:
        return cam_cache[task]
    import glob

    for p in sorted(glob.glob(os.path.join(replay_dir, task, "ep*.pkl.gz"))):
        rec = pickle.load(gzip.open(p))
        cam = next((s["cam"] for s in rec["snaps"] if s.get("cam")), None)
        if cam:
            cam_cache[task] = cam
            return cam
    raise RuntimeError(f"no camera calibration for {task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--owl", required=True)
    ap.add_argument("--sam2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_suite", type=int, default=None)
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spec", choices=["llm", "rule"], default="llm")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cam_dir", default=None, help="H5 front-camera calibration (simplememvla.bg.unsup.h5cam)")
    ap.add_argument("--cam_from_replay", action="store_true", help="legacy: calibration from the oracle replay")
    a = ap.parse_args()
    if bool(a.cam_dir) == a.cam_from_replay:
        ap.error("give exactly one of --cam_dir (H5) or --cam_from_replay (legacy)")
    try:
        from setproctitle import setproctitle

        setproctitle("hoyong-bgunsup")
    except Exception:
        pass
    from simplememvla.bg.oracle_checks import SUITE
    from simplememvla.bg.oracle_robomme import build_episode
    from simplememvla.bg.unsup.builder import UnsupBuilder, detect_demo_end
    from simplememvla.bg.unsup.instruction import LLMSpec, rule_spec
    from simplememvla.bg.unsup.metrics import compare_episode
    from simplememvla.bg.unsup.proprio import gripper_events, tcp_dwell_events
    from simplememvla.bg.unsup.visual import Perception, motion_windows, schedule

    for d in ("tracks", "rows"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    index = {int(k): v for k, v in json.load(open(os.path.join(a.npz_dir, "index.json"))).items()}
    if a.episodes:
        eps = a.episodes
    else:
        by_suite: dict[str, list[int]] = {}
        for ep in sorted(index):
            by_suite.setdefault(SUITE[index[ep]["task"]], []).append(ep)
        eps = []
        for s, lst in sorted(by_suite.items()):
            # spread over the suite's tasks: round-robin over tasks, first episodes of each
            tasks = sorted({index[e]["task"] for e in lst})
            per = {t: [e for e in lst if index[e]["task"] == t] for t in tasks}
            k = 0
            while len([e for e in eps if SUITE[index[e]["task"]] == s]) < (a.per_suite or len(lst)):
                t = tasks[k % len(tasks)]
                i = k // len(tasks)
                if i < len(per[t]):
                    eps.append(per[t][i])
                k += 1
                if k > 10 * len(lst):
                    break
    spec_fn = rule_spec if a.spec == "rule" else LLMSpec(a.llm, os.path.join(a.out, "spec_cache.json"), a.device)
    perc = None
    cam_cache: dict = {}
    if a.cam_dir:
        from simplememvla.bg.unsup.h5cam import load_cam

        def episode_cam(task, ep):
            return load_cam(a.cam_dir, task, ep, cam_cache)
    else:
        def episode_cam(task, ep):
            return episode_cam_replay(a.replay, task, ep, cam_cache)
    mlog = open(os.path.join(a.out, "metrics.jsonl"), "a")
    for ep in eps:
        task = index[ep]["task"]
        rows_p = os.path.join(a.out, "rows", f"ep{ep:04d}.pkl")
        if os.path.exists(rows_p):
            continue
        t0 = time.time()
        d = np.load(os.path.join(a.npz_dir, f"ep{ep:04d}.npz"), allow_pickle=True)
        S, instr = d["state"], str(d["instruction"])
        spec = spec_fn(instr)
        stick = task in ("PatternLock", "RouteStick")
        tr_p = os.path.join(a.out, "tracks", f"ep{ep:04d}.pkl")
        if os.path.exists(tr_p):
            tracks = pickle.load(open(tr_p, "rb"))
        else:
            frames = decode_front(a.root, ep)
            ev = tcp_dwell_events(S) if stick else gripper_events(S)
            sample = schedule(len(S), [e["frame"] for e in ev], motion_windows(frames))
            if perc is None:
                perc = Perception(a.owl, a.sam2, device=a.device)
            objs = list(spec.get("objects", []))
            if any("cube" in o for o in objs):  # detector vocabulary: every cube colour the scene may contain
                objs = [o for o in objs if "cube" not in o] + ["red cube", "green cube", "blue cube"]
            cats = list(dict.fromkeys(objs + (["highlight"] if "highlight" in instr else [])))
            cam = episode_cam(task, ep)
            from simplememvla.bg.unsup.builder import project
            from simplememvla.bg.unsup.proprio import tcp_world

            tcp_uv = {f: project(cam, tcp_world(S[f, :7], stick=stick)) for f in sample}
            tracks = perc.run(frames, sample, cats, tcp_uv=tcp_uv)
            tmp = tr_p + ".tmp"
            pickle.dump(tracks, open(tmp, "wb"))
            os.replace(tmp, tr_p)
        cam = episode_cam(task, ep)
        b = UnsupBuilder(task, instr, spec, S, tracks, cam, stick=stick, demo_end=detect_demo_end(S))
        rows = b.labels()
        pickle.dump({"rows": rows, "spec": spec, "events": b.events}, open(rows_p + ".tmp", "wb"))
        os.replace(rows_p + ".tmp", rows_p)
        rp = os.path.join(a.replay, task, f"ep{ep:04d}.pkl.gz")
        rec_m = {"ep": ep, "task": task, "suite": SUITE[task], "seconds": round(time.time() - t0, 1),
                 "spec_source": a.spec, "n_tracks": len(tracks),
                 "cam_source": "h5" if a.cam_dir else "replay"}
        if os.path.exists(rp):
            rec = pickle.load(gzip.open(rp))
            rec["instruction"] = instr
            orows, _ = build_episode(rec, task, len(S))
            rec_m.update(compare_episode([r["bg_json"] for r in orows], [r["bg_json"] for r in rows]))
            rec_m["example"] = {"t": len(S) - 1, "oracle": orows[-1]["bg_mnemonic"], "unsup": rows[-1]["bg_mnemonic"]}
        mlog.write(json.dumps(rec_m, default=str) + "\n")
        mlog.flush()
        print(json.dumps({k: rec_m[k] for k in ("ep", "task", "seconds", "n_tracks") if k in rec_m}), flush=True)


if __name__ == "__main__":
    main()
