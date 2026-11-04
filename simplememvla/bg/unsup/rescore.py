"""Re-score existing UnsupBG pilot rows against the oracle with the exact AND the ID-aligned metric (report 03).

Reads <out>/rows/ep%04d.pkl written by unsup/pilot.py (rows are not rebuilt or changed) and the oracle replay, and
appends one record per episode to <out>/<name>; the aligned scores are stored under ``aligned``.

    python -m simplememvla.bg.unsup.rescore --out <pilot out> --npz_dir <npz> --replay <replay> [--name metrics_aligned.jsonl]
"""
import argparse
import gzip
import json
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--name", default="metrics_aligned.jsonl")
    a = ap.parse_args()
    from simplememvla.bg.oracle_checks import SUITE
    from simplememvla.bg.oracle_robomme import build_episode
    from simplememvla.bg.unsup.metrics import compare_episode

    index = {int(k): v for k, v in json.load(open(os.path.join(a.npz_dir, "index.json"))).items()}
    dst = os.path.join(a.out, a.name)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; pass another --name (files are never overwritten)")
    with open(dst, "w") as log:
        for fn in sorted(os.listdir(os.path.join(a.out, "rows"))):
            if not fn.endswith(".pkl"):
                continue
            ep = int(fn[2:6])
            task = index[ep]["task"]
            rows = pickle.load(open(os.path.join(a.out, "rows", fn), "rb"))["rows"]
            d = np.load(os.path.join(a.npz_dir, f"ep{ep:04d}.npz"), allow_pickle=True)
            rec = pickle.load(gzip.open(os.path.join(a.replay, task, f"ep{ep:04d}.pkl.gz")))
            rec["instruction"] = str(d["instruction"])
            orows, _ = build_episode(rec, task, len(d["state"]))
            oj, uj = [r["bg_json"] for r in orows], [r["bg_json"] for r in rows]
            r = {"ep": ep, "task": task, "suite": SUITE[task]}
            r.update(compare_episode(oj, uj))
            r["aligned"] = compare_episode(oj, uj, align=True)
            log.write(json.dumps(r, default=str) + "\n")
            log.flush()
            print(ep, task, r["aligned"]["id_map"], flush=True)


if __name__ == "__main__":
    main()
