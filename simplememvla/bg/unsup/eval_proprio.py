"""[CPU] UnsupBG proprio channel vs BG-Oracle events (part of report 03).

Matches proprio events to oracle sub-goal completions in time:
    grasp   <-> oracle "picked"   (grasp precedes the completion: window [-0.5 s, +4 s] around grasp)
    release <-> oracle "placed"/"inserted"/"pushed" (window [-1 s, +3 s])
    close   <-> oracle "pressed"  (window [-1 s, +4 s])
Reports per task precision / recall / median |dt|, plus the FK error of the event TCP vs the simulator TCP.

    python -m simplememvla.bg.unsup.eval_proprio --replay <replay_dir> --npz_dir <episodes_npz> [--out json]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FPS = 20.0
PAIR = {"grasp": ({"picked"}, -0.5, 4.0), "release": ({"placed", "inserted", "pushed"}, -1.0, 3.0),
        "close": ({"pressed"}, -1.0, 4.0)}


def _one(path_npz):
    path, npz_dir = path_npz
    from simplememvla.bg.oracle_robomme import build_episode
    from simplememvla.bg.unsup.proprio import gripper_events

    task = path.split("/")[-2]
    ep = int(os.path.basename(path)[2:6])
    rec = pickle.load(gzip.open(path))
    d = np.load(os.path.join(npz_dir, f"ep{ep:04d}.npz"), allow_pickle=True)
    rec["instruction"] = str(d["instruction"])
    S = d["state"]
    _, b = build_episode(rec, task, len(S))
    oracle = [e for e in b.events if e.get("verb") in ("picked", "placed", "inserted", "pushed", "pressed")]
    pev = gripper_events(S)
    out = {}
    d2s = rec.get("data_to_snap") or list(range(len(S)))
    fk = []
    for e in pev:
        tcp = rec["snaps"][d2s[e["frame"]]]["poses"].get("panda_hand_tcp")
        if tcp is not None:
            fk.append(float(np.linalg.norm(np.array(e["tcp"]) - np.array(tcp))))
    for kind, (verbs, lo, hi) in PAIR.items():
        P = [e["frame"] / FPS for e in pev if e["type"] == kind]
        O = [e["t"] for e in oracle if e.get("verb") in verbs]
        used, tp, dts = set(), 0, []
        for p in P:
            best = None
            for j, o in enumerate(O):
                if j in used or not (lo <= o - p <= hi):
                    continue
                if best is None or abs(o - p) < abs(O[best] - p):
                    best = j
            if best is not None:
                used.add(best)
                tp += 1
                dts.append(O[best] - p)
        out[kind] = {"pred": len(P), "oracle": len(O), "tp": tp, "dt": dts}
    return {"ep": ep, "task": task, "events": out, "fk_err": fk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.replay, "*", "ep*.pkl.gz")))
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(_one, [(p, a.npz_dir) for p in paths], chunksize=1))
    table = {}
    for r in res:
        t = table.setdefault(r["task"], {k: {"pred": 0, "oracle": 0, "tp": 0, "dt": []} for k in PAIR})
        for k, v in r["events"].items():
            for f in ("pred", "oracle", "tp"):
                t[k][f] += v[f]
            t[k]["dt"] += v["dt"]
        t.setdefault("fk", []).extend(r["fk_err"])
        t["n_eps"] = t.get("n_eps", 0) + 1
    summary = {}
    for task, t in sorted(table.items()):
        row = {"n_eps": t["n_eps"], "fk_err_mean_m": float(np.mean(t["fk"])) if t["fk"] else None}
        for k in PAIR:
            v = t[k]
            row[k] = {"precision": v["tp"] / v["pred"] if v["pred"] else None,
                      "recall": v["tp"] / v["oracle"] if v["oracle"] else None,
                      "n_oracle": v["oracle"], "n_pred": v["pred"],
                      "median_dt_s": float(np.median(v["dt"])) if v["dt"] else None}
        summary[task] = row
        print(task, json.dumps(row))
    if a.out:
        json.dump({"summary": summary, "episodes": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
