"""Replay LeRobot RoboMME training episodes in the simulator and store privileged snapshots (resumable).

    python -m simplememvla.bg.replay_robomme_cli --npz_dir <episodes_npz> --out <replay_dir> \
        [--tasks VideoUnmask ...] [--episodes 0 1 ...] [--per_task N] [--workers 8]

<episodes_npz>/index.json maps LeRobot episode -> (task, h5 episode); ep%04d.npz holds state/action/demo arrays
(extracted from the LeRobot parquet). Each finished episode is written atomically to <out>/<task>/ep%04d.pkl.gz
with the snapshots, the task list and a fidelity record:
    n_frames_replay / n_frames_data, qpos max |diff| vs. observation.state, first frame with diff > 1e-2,
    demo flags equal, all sub-goals completed in replay (task_index == len(task_list)).
Already written episodes are skipped, so the command can be re-run after any interruption.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def _title(name):
    try:
        from setproctitle import setproctitle

        setproctitle(name[:20])
    except Exception:
        pass


def replay_one(npz_dir: str, out_dir: str, ep: int, task: str, h5_ep: int) -> dict:
    _title(f"hoyong-bgrep{ep}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from simplememvla.bg.robomme_replay import ReplayRecorder, frame_qpos8

    path = os.path.join(out_dir, task, f"ep{ep:04d}.pkl.gz")
    if os.path.exists(path):
        return {"ep": ep, "skipped": True}
    d = np.load(os.path.join(npz_dir, f"ep{ep:04d}.npz"), allow_pickle=True)
    S, A, D = d["state"], d["action"], d["demo"]
    t0 = time.time()
    rec = ReplayRecorder(task, split="train")
    rec.reset(h5_ep, keep_no_record=True)
    n_reset = len(rec.snaps)
    n_demo = int(D.sum())
    if not D[:n_demo].all():
        raise RuntimeError(f"ep {ep}: demo frames are not a prefix")
    # data demo frames <-> reset snapshots: monotonic nearest-qpos match (the recorded data keeps only part of the
    # "NO RECORD" reset motion, e.g. none of the stick tasks' leading touch, so indices differ by task)
    qr = np.stack([frame_qpos8(x) for x in rec.snaps[: n_reset - 1]]) if n_reset > 1 else np.zeros((0, 8))
    data_to_snap = []
    k = 0
    for j in range(n_demo):
        if k >= len(qr):
            data_to_snap.append(len(qr) - 1)
            continue
        d_ = np.abs(qr[k:] - S[j]).max(1)
        hit = np.nonzero(d_ < 1e-3)[0]
        k = k + int(hit[0]) if hit.size else k + int(np.argmin(d_))
        data_to_snap.append(k)
        k += 1
    err = None
    try:
        for t in range(n_demo, len(S) - 1):  # first execution frame <-> the init step taken inside reset()
            rec.step(A[t])
    except Exception as e:  # keep what was recorded; fidelity marks the failure
        err = f"{type(e).__name__}: {e}"
    rec.close()
    data_to_snap += list(range(n_reset - 1, n_reset - 1 + (len(rec.snaps) - (n_reset - 1))))
    data_to_snap = data_to_snap[: len(S)]
    n = len(data_to_snap)
    q = np.stack([frame_qpos8(rec.snaps[i]) for i in data_to_snap]) if n else np.zeros((0, 8))
    diff = np.abs(q - S[:n]).max(1) if n else np.zeros(0)
    fid = {
        "ep": ep, "task": task, "h5_episode": h5_ep, "n_replay": len(rec.snaps), "n_data": int(len(S)),
        "n_reset": n_reset, "n_demo_data": n_demo, "n_mapped": n,
        "qpos_maxdiff": float(diff.max()) if n else None,
        "qpos_maxdiff_demo": float(diff[:n_demo].max()) if n_demo else None,
        "first_diff_gt_1e-2": int(np.argmax(diff > 1e-2)) if (diff > 1e-2).any() else None,
        "demo_equal": bool((np.array([rec.snaps[i]["is_demo"] for i in data_to_snap]) == D[:n]).all()),
        "all_subgoals_done": bool(rec.snaps and rec.snaps[-1]["task_index"] >= len(rec.task_list)),
        "seconds": round(time.time() - t0, 1), "error": err,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump({"snaps": rec.snaps, "task_list": rec.task_list, "fidelity": fid, "data_to_snap": data_to_snap,
                     "instruction": str(d["instruction"])}, f)
    os.replace(tmp, path)
    return fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--per_task", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    _title("hoyong-bgreplay")
    index = {int(k): v for k, v in json.load(open(os.path.join(a.npz_dir, "index.json"))).items()}
    jobs = []
    seen: dict[str, int] = {}
    for ep in sorted(index):
        v = index[ep]
        if a.tasks and v["task"] not in a.tasks:
            continue
        if a.episodes is not None and ep not in a.episodes:
            continue
        if a.per_task is not None and seen.get(v["task"], 0) >= a.per_task:
            continue
        seen[v["task"]] = seen.get(v["task"], 0) + 1
        jobs.append((ep, v["task"], v["h5_episode"]))
    log = open(os.path.join(a.out, "fidelity.jsonl"), "a") if os.path.isdir(a.out) else None
    if log is None:
        os.makedirs(a.out, exist_ok=True)
        log = open(os.path.join(a.out, "fidelity.jsonl"), "a")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(replay_one, a.npz_dir, a.out, ep, t, h): ep for ep, t, h in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception:
                r = {"ep": futs[fut], "crash": traceback.format_exc()[-800:]}
            if not r.get("skipped"):
                log.write(json.dumps(r) + "\n")
                log.flush()
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
