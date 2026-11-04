"""Front-camera calibration for UnsupBG, read from the RoboMME H5 observations (report 03 §5).

UnsupBG uses the calibration only to project the FK end-effector position into the front image. Up to freeze v1 it was
read from the oracle simulator replay; from freeze v2 it is read from the dataset's own H5 observations:

    K          <h5>/episode_<i>/setup/front_camera_intrinsic            (3, 3)
    extrinsic  <h5>/episode_<i>/timestep_0/obs/front_camera_extrinsic   (3, 4)   (first observation of the episode)

    # [CPU] one json per task (never overwrites; tasks whose json exists are skipped)
    python -m simplememvla.bg.unsup.h5cam extract --h5_dir <dir with record_dataset_<Task>.h5> \
        --npz_dir <episodes_npz> --out_dir <cams dir>
    # [CPU] audit only: H5 calibration vs. the replay calibration used up to freeze v1
    python -m simplememvla.bg.unsup.h5cam compare --cam_dir <cams dir> --replay <replay dir> --out compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load_cam(cam_dir: str, task: str, ep: int, cache: dict | None = None) -> dict:
    """{"K": 3x3, "extrinsic": 3x4} of LeRobot episode ``ep`` from ``<cam_dir>/<task>.json``."""
    cache = {} if cache is None else cache
    if task not in cache:
        p = os.path.join(cam_dir, f"{task}.json")
        if not os.path.exists(p):
            raise RuntimeError(f"no H5 camera calibration for {task}: {p} (run h5cam extract)")
        cache[task] = json.load(open(p))["episodes"]
    rec = cache[task].get(str(ep))
    if rec is None:
        raise RuntimeError(f"no H5 camera calibration for {task} ep {ep}")
    return {"K": rec["K"], "extrinsic": rec["extrinsic"]}


def _extract_task(h5_path: str, eps: dict[int, int]) -> dict:
    import h5py

    out = {}
    with h5py.File(h5_path, "r") as f:
        for ep, h5_ep in sorted(eps.items()):
            g = f[f"episode_{h5_ep}"]
            K = np.asarray(g["setup/front_camera_intrinsic"][()], float)
            E0 = np.asarray(g["timestep_0/obs/front_camera_extrinsic"][()], float)
            # audit only: is the front camera static within the episode?
            dev = 0.0
            for k in g.keys():
                if k.startswith("timestep_"):
                    dev = max(dev, float(np.abs(np.asarray(g[k]["obs/front_camera_extrinsic"][()], float) - E0).max()))
            out[str(ep)] = {"h5_episode": int(h5_ep), "K": K.tolist(), "extrinsic": E0.tolist(),
                            "extrinsic_max_dev_within_episode": dev}
    return out


def cmd_extract(a) -> None:
    index = {int(k): v for k, v in json.load(open(os.path.join(a.npz_dir, "index.json"))).items()}
    by_task: dict[str, dict[int, int]] = {}
    for ep, v in index.items():
        by_task.setdefault(v["task"], {})[ep] = int(v["h5_episode"])
    os.makedirs(a.out_dir, exist_ok=True)
    for task in sorted(by_task):
        if a.tasks and task not in a.tasks:
            continue
        dst = os.path.join(a.out_dir, f"{task}.json")
        h5 = os.path.join(a.h5_dir, f"record_dataset_{task}.h5")
        if os.path.exists(dst):
            print(f"{task}: {dst} exists, skipped")
            continue
        if not os.path.exists(h5):
            print(f"{task}: {h5} missing, skipped")
            continue
        eps = _extract_task(h5, by_task[task])
        tmp = dst + ".tmp"
        json.dump({"task": task, "source": os.path.abspath(h5), "episodes": eps}, open(tmp, "w"))
        os.replace(tmp, dst)
        Ks = {json.dumps(r["K"]) for r in eps.values()}
        Es = {json.dumps(np.round(r["extrinsic"], 6).tolist()) for r in eps.values()}
        dev = max(r["extrinsic_max_dev_within_episode"] for r in eps.values())
        print(f"{task}: {len(eps)} eps, distinct K {len(Ks)}, distinct extrinsic {len(Es)}, max dev in episode {dev:g}")


def cmd_compare(a) -> None:
    import glob
    import gzip
    import pickle

    res = {}
    for p in sorted(glob.glob(os.path.join(a.cam_dir, "*.json"))):
        task = os.path.basename(p)[:-5]
        eps = json.load(open(p))["episodes"]
        n, dK, dE, missing = 0, 0.0, 0.0, 0
        for ep, r in sorted(eps.items(), key=lambda x: int(x[0])):
            rp = os.path.join(a.replay, task, f"ep{int(ep):04d}.pkl.gz")
            if not os.path.exists(rp):
                missing += 1
                continue
            rec = pickle.load(gzip.open(rp))
            cam = next((s["cam"] for s in rec["snaps"] if s.get("cam")), None)
            if cam is None:
                missing += 1
                continue
            n += 1
            dK = max(dK, float(np.abs(np.asarray(cam["K"], float) - np.asarray(r["K"])).max()))
            dE = max(dE, float(np.abs(np.asarray(cam["extrinsic"], float)[:3] - np.asarray(r["extrinsic"])).max()))
        res[task] = {"n_compared": n, "n_no_replay": missing, "max_abs_diff_K": dK, "max_abs_diff_extrinsic": dE}
        print(task, res[task])
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; files are never overwritten")
    json.dump(res, open(a.out, "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--h5_dir", required=True)
    e.add_argument("--npz_dir", required=True)
    e.add_argument("--out_dir", required=True)
    e.add_argument("--tasks", nargs="*", default=None)
    c = sp.add_parser("compare")
    c.add_argument("--cam_dir", required=True)
    c.add_argument("--replay", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
