"""Collect UnsupBG per-episode rows into an immutable label sidecar (source = unsup).

    python -m simplememvla.bg.unsup.write_labels --rows <unsup>/rows --npz_dir <episodes_npz> --out <label_root> \
        [--exclude_ids <oracle label dir>/excluded_ids.json] [--allow_missing]
The episode set must equal the oracle's (same exclusions) so that R3 and R4 train on identical anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_ids", default=None)
    ap.add_argument("--max_interactions", type=int, default=4)
    ap.add_argument("--allow_missing", action="store_true")
    a = ap.parse_args()
    import pandas as pd

    from simplememvla.bg.labels import write_labels

    index = {int(k): v for k, v in json.load(open(os.path.join(a.npz_dir, "index.json"))).items()}
    excl = set(json.load(open(a.exclude_ids))) if a.exclude_ids else set()
    rows, missing, specs = [], [], {}
    for ep in sorted(index):
        if ep in excl:
            continue
        p = os.path.join(a.rows, f"ep{ep:04d}.pkl")
        if not os.path.exists(p):
            missing.append(ep)
            continue
        r = pickle.load(open(p, "rb"))
        specs[index[ep]["task"]] = r.get("spec")
        for x in r["rows"]:
            rows.append(dict(x, episode_index=ep, task=index[ep]["task"]))
    if missing and not a.allow_missing:
        raise SystemExit(f"{len(missing)} episodes have no UnsupBG rows (e.g. {missing[:5]})")
    df = pd.DataFrame(rows)
    from simplememvla.bg.serialize import FORMAT_VERSION

    h = hashlib.sha256(pd.util.hash_pandas_object(
        df[["episode_index", "frame_index", "bg_json", "bg_mnemonic", "bg_full"]], index=False).values.tobytes()
        + FORMAT_VERSION.encode()).hexdigest()[:12]
    out = os.path.join(a.out, f"unsup_{h}")
    digest = write_labels(out, rows, {"source": "unsup", "max_interactions": a.max_interactions,
                                      "builder": "simplememvla.bg.unsup v1", "content_hash12": h,
                                      "missing_episodes": missing, "excluded_episodes": sorted(excl),
                                      "spec_examples": specs})
    print(out, digest)


if __name__ == "__main__":
    main()
