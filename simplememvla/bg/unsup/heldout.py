"""Held-out evaluation of UnsupBG against the oracle (report 03 §5-3).

The pilot episodes were used while the UnsupBG rules were written, so their scores are development-set scores.
This module fixes (1) an evaluation set that excludes every episode looked at during development and (2) a
fingerprint of the UnsupBG code/configuration, so that the held-out score is computed once with frozen rules.

    # [CPU] choose the evaluation set (never overwrites)
    python -m simplememvla.bg.unsup.heldout select --npz_dir <episodes_npz> \
        --exclude_ids <oracle labels>/excluded_ids.json --dev_json <unsup_pilot_eps.json> --out evalset.json
    # [CPU] freeze the rules (never overwrites; a new freeze = a new file name, recorded in the report)
    python -m simplememvla.bg.unsup.heldout freeze --out freeze.json --llm Qwen/Qwen2.5-7B-Instruct ...
    # [CPU] exit 1 if the code differs from the freeze
    python -m simplememvla.bg.unsup.heldout verify --freeze freeze.json --llm ... --owl ... --sam2 ...

Development episodes (always excluded):
    * the 20 pilot episodes (``--dev_json``), including ep1500 used for perception debugging
    * the first episode of every task (the replay/proprio probes and, most likely, the OWLv2 prompt check frames)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

# files whose content defines the UnsupBG labels (the metric code is listed separately: changing it does not
# change the labels, but the evaluation must be reproducible too)
RULE_FILES = ["simplememvla/bg/unsup/builder.py", "simplememvla/bg/unsup/proprio.py",
              "simplememvla/bg/unsup/visual.py", "simplememvla/bg/unsup/instruction.py",
              "simplememvla/bg/unsup/pilot.py", "simplememvla/bg/schema.py", "simplememvla/bg/serialize.py",
              "simplememvla/bg/unsup/h5cam.py"]   # h5cam.py: from freeze v2 (camera calibration from the H5)
METRIC_FILES = ["simplememvla/bg/unsup/metrics.py", "simplememvla/bg/unsup/rescore.py",
                "simplememvla/bg/unsup/summarize.py", "simplememvla/bg/oracle_robomme.py"]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fingerprint(llm: str, owl: str, sam2: str, spec: str) -> dict:
    root = _repo_root()
    return {"rule_files": {f: _sha(os.path.join(root, f)) for f in RULE_FILES},
            "metric_files": {f: _sha(os.path.join(root, f)) for f in METRIC_FILES},
            "models": {"llm": llm, "owl": os.path.basename(owl.rstrip("/")), "sam2": os.path.basename(sam2.rstrip("/"))},
            "spec": spec}


def _write_new(path: str, obj) -> None:
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; files are never overwritten (use a new name)")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def cmd_select(a) -> None:
    index = {int(k): v for k, v in json.load(open(os.path.join(a.npz_dir, "index.json"))).items()}
    excluded = set(json.load(open(a.exclude_ids))) if a.exclude_ids else set()
    dev = set()
    if a.dev_json:
        for v in json.load(open(a.dev_json)).values():
            dev.update(int(x) for x in v)
    first_of_task = {}
    for e in sorted(index):
        first_of_task.setdefault(index[e]["task"], e)
    first_of_task = set(first_of_task.values())
    dev |= first_of_task
    by_task: dict[str, list[int]] = {}
    for e in sorted(index):
        if e in excluded or e in dev:
            continue
        by_task.setdefault(index[e]["task"], []).append(e)
    rng = random.Random(a.seed)
    chosen = {t: sorted(rng.sample(eps, a.per_task)) for t, eps in sorted(by_task.items())}
    out = {"seed": a.seed, "per_task": a.per_task,
           "episodes": sorted(e for v in chosen.values() for e in v), "by_task": chosen,
           "excluded_dev": sorted(dev), "excluded_failed_replay": sorted(excluded),
           "note": "chosen before any held-out score was computed; do not re-draw after seeing results"}
    _write_new(a.out, out)
    print(json.dumps({t: v for t, v in chosen.items()}))
    print(f"{len(out['episodes'])} episodes -> {a.out}")


def cmd_freeze(a) -> None:
    fp = fingerprint(a.llm, a.owl, a.sam2, a.spec)
    fp["note"] = a.note
    _write_new(a.out, fp)
    print(f"frozen -> {a.out}")


def cmd_verify(a) -> None:
    ref = json.load(open(a.freeze))
    cur = fingerprint(a.llm, a.owl, a.sam2, a.spec)
    bad = []
    for grp in ("rule_files", "metric_files"):
        for f, h in ref[grp].items():
            if cur[grp].get(f) != h:
                bad.append(f)
    for k, v in ref["models"].items():
        if cur["models"].get(k) != v:
            bad.append(f"model:{k} {cur['models'].get(k)} != {v}")
    if cur["spec"] != ref["spec"]:
        bad.append(f"spec {cur['spec']} != {ref['spec']}")
    if bad:
        print("UnsupBG differs from the freeze:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(1)
    print("UnsupBG matches the freeze")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--npz_dir", required=True)
    s.add_argument("--exclude_ids", default=None)
    s.add_argument("--dev_json", default=None)
    s.add_argument("--per_task", type=int, default=3)
    s.add_argument("--seed", type=int, default=20261105)
    s.add_argument("--out", required=True)
    for name in ("freeze", "verify"):
        p = sp.add_parser(name)
        p.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
        p.add_argument("--owl", default="owlv2-base-patch16-ensemble")
        p.add_argument("--sam2", default="sam2.1-hiera-small")
        p.add_argument("--spec", default="llm", choices=["llm", "rule"])
        if name == "freeze":
            p.add_argument("--out", required=True)
            p.add_argument("--note", default="")
        else:
            p.add_argument("--freeze", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "freeze": cmd_freeze, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
