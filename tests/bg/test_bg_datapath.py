"""[CPU] BG training data path: answer = sub-task + "\\n" + BG text from the sidecar, the collator masks split the
answer exactly at "Belief:", condition_span=subtask_only keeps only the sub-task tokens, holdout / anchor stride.

    python -m tests.bg.test_bg_datapath --root <robomme_lerobot> --processor <ckpt> --labels <label dir> \
        --episodes 0 100 ... [--out report.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--episodes", nargs="+", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from simplememvla.compat import apply_fla_torch_compat

    apply_fla_torch_compat()
    from transformers import AutoProcessor, HfArgumentParser

    from simplememvla.bg import BG_PREFIX
    from simplememvla.bg.args import BGDataArguments
    from simplememvla.bg.collator import BGDataCollator
    from simplememvla.bg.dataset import BGSimpleMemVLADataset
    from simplememvla.bg.serialize import parse
    from train import resolve_benchmark_defaults

    proc = AutoProcessor.from_pretrained(a.processor, trust_remote_code=True, padding_side="right",
                                         model_max_length=8192)
    tok = proc.tokenizer
    import os

    eps = [str(e) for e in a.episodes]
    res = {}
    excl_file = os.path.join(a.labels, "excluded_ids.json")
    excl_arg = ["--exclude_episodes_file", excl_file] if os.path.exists(excl_file) else []
    # guard: without the exclusion file the dataset must refuse anchors of excluded episodes
    try:
        (g,) = HfArgumentParser((BGDataArguments,)).parse_args_into_dataclasses(
            ["--benchmark", "robomme", "--root", a.root, "--bg_mode", "oracle", "--bg_label_dir", a.labels,
             "--episodes", *eps])
        resolve_benchmark_defaults(g)
        BGSimpleMemVLADataset(g, None if False else resolve_benchmark_defaults(g))
        res["guard_raises"] = False
    except ValueError:
        res["guard_raises"] = True
    for cond in ("all", "subtask_only"):
        argv = ["--benchmark", "robomme", "--root", a.root, "--bg_mode", "oracle", "--bg_label_dir", a.labels,
                "--condition_span", cond, "--image_aug", "False", "--episodes", *eps] + excl_arg
        (da,) = HfArgumentParser((BGDataArguments,)).parse_args_into_dataclasses(argv)
        da.bg_validate()
        spec = resolve_benchmark_defaults(da)
        ds = BGSimpleMemVLADataset(da, spec)
        col = BGDataCollator(proc, max_length=8192, condition_span=cond, emit_bg_masks=True)
        idx = np.linspace(0, len(ds) - 1, a.n).round().astype(int)
        fails = []
        bg_tok, sub_tok = [], []
        for i in idx:
            f = ds[int(i)]
            ans = f["messages"][-1]["content"]
            b = col([f])
            ids, lab = b["input_ids"][0], b["labels"][0]
            bgm, cm = b["bg_label_mask"][0], b["condition_mask"][0]
            span = lab.ne(-100)
            bg_text = tok.decode(ids[bgm], skip_special_tokens=True).strip()
            sub_text = tok.decode(ids[span & ~bgm], skip_special_tokens=True).strip()
            ok = (ans.split("\n", 1)[1] == bg_text and ans.split("\n", 1)[0] == sub_text and bg_text.startswith(BG_PREFIX))
            try:
                parse(bg_text)
            except Exception as e:  # noqa: BLE001
                ok = False
            if cond == "all":
                ok &= bool((cm == span).all())
            else:
                ok &= bool((cm <= (span & ~bgm)).all()) and tok.decode(ids[cm]).strip() == ans.split("\n", 1)[0]
            if not ok:
                fails.append({"i": int(i), "answer": ans[:200], "bg_dec": bg_text[:200], "sub_dec": sub_text})
            bg_tok.append(int(bgm.sum()))
            sub_tok.append(int((span & ~bgm).sum()))
        res[cond] = {"samples": len(idx), "fails": fails, "bg_tokens_mean": float(np.mean(bg_tok)),
                     "subtask_tokens_mean": float(np.mean(sub_tok))}
    # holdout + anchor stride
    argv = ["--benchmark", "robomme", "--root", a.root, "--bg_mode", "oracle", "--bg_label_dir", a.labels,
            "--episodes", *eps, "--holdout_episodes_per_task", "1", "--anchor_stride", "4", "--anchor_phase_by_epoch",
            "True"] + (["--exclude_episodes_file", excl_file] if os.path.exists(excl_file) else [])
    (da,) = HfArgumentParser((BGDataArguments,)).parse_args_into_dataclasses(argv)
    spec = resolve_benchmark_defaults(da)
    ds = BGSimpleMemVLADataset(da, spec, output_dir="/tmp/bg_holdout_test")
    fr0 = ds._columns(["frame_index"])["frame_index"][ds._anchor_indices]
    ds.set_epoch(1)
    fr1 = ds._columns(["frame_index"])["frame_index"][ds._anchor_indices]
    held = sorted(e for v in ds.holdout.values() for e in v)
    ep_col = ds._columns(["episode_index"])["episode_index"]
    excl = json.load(open(excl_file)) if os.path.exists(excl_file) else []
    res["holdout_stride"] = {"held": held, "held_in_anchors": int(np.isin(ep_col[ds._anchor_indices], held).sum()),
                             "excluded_in_anchors": int(np.isin(ep_col[ds._anchor_indices], excl).sum()),
                             "anchor_episodes": sorted(set(ep_col[ds._anchor_indices].tolist())),
                             "phase0_ok": bool((fr0 % 4 == 0).all()), "phase1_ok": bool((fr1 % 4 == 1).all()),
                             "len": len(ds)}
    res["pass"] = (res["guard_raises"] and not res["all"]["fails"] and not res["subtask_only"]["fails"]
                   and res["holdout_stride"]["held_in_anchors"] == 0 and res["holdout_stride"]["phase0_ok"]
                   and res["holdout_stride"]["excluded_in_anchors"] == 0 and res["holdout_stride"]["len"] > 0
                   and res["holdout_stride"]["phase1_ok"])
    print(json.dumps(res, indent=1)[:3000])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
