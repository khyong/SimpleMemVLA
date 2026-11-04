"""[CPU] Test 1: bg_mode=none is identical to the official pipeline.

For N samples (default 100, fixed indices spread over the dataset) build the official
SimpleMemVLADataset + SimpleMemVLADataCollator batch and the BG one (BGSimpleMemVLADataset + BGDataCollator with
all BG options at defaults, through train_bg's argument path) and require every batch tensor to be identical:
input_ids, attention_mask, mm_token_type_ids, labels, pixel values, grids, actions, masks, state; plus the
condition mask (official: labels != -100) and the batch key set.

    python -m tests.bg.test_identity --root <robomme_lerobot> --processor <ckpt dir> [--n 100] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch


def _seed(i: int) -> None:
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from simplememvla.compat import apply_fla_torch_compat

    apply_fla_torch_compat()
    from transformers import AutoProcessor, HfArgumentParser

    from configs.sft_params import DataArguments
    from simplememvla.bg.args import BGDataArguments
    from simplememvla.bg.collator import BGDataCollator
    from simplememvla.bg.dataset import BGSimpleMemVLADataset
    from simplememvla.data.collator import SimpleMemVLADataCollator
    from simplememvla.data.dataset import SimpleMemVLADataset
    from train import resolve_benchmark_defaults

    argv = ["--benchmark", "robomme", "--root", a.root]
    (off_args,) = HfArgumentParser((DataArguments,)).parse_args_into_dataclasses(argv)
    (bg_args,) = HfArgumentParser((BGDataArguments,)).parse_args_into_dataclasses(argv)
    bg_args.bg_validate()
    spec = resolve_benchmark_defaults(off_args)
    resolve_benchmark_defaults(bg_args)
    proc = AutoProcessor.from_pretrained(a.processor, trust_remote_code=True, padding_side="right",
                                         model_max_length=8192)
    ds_off = SimpleMemVLADataset(off_args, spec)
    ds_bg = BGSimpleMemVLADataset(bg_args, spec)
    col_off = SimpleMemVLADataCollator(proc, max_length=8192)
    col_bg = BGDataCollator(proc, max_length=8192, condition_span="all", emit_bg_masks=False)
    assert len(ds_off) == len(ds_bg), (len(ds_off), len(ds_bg))
    idx = np.linspace(0, len(ds_off) - 1, a.n).round().astype(int).tolist()
    mismatches, checked = [], 0
    for b0 in range(0, len(idx), a.batch):
        chunk = idx[b0:b0 + a.batch]
        feats_off, feats_bg = [], []
        for i in chunk:
            _seed(1000 + i)
            feats_off.append(ds_off[i])
            _seed(1000 + i)
            feats_bg.append(ds_bg[i])
        bo, bb = col_off(feats_off), col_bg(feats_bg)
        if set(bo) != set(bb):
            mismatches.append({"batch": chunk, "keys": [sorted(bo), sorted(bb)]})
        for k in bo:
            if not torch.equal(bo[k], bb.get(k, torch.empty(0))):
                mismatches.append({"batch": chunk, "key": k})
        cond_off = bo["labels"].ne(-100)
        cond_bg = bb.get("condition_mask", bb["labels"].ne(-100))
        if not torch.equal(cond_off, cond_bg):
            mismatches.append({"batch": chunk, "key": "condition_mask"})
        checked += len(chunk)
    rep = {"samples": checked, "indices": idx, "mismatches": mismatches, "pass": not mismatches}
    print(json.dumps({k: v for k, v in rep.items() if k != "indices"}, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    raise SystemExit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
