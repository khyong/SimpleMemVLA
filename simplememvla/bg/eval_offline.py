"""[GPU] Offline BG validation on held-out training episodes, per checkpoint.

    python -m simplememvla.bg.eval_offline --checkpoints ck1 ck2 ... --root <robomme_lerobot> --labels <label dir> \
        --holdout <run_dir>/holdout_episodes.json [--frames_per_episode 20] [--out result.json]

For sampled anchor frames of the held-out episodes (``holdout_episodes.json`` written by train_bg.py) it runs the
training forward with labels and reports
    subtask_ce, bg_ce             token-mean CE of the two answer parts
    bg_token_acc                  teacher-forced next-token accuracy on BG tokens
    bg_fact_precision / recall    facts of the teacher-forced argmax BG text vs. the label BG text (times and #k
                                  ignored, see bg/unsup/metrics.canon), parse failures counted as empty
    bg_parse_rate                 fraction of teacher-forced BG texts that parse
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--frames_per_episode", type=int, default=20)
    ap.add_argument("--bg_filter", default="mnemonic")
    ap.add_argument("--attn_implementation", default="flash_attention_2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from simplememvla.compat import apply_fla_torch_compat

    apply_fla_torch_compat()
    import torch
    from transformers import AutoProcessor, HfArgumentParser

    from robomme_sim.policy import _build_vla
    from simplememvla.bg.args import BGDataArguments
    from simplememvla.bg.collator import BGDataCollator
    from simplememvla.bg.dataset import BGSimpleMemVLADataset
    from simplememvla.bg.schema import BeliefGraph  # noqa: F401
    from simplememvla.bg.serialize import BGParseError, parse
    from train import resolve_benchmark_defaults

    hold = json.load(open(a.holdout))["episodes"]
    eps = sorted(e for v in hold.values() for e in v)
    argv = ["--benchmark", "robomme", "--root", a.root, "--bg_mode", json.load(open(os.path.join(a.labels, "manifest.json")))["source"],
            "--bg_label_dir", a.labels, "--bg_filter", a.bg_filter, "--image_aug", "False",
            "--episodes", *map(str, eps)]
    (da,) = HfArgumentParser((BGDataArguments,)).parse_args_into_dataclasses(argv)
    spec = resolve_benchmark_defaults(da)
    ds = BGSimpleMemVLADataset(da, spec)
    ep_col = ds._columns(["episode_index"])["episode_index"][ds._anchor_indices]
    rng = np.random.default_rng(0)
    picks = []
    for e in eps:
        idx = np.nonzero(ep_col == e)[0]
        if len(idx):
            picks += sorted(rng.choice(idx, min(a.frames_per_episode, len(idx)), replace=False).tolist())

    def canon_text(text):
        try:
            fs = parse(text)
        except BGParseError:
            return None
        out = []
        for f in fs:
            k = f[0]
            if k == "int":
                out.append(("int", f[2], f[3], f[4]))
            elif k in ("target", "order"):
                out.append(f[:3])
            elif k in ("rel", "hidden"):
                out.append(f[:-1])
            else:
                out.append(f)
        return out

    results = {}
    for ck in a.checkpoints:
        proc = AutoProcessor.from_pretrained(ck, trust_remote_code=True, padding_side="right", model_max_length=8192)
        col = BGDataCollator(proc, max_length=8192, condition_span="all", emit_bg_masks=True)
        vla = _build_vla(ck, torch.bfloat16, attn_implementation=a.attn_implementation).eval().cuda()
        lm_head = vla.backbone.get_output_embeddings()
        tok = proc.tokenizer
        sub_ce, bg_ce, acc, tp, npred, ngold, parsed, n = [], [], [], 0, 0, 0, 0, 0
        for i in picks:
            b = col([ds[int(i)]])
            b = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in b.items()}
            with torch.no_grad():
                pos = vla._position_ids(b["input_ids"], b["attention_mask"], b)
                h, _ = vla._backbone_forward(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                             position_ids=pos, pixel_values_videos=b["pixel_values_videos"],
                                             video_grid_thw=b["video_grid_thw"], pixel_values=b.get("pixel_values"),
                                             image_grid_thw=b.get("image_grid_thw"),
                                             mm_token_type_ids=b["mm_token_type_ids"])
                lab = b["labels"][0, 1:]
                m = lab != -100
                logits = lm_head(h[0, :-1][m]).float()
                ce = torch.nn.functional.cross_entropy(logits, lab[m], reduction="none")
                isbg = b["bg_label_mask"][0, 1:][m]
                pred = logits.argmax(-1)
            sub_ce.append(float(ce[~isbg].mean()))
            if bool(isbg.any()):
                bg_ce.append(float(ce[isbg].mean()))
                acc.append(float((pred[isbg] == lab[m][isbg]).float().mean()))
                gold_text = tok.decode(lab[m][isbg], skip_special_tokens=True).strip()
                pred_text = tok.decode(pred[isbg], skip_special_tokens=True).strip()
                g, p = canon_text(gold_text), canon_text(pred_text)
                n += 1
                parsed += int(p is not None)
                p = p or []
                g = g or []
                from collections import Counter as MC

                tp += sum((MC(map(str, p)) & MC(map(str, g))).values())
                npred += len(p)
                ngold += len(g)
        results[ck] = {"frames": len(picks), "subtask_ce": float(np.mean(sub_ce)) if sub_ce else None,
                       "bg_ce": float(np.mean(bg_ce)) if bg_ce else None,
                       "bg_token_acc": float(np.mean(acc)) if acc else None,
                       "bg_fact_precision": tp / npred if npred else None,
                       "bg_fact_recall": tp / ngold if ngold else None, "bg_parse_rate": parsed / n if n else None}
        print(ck, json.dumps(results[ck]), flush=True)
        del vla
        torch.cuda.empty_cache()
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
