"""RoboMME closed-loop evaluation with BG options (branch bgmem). Wraps the official robomme_sim.eval_success
without modifying it: the same argument list plus

  --condition_span all|subtask_only    action-head condition span (default: checkpoint config, else all)
  --max_subtask_tokens_auto LABEL_DIR  answer budget = p99(sub-task + BG tokens over the labels) + margin
  --bg_log PATH.jsonl                  one line per decision and row: task, episode, decision, frame, sub-task,
                                       BG, parse result, token counts, prefill/decode/denoise seconds, runaway
  --bg_override none|oracle|file       teacher-force the BG text after the generated sub-task
  --bg_override_file PATH.json         for 'file': {"<task>": "<BG text>"} or {"<task>/<episode>": "<BG text>"}
  --bg_edit_spec PATH.json             edits applied to the forced text: [{"pattern": regex, "repl": str}, ...]
  --bg_seed_offset                     unused (evaluation seeds stay the official frozen episodes)

The evaluation loop stays synchronous (generate_batch then pool.step), so the extra logging cannot change
success rates; times are recorded only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading

import numpy as np

from robomme_sim import eval_success as ES

_LOCK = threading.Lock()


def _extra_args(argv):
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--condition_span", default=None, choices=[None, "all", "subtask_only"])
    ap.add_argument("--max_subtask_tokens_auto", default=None)
    ap.add_argument("--max_subtask_tokens_margin", type=int, default=16)
    ap.add_argument("--bg_log", default=None)
    ap.add_argument("--bg_override", default="none", choices=["none", "oracle", "file"])
    ap.add_argument("--bg_override_file", default=None)
    ap.add_argument("--bg_edit_spec", default=None)
    return ap.parse_known_args(argv)


def auto_answer_budget(label_dir: str, tokenizer, margin: int = 16, sample: int = 20000) -> int:
    """p99 of tokenized (sub-task + '\\n' + BG) answers over a label sample, plus a margin."""
    import pandas as pd

    from simplememvla.bg.labels import LABEL_FILE

    df = pd.read_parquet(os.path.join(label_dir, LABEL_FILE), columns=["bg_mnemonic"])
    texts = df["bg_mnemonic"].sample(min(sample, len(df)), random_state=0).tolist()
    # sub-task strings are <= 20 tokens in RoboMME; measured separately in reports/06 length table
    lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    return int(np.percentile(lens, 99)) + 24 + int(margin)


class _Ctx:
    """Per-group context: which env/episode each buffer belongs to, the decision counter, the oracle trackers."""
    def __init__(self):
        self.task = None
        self.specs = []
        self.pool = None
        self.decision = 0
        self.trackers = {}


def main():
    extra, rest = _extra_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    args = ES.parse_args()
    if getattr(args, "pipelined", False):
        raise SystemExit("eval_bg supports the default batched path only (drop --pipelined)")
    if int(getattr(args, "num_gpus", 1)) != 1:
        raise SystemExit("eval_bg runs one GPU per process (--num_gpus 1); shard tasks across processes instead")
    ctx = _Ctx()
    log_f = open(extra.bg_log, "a") if extra.bg_log else None
    edits = json.load(open(extra.bg_edit_spec)) if extra.bg_edit_spec else []
    file_texts = json.load(open(extra.bg_override_file)) if extra.bg_override_file else {}
    if extra.bg_override == "file" and not file_texts:
        raise SystemExit("--bg_override file needs --bg_override_file")

    from robomme_sim.bg_policy import BGBatchedEvalPolicy

    orig_build = ES.build_policy

    def build_policy(a):
        batched, buffer_factory, normalize_state = orig_build(a)
        cfg = batched.model.config
        cond = extra.condition_span or getattr(cfg, "condition_span", None) or "all"
        budget = a.max_subtask_tokens
        if extra.max_subtask_tokens_auto:
            budget = auto_answer_budget(extra.max_subtask_tokens_auto, batched.processor.tokenizer,
                                        extra.max_subtask_tokens_margin)
            print(f"[eval-bg] max_subtask_tokens auto = {budget}", flush=True)
        bg = BGBatchedEvalPolicy(model=batched.model, processor=batched.processor,
                                 unnormalize_action=batched.unnormalize, normalize_state=batched.normalize_state,
                                 num_denoising_steps=batched.num_denoising_steps,
                                 eval_temperature=batched.eval_temperature, max_subtask_tokens=budget,
                                 device=batched.device, condition_span=cond)
        counter = {"n": 0}

        def tagged_factory():
            pol = buffer_factory()
            pol._bg_slot = counter["n"]
            counter["n"] += 1
            return pol

        bg._reset_counter = counter
        orig_gen = bg.generate_batch

        def generate_batch(processed_list, state_list):
            slots = [p["policy"]._bg_slot for p in processed_list]
            texts = []
            for s, p in zip(slots, processed_list):
                ep = ctx.specs[s]["episode"] if s < len(ctx.specs) else None
                txt = None
                if extra.bg_override == "oracle":
                    tr = ctx.trackers.get(s)
                    if tr is None:
                        raise RuntimeError("oracle override: no tracker for this env (see bg/oracle_robomme.py)")
                    txt = tr.text()
                elif extra.bg_override == "file":
                    txt = file_texts.get(f"{ctx.task}/{ep}", file_texts.get(ctx.task))
                if txt is not None:
                    for e in edits:
                        txt = re.sub(e["pattern"], e["repl"], txt)
                texts.append(txt)
            bg.override_texts = texts if any(t is not None for t in texts) else None
            out = orig_gen(processed_list, state_list)
            if log_f is not None:
                from simplememvla.bg.serialize import try_parse_answer

                with _LOCK:
                    for s, p, inf in zip(slots, processed_list, bg.last_info):
                        _, facts, err = try_parse_answer(inf["answer"])
                        rec = dict(inf, task=ctx.task, slot=s,
                                   episode=ctx.specs[s]["episode"] if s < len(ctx.specs) else None,
                                   decision=ctx.decision,
                                   frame=int(p["policy"]._frames_seen[p["policy"].image_keys[0]]) - 1,
                                   bg_parse_ok=facts is not None, bg_parse_error=err,
                                   cond_norm=float(bg.last_condition[0].float().norm()))
                        log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
            ctx.decision += 1
            return out

        bg.generate_batch = generate_batch
        return bg, tagged_factory, normalize_state

    orig_run_group = ES.run_group

    def run_group(a, pool, batched, buffer_factory, normalize_state, task, specs, **kw):
        ctx.task, ctx.specs, ctx.pool, ctx.decision = task, specs, pool, 0
        batched._reset_counter["n"] = 0
        ctx.trackers = {}
        if extra.bg_override == "oracle":
            from simplememvla.bg.oracle_robomme import attach_online_trackers

            ctx.trackers = attach_online_trackers(pool, specs)
        return orig_run_group(a, pool, batched, buffer_factory, normalize_state, task, specs, **kw)

    ES.build_policy = build_policy
    ES.run_group = run_group
    try:
        ES.main()
    finally:
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
