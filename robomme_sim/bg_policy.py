"""BG-aware batched RoboMME policy (branch bgmem). Default (non-pipelined) path only.

``BGBatchedEvalPolicy.generate_batch`` is the official ``BatchedEvalPolicy.generate_batch`` with four additions:

1. Timing per decision (CUDA events): prefill (vision features + prompt prefill), decode (answer tokens) and
   denoise (action head), plus answer token counts.
2. BG override (teacher forcing): for rows with an override text the sub-task is generated normally; as soon as
   the model emits the separator (first token whose decoded text contains "\\n"), the remaining answer is
   replaced by the forced tokens ``tokenize(<sub-task>\\n<BG>)[len(tokenize(<sub-task>)):] + <|im_end|>``,
   fed through the same KV-cached decode loop so the condition hidden states are those of the forced text.
3. ``condition_span=subtask_only``: the action head reads only the sub-task tokens (separator and BG excluded),
   matching the training ablation.
4. Runaway count: rows that hit ``max_subtask_tokens`` without <|im_end|>.

With no override, ``condition_span=all`` and the same ``max_subtask_tokens`` the decode / conditioning is the
official one token for token; the extra work is only timing and bookkeeping.
``self.last_info`` holds one dict per row of the last call (answer text, sub-task, BG, token counts, timings,
forced flag, runaway flag).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from robomme_sim.batched_policy import BatchedEvalPolicy
from simplememvla.bg import BG_PREFIX
from simplememvla.bg.serialize import split_answer


class _Timer:
    def __init__(self, device):
        self.cuda = torch.cuda.is_available() and getattr(device, "type", str(device)) != "cpu"
        self.marks: dict[str, float | torch.cuda.Event] = {}

    def mark(self, name):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.marks[name] = ev
        else:
            self.marks[name] = time.perf_counter()

    def elapsed(self, a, b) -> float:
        if self.cuda:
            torch.cuda.synchronize()
            return float(self.marks[a].elapsed_time(self.marks[b])) / 1000.0
        return float(self.marks[b] - self.marks[a])


class BGBatchedEvalPolicy(BatchedEvalPolicy):

    def __init__(self, *args, condition_span: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        if condition_span not in ("all", "subtask_only"):
            raise ValueError(f"condition_span {condition_span!r}")
        self.condition_span = condition_span
        self.override_texts: list[str | None] | None = None  # set by the caller per call (row order)
        self.last_info: list[dict] = []
        self._nl_cache: dict[int, bool] = {}

    def _has_newline(self, tid: int) -> bool:
        v = self._nl_cache.get(tid)
        if v is None:
            v = "\n" in self.processor.tokenizer.decode([tid])
            self._nl_cache[tid] = v
        return v

    def _forced_tail(self, sub_ids: list[int], bg_text: str) -> list[int]:
        tok = self.processor.tokenizer
        sub_text = tok.decode(sub_ids, skip_special_tokens=True).rstrip("\n")
        full = tok.encode(f"{sub_text}\n{bg_text}", add_special_tokens=False)
        head = tok.encode(sub_text, add_special_tokens=False)
        if full[: len(head)] == head:
            tail = full[len(head):]
        else:  # merge across the boundary: fall back to separately tokenized pieces
            tail = tok.encode("\n", add_special_tokens=False) + tok.encode(bg_text, add_special_tokens=False)
        return tail + [self.im_end_id]

    @torch.no_grad()
    def generate_batch(self, processed_list: list[dict], state_norm_list: list) -> list[tuple[np.ndarray, str]]:
        from simplememvla.data.collator import subtask_span_mask

        self.model.eval()
        N = len(processed_list)
        dev = self.device
        tok = self.processor.tokenizer
        pad = self.pad_token_id
        overrides = list(self.override_texts) if self.override_texts is not None else [None] * N
        if len(overrides) != N:
            raise ValueError(f"override_texts has {len(overrides)} rows, batch has {N}")
        timer = _Timer(dev)
        timer.mark("t0")

        qwen = self.model.backbone.model
        lm = qwen.language_model
        lm_head = self.model.backbone.get_output_embeddings()
        embed = qwen.get_input_embeddings()
        video_token_id = int(qwen.config.video_token_id)
        image_token_id = int(qwen.config.image_token_id)

        miss = []
        for p in processed_list:
            pol = p["policy"]
            for k, rows in p["video_patch_pixels"].items():
                if k not in pol._patch_feat_cache:
                    miss.append((pol, k, rows))
        if miss:
            gh, gw = processed_list[0]["policy"]._prompt_cache["grid_hw"]
            pixels = torch.cat([rows for _, _, rows in miss], dim=0).to(dev)
            grid = torch.tensor([[1, gh, gw]] * len(miss), device=dev, dtype=torch.long)
            feats = qwen.get_video_features(pixels, grid, return_dict=True).pooler_output
            for (pol, k, _), f in zip(miss, feats):
                pol._patch_feat_cache[k] = f

        has_images = "pixel_values" in processed_list[0]
        img_feats_per_env = [None] * N
        img_grids_cat = None
        if has_images:
            pix = torch.cat([p["pixel_values"] for p in processed_list], dim=0).to(dev)
            img_grids_cat = torch.cat([p["image_grid_thw"] for p in processed_list], dim=0).to(dev)
            feats = qwen.get_image_features(pix, img_grids_cat, return_dict=True).pooler_output
            idx = 0
            for i, p in enumerate(processed_list):
                n_img = p["image_grid_thw"].shape[0]
                img_feats_per_env[i] = torch.cat(feats[idx: idx + n_img], dim=0)
                idx += n_img

        maxP = max(p["input_ids"].shape[1] for p in processed_list)
        iid = torch.full((N, maxP), pad, dtype=torch.long, device=dev)
        att = torch.zeros((N, maxP), dtype=torch.long, device=dev)
        mmtt = torch.zeros((N, maxP), dtype=torch.long, device=dev)
        prompt_lens = []
        for i, p in enumerate(processed_list):
            L = p["input_ids"].shape[1]
            iid[i, maxP - L:] = p["input_ids"][0].to(dev)
            att[i, maxP - L:] = 1
            mmtt[i, maxP - L:] = p["mm_token_type_ids"][0].to(dev)
            prompt_lens.append(L)
        emb = embed(iid)
        for i, p in enumerate(processed_list):
            pol = p["policy"]
            vid_feat = torch.cat([pol._patch_feat_cache[k] for k in p["video_patch_keys"]], dim=0)
            vmask = iid[i] == video_token_id
            if int(vmask.sum()) != vid_feat.shape[0]:
                raise RuntimeError("video token count != cached features")
            emb[i][vmask] = vid_feat.to(emb.dtype)
            if has_images:
                imask = iid[i] == image_token_id
                emb[i][imask] = img_feats_per_env[i].to(emb.dtype)

        grids_cat = torch.cat([p["video_grid_thw"] for p in processed_list], dim=0).to(dev)
        position_ids, rope_deltas = qwen.get_rope_index(
            iid, image_grid_thw=img_grids_cat, video_grid_thw=grids_cat, attention_mask=att,
            mm_token_type_ids=mmtt)
        out = lm(inputs_embeds=emb, attention_mask=att, position_ids=position_ids, use_cache=True)
        cache = out.past_key_values
        next_tok = lm_head(out.last_hidden_state[:, -1]).float().argmax(-1)
        timer.mark("prefill")

        max_new = self.max_subtask_tokens
        sampled = [[int(next_tok[i])] for i in range(N)]
        eos_seen = [sampled[i][0] == self.im_end_id for i in range(N)]
        forced = [False] * N
        sub_len = [None] * N          # number of generated tokens before the forced tail
        newline_fed = [False] * N
        runaway = [False] * N
        dec_hidden = []
        cur_att = att
        valid_len = att.sum(-1)
        delta = rope_deltas[:, 0].to(dev)

        def maybe_force(i: int) -> None:
            if overrides[i] is None or forced[i] or eos_seen[i]:
                return
            if self._has_newline(sampled[i][-1]):
                head = sampled[i][:-1]
                sub_len[i] = len(head)
                sampled[i] = head + self._forced_tail(head, overrides[i])
                forced[i] = True
                eos_seen[i] = True

        for i in range(N):
            maybe_force(i)

        def budget(i: int) -> int:
            return len(sampled[i]) if forced[i] else max_new

        def row_done(i, steps_done):
            if eos_seen[i]:
                return newline_fed[i] and steps_done >= len(sampled[i]) + 1
            return len(sampled[i]) >= max_new and steps_done >= len(sampled[i])

        step = 0
        while True:
            feed = torch.full((N, 1), pad, dtype=torch.long, device=dev)
            for i in range(N):
                if step < len(sampled[i]):
                    feed[i, 0] = sampled[i][step]
                elif eos_seen[i] and not newline_fed[i] and self.newline_id is not None:
                    feed[i, 0] = self.newline_id
                    newline_fed[i] = True
            cur_att = torch.cat([cur_att, torch.ones((N, 1), dtype=cur_att.dtype, device=dev)], dim=-1)
            valid_len = valid_len + 1
            pos = (delta + valid_len - 1).view(1, N, 1).expand(3, N, 1)
            out = lm(input_ids=feed, attention_mask=cur_att, position_ids=pos, past_key_values=cache,
                     use_cache=True)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]
            dec_hidden.append(h)
            nxt = lm_head(h).float().argmax(-1)
            for i in range(N):
                if eos_seen[i] or step != len(sampled[i]) - 1:
                    continue
                if len(sampled[i]) < max_new:
                    t = int(nxt[i])
                    sampled[i].append(t)
                    if t == self.im_end_id:
                        eos_seen[i] = True
                    else:
                        maybe_force(i)
                else:
                    sampled[i].append(self.im_end_id)
                    eos_seen[i] = True
                    runaway[i] = True
            step += 1
            if all(row_done(i, step) for i in range(N)):
                break
            if step >= max(budget(i) for i in range(N)) + 2:  # official: range(max_new + 2)
                break
        timer.mark("decode")

        full_ids_list, mmtt_full_list, span_list, text_list, gen_lens = [], [], [], [], []
        for i in range(N):
            prompt_ids = processed_list[i]["input_ids"][0].to(dev)
            gen_ids = torch.tensor(sampled[i], dtype=prompt_ids.dtype, device=dev)
            full = torch.cat([prompt_ids, gen_ids])
            if self.newline_id is not None and gen_ids.numel() > 0 and int(full[-1]) == self.im_end_id:
                full = torch.cat([full, torch.tensor([self.newline_id], device=dev, dtype=full.dtype)])
            full_ids_list.append(full)
            gen_lens.append(full.shape[0] - prompt_lens[i])
            mmtt_full_list.append(torch.cat([
                processed_list[i]["mm_token_type_ids"][0].to(dev),
                torch.zeros(full.shape[0] - prompt_lens[i], dtype=mmtt.dtype, device=dev)]))
            span = subtask_span_mask(full, self.think_close_id, self.whitespace_ids).to(dev)
            if self.condition_span == "subtask_only":
                span = self._subtask_only_span(full, span, prompt_lens[i])
            span_list.append(span)
            text_list.append(tok.decode(gen_ids, skip_special_tokens=True).strip())

        maxF = max(f.shape[0] for f in full_ids_list)
        fiid = torch.full((N, maxF), pad, dtype=torch.long, device=dev)
        fatt = torch.zeros((N, maxF), dtype=torch.long, device=dev)
        fmmtt = torch.zeros((N, maxF), dtype=torch.long, device=dev)
        fspan = torch.zeros((N, maxF), dtype=torch.bool, device=dev)
        for i in range(N):
            Lf = full_ids_list[i].shape[0]
            fiid[i, :Lf] = full_ids_list[i]
            fatt[i, :Lf] = 1
            fmmtt[i, :Lf] = mmtt_full_list[i]
            fspan[i, :Lf] = span_list[i]
            if bool(span_list[i][: prompt_lens[i]].any()):
                raise RuntimeError("subtask span leaked into the prompt region")
        mm_full = {"image_grid_thw": img_grids_cat, "video_grid_thw": grids_cat, "mm_token_type_ids": fmmtt}
        position_ids_full = self._position_ids(fiid, fatt, mm_full)
        dec_stack = torch.stack(dec_hidden, dim=1)
        hidden_full = dec_stack.new_zeros(N, maxF, dec_stack.shape[-1])
        for i in range(N):
            L, g = prompt_lens[i], gen_lens[i]
            if g > dec_stack.shape[1]:
                raise RuntimeError("decode hidden capture shorter than generated span")
            hidden_full[i, L: L + g] = dec_stack[i, :g]
        cond_hidden, cond_token, cond_mask, cond_pos = self._gather_condition(
            hidden_states=hidden_full, input_ids=fiid, span=fspan, position_ids=position_ids_full,
            attention_mask=fatt)
        self.last_condition = (cond_hidden.detach(), cond_mask.detach())

        state_batch = None
        if self.use_proprio:
            state_batch = torch.stack([s[0] if s.dim() == 2 else s for s in state_norm_list]).to(dev)
        actions_b = self.model.action_head.sample(
            cond_hidden, condition_position_ids=cond_pos, condition_attention_mask=cond_mask,
            condition_token_embeds=cond_token, num_steps=self.num_denoising_steps,
            temperature=self.eval_temperature, state=state_batch)
        timer.mark("denoise")
        t_pre, t_dec, t_den = (timer.elapsed("t0", "prefill"), timer.elapsed("prefill", "decode"),
                               timer.elapsed("decode", "denoise"))

        results, info = [], []
        for i in range(N):
            norm_a = actions_b[i, :, : self.action_dim].float().cpu()
            actions_unnorm = self.unnormalize.unnormalize(norm_a).numpy().astype(np.float32)
            sub, bg = split_answer(text_list[i])
            results.append((actions_unnorm, text_list[i]))
            info.append({
                "answer": text_list[i], "subtask": sub, "bg": bg, "forced": forced[i],
                "override": overrides[i], "runaway": runaway[i],
                "answer_tokens": len(sampled[i]), "subtask_tokens": sub_len[i],
                "cond_tokens": int(span_list[i].sum()), "prompt_tokens": prompt_lens[i],
                "t_prefill": t_pre, "t_decode": t_dec, "t_denoise": t_den, "batch": N,
            })
        if any(runaway):
            print(f"[eval-bg] WARNING: {sum(runaway)}/{N} row(s) hit the {max_new}-token answer budget", flush=True)
        self.last_info = info
        return results

    def _subtask_only_span(self, full: torch.Tensor, span: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Keep the answer tokens before the separator newline that precedes "Belief:"."""
        idx = span.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return span
        tok = self.processor.tokenizer
        ids = full.tolist()
        out = torch.zeros_like(span)
        for j in idx.tolist():
            if self._has_newline(ids[j]) or ids[j] == self.im_end_id:
                break
            out[j] = True
        if not bool(out.any()):  # degenerate: no sub-task text, fall back to the full span
            return span
        _ = tok, BG_PREFIX
        return out
