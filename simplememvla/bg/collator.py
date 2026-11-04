"""Collator for BG answers: official encoding + optional condition / BG masks.

Answer span (official ``subtask_span_mask``: everything after ``</think>`` and its whitespace, i.e. the answer,
``<|im_end|>`` and the trailing newline) is split at the first token of the ``Belief:`` prefix:

    [ sub-task tokens + separator "\\n" ][ "Belief: ..." tokens + <|im_end|> + "\\n" ]
      subtask part                         BG part (bg_label_mask)

* ``condition_span=all``          condition mask = labels != -100 (official)
* ``condition_span=subtask_only`` condition mask = sub-task tokens only (separator excluded)
* ``vl_loss_reduction=span_mean`` uses ``bg_label_mask`` to average the two CE parts separately

With bg_mode=none (no BG in any answer) and the default options the batch is exactly the official one: no key
is added.
"""
from __future__ import annotations

from typing import Any

import torch

from simplememvla.bg import BG_PREFIX
from simplememvla.data.collator import SimpleMemVLADataCollator


def find_subsequence(seq: torch.Tensor, sub: list[int], start: int = 0) -> int:
    n, m = int(seq.shape[0]), len(sub)
    if m == 0:
        return -1
    s = seq.tolist()
    for i in range(start, n - m + 1):
        if s[i:i + m] == sub:
            return i
    return -1


class BGDataCollator(SimpleMemVLADataCollator):

    def __init__(self, processor, max_length: int | None = None, condition_span: str = "all",
                 emit_bg_masks: bool = False):
        super().__init__(processor, max_length=max_length)
        self.condition_span = condition_span
        self.emit_bg_masks = bool(emit_bg_masks) or condition_span != "all"
        self.bg_prefix_ids = self.tokenizer.encode(BG_PREFIX, add_special_tokens=False)
        nl = self.tokenizer.encode("\n", add_special_tokens=False)
        self.newline_id = nl[0] if len(nl) == 1 else None

    def bg_split(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(subtask_mask, bg_mask) inside the answer span; bg_mask empty when the answer has no BG."""
        span = labels.ne(-100)
        idx = span.nonzero(as_tuple=True)[0]
        sub = torch.zeros_like(span)
        bg = torch.zeros_like(span)
        if idx.numel() == 0:
            return sub, bg
        start = int(idx[0])
        pos = find_subsequence(input_ids, self.bg_prefix_ids, start)
        if pos < 0:
            sub[span] = True
            return sub, bg
        bg[pos:] = span[pos:]
        sub[start:pos] = True
        return sub, bg

    def _encode_one(self, feature: dict[str, Any]) -> dict[str, torch.Tensor]:
        out = super()._encode_one(feature)
        if self.emit_bg_masks:
            sub, bg = self.bg_split(out["input_ids"], out["labels"])
            if self.condition_span == "subtask_only":
                cond = sub.clone()
                # drop the separator newline that precedes "Belief:" (it belongs to neither sentence)
                last = cond.nonzero(as_tuple=True)[0]
                if bg.any() and last.numel() and self.newline_id is not None \
                        and int(out["input_ids"][int(last[-1])]) == self.newline_id:
                    cond[int(last[-1])] = False
            else:
                cond = out["labels"].ne(-100)
            out["condition_mask"] = cond
            out["bg_label_mask"] = bg
        return out

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not self.emit_bg_masks:
            return super().__call__(features)
        # encode once, then reuse the official padding code path for the shared keys
        encoded = [self._encode_one(f) for f in features]
        cache = iter(encoded)
        self._encode_one = lambda _f: next(cache)  # type: ignore[method-assign]  (instance shadow)
        try:
            batch = super().__call__(features)
        finally:
            del self._encode_one  # back to the class method
        for key in ("condition_mask", "bg_label_mask"):
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [e[key] for e in encoded], batch_first=True, padding_value=False)
        return batch
