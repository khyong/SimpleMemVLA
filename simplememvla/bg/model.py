"""SimpleMemVLA with BG-aware training forward.

``forward`` additionally accepts
    condition_mask   (B, L) bool  tokens the action head reads; default (None) = labels != -100 (official)
    bg_label_mask    (B, L) bool  BG tokens inside the answer span; enables separate sub-task / BG CE logging
and the config field ``vl_loss_reduction`` (token_mean = official; span_mean = mean CE of the sub-task tokens +
mean CE of the BG tokens). Without the two masks the computation is the official one, op for op.
The per-part CE values of the last forward are kept in ``self.bg_ce_parts`` (detached floats) for logging.
"""
from __future__ import annotations

import torch

from simplememvla.model.modeling_simplememvla import SimpleMemVLAForActionPrediction


class BGSimpleMemVLAForActionPrediction(SimpleMemVLAForActionPrediction):

    def __init__(self, config, backbone=None):
        super().__init__(config, backbone=backbone)
        self.vl_loss_reduction = getattr(config, "vl_loss_reduction", "token_mean")
        self._cond_override: torch.Tensor | None = None
        self._bg_mask: torch.Tensor | None = None
        self.bg_ce_parts: dict[str, float] = {}

    # the official forward computes span = labels.ne(-100); swap in the condition mask when given
    def _gather_condition(self, hidden_states, input_ids, span, position_ids, attention_mask):
        if self._cond_override is not None:
            span = self._cond_override
        return super()._gather_condition(hidden_states, input_ids, span, position_ids, attention_mask)

    def _sparse_ce_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._bg_mask is None:
            return super()._sparse_ce_loss(hidden_states, labels)
        lm_head = self.backbone.get_output_embeddings()
        if lm_head is None:
            lm_head = self.backbone.lm_head
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:].to(shift_hidden.device)
        mask = shift_labels != -100
        if not bool(mask.any()):
            return (hidden_states.sum() + lm_head.weight.sum()) * 0.0
        bg = self._bg_mask[:, 1:].to(shift_hidden.device) & mask
        sub = mask & ~bg
        logits = lm_head(shift_hidden[mask]).float()
        tok = torch.nn.functional.cross_entropy(logits, shift_labels[mask], reduction="none")
        is_bg = bg[mask]
        parts = {}
        sub_ce = tok[~is_bg].mean() if bool((~is_bg).any()) else None
        bg_ce = tok[is_bg].mean() if bool(is_bg.any()) else None
        if sub_ce is not None:
            parts["subtask_ce"] = float(sub_ce.detach())
        if bg_ce is not None:
            parts["bg_ce"] = float(bg_ce.detach())
        parts["bg_tokens"] = float(is_bg.sum())
        self.bg_ce_parts = parts
        if self.vl_loss_reduction == "span_mean":
            total = tok.new_zeros(())
            for v in (sub_ce, bg_ce):
                if v is not None:
                    total = total + v
            return total
        return tok.mean()

    def forward(self, input_ids, attention_mask, position_ids=None, actions=None, action_mask=None,
                labels=None, state=None, condition_mask=None, bg_label_mask=None, **kwargs):
        self._cond_override = condition_mask
        self._bg_mask = bg_label_mask
        try:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                   actions=actions, action_mask=action_mask, labels=labels, state=state, **kwargs)
        finally:
            self._cond_override = None
            self._bg_mask = None
