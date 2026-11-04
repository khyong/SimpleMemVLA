"""Argument dataclasses for BG training. Every default reproduces the official behaviour exactly."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from configs.sft_params import DataArguments


@dataclass
class BGDataArguments(DataArguments):
    bg_mode: str = field(default="none", metadata={"help": "none | unsup | oracle"})
    bg_label_dir: Optional[str] = field(
        default=None, metadata={"help": "dir with labels.parquet (episode_index, frame_index -> BG) + manifest.json"})
    bg_filter: str = field(default="mnemonic", metadata={"help": "mnemonic | full"})
    bg_max_interactions: int = field(default=4)
    bg_dropout_prob: float = field(default=0.0, metadata={"help": "answer = sub-task only with this probability"})
    bg_corrupt_prob: float = field(
        default=0.0, metadata={"help": "BG replaced by the BG of another frame of the same task with this probability"})
    condition_span: str = field(default="all", metadata={"help": "all | subtask_only (action head condition span)"})
    vl_loss_reduction: str = field(default="token_mean", metadata={"help": "token_mean (official) | span_mean"})
    anchor_stride: int = field(default=1, metadata={"help": "train only on frames with frame_index % stride == phase"})
    anchor_phase_by_epoch: bool = field(default=False, metadata={"help": "phase = epoch % stride (else 0)"})
    holdout_episodes_per_task: int = field(default=0)
    holdout_seed: int = field(default=20261104)
    exclude_episodes_file: Optional[str] = field(
        default=None, metadata={"help": "JSON list of episode indices removed from the anchors in EVERY bg_mode "
                                        "(oracle replay failures; keeps R2/R3/R4 on identical data)"})
    min_anchor_frame: int = field(default=0, metadata={"help": "test knob: anchors with frame_index >= this only "
                                                             "(full-history samples for the memory test)"})

    def bg_validate(self) -> None:
        from simplememvla.bg import BG_FILTERS, BG_MODES, CONDITION_SPANS, VL_LOSS_REDUCTIONS

        if self.bg_mode not in BG_MODES:
            raise ValueError(f"bg_mode {self.bg_mode!r} not in {BG_MODES}")
        if self.bg_filter not in BG_FILTERS:
            raise ValueError(f"bg_filter {self.bg_filter!r} not in {BG_FILTERS}")
        if self.condition_span not in CONDITION_SPANS:
            raise ValueError(f"condition_span {self.condition_span!r} not in {CONDITION_SPANS}")
        if self.vl_loss_reduction not in VL_LOSS_REDUCTIONS:
            raise ValueError(f"vl_loss_reduction {self.vl_loss_reduction!r} not in {VL_LOSS_REDUCTIONS}")
        if self.bg_mode != "none" and not self.bg_label_dir:
            raise ValueError("bg_mode != none requires --bg_label_dir")
        if self.bg_mode == "none" and (self.bg_dropout_prob or self.bg_corrupt_prob):
            raise ValueError("bg_dropout_prob / bg_corrupt_prob need bg_mode != none")
        if self.bg_mode == "none" and (self.condition_span != "all" or self.vl_loss_reduction != "token_mean"):
            raise ValueError("condition_span / vl_loss_reduction ablations need a BG span (bg_mode != none)")
        if self.anchor_stride < 1:
            raise ValueError("anchor_stride must be >= 1")
        for p in ("bg_dropout_prob", "bg_corrupt_prob"):
            if not 0.0 <= getattr(self, p) <= 1.0:
                raise ValueError(f"{p} must be in [0, 1]")
