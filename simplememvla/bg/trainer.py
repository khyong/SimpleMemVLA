"""Trainer that additionally logs the sub-task CE and the BG CE (window means, rank 0)."""
from __future__ import annotations

import time

import torch

from simplememvla.training.sft_trainer import SimpleMemVLATrainer


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


class BGTrainer(SimpleMemVLATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bg_sums: dict[str, float] = {}
        self._bg_counts: dict[str, int] = {}
        self._t_last, self._step_last = None, 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if self.args.process_index == 0 and model.training:
            for k, v in getattr(_unwrap(model), "bg_ce_parts", {}).items():
                self._bg_sums[k] = self._bg_sums.get(k, 0.0) + v
                self._bg_counts[k] = self._bg_counts.get(k, 0) + 1
            _unwrap(model).bg_ce_parts = {}
        return out

    def log(self, logs: dict, *args, **kwargs):
        if "loss" in logs and self._bg_counts:
            logs = {**logs, **{k: self._bg_sums[k] / self._bg_counts[k] for k in self._bg_sums}}
            self._bg_sums, self._bg_counts = {}, {}
        if "loss" in logs and torch.cuda.is_available():
            now, step = time.time(), self.state.global_step
            logs["mem/peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
            logs["mem/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            if self._t_last is not None and step > self._step_last:
                bs = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
                logs["perf/samples_per_s"] = (step - self._step_last) * bs / (now - self._t_last)
            self._t_last, self._step_last = now, step
        return super().log(logs, *args, **kwargs)
