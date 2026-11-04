"""SimpleMemVLA dataset with an optional Belief Graph in the assistant answer.

With ``bg_mode=none``, ``anchor_stride=1`` and ``holdout_episodes_per_task=0`` every call is forwarded to the
official ``SimpleMemVLADataset`` unchanged (verified sample-by-sample by ``tests/test_bg_identity.py``).
Otherwise the assistant answer becomes ``<sub-task>\\n<BG text>`` (BG_SEPARATOR), the BG text coming from the
sidecar label store keyed by (episode_index, frame_index).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simplememvla.bg import BG_SEPARATOR
from simplememvla.data.dataset import SimpleMemVLADataset


def _scalar(item: dict, key: str) -> int:
    return int(torch.as_tensor(item[key]).reshape(-1)[0])


def episode_task_names(root: str | Path) -> list[str]:
    """LeRobot episode_index -> RoboMME task name, from meta/_conversion_state.json (conversion order)."""
    state = json.loads((Path(root) / "meta" / "_conversion_state.json").read_text())
    return [str(e).split("/")[0] for e in state["done_episodes"]]


def select_holdout(episodes_by_task: dict[str, list[int]], per_task: int, seed: int) -> dict[str, list[int]]:
    """Deterministic per-task holdout (seeded shuffle of the training episode ids; never looks at eval data)."""
    out = {}
    for task in sorted(episodes_by_task):
        eps = sorted(episodes_by_task[task])
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(eps)
        out[task] = sorted(eps[:per_task])
    return out


class BGSimpleMemVLADataset(SimpleMemVLADataset):

    def __init__(self, data_args, spec, output_dir: str | None = None):
        super().__init__(data_args, spec)
        self.bg_mode = getattr(data_args, "bg_mode", "none")
        self.bg_dropout_prob = float(getattr(data_args, "bg_dropout_prob", 0.0))
        self.bg_corrupt_prob = float(getattr(data_args, "bg_corrupt_prob", 0.0))
        self.anchor_stride = int(getattr(data_args, "anchor_stride", 1))
        self.anchor_phase_by_epoch = bool(getattr(data_args, "anchor_phase_by_epoch", False))
        self.holdout_per_task = int(getattr(data_args, "holdout_episodes_per_task", 0))
        self.labels = None
        if self.bg_mode != "none":
            from simplememvla.bg.labels import BGLabelStore

            self.labels = BGLabelStore(data_args.bg_label_dir, bg_filter=data_args.bg_filter)
            src = self.labels.manifest.get("source")
            if src != self.bg_mode:
                raise ValueError(f"bg_mode={self.bg_mode} but labels in {data_args.bg_label_dir} are {src!r}")
        self._phase = 0
        self.holdout: dict[str, list[int]] = {}
        self._base_anchors = None
        self.min_anchor_frame = int(getattr(data_args, "min_anchor_frame", 0))
        self.exclude_episodes = []
        if getattr(data_args, "exclude_episodes_file", None):
            self.exclude_episodes = sorted(int(e) for e in json.loads(Path(data_args.exclude_episodes_file).read_text()))
        if (self.anchor_stride > 1 or self.holdout_per_task > 0 or self.min_anchor_frame > 0
                or self.exclude_episodes):
            self._build_anchor_filter(data_args, output_dir)
        if self.labels is not None:
            bad = set(self.labels.manifest.get("excluded_episodes", [])) - set(self.exclude_episodes)
            if bad:
                ep_all = self._columns(["episode_index"])["episode_index"]
                anchors = self._anchor_indices if self._anchor_indices is not None else np.arange(len(ep_all))
                hit = sorted(set(ep_all[anchors].tolist()) & bad)
                if hit:
                    raise ValueError(
                        f"{len(hit)} anchor episode(s) have no BG label (excluded by the label builder, e.g. "
                        f"{hit[:5]}); pass --exclude_episodes_file {data_args.bg_label_dir}/excluded_ids.json "
                        "(and the same file to the R2 control run)")

    # --------------------------------------------------------------- anchors
    def _columns(self, names):
        hf = self.dataset.reader.hf_dataset
        return {n: np.asarray(hf.data.column(n).to_numpy()).reshape(-1) for n in names}

    def _build_anchor_filter(self, data_args, output_dir):
        cols = self._columns(["episode_index", "frame_index"])
        ep, fr = cols["episode_index"], cols["frame_index"]
        base = self._anchor_indices if self._anchor_indices is not None else np.arange(len(ep))
        keep = np.ones(len(base), dtype=bool)
        if self.min_anchor_frame > 0:
            keep &= fr[base] >= self.min_anchor_frame
        if self.exclude_episodes:
            keep &= ~np.isin(ep[base], np.asarray(self.exclude_episodes, dtype=np.int64))
        if self.holdout_per_task > 0:
            names = episode_task_names(self.root)
            by_task: dict[str, list[int]] = {}
            for e in np.unique(ep):
                by_task.setdefault(names[int(e)], []).append(int(e))
            self.holdout = select_holdout(by_task, self.holdout_per_task, int(data_args.holdout_seed))
            held = np.array(sorted(e for v in self.holdout.values() for e in v), dtype=np.int64)
            keep &= ~np.isin(ep[base], held)
            if output_dir:
                p = Path(output_dir) / "holdout_episodes.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({"seed": int(data_args.holdout_seed), "per_task": self.holdout_per_task,
                                           "episodes": self.holdout}, indent=1, sort_keys=True))
                tmp.replace(p)
        self._base_anchors = base[keep]
        self._base_frames = fr[self._base_anchors]
        if self.anchor_stride > 1:
            per_phase = [self._base_anchors[self._base_frames % self.anchor_stride == p]
                         for p in range(self.anchor_stride)]
            # every phase exposes the same length so the sampler's permutation size never changes
            self._phase_len = min(len(a) for a in per_phase)
            self._per_phase = [a[: self._phase_len] for a in per_phase]
        else:
            self._phase_len = len(self._base_anchors)
            self._per_phase = [self._base_anchors]
        self._anchor_indices = self._per_phase[0]

    def set_epoch(self, epoch: int) -> None:
        if self.anchor_stride > 1 and self.anchor_phase_by_epoch:
            self._phase = int(epoch) % self.anchor_stride
            self._anchor_indices = self._per_phase[self._phase]

    # --------------------------------------------------------------- answer
    def _bg_text(self, item: dict[str, Any]) -> str | None:
        ep, fr = _scalar(item, "episode_index"), _scalar(item, "frame_index")
        row = self.labels.row(ep, fr)
        if self.bg_dropout_prob and random.random() < self.bg_dropout_prob:
            return None
        if self.bg_corrupt_prob and random.random() < self.bg_corrupt_prob:
            pool = self.labels.rows_of_task(self.labels.task[row])
            other = int(pool[random.randrange(len(pool))])
            return str(self.labels.text[other])
        return str(self.labels.text[row])

    def _get_subtask(self, item: dict[str, Any]) -> str:
        subtask = super()._get_subtask(item)
        if self.bg_mode == "none":
            return subtask
        bg = self._bg_text(item)
        return subtask if bg is None else f"{subtask}{BG_SEPARATOR}{bg}"
