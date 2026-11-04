"""BG label sidecar: ``<bg_label_dir>/labels.parquet`` + ``manifest.json``.

labels.parquet columns
    episode_index  int64   LeRobot episode index
    frame_index    int64   LeRobot frame index within the episode
    task           string  RoboMME task name (env id)
    bg_json        string  BeliefGraph JSON (schema.BeliefGraph.to_json)
    bg_mnemonic    string  serialize(bg, "mnemonic", K)
    bg_full        string  serialize(bg, "full", K)
manifest.json
    source (oracle|unsup), max_interactions K, label_hash (sha256 of the parquet bytes), builder version, notes

The label hash is computed by ``write_labels`` and checked by ``BGLabelStore`` so a run always records exactly
which labels it trained on.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

LABEL_FILE = "labels.parquet"
MANIFEST = "manifest.json"


def file_sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_labels(out_dir: str | os.PathLike, rows: list[dict], manifest: dict) -> str:
    """Atomically write labels.parquet + manifest.json; returns the label hash. Refuses to overwrite."""
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / LABEL_FILE
    if final.exists():
        raise FileExistsError(f"{final} exists; labels are immutable (write a new version dir)")
    df = pd.DataFrame(rows).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    tmp = out / (LABEL_FILE + ".tmp")
    df.to_parquet(tmp, index=False)
    digest = file_sha256(tmp)
    os.replace(tmp, final)
    man = dict(manifest)
    man.update(label_hash=digest, n_rows=int(len(df)), n_episodes=int(df["episode_index"].nunique()))
    tmpm = out / (MANIFEST + ".tmp")
    tmpm.write_text(json.dumps(man, indent=2, sort_keys=True))
    os.replace(tmpm, out / MANIFEST)
    return digest


class BGLabelStore:
    def __init__(self, label_dir: str, bg_filter: str = "mnemonic", verify_hash: bool = True):
        import pandas as pd

        d = Path(label_dir)
        self.manifest = json.loads((d / MANIFEST).read_text())
        path = d / LABEL_FILE
        if verify_hash:
            got = file_sha256(path)
            if got != self.manifest["label_hash"]:
                raise ValueError(f"label hash mismatch for {path}: {got} != {self.manifest['label_hash']}")
        self.label_hash = self.manifest["label_hash"]
        col = {"mnemonic": "bg_mnemonic", "full": "bg_full"}[bg_filter]
        df = pd.read_parquet(path, columns=["episode_index", "frame_index", "task", col])
        self.ep = df["episode_index"].to_numpy(np.int64)
        self.fr = df["frame_index"].to_numpy(np.int64)
        self.task = df["task"].to_numpy(object)
        self.text = df[col].to_numpy(object)
        # (episode, frame) -> row via a sorted int64 key
        self.key = self.ep * 1_000_000 + self.fr
        order = np.argsort(self.key, kind="stable")
        self.key, self.ep, self.fr, self.task, self.text = (
            self.key[order], self.ep[order], self.fr[order], self.task[order], self.text[order])
        self._by_task: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.key)

    def row(self, episode_index: int, frame_index: int) -> int:
        k = int(episode_index) * 1_000_000 + int(frame_index)
        i = int(np.searchsorted(self.key, k))
        if i >= len(self.key) or int(self.key[i]) != k:
            raise KeyError(f"no BG label for episode {episode_index} frame {frame_index}")
        return i

    def text_at(self, episode_index: int, frame_index: int) -> str:
        return str(self.text[self.row(episode_index, frame_index)])

    def rows_of_task(self, task: str) -> np.ndarray:
        if task not in self._by_task:
            self._by_task[task] = np.nonzero(self.task == task)[0]
        return self._by_task[task]
