"""Generic manifest-driven audio classification datasets.

Every benchmark in the paper is exposed through one uniform interface. A
dataset is described by a manifest CSV with columns::

    filepath,label[,fold]

* single-label tasks (P.A.S.T., WMM, ESC-50, SPC-2): ``label`` is the class
  name or integer id.
* multi-label tasks (AS-20K, BirdSet): ``label`` is a ``|``-separated list of
  integer label ids (e.g. ``3|17|204``).
* ESC-50 optionally carries a ``fold`` column for 5-fold cross-validation.

Each item returns a dict with the *raw* fbank (physical log-Mel magnitudes,
needed by energy-based masking), the normalized fbank (encoder input), and the
target vector.
"""

import csv
import random
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..utils.audio import extract_fbank, load_waveform, normalize_fbank


def read_manifest(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def write_manifest(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_label_maps(rows: Sequence[dict], multilabel: bool):
    if multilabel:
        ids = set()
        for r in rows:
            ids.update(int(x) for x in str(r["label"]).split("|"))
        return {}, max(ids) + 1
    names = sorted({r["label"] for r in rows})
    return {n: i for i, n in enumerate(names)}, len(names)


class AudioClassificationDataset(Dataset):
    """Unified classification dataset driven by a manifest CSV.

    Args:
        manifest: path to CSV (or a pre-loaded row list).
        multilabel: parse ``label`` as ``|``-separated ids and emit a multi-hot
            target (used for AS-20K / BirdSet).
        num_classes: override the inferred class count.
        training: random time crop (train) vs center crop (eval).
        root: optional root directory prepended to manifest filepaths.
        fold: if the manifest has a ``fold`` column, keep only this fold
            (ESC-50 5-fold protocol).
    """

    def __init__(
        self,
        manifest,
        multilabel: bool = False,
        num_classes: Optional[int] = None,
        training: bool = True,
        root: str = "",
        fold: Optional[int] = None,
    ) -> None:
        self.rows = read_manifest(manifest) if isinstance(manifest, str) else list(manifest)
        if fold is not None and "fold" in self.rows[0]:
            self.rows = [r for r in self.rows if int(r["fold"]) == fold]
        self.root = Path(root)
        self.multilabel = multilabel
        self.label_map, inferred = build_label_maps(self.rows, multilabel)
        self.num_classes = num_classes or inferred
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def target_of(self, row: dict) -> torch.Tensor:
        if self.multilabel:
            t = torch.zeros(self.num_classes)
            for x in str(row["label"]).split("|"):
                t[int(x)] = 1.0
            return t
        return torch.tensor(int(row["label"]) if str(row["label"]).isdigit() else self.label_map[row["label"]])

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        path = self.root / row["filepath"] if self.root else Path(row["filepath"])
        wav = load_waveform(str(path))
        raw = extract_fbank(wav, training=self.training)
        return {"raw": raw, "fbank": normalize_fbank(raw), "target": self.target_of(row)}


class UnlabeledAudioDataset(Dataset):
    """Unlabeled pool for self-supervised pre-training (P.A.S.T. 30 h)."""

    def __init__(self, manifest, training: bool = True, root: str = "") -> None:
        self.rows = read_manifest(manifest) if isinstance(manifest, str) else list(manifest)
        self.root = Path(root)
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        path = self.root / self.rows[idx]["filepath"] if self.root else Path(self.rows[idx]["filepath"])
        wav = load_waveform(str(path))
        raw = extract_fbank(wav, training=self.training)
        return {"raw": raw, "fbank": normalize_fbank(raw)}


class KShotDataset(AudioClassificationDataset):
    """Few-shot subset: at most ``shots`` instances per class.

    Reproduces the few-shot protocol of Fig. 3 (5 / 25 / 50 / 100 shots per
    class, plus "All"). The sampler is seeded so that every masking strategy
    is evaluated on identical shot sets.
    """

    def __init__(self, manifest, shots: int, seed: int = 42, **kwargs) -> None:
        super().__init__(manifest, **kwargs)
        rng = random.Random(seed)
        by_class = {}
        for i, row in enumerate(self.rows):
            by_class.setdefault(self.target_of(row).item(), []).append(i)
        chosen = []
        for _, pool in sorted(by_class.items()):
            rng.shuffle(pool)
            chosen.extend(pool[:shots])
        chosen.sort()
        self.rows = [self.rows[i] for i in chosen]
