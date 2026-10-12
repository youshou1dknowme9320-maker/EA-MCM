"""Evaluation metrics used in the paper.

* ``accuracy``            - top-1 accuracy (P.A.S.T., ESC-50, SPC-2, WMM).
* ``macro_f1``            - macro-averaged F1 (WMM headline metric, P.A.S.T.
                            ablation tables).
* ``mAP``                 - multi-label mean average precision (AS-20K).
* ``cmAP``                - clip-wise mean average precision (BirdSet): AP is
                            computed per class over clip-level binary labels
                            and averaged, i.e. the standard multi-label mAP at
                            clip granularity.
* ``min_class_recall``    - recall of the rarest class ("Min-Rec" in the
                            long-tail analysis).
"""

from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item() * 100.0


@torch.no_grad()
def macro_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    classes = targets.unique()
    f1s = []
    for c in classes.tolist():
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-12))
    return float(np.mean(f1s)) * 100.0


@torch.no_grad()
def min_class_recall(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    recalls = []
    for c in targets.unique().tolist():
        mask = targets == c
        recalls.append(((preds == c) & mask).sum().item() / (mask.sum().item() + 1e-12))
    return min(recalls) * 100.0


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """AP with the interpolation-free step definition (sum (R_n - R_{n-1}) P_n)."""
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    total = max(labels.sum(), 1)
    precision = tp / (np.arange(len(labels)) + 1)
    recall = tp / total
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return ap


@torch.no_grad()
def mAP(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Multi-label mean AP over classes (used for AS-20K mAP and BirdSet cmAP)."""
    scores = torch.sigmoid(logits).cpu().numpy()
    labels = targets.cpu().numpy()
    aps = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue  # AP undefined for absent classes
        aps.append(_average_precision(scores[:, c], labels[:, c]))
    return float(np.mean(aps)) * 100.0


#: alias: BirdSet's cmAP equals clip-level mAP
cmAP = mAP


def summarize(task: str, logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Dispatch metrics by benchmark task type.

    task in {``single``, ``multi``}.
    """
    if task == "multi":
        return {"mAP": mAP(logits, targets)}
    return {
        "acc": accuracy(logits, targets),
        "macro_f1": macro_f1(logits, targets),
        "min_recall": min_class_recall(logits, targets),
    }


def mean_ci95(values) -> float:
    """Mean and 95% confidence half-width over n>=2 runs (n=6 in the paper)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 2:
        return 0.0
    half = 1.96 * v.std(ddof=1) / np.sqrt(n)
    return float(half)
