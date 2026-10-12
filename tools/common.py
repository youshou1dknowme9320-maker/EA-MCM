"""Shared helpers for the fine-tuning / evaluation toolchain."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eamcm.datasets import (
    AS20KDataset,
    BirdSetDataset,
    ESC50Dataset,
    PASTDataset,
    SPC2Dataset,
    WMMDataset,
)

# dataset, task, #classes (None = inferred from manifest), headline metric
DATASET_SPECS = {
    "past":    dict(task="single", num_classes=6,  metric="acc",      wrapper=PASTDataset),
    "esc50":   dict(task="single", num_classes=50, metric="acc",      wrapper=ESC50Dataset),
    "spc2":    dict(task="single", num_classes=35, metric="acc",      wrapper=SPC2Dataset),
    "wmm":     dict(task="single", num_classes=None, metric="macro_f1", wrapper=WMMDataset),
    "as20k":   dict(task="multi",  num_classes=527, metric="mAP",   wrapper=AS20KDataset),
    "birdset": dict(task="multi",  num_classes=None, metric="cmAP",  wrapper=BirdSetDataset),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(spec: str, manifest: str, split: str = "train", training: bool = True,
                  fold: int = 1, root: str = "", batch_size: int = 64,
                  num_workers: int = 8):
    """Build a DataLoader for a downstream benchmark.

    For ESC-50, ``fold`` selects the held-out fold (5-fold cross-validation);
    the train split uses the complementary folds.
    """
    cfg = DATASET_SPECS[spec]
    kwargs = dict(num_classes=cfg["num_classes"], training=training, root=root)
    if spec == "esc50":
        if training:
            # training uses the four complementary folds of the held-out fold
            ds = cfg["wrapper"](manifest, **kwargs)
            ds.rows = [r for r in ds.rows if int(r.get("fold", fold)) != fold]
        else:
            ds = cfg["wrapper"](manifest, fold=fold, **kwargs)
    else:
        ds = cfg["wrapper"](manifest, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, pin_memory=True, drop_last=training)


def esc50_train_loader(manifest: str, fold: int, root: str = "", batch_size: int = 64,
                       num_workers: int = 8, center_crop: bool = False):
    """ESC-50 train loader over the *complementary* folds.

    Unlike :func:`build_dataset` this supports deterministic center crops
    (``center_crop=True``) needed by linear probing / feature caching, and it
    never applies the held-out-fold filter to the train split.
    """
    from eamcm.datasets import AudioClassificationDataset

    ds = AudioClassificationDataset(manifest, num_classes=50, training=not center_crop, root=root)
    ds.rows = [r for r in ds.rows if int(r.get("fold", fold)) != fold]
    return DataLoader(ds, batch_size=batch_size, shuffle=not center_crop,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, loader, device, task: str) -> dict:
    """Full-pass evaluation; returns the metric dict for the task type."""
    from eamcm.utils.metrics import accuracy, mAP, macro_f1, min_class_recall

    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        logits = model(batch["fbank"].to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_targets.append(batch["target"])
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    if task == "multi":
        return {"mAP": mAP(logits, targets)}
    return {
        "acc": accuracy(logits, targets),
        "macro_f1": macro_f1(logits, targets),
        "min_recall": min_class_recall(logits, targets),
    }


def load_encoder_from_checkpoint(encoder, path: str) -> None:
    """Load pre-trained encoder weights from a pretrain.py artifact."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    encoder_state = {k: v for k, v in state.items() if not k.startswith("decoder")}
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    # decoder_* keys are expected to be missing here
    unexpected = [k for k in unexpected if not k.startswith("decoder")]
    print(f"[load_encoder] {path}  missing={len(missing)} unexpected={len(unexpected)}")


def mean_ci95(values):
    """Report mean +- 95% CI half-width across repeated runs (n=6 in the paper)."""
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    half = 0.0 if len(v) < 2 else 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(half)
