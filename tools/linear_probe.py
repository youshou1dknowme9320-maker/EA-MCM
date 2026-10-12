#!/usr/bin/env python
"""Linear probing with a frozen pre-trained encoder.

Globally-average-pooled encoder features are extracted once and cached in
memory; only a linear classification head is then trained. This isolates
representation quality from head/adapter capacity and is the cheapest protocol
for comparing masking strategies::

    python tools/linear_probe.py --dataset past \
        --train-manifest data/past/train.csv --test-manifest data/past/test.csv \
        --pretrained work_dirs/pretrain_eamcm/encoder_last.pth --runs 6
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from common import (
    DATASET_SPECS,
    build_dataset,
    esc50_train_loader,
    load_encoder_from_checkpoint,
    mean_ci95,
    set_seed,
)
from eamcm.models import AudioClassifier, build_audiomae
from eamcm.utils.metrics import accuracy, macro_f1, mAP, min_class_recall
from eamcm.utils.sched import adjust_lr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probing (frozen encoder)")
    p.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--fold", type=int, default=1, help="ESC-50 held-out fold")
    p.add_argument("--pretrained", required=True, help="encoder from pretrain.py")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--feature-batch", type=int, default=64, help="encoder extraction batch")
    p.add_argument("--batch-size", type=int, default=512, help="head batch over cached features")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="work_dirs/linear_probe")
    return p.parse_args()


@torch.no_grad()
def extract_features(model: AudioClassifier, loader, device):
    model.eval()
    feats, tgts = [], []
    for batch in loader:
        x = batch["fbank"].to(device, non_blocking=True)
        feats.append(model.forward_features(x).float().cpu())
        tgts.append(batch["target"])
    return torch.cat(feats), torch.cat(tgts)


def train_head(train_feats, train_targets, num_classes, multilabel, args, seed, device):
    set_seed(seed)
    head = torch.nn.Sequential(
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(train_feats.shape[1], num_classes),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = train_feats.shape[0]
    steps_per_epoch = max(1, (n + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * min(5, args.epochs)
    step = 0
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            x = train_feats[idx].to(device)
            y = train_targets[idx].to(device).float() if multilabel else train_targets[idx].to(device)
            adjust_lr(optimizer, args.lr, step, total_steps, warmup_steps)
            logits = head(x)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                if multilabel
                else torch.nn.functional.cross_entropy(logits, y)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
    return head


@torch.no_grad()
def eval_head(head, feats, targets, multilabel, device) -> dict:
    head.eval()
    logits = []
    for i in range(0, feats.shape[0], 1024):
        logits.append(head(feats[i : i + 1024].to(device)).float().cpu())
    logits = torch.cat(logits)
    if multilabel:
        return {"mAP": mAP(logits, targets)}
    return {
        "acc": accuracy(logits, targets),
        "macro_f1": macro_f1(logits, targets),
        "min_recall": min_class_recall(logits, targets),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    spec = DATASET_SPECS[args.dataset]

    encoder = build_audiomae()
    load_encoder_from_checkpoint(encoder, args.pretrained)
    num_classes = spec["num_classes"]
    if num_classes is None:
        from eamcm.datasets.audio_dataset import build_label_maps, read_manifest

        _, num_classes = build_label_maps(read_manifest(args.train_manifest), spec["task"] == "multi")
        print(f"[classes] inferred {num_classes} from manifest")

    probe = AudioClassifier(encoder, num_classes=num_classes).to(device)
    for p_ in probe.parameters():
        p_.requires_grad = False
    probe.eval()

    if args.dataset == "esc50":
        train_loader = esc50_train_loader(args.train_manifest, args.fold, args.data_root,
                                          args.feature_batch, args.num_workers, center_crop=True)
    else:
        train_loader = build_dataset(args.dataset, args.train_manifest, "train", False,
                                     args.fold, args.data_root, args.feature_batch, args.num_workers)
    test_loader = build_dataset(args.dataset, args.test_manifest, "test", False,
                                args.fold, args.data_root, args.feature_batch, args.num_workers)

    print("[features] extracting train ...")
    train_feats, train_targets = extract_features(probe, train_loader, device)
    print("[features] extracting test ...")
    test_feats, test_targets = extract_features(probe, test_loader, device)
    print(f"[features] train {tuple(train_feats.shape)}  test {tuple(test_feats.shape)}")

    results = []
    for run_idx in range(args.runs):
        t0 = time.time()
        head = train_head(train_feats, train_targets, num_classes,
                          spec["task"] == "multi", args, args.seed + run_idx, device)
        metrics = eval_head(head, test_feats, test_targets, spec["task"] == "multi", device)
        results.append(metrics)
        print(f"[run{run_idx}] {metrics}  ({time.time() - t0:.0f}s)")

    metric_key = spec["metric"]
    values = [r[metric_key] for r in results]
    mean, ci = mean_ci95(values)
    summary = {
        "dataset": args.dataset, "protocol": "linear_probe", "metric": metric_key,
        "runs": args.runs, "values": values, "mean": mean, "ci95": ci, "all_metrics": results,
    }
    out = Path(args.output_dir) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {metric_key} = {mean:.2f} ± {ci:.2f} (n={args.runs})  -> {out}")


if __name__ == "__main__":
    main()
