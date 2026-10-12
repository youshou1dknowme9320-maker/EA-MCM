#!/usr/bin/env python
"""Supervised fine-tuning of a pre-trained EA-MCM / baseline encoder.

Covers the main benchmark protocol (Table 1), the orthogonality study
(Table 2: +SpecAugment, +Audio Mixup), and repeated runs (n = 6) reporting
mean +- 95% CI::

    python tools/finetune.py --dataset past \
        --train-manifest data/past/train.csv --val-manifest data/past/val.csv \
        --test-manifest data/past/test.csv \
        --pretrained work_dirs/pretrain_eamcm/encoder_last.pth \
        --runs 6 --seed 42

    # Table 2 Group 2/3 orthogonality
    python tools/finetune.py --dataset past ... --specaugment
    python tools/finetune.py --dataset past ... --mixup
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from common import DATASET_SPECS, build_dataset, evaluate_classifier, load_encoder_from_checkpoint, mean_ci95, set_seed
from eamcm.datasets import KShotDataset, read_manifest
from eamcm.models import AudioClassifier, build_audiomae
from eamcm.utils.augment import SpecAugment, audio_mixup
from eamcm.utils.sched import GradScalerAMP, adjust_lr, build_optimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised fine-tuning")
    p.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", default="")
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--fold", type=int, default=1, help="ESC-50 held-out fold")
    p.add_argument("--pretrained", required=True, help="encoder from pretrain.py")
    p.add_argument("--freeze-encoder", action="store_true", help="linear probing mode")
    # optimization
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--clip-grad", type=float, default=3.0)
    p.add_argument("--dropout", type=float, default=0.1)
    # orthogonality (Table 2)
    p.add_argument("--specaugment", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    # runs / io
    p.add_argument("--runs", type=int, default=1, help="independent runs (paper: 6)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--output-dir", default="work_dirs/finetune")
    return p.parse_args()


def run_once(args, run_idx: int) -> dict:
    set_seed(args.seed + run_idx)
    device = torch.device(args.device)
    spec = DATASET_SPECS[args.dataset]

    encoder = build_audiomae()
    load_encoder_from_checkpoint(encoder, args.pretrained)
    if args.freeze_encoder:
        for p_ in encoder.parameters():
            p_.requires_grad = False
        encoder.eval()
    model = AudioClassifier(encoder, num_classes=spec["num_classes"] or 1,
                            multilabel=spec["task"] == "multi",
                            dropout=args.dropout).to(device)
    if spec["num_classes"] is None:  # infer class count from the manifest
        from eamcm.datasets.audio_dataset import build_label_maps, read_manifest

        _, n_cls = build_label_maps(read_manifest(args.train_manifest), spec["task"] == "multi")
        model.head[-1] = torch.nn.Linear(model.head[-1].in_features, n_cls).to(device)
        print(f"[classes] inferred {n_cls} from manifest")

    train_loader = build_dataset(args.dataset, args.train_manifest, "train", True,
                                 args.fold, args.data_root, args.batch_size, args.num_workers)
    val_loader = (build_dataset(args.dataset, args.val_manifest, "val", False,
                                args.fold, args.data_root, args.batch_size, args.num_workers)
                  if args.val_manifest else None)
    test_loader = build_dataset(args.dataset, args.test_manifest, "test", False,
                                args.fold, args.data_root, args.batch_size, args.num_workers)

    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = GradScalerAMP(args.amp)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    specaug = SpecAugment() if args.specaugment else None

    best_score, best_path = -1.0, Path(args.output_dir) / f"run{run_idx}_best.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0
    metric_key = spec["metric"]

    for epoch in range(args.epochs):
        model.train(not args.freeze_encoder)
        running = 0.0
        for batch in train_loader:
            raw, fbank = batch["raw"], batch["fbank"].to(device, non_blocking=True)
            target = batch["target"].to(device)
            adjust_lr(optimizer, args.lr, global_step, total_steps, warmup_steps, args.min_lr)

            if args.mixup:
                perm = torch.randperm(fbank.shape[0], device=device)
                fbank, soft_target = audio_mixup(fbank, fbank[perm], target, target[perm], args.mixup_alpha)
            if specaug is not None:
                fbank = torch.stack([specaug(f) for f in fbank])

            with torch.autocast("cuda", enabled=args.amp):
                logits = model(fbank)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, soft_target.float())
                        if (args.mixup and spec["task"] == "multi")
                        else (torch.nn.functional.cross_entropy(logits, soft_target)
                              if args.mixup else model.criterion(logits, target)))

            optimizer.zero_grad(set_to_none=True)
            scaler.backward(loss)
            scaler.step(optimizer, clip_grad=args.clip_grad, model=model if not args.freeze_encoder else None)
            global_step += 1
            running += loss.item()

        line = {"epoch": epoch, "loss": running / max(1, len(train_loader))}

        if val_loader is not None:
            val_metrics = evaluate_classifier(model, val_loader, device, spec["task"])
            line["val"] = val_metrics
            score = val_metrics.get(metric_key if metric_key in val_metrics else "mAP", 0.0)
            if score > best_score:
                best_score = score
                torch.save({"encoder": encoder.state_dict(), "head": model.head.state_dict()}, best_path)
        print(f"[run{run_idx}] {line}")

    if best_path.exists():  # restore best val weights before testing
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        model.head.load_state_dict(ckpt["head"])

    test_metrics = evaluate_classifier(model, test_loader, device, spec["task"])
    print(f"[run{run_idx}] TEST {test_metrics}")
    return test_metrics


def main() -> None:
    args = parse_args()
    results = []
    for run_idx in range(args.runs):
        t0 = time.time()
        results.append(run_once(args, run_idx))
        print(f"[run{run_idx}] took {time.time() - t0:.0f}s")

    metric_key = DATASET_SPECS[args.dataset]["metric"]
    values = [r[metric_key] for r in results]
    mean, ci = mean_ci95(values)
    summary = {"dataset": args.dataset, "metric": metric_key,
               "runs": args.runs, "values": values, "mean": mean, "ci95": ci,
               "all_metrics": results}
    out = Path(args.output_dir) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {metric_key} = {mean:.2f} ± {ci:.2f} (n={args.runs})  -> {out}")


if __name__ == "__main__":
    main()
