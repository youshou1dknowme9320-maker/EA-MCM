#!/usr/bin/env python
"""Few-shot fine-tuning evaluation (Fig. 3 of the paper).

For each K in {5, 25, 50, 100, All}, the encoder is fine-tuned on at most K
labelled instances per class. Independent runs draw different shot sets and
head initializations; results are reported as mean ± 95% CI::

    python tools/few_shot.py --dataset past \
        --train-manifest data/past/train.csv --test-manifest data/past/test.csv \
        --pretrained work_dirs/pretrain_eamcm/encoder_last.pth \
        --shots 5,25,50,100,0 --runs 6

The shot sampler is seeded per run, so every pre-trained encoder is evaluated
on *identical* shot sets when ``--seed`` is kept fixed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from common import (
    DATASET_SPECS,
    build_dataset,
    evaluate_classifier,
    load_encoder_from_checkpoint,
    mean_ci95,
    set_seed,
)
from eamcm.datasets import KShotDataset, read_manifest
from eamcm.datasets.audio_dataset import build_label_maps
from eamcm.models import AudioClassifier, build_audiomae
from eamcm.utils.sched import GradScalerAMP, adjust_lr, build_optimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot fine-tuning (Fig. 3)")
    p.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--fold", type=int, default=1, help="ESC-50 held-out fold")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--shots", default="5,25,50,100,0", help="comma-separated K list; 0 = All")
    p.add_argument("--runs", type=int, default=6)
    # optimization (short schedule: few-shot subsets are tiny)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--clip-grad", type=float, default=3.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--output-dir", default="work_dirs/few_shot")
    return p.parse_args()


def canonical_rows(manifest_path: str, task: str, fold: int, dataset: str):
    """Rows with integer labels + the full class count (stable target space).

    The label space is fixed from the *full* manifest before sub-sampling so
    that every shot set shares one consistent target mapping.
    """
    rows = read_manifest(manifest_path)
    if dataset == "esc50":
        rows = [r for r in rows if int(r.get("fold", fold)) != fold]
    label_map, n_cls = build_label_maps(rows, task == "multi")
    if task == "single":
        for r in rows:
            r["label"] = str(label_map[r["label"]])
    return rows, n_cls


def run_once(args, shots: int, run_idx: int, num_classes: int, train_rows, test_loader, device) -> dict:
    set_seed(args.seed + run_idx)
    spec = DATASET_SPECS[args.dataset]

    train_ds = KShotDataset(train_rows, shots=shots, seed=args.seed + run_idx,
                            num_classes=num_classes, training=True, root=args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"[run{run_idx}] {len(train_ds)} shots across {num_classes} classes")

    encoder = build_audiomae()
    load_encoder_from_checkpoint(encoder, args.pretrained)
    model = AudioClassifier(encoder, num_classes=num_classes,
                            multilabel=spec["task"] == "multi", dropout=args.dropout).to(device)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = GradScalerAMP(args.amp)
    total_steps = max(1, len(train_loader)) * args.epochs
    warmup_steps = max(1, len(train_loader)) * args.warmup_epochs
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for batch in train_loader:
            fbank = batch["fbank"].to(device, non_blocking=True)
            target = batch["target"].to(device)
            adjust_lr(optimizer, args.lr, global_step, total_steps, warmup_steps, args.min_lr)
            with torch.autocast("cuda", enabled=args.amp):
                logits = model(fbank)
                loss = model.criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            scaler.backward(loss)
            scaler.step(optimizer, clip_grad=args.clip_grad, model=model)
            global_step += 1
            running += loss.item()
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"[run{run_idx}] ep {epoch + 1}/{args.epochs} loss {running / max(1, len(train_loader)):.4f}")
    return evaluate_classifier(model, test_loader, device, spec["task"])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    spec = DATASET_SPECS[args.dataset]

    num_classes = spec["num_classes"]
    if num_classes is None:
        _, num_classes = build_label_maps(read_manifest(args.train_manifest), spec["task"] == "multi")
        print(f"[classes] inferred {num_classes} from manifest")

    train_rows, _ = canonical_rows(args.train_manifest, spec["task"], args.fold, args.dataset)
    test_loader = build_dataset(args.dataset, args.test_manifest, "test", False,
                                args.fold, args.data_root, 64, args.num_workers)

    shots_list = []
    for tok in args.shots.split(","):
        k = int(tok.strip())
        shots_list.append((k, "All" if k <= 0 else str(k)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for k, name in shots_list:
        results = []
        for run_idx in range(args.runs):
            t0 = time.time()
            metrics = run_once(args, k if k > 0 else 10 ** 9, run_idx, num_classes,
                               train_rows, test_loader, device)
            results.append(metrics)
            print(f"[K={name} run{run_idx}] {metrics}  ({time.time() - t0:.0f}s)")
        values = [r[spec["metric"]] for r in results]
        mean, ci = mean_ci95(values)
        report[name] = {"metric": spec["metric"], "values": values,
                        "mean": mean, "ci95": ci, "all": results}
        print(f"[K={name}] {spec['metric']} = {mean:.2f} ± {ci:.2f}")

    with open(out / "few_shot.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "runs": args.runs, "report": report}, f, indent=2)
    print(f"[saved] {out / 'few_shot.json'}")


if __name__ == "__main__":
    main()
