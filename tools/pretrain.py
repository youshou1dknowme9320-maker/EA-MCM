#!/usr/bin/env python
"""Self-supervised pre-training on the P.A.S.T. unlabeled pool (~30 h).

Reproduces the in-domain SSL stage (Group B of Table 1). Any registered
masking strategy can be trained::

    # EA-MCM (proposed): rho_total=75%, rho_anchor=40%, V=8
    python tools/pretrain.py --strategy ea_mcm \
        --manifest data/past/unlabeled.csv \
        --init-checkpoint ckpt/audiomae_audioset.pth \
        --epochs 300 --batch-size 128 --num-views 8

    # single-view baselines
    python tools/pretrain.py --strategy random --num-views 1 ...

    # multi-view baseline with independent masks (cross-view leakage!)
    python tools/pretrain.py --strategy naive_random_multimasking --num-views 8 ...

The script reports ms/batch (the "Mask Time" column of Table 1) and saves
checkpoints every ``--save-every`` epochs.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from eamcm.datasets import UnlabeledAudioDataset
from eamcm.masking import build_masking, seed_generator
from eamcm.models import AudioMAE, build_audiomae, load_audio_mae_pretrained
from eamcm.utils.sched import GradScalerAMP, adjust_lr, build_optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EA-MCM SSL pre-training")
    p.add_argument("--strategy", required=True,
                   choices=["random", "time_freq", "inverse_block", "attmask_low",
                            "attmask_hint", "hard_patch", "att_guided",
                            "naive_random_multimasking", "ea_mcm"])
    p.add_argument("--manifest", required=True, help="CSV with a `filepath` column")
    p.add_argument("--data-root", default="")
    # masking hyper-parameters (paper defaults)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--anchor-ratio", type=float, default=0.40, help="EA-MCM rho_anchor")
    p.add_argument("--num-views", type=int, default=8, help="V; use 1 for single-view")
    p.add_argument("--attmask-mode", default="hint", choices=["hint", "low"])
    p.add_argument("--teacher-source", default="energy", choices=["energy", "attention"],
                   help="AttMask teacher signal; 'attention' needs --teacher-checkpoint")
    p.add_argument("--teacher-checkpoint", default="")
    p.add_argument("--stretched-patch", action="store_true",
                   help="use 32(mel) x 8(frame) stretched patches (supplement Sec. B.2 ablation)")
    p.add_argument("--patch-size", type=int, default=16,
                   help="square patch size (time x mel); ignored with --stretched-patch")
    # optimization (paper: AdamW, 300 epochs)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-5, help="continued pre-training LR")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--clip-grad", type=float, default=3.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    # init & io
    p.add_argument("--init-checkpoint", default="", help="Audio-MAE (AudioSet) weights: AS->P.A.S.T.")
    p.add_argument("--resume", default="")
    p.add_argument("--output-dir", default="work_dirs/pretrain")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def patch_shapes(args):
    """(patch_time, patch_mel); both settings keep 512 tokens on a 64x8 grid."""
    if args.stretched_patch:
        return (8, 32)
    return (args.patch_size, args.patch_size)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    patch_size = patch_shapes(args)
    model = build_audiomae(patch_size=patch_size).to(device)
    if args.init_checkpoint:
        load_audio_mae_pretrained(model, args.init_checkpoint)
        print(f"[init] loaded AudioSet-pretrained weights from {args.init_checkpoint}")

    # frozen teacher for attention/loss-guided strategies
    needs_teacher = (
        args.teacher_source == "attention"
        or args.strategy in ("att_guided",)
        or (args.strategy == "hard_patch" and args.teacher_source == "attention")
    )
    teacher = None
    if needs_teacher:
        teacher = copy.deepcopy(model).to(device).eval()
        if args.teacher_checkpoint:
            load_audio_mae_pretrained(teacher, args.teacher_checkpoint)
        for p_ in teacher.parameters():
            p_.requires_grad = False

    strategy = build_masking(
        args.strategy,
        img_size=model.patch_embed.img_size,
        patch_size=model.patch_embed.patch_size,
        mask_ratio=args.mask_ratio,
        anchor_ratio=args.anchor_ratio,
        num_views=args.num_views,
        generator=seed_generator(args.seed),
        **({"teacher_source": args.teacher_source}
           if args.strategy.startswith("attmask") else {}),
    )
    print(f"[strategy] {strategy.name}  views={strategy.num_views}  "
          f"budget={strategy.budget()}/{strategy.num_patches}")

    dataset = UnlabeledAudioDataset(args.manifest, training=True, root=args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"[data] {len(dataset)} unlabeled clips")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = GradScalerAMP(args.amp)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    start_epoch, global_step = 0, 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, global_step = ckpt["epoch"], ckpt["global_step"]
        print(f"[resume] epoch {start_epoch} from {args.resume}")

    stats_path = Path(args.output_dir) / "train_stats.jsonl"
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss, epoch_time, n_batches = 0.0, 0.0, 0
        t_epoch = time.time()
        for batch in loader:
            t0 = time.time()
            raw = batch["raw"].to(device, non_blocking=True)          # (B, 1, T, F)
            norm = batch["fbank"].to(device, non_blocking=True)
            b = raw.shape[0]

            masks = strategy(raw, norm, teacher=teacher)              # list of (B, N)
            v = len(masks)
            mask_cat = torch.stack(masks).flatten(0, 1)               # (V*B, N)
            norm_rep = norm.unsqueeze(0).expand(v, *norm.shape).flatten(0, 1)
            raw_rep = raw.unsqueeze(0).expand(v, *raw.shape).flatten(0, 1)

            adjust_lr(optimizer, args.lr, global_step, total_steps, warmup_steps, args.min_lr)
            with torch.autocast("cuda", enabled=args.amp):
                loss, pred, _ = model(norm_rep, mask_cat, targets=norm_rep)

            optimizer.zero_grad(set_to_none=True)
            scaler.backward(loss)
            scaler.step(optimizer, clip_grad=args.clip_grad, model=model)

            # hard-patch mining memory update (EMA, no extra forward)
            if strategy.name == "hard_patch" and strategy.teacher_mode == "memory":
                with torch.no_grad():
                    target = model.patchify(norm_rep)
                    if model.norm_pix_loss:
                        mu, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
                        target = (target - mu) / (var + 1e-6) ** 0.5
                    per_patch = ((pred.float() - target) ** 2).mean(-1)
                strategy.update_memory(per_patch.view(v, b, -1).mean(0))

            global_step += 1
            n_batches += 1
            epoch_loss += loss.item()
            epoch_time += (time.time() - t0) * 1000.0

            if global_step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                ms_per_batch = epoch_time / n_batches
                print(f"ep {epoch:3d} step {global_step:6d} loss {loss.item():.4f} "
                      f"lr {lr_now:.2e} {ms_per_batch / v:.1f} ms/batch(view)")

        msg = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, n_batches),
            "sec_per_epoch": round(time.time() - t_epoch, 1),
            "ms_per_view_batch": round(epoch_time / max(1, n_batches) / v, 2),
        }
        print(f"[epoch {epoch}] {msg}")
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = Path(args.output_dir) / f"checkpoint-{epoch + 1}.pth"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
                "strategy": strategy.name,
            }, ckpt_path)
            torch.save(model.state_dict(), Path(args.output_dir) / "encoder_last.pth")
            print(f"[save] {ckpt_path}")

    print("[done] pre-training finished")


if __name__ == "__main__":
    main()
