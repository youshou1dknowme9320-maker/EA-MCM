#!/usr/bin/env python
"""Render masking-strategy overlays on log-Mel spectrograms (demo figures).

For every (audio, strategy) pair the script saves a PNG showing
  * the input log-Mel spectrogram,
  * the first two generated views (red = masked patches),
  * for multi-view strategies: the union of all V views and the per-patch
    *cross-view exposure map* (fraction of views in which each patch stays
    visible). Naive multi-masking exposes focal patches in nearly every view
    (exposure -> 1, the leak of Eq. 5), while EA-MCM pins its anchor set to
    exactly zero exposure,
  * for EA-MCM: the energy anchor set A* in gold.

    python tools/visualize_mask.py --audio data/past/clips/session_01/x.wav \
        --output figures/mask_vis
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eamcm.masking import build_masking, seed_generator
from eamcm.models import build_audiomae, load_audio_mae_pretrained
from eamcm.utils.audio import extract_fbank, load_waveform, normalize_fbank

STRATEGY_NAMES = [
    "random", "time_freq", "inverse_block", "attmask_low", "attmask_hint",
    "hard_patch", "att_guided", "naive_random_multimasking", "ea_mcm",
]


def draw_mask_panel(ax, spec, grid, ph, pf, color, alpha, title):
    ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    if grid is not None:
        big = np.repeat(np.repeat(grid.astype(float), ph, axis=0), pf, axis=1)
        rgba = np.zeros((big.shape[0], big.shape[1], 4))
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
        rgba[..., 3] = big * alpha
        ax.imshow(rgba, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_exposure_panel(ax, exposure, title):
    ax.imshow(exposure, origin="lower", aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    p = argparse.ArgumentParser(description="Masking visualization")
    p.add_argument("--audio", nargs="+", required=True)
    p.add_argument("--strategies", nargs="+", default=STRATEGY_NAMES, choices=STRATEGY_NAMES)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--anchor-ratio", type=float, default=0.40)
    p.add_argument("--num-views", type=int, default=8)
    p.add_argument("--teacher-checkpoint", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="figures/mask_vis")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    teacher = None
    if args.teacher_checkpoint:
        teacher = build_audiomae().to(device).eval()
        load_audio_mae_pretrained(teacher, args.teacher_checkpoint)
        for p_ in teacher.parameters():
            p_.requires_grad = False

    for audio_path in args.audio:
        wav = load_waveform(str(audio_path))
        raw = extract_fbank(wav, training=False)
        norm = normalize_fbank(raw)
        raw_b = raw.unsqueeze(0).to(device)
        norm_b = norm.unsqueeze(0).to(device)
        spec = raw.squeeze(0).numpy()
        t, f = spec.shape

        for name in args.strategies:
            strategy = build_masking(
                name,
                img_size=(t, f), patch_size=(16, 16),
                mask_ratio=args.mask_ratio, anchor_ratio=args.anchor_ratio,
                num_views=args.num_views, generator=seed_generator(args.seed),
                **({"teacher_source": "energy"} if name.startswith("attmask") else {}),
            )
            with torch.no_grad():
                masks = strategy(raw_b, norm_b, teacher=teacher)
            v = len(masks)
            masks_np = [m[0].cpu().numpy() for m in masks]
            gh, gw = strategy.grid
            ph, pf = t // gh, f // gw

            red, gold, green = (1.0, 0.1, 0.1), (1.0, 0.84, 0.0), (0.0, 1.0, 0.3)
            panels = [("spec", None, None, 0.0, "input log-Mel")]
            panels.append(("mask", masks_np[0].reshape(gh, gw), red, 0.5, "view 1 (red = masked)"))
            if v > 1:
                panels.append(("mask", masks_np[1].reshape(gh, gw), red, 0.5, "view 2 (red = masked)"))
                panels.append(("mask", np.any(masks_np, axis=0).reshape(gh, gw), red, 0.5,
                               f"union of {v} views"))
                exposure = (1.0 - np.mean(masks_np, axis=0)).reshape(gh, gw)
                panels.append(("exposure", exposure, None, 0.0, "cross-view exposure"))
            if name == "ea_mcm":
                anchor, _ = strategy.decompose(raw_b)
                panels.append(("mask", anchor[0].cpu().numpy().reshape(gh, gw), gold, 0.65,
                               "anchor set A* (gold)"))
            if v > 1 and name == "ea_mcm":
                panels.append(("mask", (~np.any(masks_np, axis=0)).reshape(gh, gw), green, 0.6,
                               "visible in EVERY view"))

            fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 3.0))
            for ax, (kind, grid, color, alpha, title) in zip(np.atleast_1d(axes), panels):
                if kind == "exposure":
                    draw_exposure_panel(ax, grid, title)
                else:
                    draw_mask_panel(ax, spec, grid, ph, pf, color, alpha, title)
            fig.suptitle(f"{Path(audio_path).stem} — {name}", fontsize=10)
            fig.tight_layout()
            dest = out / f"{Path(audio_path).stem}_{name}.png"
            fig.savefig(dest, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {dest}")


if __name__ == "__main__":
    main()
