#!/usr/bin/env python
"""Robustness under additive FSD50K interference (Table 2, Group 4).

Every test clip is mixed with Fan (broadband), Buzz (harmonic machinery) or
Boom (impulsive transient) interference at 0 dB SNR and classified with a
checkpoint produced by tools/finetune.py::

    python tools/robustness_eval.py --dataset past \
        --test-manifest data/past/test.csv \
        --checkpoint work_dirs/finetune/run0_best.pth \
        --noise-dir data/fsd50k/interference --noise-types clean,fan,buzz,boom
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from common import DATASET_SPECS, evaluate_classifier, set_seed
from eamcm.datasets import AudioClassificationDataset
from eamcm.datasets.audio_dataset import build_label_maps, read_manifest
from eamcm.models import AudioClassifier, build_audiomae
from eamcm.utils.augment import NoiseInjector
from eamcm.utils.audio import extract_fbank, load_waveform, normalize_fbank


class NoisyAudioDataset(AudioClassificationDataset):
    """Test-time additive-noise wrapper (deterministic center crop)."""

    def __init__(self, manifest, noise_dir: str = "", noise_type: str = "",
                 snr_db: float = 0.0, **kwargs) -> None:
        super().__init__(manifest, training=False, **kwargs)
        self.injector = NoiseInjector(noise_dir, noise_type, snr_db) if noise_dir else None

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        path = self.root / row["filepath"] if self.root else Path(row["filepath"])
        wav = load_waveform(str(path))
        if self.injector is not None:
            wav = self.injector(wav)
        raw = extract_fbank(wav, training=False)
        return {"raw": raw, "fbank": normalize_fbank(raw), "target": self.target_of(row)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Additive-noise robustness evaluation")
    p.add_argument("--dataset", default="past", choices=list(DATASET_SPECS))
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--fold", type=int, default=1, help="ESC-50 held-out fold")
    p.add_argument("--checkpoint", required=True, help="run*_best.pth from tools/finetune.py")
    p.add_argument("--noise-dir", default="", help="directory with FSD50K interference wavs")
    p.add_argument("--noise-types", default="clean,fan,buzz,boom")
    p.add_argument("--snr-db", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="work_dirs/robustness/robustness.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    spec = DATASET_SPECS[args.dataset]

    num_classes = spec["num_classes"]
    if num_classes is None:
        _, num_classes = build_label_maps(read_manifest(args.test_manifest), spec["task"] == "multi")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = build_audiomae()
    encoder.load_state_dict(ckpt["encoder"])
    model = AudioClassifier(encoder, num_classes=num_classes,
                            multilabel=spec["task"] == "multi").to(device)
    model.head.load_state_dict(ckpt["head"])
    model.eval()
    print(f"[ckpt] loaded {args.checkpoint}")

    fold = args.fold if args.dataset == "esc50" else None
    results = []
    for ntype in [t.strip() for t in args.noise_types.split(",") if t.strip()]:
        ds = NoisyAudioDataset(
            args.test_manifest,
            noise_dir=args.noise_dir if ntype != "clean" else "",
            noise_type=ntype,
            snr_db=args.snr_db,
            num_classes=num_classes,
            multilabel=spec["task"] == "multi",
            root=args.data_root,
            fold=fold,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        metrics = evaluate_classifier(model, loader, device, spec["task"])
        results.append({"condition": ntype, **metrics})
        print(f"[{ntype:>6}] {metrics}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "snr_db": args.snr_db,
                   "metric": spec["metric"], "results": results}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
