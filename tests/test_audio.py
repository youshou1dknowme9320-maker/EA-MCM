"""Unit tests for the audio front-end, metrics, and P.A.S.T. session split.

Run directly (no pytest required)::

    python tests/test_audio.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from eamcm.datasets.audio_dataset import KShotDataset, read_manifest, write_manifest
from eamcm.utils.audio import (
    NUM_FRAMES,
    NUM_MEL_BINS,
    TARGET_SR,
    extract_fbank,
    mix_audio,
    normalize_fbank,
    split_session_manifest,
)
from eamcm.utils.metrics import accuracy, macro_f1, mAP


def _sine(freq: float = 440.0, seconds: float = 2.0) -> torch.Tensor:
    t = torch.arange(int(TARGET_SR * seconds), dtype=torch.float32) / TARGET_SR
    return 0.5 * torch.sin(2 * math.pi * freq * t)


def test_fbank_shape_short_clip_padded():
    """Short clips are zero-padded to the fixed 1024 x 128 input."""
    feats = extract_fbank(_sine(seconds=0.5), training=False)
    assert feats.shape == (1, NUM_FRAMES, NUM_MEL_BINS)


def test_fbank_center_crop_deterministic():
    """Eval mode center-crops deterministically; train mode crops randomly."""
    wav = _sine(seconds=30.0)  # 30 s -> ~3000 frames > 1024
    a = extract_fbank(wav, training=False)
    b = extract_fbank(wav, training=False)
    assert torch.equal(a, b), "eval extraction must be deterministic"
    c = extract_fbank(wav, training=True)
    assert c.shape == a.shape and not torch.equal(a, c)


def test_normalize_fbank_uses_audiomae_stats():
    x = torch.zeros(1, 8, 8)
    y = normalize_fbank(x)
    assert torch.allclose(y, (x - (-4.268)) / (4.569 + 1e-5))


def test_mix_audio_snr():
    sig = _sine(seconds=2.0)
    noise = 0.5 * _sine(freq=100.0, seconds=2.0)
    for snr_db in (0.0, 6.0, -3.0):
        mixed = mix_audio(sig, noise, snr_db=snr_db)
        assert mixed.shape == sig.shape
        p_sig = (sig**2).mean()
        p_noise = ((mixed - sig) ** 2).mean()
        ratio_db = 10 * math.log10(p_sig / p_noise)
        assert abs(ratio_db - snr_db) < 0.5, f"SNR mismatch: {ratio_db:.2f} vs {snr_db}"


def test_session_split_isolation():
    rows = [
        {"filepath": f"session_{s:02d}/clip_{i:04d}.wav", "label": "cough",
         "session": f"session_{s:02d}"}
        for s in range(10)
        for i in range(5)
    ]
    splits = split_session_manifest(rows, (0.8, 0.1, 0.1), seed=42)
    train_sessions = {r["filepath"].split("/")[0] for r in splits["train"]}
    test_sessions = {r["filepath"].split("/")[0] for r in splits["test"]}
    val_sessions = {r["filepath"].split("/")[0] for r in splits["val"]}
    assert not (train_sessions & test_sessions)
    assert not (train_sessions & val_sessions)
    assert not (val_sessions & test_sessions)
    assert sum(len(v) for v in splits.values()) == len(rows)


def test_kshot_dataset_caps_per_class():
    tmp = Path(__file__).parent / "_tmp_manifest.csv"
    rows = [{"filepath": f"{c}_{i}.wav", "label": c} for c in ("a", "b") for i in range(20)]
    write_manifest(str(tmp), rows)
    ds = KShotDataset(str(tmp), shots=5, seed=42)
    counts = {}
    for r in ds.rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    assert len(ds) == 10 and set(counts.values()) == {5}
    ds_all = KShotDataset(str(tmp), shots=10 ** 9, seed=42)
    assert len(ds_all) == 40
    tmp.unlink()


def test_metrics():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1, 1, 1])
    assert accuracy(logits, targets) == 75.0
    # class 0: P=1.0 R=0.5 -> F1=2/3 ; class 1: P=2/3 R=1.0 -> F1=0.8
    assert abs(macro_f1(logits, targets) - (2 / 3 + 0.8) / 2 * 100) < 1e-6

    ml_logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    ml_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert mAP(ml_logits, ml_targets) == 100.0
    ml_targets2 = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    assert 0.0 < mAP(ml_logits, ml_targets2) < 100.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} audio/metric tests passed.")
