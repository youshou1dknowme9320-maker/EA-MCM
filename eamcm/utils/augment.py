"""Fine-tuning augmentations used in the orthogonality study (Table 2).

* :class:`SpecAugment` - time/frequency masking on the fbank (Park et al. 2019).
* :func:`audio_mixup` - waveform-level mixup with soft targets (Zhang et al. 2018).
* :func:`add_noise_at_snr` - additive FSD50K interference (Fan/Buzz/Boom at
  0 dB SNR) for the robustness protocol.
"""

import random
from pathlib import Path
from typing import Optional

import torch

from .audio import load_waveform, mix_audio


class SpecAugment:
    """Apply masked time strips and frequency strips to a (1, T, F) fbank."""

    def __init__(
        self,
        num_time_masks: int = 2,
        time_width: int = 48,
        num_freq_masks: int = 2,
        freq_width: int = 16,
    ) -> None:
        self.num_time_masks = num_time_masks
        self.time_width = time_width
        self.num_freq_masks = num_freq_masks
        self.freq_width = freq_width

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        x = fbank.clone()
        t, f = x.shape[-2], x.shape[-1]
        for _ in range(self.num_time_masks):
            w = random.randint(0, self.time_width)
            if w == 0:
                continue
            start = random.randint(0, max(0, t - w))
            x[..., start : start + w, :] = 0.0
        for _ in range(self.num_freq_masks):
            w = random.randint(0, self.freq_width)
            if w == 0:
                continue
            start = random.randint(0, max(0, f - w))
            x[..., :, start : start + w] = 0.0
        return x


def audio_mixup(
    fbank_a: torch.Tensor,
    fbank_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    alpha: float = 0.4,
):
    """Beta(alpha, alpha) mixup on the fbank; returns (fbank, soft_targets)."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    fbank = lam * fbank_a + (1 - lam) * fbank_b
    target = lam * target_a + (1 - lam) * target_b
    return fbank, target


class NoiseInjector:
    """Additive-noise robustness protocol (Table 2).

    Interference clips come from FSD50K: ``fan`` (broadband), ``buzz``
    (harmonic machinery), ``boom`` (impulsive transient). Clips are mixed at
    0 dB SNR against each utterance.
    """

    def __init__(self, noise_dir: str, noise_type: str, snr_db: float = 0.0) -> None:
        self.files = sorted(Path(noise_dir).rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No .wav interference files under {noise_dir}")
        self.snr_db = snr_db
        self.noise_type = noise_type

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        noise = load_waveform(str(random.choice(self.files)))
        return mix_audio(wav, noise, self.snr_db)
