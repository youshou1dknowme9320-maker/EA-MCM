"""Audio front-end: unified 128-band log-Mel spectrogram extraction.

All benchmarks (P.A.S.T., BirdSet, WMM, ESC-50, SPC-2, AS-20K) are resampled
to 16 kHz and converted to 128-band log-Mel spectrograms following the
Audio-MAE protocol (Kaldi-compatible fbank, 25 ms frame / 10 ms shift).
Variable-length waveforms are zero-padded or center-cropped to 1024 frames
(10.24 s) so that a ViT-Base with 16x16 patches always yields a 64x8 grid of
512 tokens.
"""

import math
import random
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

TARGET_SR = 16000
NUM_MEL_BINS = 128
NUM_FRAMES = 1024  # 10.24 s at 10 ms hop


def _load_fallback(path: str) -> Tuple[torch.Tensor, int]:
    """Load audio without the torchaudio I/O backend (Windows w/o FFmpeg)."""
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr
    except ImportError:
        pass
    import wave

    with wave.open(path, "rb") as f:
        sr, n, ch, width = f.getframerate(), f.getnframes(), f.getnchannels(), f.getsampwidth()
        raw = f.readframes(n)
    if width == 2:
        data = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    elif width == 4:
        data = torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    elif width == 1:
        data = (torch.frombuffer(bytearray(raw), dtype=torch.uint8).float() - 128.0) / 128.0
    else:
        raise RuntimeError(f"unsupported WAV sample width {width}: {path}")
    return data.view(-1, ch).T.contiguous(), sr


def load_waveform(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load an audio file as mono float tensor at ``target_sr``.

    Prefers the torchaudio I/O backend and falls back to soundfile (or the
    stdlib ``wave`` module for PCM WAV) on platforms where no backend is
    registered (e.g. Windows without FFmpeg).
    """
    try:
        wav, sr = torchaudio.load(path)
    except RuntimeError:
        wav, sr = _load_fallback(path)
    wav = wav.mean(dim=0)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def extract_fbank(
    wav: torch.Tensor,
    num_mel_bins: int = NUM_MEL_BINS,
    target_len: int = NUM_FRAMES,
    training: bool = True,
) -> torch.Tensor:
    """Waveform -> (1, ``target_len``, ``num_mel_bins``) log-Mel spectrogram.

    Follows the Audio-MAE front-end: Kaldi fbank with 25 ms window and 10 ms
    shift, then random cropping (training) or center cropping (eval) to
    ``target_len`` frames with zero padding for shorter clips.
    """
    feats = kaldi.fbank(
        wav.unsqueeze(0) if wav.dim() == 1 else wav,
        sample_frequency=float(TARGET_SR),
        num_mel_bins=num_mel_bins,
        frame_length=25.0,
        frame_shift=10.0,
    )  # (T, num_mel_bins)

    t = feats.shape[0]
    if t < target_len:
        pad = torch.zeros(target_len - t, num_mel_bins)
        feats = torch.cat([feats, pad], dim=0)
    elif t > target_len:
        if training:
            start = random.randint(0, t - target_len)
        else:
            start = (t - target_len) // 2
        feats = feats[start : start + target_len]

    return feats.unsqueeze(0)  # (1, T, F)


def normalize_fbank(feats: torch.Tensor, mean: float = -4.268, std: float = 4.569) -> torch.Tensor:
    """Per-dataset-free global normalization (Audio-MAE fbank statistics)."""
    return (feats - mean) / (std + 1e-5)


def mix_audio(wav_a: torch.Tensor, wav_b: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add ``wav_b`` (noise/interference) on top of ``wav_a`` at ``snr_db``.

    Used by the robustness protocol (FSD50K Fan / Buzz / Boom at 0 dB SNR).
    Both waveforms are truncated or zero-padded to the same length.
    """
    n = max(wav_a.numel(), wav_b.numel())
    a = torch.zeros(n)
    b = torch.zeros(n)
    a[: wav_a.numel()] = wav_a
    b[: wav_b.numel()] = wav_b

    p_sig = torch.mean(a**2) + 1e-10
    p_noise = torch.mean(b**2) + 1e-10
    scale = math.sqrt(p_sig / p_noise / (10 ** (snr_db / 10)))
    return a + scale * b


def split_session_manifest(
    manifest_rows: list,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    session_key: str = "session",
) -> dict:
    """Session-level train/val/test split for P.A.S.T.

    A *session* is a continuous recording block captured under a fixed
    microphone setup on a separate day. Entire sessions (and therefore all
    clips they contain) are assigned to exactly one split, preventing
    acoustic-profile cross-contamination between train and test.
    """
    sessions = sorted({row[session_key] for row in manifest_rows})
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n = len(sessions)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    split_of = {}
    for i, s in enumerate(sessions):
        if i < n_train:
            split_of[s] = "train"
        elif i < n_train + n_val:
            split_of[s] = "val"
        else:
            split_of[s] = "test"
    out = {"train": [], "val": [], "test": []}
    for row in manifest_rows:
        out[split_of[row[session_key]]].append(row)
    return out
