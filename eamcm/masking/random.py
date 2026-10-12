"""Single-view data-agnostic baselines: Random, Time-Freq, Inverse Block."""

from typing import List, Optional

import torch

from .base import MaskingStrategy

try:
    from ..models.mae_audio import AudioMAE
except ImportError:  # pragma: no cover
    AudioMAE = None


class RandomMasking(MaskingStrategy):
    """Uniform random masking (He et al. 2022), single view."""

    name = "random"

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        return [self.random_mask(b, device)]


class TimeFreqMasking(MaskingStrategy):
    """Axis-aligned strip masking on the patch grid (SpecAugment-like).

    The masking budget is split between the time and the frequency axis; each
    strip has a random start and a random width. Unlike random masking this
    never samples isolated single patches, but it still ignores spectral
    content and therefore leaves focal formants exposed.
    """

    name = "time_freq"

    def __init__(self, freq_share: float = 0.5, max_strip: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freq_share = freq_share
        self.max_strip = max_strip

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        gh, gw = self.grid
        total = self.budget()
        k_freq = int(total * self.freq_share)
        k_time = total - k_freq

        mask = torch.zeros(b, self.num_patches, dtype=torch.bool, device=device)

        def _axis_mask(k_budget: int, axis_len: int, other_len: int) -> torch.Tensor:
            """Mask ``k_budget`` grid cells as contiguous strips along one axis."""
            remain = k_budget
            out = torch.zeros(b, gh, gw, dtype=torch.bool, device=device)
            guard = 0
            while remain > 0 and guard < 1000:
                guard += 1
                max_len = max(1, int(self.max_strip * axis_len))
                length = int(torch.randint(1, max_len + 1, (1,)).item())
                start = int(torch.randint(0, axis_len, (1,)).item())
                length = min(length, axis_len - start, remain)
                if axis_len == gh:  # time strips: full freq extent at given rows
                    cols = int(torch.randint(0, other_len, (1,)).item())
                    span = min(remain // max(length, 1) + 1, other_len - cols)
                    out[:, start : start + length, cols : cols + max(span, 1)] = True
                    covered = out.sum(dim=(1, 2))
                else:  # freq strips
                    rows = int(torch.randint(0, other_len, (1,)).item())
                    span = min(remain // max(length, 1) + 1, other_len - rows)
                    out[:, rows : rows + max(span, 1), start : start + length] = True
                    covered = out.sum(dim=(1, 2))
                remain = k_budget - int(covered.max().item())
            return out.reshape(b, self.num_patches)

        mask = _axis_mask(k_time, gh, gw) | _axis_mask(k_freq, gw, gh)
        # trim potential overshoot, then top up with random patches if short
        over = mask.sum(dim=1) - total
        if (over > 0).any():
            for i in torch.nonzero(over > 0).flatten():
                idx = torch.nonzero(mask[i]).flatten()
                drop = idx[torch.randperm(idx.numel(), device=device)[: over[i]]]
                mask[i, drop] = False
        short = total - mask.sum(dim=1)  # (B,) remaining budget per sample (>= 0)
        if (short > 0).any():
            fill = self.rand(b, self.num_patches).to(device).masked_fill(mask, -1.0)
            order = fill.argsort(dim=1, descending=True)  # unmasked cells first
            is_new = ~torch.gather(mask, 1, order)
            take = is_new & (torch.cumsum(is_new.to(torch.int64), dim=1) <= short.unsqueeze(1))
            extra = torch.zeros_like(mask)
            extra.scatter_(1, order, take)
            mask = mask | extra
        return [mask]


class InverseBlockMasking(MaskingStrategy):
    """Inverse block masking following EAT (Chen et al. 2024).

    A small *seed* neighborhood around a randomly chosen patch stays visible;
    masking then expands outward ring by ring (nearest patches first) until
    the budget is met, and any remaining budget is filled uniformly at random.
    The model must therefore reconstruct the near context from a small seed
    instead of interpolating a hidden block from distant context.
    """

    name = "inverse_block"

    def __init__(self, seed_size: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seed_size = seed_size  # visible seed is seed_size x seed_size patches

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        gh, gw = self.grid

        # grid coordinates of every patch (row-major), shape (N, 2)
        rows = torch.arange(gh, device=device).view(gh, 1).expand(gh, gw).reshape(-1)
        cols = torch.arange(gw, device=device).view(1, gw).expand(gh, gw).reshape(-1)
        pos = torch.stack([rows, cols], dim=1).float()  # (N, 2)

        seed_idx = torch.randint(0, self.num_patches, (b,), device=device)
        seed_pos = pos[seed_idx]  # (B, 2)
        d = (pos.unsqueeze(0) - seed_pos.unsqueeze(1)).norm(dim=-1)  # (B, N)

        # visible seed neighborhood (small circular region around the seed)
        seed_radius = (self.seed_size - 1) / 2.0
        visible = d <= seed_radius + 1e-6

        # mask nearest-first from the seed boundary outward
        d_masked = d.masked_fill(visible, float("inf"))
        k = self.budget()
        nearest = d_masked.argsort(dim=1)[:, :k]
        mask = torch.zeros(b, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, nearest, True)
        # visible seed overrides masking so the seed truly remains visible
        mask &= ~visible
        deficit = (mask.sum(dim=1) < k)
        if deficit.any():
            fill = self.rand(b, self.num_patches).to(device).masked_fill(mask | visible, -1.0)
            extra = fill.topk(k, dim=1).indices
            top = torch.zeros_like(mask)
            top.scatter_(1, extra, True)
            mask = torch.where(deficit.unsqueeze(1), top, mask)
        return [mask]
