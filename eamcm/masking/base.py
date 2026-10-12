"""Masking strategy base definitions and shared utilities.

All strategies share one interface::

    masks = strategy(raw_fbank, norm_fbank, teacher=None)

``raw_fbank``  : (B, 1, T, F) *unnormalized* log-Mel spectrogram (energy
                 strategies use the physical acoustic power, Eq. 3 of the paper).
``norm_fbank`` : (B, 1, T, F) normalized spectrogram fed to the encoder
                 (attention teachers use it).
``teacher``    : optional frozen AudioMAE used by attention/loss-aware
                 strategies.

Return value: a **list** of boolean masks (B, N), True = masked.
Single-view strategies return a list of length 1; multi-view strategies
(EA-MCM, Naive Random Multi-Masking) return ``num_views`` masks.
"""

from typing import List, Optional, Tuple

import torch

try:
    from ..models.mae_audio import AudioMAE
except ImportError:  # pragma: no cover
    AudioMAE = None


class MaskingStrategy:
    """Base class. Subclasses set :attr:`name` and implement :meth:`get_masks`."""

    name: str = "base"
    #: multi-view strategies return ``num_views`` masks per batch
    multi_view: bool = False

    def __init__(
        self,
        img_size: Tuple[int, int] = (1024, 128),
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        num_views: int = 1,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> None:
        # strategies without extra hyper-parameters silently accept (and
        # ignore) strategy-specific kwargs such as ``anchor_ratio`` so a
        # single call-site can pass the full hyper-parameter set.
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid[0] * self.grid[1]
        self.mask_ratio = mask_ratio
        self.num_views = num_views if self.multi_view else 1
        self.generator = generator

    # ------------------------------------------------------------------ api
    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def __call__(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: Optional[torch.Tensor] = None,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        if norm_fbank is None:
            norm_fbank = raw_fbank
        return self.get_masks(raw_fbank, norm_fbank, teacher)

    # -------------------------------------------------------------- helpers
    def rand(self, *shape, device="cpu") -> torch.Tensor:
        """Uniform noise honoring the strategy generator (reproducibility)."""
        if self.generator is not None and device == "cpu":
            return torch.rand(*shape, generator=self.generator)
        return torch.rand(*shape, device=device)

    def budget(self) -> int:
        """Number of masked patches under the total mask ratio."""
        return int(round(self.mask_ratio * self.num_patches))

    def random_mask(self, batch: int, device) -> torch.Tensor:
        """Uniform random mask (He et al. 2022) of size (B, N)."""
        noise = self.rand(batch, self.num_patches, device=device)
        ids_shuffle = noise.argsort(dim=1)
        mask = torch.zeros(batch, self.num_patches, dtype=torch.bool, device=ids_shuffle.device)
        mask.scatter_(1, ids_shuffle[:, : self.budget()], True)
        return mask

    @staticmethod
    def fold_patches(x: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """(B, 1, T, F) -> (B, N, P) patch tokens (row-major over the grid)."""
        b, c, t, f = x.shape
        gh, gw = grid
        ph, pf = t // gh, f // gw
        return (
            x.reshape(b, c, gh, ph, gw, pf)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, gh * gw, ph * pf * c)
        )


def patch_energy(raw_fbank: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    """Acoustic power operator of Eq. (3):  E(P_i) = || exp(P_i) ||_1.

    The per-sample global maximum is subtracted before the exponential for
    numerical stability. Because this shift is constant within a sample, the
    induced patch *ranking* (and therefore the anchor set of Eq. 4) is
    mathematically identical to the unshifted formulation.

    Returns:
        (B, N) non-negative patch energies.
    """
    x = raw_fbank.squeeze(1)  # (B, T, F)
    x = x - x.amax(dim=(1, 2), keepdim=True)
    x = x.exp()
    b, t, f = x.shape
    gh, gw = grid
    ph, pf = t // gh, f // gw
    x = x.reshape(b, gh, ph, gw, pf).sum(dim=(2, 4))
    return x.reshape(b, gh * gw)


def topk_mask(scores: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """(B, N) scores -> (B, N) bool mask selecting top-k per row."""
    idx = scores.topk(k, dim=1, largest=largest).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask
