"""EA-MCM: Energy-Anchored Multi-Context Masking (the proposed method).

Pipeline (Sec. 2.2-2.3 of the paper):

1. **Energy-driven decoupling** - each patch is scored by its acoustic power
   ``E(P_i) = ||exp(P_i)||_1``; the top ``K_anchor = floor(rho_anchor * N)``
   patches form the high-energy focal anchor set ``A*`` (Eq. 4), the rest form
   the ambient context ``Omega_context``.
2. **Deterministic anchor lockdown** - ``A*`` is masked in *every* generated
   view, which reduces the cross-view exposure probability of any focal patch
   to exactly zero (Eq. 5-7).
3. **Stochastic ambient masking** - per view ``v``, a dynamic context mask
   ``M_dynamic^(v) ~ U(Omega_context, K_dynamic)`` with
   ``K_dynamic = floor((rho_total - rho_anchor) * N)`` is sampled exclusively
   from the ambient context.

The composite mask is ``M_total^(v) = A* | M_dynamic^(v)``.
"""

from typing import List, Optional

import torch

from .base import MaskingStrategy, patch_energy

try:
    from ..models.mae_audio import AudioMAE
except ImportError:  # pragma: no cover
    AudioMAE = None


class EAMCM(MaskingStrategy):
    name = "ea_mcm"
    multi_view = True

    def __init__(
        self,
        img_size=(1024, 128),
        patch_size=(16, 16),
        mask_ratio: float = 0.75,
        anchor_ratio: float = 0.40,
        num_views: int = 8,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> None:
        super().__init__(img_size, patch_size, mask_ratio, num_views, generator)
        if anchor_ratio >= mask_ratio:
            raise ValueError("anchor_ratio must be smaller than mask_ratio")
        self.anchor_ratio = anchor_ratio
        self.k_anchor = int(round(anchor_ratio * self.num_patches))
        self.k_dynamic = self.budget() - self.k_anchor
        if self.k_dynamic <= 0:
            raise ValueError("Empty context pool: lower anchor_ratio or raise mask_ratio")

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device

        # --- 1. acoustic power decoupling (Eq. 3-4) ---
        energy = patch_energy(raw_fbank, self.grid)  # (B, N)
        anchor_mask = torch.zeros(b, self.num_patches, dtype=torch.bool, device=device)
        anchor_ids = energy.topk(self.k_anchor, dim=1).indices  # argmax cumulative energy
        anchor_mask.scatter_(1, anchor_ids, True)

        # --- 2./3. locked anchor + stochastic ambient masks (Eq. 6-7) ---
        context_scores = self.rand(b, self.num_patches).to(device)
        context_scores = context_scores.masked_fill(anchor_mask, -1.0)  # exclude A*
        dyn_ids = context_scores.topk(self.k_dynamic, dim=1).indices
        dyn_mask = torch.zeros_like(anchor_mask)
        dyn_mask.scatter_(1, dyn_ids, True)

        composite = anchor_mask | dyn_mask  # same anchor across all views
        return [composite.clone() for _ in range(self.num_views)]

    def decompose(self, raw_fbank: torch.Tensor):
        """Expose (anchor_mask, context_mask) for analysis / visualization."""
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        energy = patch_energy(raw_fbank, self.grid)
        anchor_mask = torch.zeros(b, self.num_patches, dtype=torch.bool, device=device)
        anchor_mask.scatter_(1, energy.topk(self.k_anchor, dim=1).indices, True)
        return anchor_mask, ~anchor_mask


class NaiveRandomMultiMasking(MaskingStrategy):
    """Multi-view baseline: V *independently* sampled uniform random masks.

    Independent sampling implies a focal patch leaks with probability
    ``1 - rho_total**V`` (Eq. 5) - approximately 100% at V=8 for rho=75%.
    """

    name = "naive_random_multimasking"
    multi_view = True

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        return [self.random_mask(b, device) for _ in range(self.num_views)]
