"""Data-aware teacher-guided baselines.

* :class:`AttMask` (Mo et al., AAAI 2025) - rank patches with a teacher signal
  and mask the most (Hint) or least (Low) salient ones. In our comparison the
  teacher signal is the *acoustic energy* (marked "(Energy)" in Table 1), so
  no auxiliary forward pass is required; an attention teacher is also
  supported to reproduce the original latency profile.
* :class:`HardPatchMasking` (Wang et al., CVPR 2023) - mask patches with the
  highest reconstruction loss predicted by a teacher ("(Loss)").
* :class:`AttGuidedMasking` - mask the highest *semantic* attention patches
  ("(Semantic)"), using a (pre-trained) teacher encoder attention rollout.
"""

from typing import List, Optional

import torch

from .base import MaskingStrategy, patch_energy, topk_mask

try:
    from ..models.mae_audio import AudioMAE
except ImportError:  # pragma: no cover
    AudioMAE = None


class AttMask(MaskingStrategy):
    """AttMask-Low / AttMask-Hint.

    Args:
        mode: ``"hint"`` masks the highest-signal (hardest) patches, ``"low"``
            masks the lowest-signal ones.
        teacher_source: ``"energy"`` uses the closed-form acoustic power of
            Eq. 3 (no extra network, as in the paper's "(Energy)" setting);
            ``"attention"`` runs a frozen teacher MAE full pass and uses CLS
            attention rollout (reproduces the original AttMask cost profile).
        random_fill: fraction of the budget re-filled uniformly at random
            (stochastic mixing as in the official AttMask implementation).
    """

    name = "attmask"

    def __init__(self, mode: str = "hint", teacher_source: str = "energy", random_fill: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        assert mode in ("hint", "low")
        assert teacher_source in ("energy", "attention")
        self.mode = mode
        self.teacher_source = teacher_source
        self.random_fill = random_fill

    def _teacher_scores(self, raw_fbank, norm_fbank, teacher) -> torch.Tensor:
        if self.teacher_source == "energy":
            return patch_energy(raw_fbank, self.grid)
        if teacher is None:
            raise ValueError("teacher='attention' requires a frozen teacher AudioMAE")
        return teacher.forward_attention(norm_fbank)

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        k = self.budget()
        scores = self._teacher_scores(raw_fbank, norm_fbank, teacher)

        # stochastic mixing: part of the budget from the teacher ranking,
        # the rest uniform at random (AttMask default).
        k_guided = k - int(round(self.random_fill * k))
        guided = topk_mask(scores, k_guided, largest=(self.mode == "hint"))

        noise = self.rand(b, self.num_patches, device=device)
        noise = noise.masked_fill(guided, -1.0)
        random_part = topk_mask(noise, k - k_guided, largest=True)

        return [guided | random_part]


class HardPatchMasking(MaskingStrategy):
    """Hard-patch mining ("(Loss)" in Table 1).

    ``teacher_mode='memory'`` (default) keeps an exponential moving average of
    the per-patch reconstruction loss observed during pre-training and masks
    the highest-loss patches - no auxiliary forward pass needed.
    ``teacher_mode='forward'`` scores patches with an actual teacher MAE
    reconstruction pass at every step (reproduces the original latency).

    The loss memory is updated by the training loop via :meth:`update_memory`.
    """

    name = "hard_patch"

    def __init__(self, teacher_mode: str = "memory", momentum: float = 0.9, random_fill: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        assert teacher_mode in ("memory", "forward")
        self.teacher_mode = teacher_mode
        self.momentum = momentum
        self.random_fill = random_fill
        self.register_memory()

    def register_memory(self) -> None:
        self._loss_memory = torch.full((self.num_patches,), 0.5)

    @torch.no_grad()
    def update_memory(self, per_patch_loss: torch.Tensor) -> None:
        """EMA update with (B, N) reconstruction losses from the current step."""
        mean_loss = per_patch_loss.mean(dim=0).detach().cpu()
        self._loss_memory = (
            self.momentum * self._loss_memory + (1 - self.momentum) * mean_loss
        )

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        k = self.budget()

        if self.teacher_mode == "forward":
            if teacher is None:
                raise ValueError("teacher_mode='forward' requires a teacher AudioMAE")
            with torch.no_grad():
                # uniform probing pass, then re-rank the probed masked patches
                probe_mask = self.random_mask(b, device)
                per_patch = teacher.forward_per_patch_loss(norm_fbank, probe_mask)
            scores = per_patch.masked_fill(~probe_mask, float("-inf"))
        else:
            scores = self._loss_memory.to(device).unsqueeze(0).expand(b, -1)

        k_guided = k - int(round(self.random_fill * k))
        guided = topk_mask(scores, k_guided, largest=True)
        noise = self.rand(b, self.num_patches, device=device).masked_fill(guided, -1.0)
        random_part = topk_mask(noise, k - k_guided, largest=True)
        return [guided | random_part]


class AttGuidedMasking(MaskingStrategy):
    """Semantic-guided masking ("(Semantic)" in Table 1).

    Runs a (supervised / pre-trained) teacher encoder on the full spectrogram
    and masks the patches that receive the highest CLS attention, forcing the
    student to reconstruct semantic focal regions. Falls back to acoustic
    energy when no teacher checkpoint is supplied.
    """

    name = "att_guided"

    def __init__(self, random_fill: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.random_fill = random_fill

    def get_masks(
        self,
        raw_fbank: torch.Tensor,
        norm_fbank: torch.Tensor,
        teacher: Optional["AudioMAE"] = None,
    ) -> List[torch.Tensor]:
        b = raw_fbank.shape[0]
        device = raw_fbank.device
        k = self.budget()
        if teacher is not None:
            scores = teacher.forward_attention(norm_fbank)
        else:
            scores = patch_energy(raw_fbank, self.grid)

        k_guided = k - int(round(self.random_fill * k))
        guided = topk_mask(scores, k_guided, largest=True)
        noise = self.rand(b, self.num_patches, device=device).masked_fill(guided, -1.0)
        random_part = topk_mask(noise, k - k_guided, largest=True)
        return [guided | random_part]
