"""Masking strategies for audio MAE pre-training.

Strategies implement a single interface: given the raw and normalized fbank,
return a list of boolean masks (B, N); multi-view strategies return
``num_views`` masks whose anchor set (EA-MCM) is shared across views.
"""

from .attmask import AttGuidedMasking, AttMask, HardPatchMasking
from .base import MaskingStrategy, patch_energy
from .ea_mcm import EAMCM, NaiveRandomMultiMasking
from .random import InverseBlockMasking, RandomMasking, TimeFreqMasking
from .registry import build_masking, seed_generator

__all__ = [
    "MaskingStrategy",
    "RandomMasking",
    "TimeFreqMasking",
    "InverseBlockMasking",
    "AttMask",
    "HardPatchMasking",
    "AttGuidedMasking",
    "NaiveRandomMultiMasking",
    "EAMCM",
    "build_masking",
    "seed_generator",
    "patch_energy",
]
