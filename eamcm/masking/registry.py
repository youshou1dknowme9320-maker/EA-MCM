"""Registry to instantiate masking strategies by name.

Usage::

    from eamcm.masking.registry import build_masking
    strategy = build_masking("ea_mcm", mask_ratio=0.75, anchor_ratio=0.40, num_views=8)
    masks = strategy(raw_fbank, norm_fbank, teacher=None)  # list of V x (B, N)
"""

from typing import Optional

import torch

from .attmask import AttGuidedMasking, AttMask, HardPatchMasking
from .base import MaskingStrategy
from .ea_mcm import EAMCM, NaiveRandomMultiMasking
from .random import InverseBlockMasking, RandomMasking, TimeFreqMasking

STRATEGIES = {
    "random": RandomMasking,
    "time_freq": TimeFreqMasking,
    "inverse_block": InverseBlockMasking,
    "attmask_low": lambda **kw: AttMask(mode="low", **kw),
    "attmask_hint": lambda **kw: AttMask(mode="hint", **kw),
    "hard_patch": HardPatchMasking,
    "att_guided": AttGuidedMasking,
    "naive_random_multimasking": NaiveRandomMultiMasking,
    "ea_mcm": EAMCM,
}


def build_masking(name: str, **kwargs) -> MaskingStrategy:
    """Build a masking strategy by registry name.

    Common kwargs: ``img_size``, ``patch_size``, ``mask_ratio``, ``num_views``,
    ``generator``; strategy-specific kwargs are forwarded (e.g. ``anchor_ratio``
    for EA-MCM, ``mode``/``teacher_source`` for AttMask).
    """
    if name not in STRATEGIES:
        raise KeyError(f"Unknown masking strategy '{name}'. Available: {sorted(STRATEGIES)}")
    if "generator" not in kwargs:
        kwargs["generator"] = None
    return STRATEGIES[name](**kwargs)


def seed_generator(seed: int) -> Optional[torch.Generator]:
    """Deterministic per-strategy generator for reproducible mask sampling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
