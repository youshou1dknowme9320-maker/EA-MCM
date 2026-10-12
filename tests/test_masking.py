"""Unit tests for the masking-strategy library.

Run directly (no pytest required)::

    python tests/test_masking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from eamcm.masking import build_masking, seed_generator
from eamcm.masking.base import patch_energy

STRATEGIES = [
    "random", "time_freq", "inverse_block", "attmask_low", "attmask_hint",
    "hard_patch", "att_guided", "naive_random_multimasking", "ea_mcm",
]

B, T, F = 2, 1024, 128


def _inputs():
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(B, 1, T, F, generator=g)
    return raw, torch.randn_like(raw)


def _build(name, **overrides):
    kwargs = dict(img_size=(T, F), patch_size=(16, 16), mask_ratio=0.75,
                  anchor_ratio=0.40, num_views=8, generator=seed_generator(0))
    if name.startswith("attmask"):
        kwargs.update(teacher_source="energy")  # mode is fixed by the registry alias
    kwargs.update(overrides)
    return build_masking(name, **kwargs)


def test_masking_budget_exact_for_all_strategies():
    """Every strategy must consume the full masking budget and no more."""
    raw, norm = _inputs()
    for name in STRATEGIES:
        s = _build(name)
        masks = s(raw, norm)
        n_views = s.num_views if s.multi_view else 1
        assert len(masks) == n_views, name
        for m in masks:
            assert m.dtype == torch.bool and m.shape == (B, s.num_patches), name
            counts = m.sum(dim=1)
            assert counts.eq(s.budget()).all(), f"{name}: budget {counts.tolist()} != {s.budget()}"


def test_ea_mcm_anchor_shared_across_views():
    """The anchor set must be identical in every view (zero cross-view leak)."""
    raw, norm = _inputs()
    s = _build("ea_mcm")
    masks = s(raw, norm)
    for m in masks[1:]:
        assert torch.equal(m, masks[0]), "EA-MCM composite masks must be identical across views"


def test_ea_mcm_anchor_never_visible():
    """Anchor patches stay masked in all views; ambient patches vary somewhere."""
    raw, norm = _inputs()
    s = _build("ea_mcm")
    anchor, context = s.decompose(raw)
    assert (anchor | context).all() and not (anchor & context).any()
    k_anchor = int(round(0.40 * s.num_patches))
    assert anchor.sum(dim=1).eq(k_anchor).all()
    masks = s(raw, norm)
    for m in masks:
        assert (m & anchor.bool()).sum(dim=1).eq(k_anchor).all(), "anchor must be fully masked"


def test_naive_multiview_leaks_focal_patches():
    """Independent multi-view sampling must expose patches EA-MCM never does."""
    raw, norm = _inputs()
    energy = patch_energy(raw, (64, 8))
    anchor = energy.topk(int(round(0.40 * 512)), dim=1).indices  # (B, k_anchor)

    def leak_per_sample(masks):
        leak = torch.zeros(B, dtype=torch.bool)
        for m in masks:  # True = masked -> ~m at anchor = focal patch visible
            leak |= (~m).gather(1, anchor).any(dim=1)
        return leak

    leak_naive = leak_per_sample(_build("naive_random_multimasking")(raw, norm))
    leak_eamcm = leak_per_sample(_build("ea_mcm")(raw, norm))

    assert leak_naive.all(), "naive multi-view should leak anchors in ~every sample"
    assert not leak_eamcm.any(), "EA-MCM must never leak anchor patches"


def test_patch_energy_ranks_high_power_first():
    """A patch block with larger linear magnitude must receive higher energy."""
    x = torch.full((1, 1, 1024, 128), -10.0)
    x[:, :, :16, :16] = 2.0  # one loud patch block (patch grid row 0, col 0)
    e = patch_energy(x, (64, 8)).reshape(64, 8)
    assert e[0, 0] == e.max()
    assert e[0, 0] > e[1, 0]


def test_generator_makes_deterministic_masks():
    raw, norm = _inputs()
    m1 = _build("ea_mcm")(raw, norm)
    m2 = _build("ea_mcm")(raw, norm)
    assert all(torch.equal(a, b) for a, b in zip(m1, m2))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} masking tests passed.")
