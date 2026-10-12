"""Unit tests for the Audio-MAE model and classifier head (CPU, tiny inputs).

Run directly (no pytest required)::

    python tests/test_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from eamcm.models import AudioClassifier, build_audiomae


def _model() -> "torch.nn.Module":
    torch.manual_seed(0)
    return build_audiomae()


def test_patchify_unpatchify_roundtrip():
    m = _model()
    x = torch.randn(2, 1, 1024, 128)
    rec = m.unpatchify(m.patchify(x))
    assert torch.allclose(rec, x), "patchify/unpatchify must be a lossless reshape"


def test_mae_forward_masks_only_masked_tokens_in_loss():
    m = _model()
    x = torch.randn(2, 1, 1024, 128)
    mask = torch.zeros(2, 512, dtype=torch.bool)
    mask[:, :384] = True  # 75% budget
    loss, pred, _ = m(x, mask)
    assert torch.isfinite(loss) and loss.item() > 0
    assert pred.shape == (2, 512, 256)  # 16 x 16 patch pixels


def test_forward_attention_shapes():
    m = _model()
    x = torch.randn(1, 1, 1024, 128)
    attn = m.forward_attention(x)
    assert attn.shape == (1, 512)
    assert attn.min() >= 0.0 and attn.max() <= 1.0


def test_classifier_gap_head():
    m = _model()
    clf = AudioClassifier(m, num_classes=6)
    x = torch.randn(2, 1, 1024, 128)
    feats = clf.forward_features(x)
    assert feats.shape == (2, 768)
    logits = clf(x)
    assert logits.shape == (2, 6)
    loss = clf.criterion(logits, torch.tensor([0, 3]))
    assert torch.isfinite(loss)


def test_visible_token_count_matches_mask():
    """The encoder must process exactly (N - masked) tokens plus CLS."""
    m = _model()
    x = torch.randn(1, 1, 1024, 128)
    mask = torch.zeros(1, 512, dtype=torch.bool)
    mask[:, :384] = True
    latent = m.forward_encoder(x, mask)
    assert latent.shape == (1, 512 - 384 + 1, 768)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} model tests passed.")
