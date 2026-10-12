"""Per-benchmark dataset wrappers with paper-faithful defaults.

+------------+--------------------------+------------+---------------------+
| Dataset    | Domain                   | Task       | Split protocol      |
+------------+--------------------------+------------+---------------------+
| P.A.S.T.   | Sus scrofa domesticus    | 6-class    | session isolation   |
| BirdSet    | avian (fine-grained)     | multi-label| standard splits     |
| WMM        | marine mammals           | multi-class| species-disjoint    |
| AS-20K     | general ambient / music  | multi-label| class-balanced      |
| ESC-50     | environmental sounds     | 50-class   | 5-fold CV           |
| SPC-2      | human speech transients  | 35-class   | standard splits     |
+------------+--------------------------+------------+---------------------+

All wrappers assume manifests prepared by ``tools/prepare_manifests.py``.
"""

from .audio_dataset import (
    AudioClassificationDataset,
    KShotDataset,
    UnlabeledAudioDataset,
)

PAST_CLASSES = ["cough", "sneeze", "earflap", "eat", "squeal", "hum"]


class PASTUnlabeled(UnlabeledAudioDataset):
    """P.A.S.T. unlabeled pool: 610 continuous recordings (~30 h) for SSL."""


class PASTDataset(AudioClassificationDataset):
    """P.A.S.T. annotated core: 1,440 session-isolated clips, 6 long-tailed classes.

    Use ``tools/split_past_sessions.py`` once to materialize session-level
    train/val/test manifests (8:1:1), then point ``manifest`` at the split CSV.
    """


class BirdSetDataset(AudioClassificationDataset):
    """BirdSet (Rauch et al. 2025): fine-grained avian multi-label cmAP."""

    def __init__(self, manifest, **kwargs) -> None:
        kwargs.setdefault("multilabel", True)
        super().__init__(manifest, **kwargs)


class WMMDataset(AudioClassificationDataset):
    """Watkins Marine Mammal (Sayigh et al. 2016): species-level classification."""

    def __init__(self, manifest, **kwargs) -> None:
        kwargs.setdefault("multilabel", False)
        super().__init__(manifest, **kwargs)


class AS20KDataset(AudioClassificationDataset):
    """AudioSet-20K subset (Gemmeke et al. 2017): 527-class multi-label mAP."""

    def __init__(self, manifest, num_classes: int = 527, **kwargs) -> None:
        kwargs.setdefault("multilabel", True)
        super().__init__(manifest, num_classes=num_classes, **kwargs)


class ESC50Dataset(AudioClassificationDataset):
    """ESC-50 (Piczak 2015): 50 environmental classes, 5-fold cross-validation."""

    def __init__(self, manifest, fold: int = 1, **kwargs) -> None:
        super().__init__(manifest, fold=fold, **kwargs)


class SPC2Dataset(AudioClassificationDataset):
    """Speech Commands V2 (Warden 2018): 35-word keyword spotting."""


__all__ = [
    "PAST_CLASSES",
    "PASTUnlabeled",
    "PASTDataset",
    "BirdSetDataset",
    "WMMDataset",
    "AS20KDataset",
    "ESC50Dataset",
    "SPC2Dataset",
    "AudioClassificationDataset",
    "UnlabeledAudioDataset",
    "KShotDataset",
]
