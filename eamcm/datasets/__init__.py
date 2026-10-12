"""Datasets for EA-MCM pre-training and downstream evaluation."""

from .audio_dataset import (
    AudioClassificationDataset,
    KShotDataset,
    UnlabeledAudioDataset,
    build_label_maps,
    read_manifest,
    write_manifest,
)
from .benchmarks import (
    AS20KDataset,
    BirdSetDataset,
    ESC50Dataset,
    PAST_CLASSES,
    PASTDataset,
    PASTUnlabeled,
    SPC2Dataset,
    WMMDataset,
)

__all__ = [
    "AudioClassificationDataset",
    "UnlabeledAudioDataset",
    "KShotDataset",
    "read_manifest",
    "write_manifest",
    "build_label_maps",
    "PAST_CLASSES",
    "PASTUnlabeled",
    "PASTDataset",
    "BirdSetDataset",
    "WMMDataset",
    "AS20KDataset",
    "ESC50Dataset",
    "SPC2Dataset",
]
