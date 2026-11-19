"""Dataset interfaces and config schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class Batch:
    """
    Simple batch container used by the training engine.

    Fields:
    - x: image tensor, shape (B, C, H, W)
    - y: integer class labels, shape (B,)
    - meta: optional dict for dataset-specific metadata (indices, paths, etc.)
    """

    x: Any
    y: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None


class DatasetLike(Protocol):
    """
    Minimal dataset protocol to support typing without committing to a framework.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


def expected_dataset_config_schema() -> Dict[str, str]:
    """
    Returns a human-readable schema description for `cfg.dataset.*`.
    """

    return {
        "dataset.name": "str: dataset identifier (e.g., cub200)",
        "dataset.root": "str: path to dataset root",
        "dataset.cub_raw_dir": "str: folder name containing CUB_200_2011 metadata",
        "dataset.cropped_dir": "str: folder name containing cropped splits",
        "dataset.train_dir": "str: train split folder under cropped_dir",
        "dataset.val_dir": "str: optional val split folder under cropped_dir",
        "dataset.test_dir": "str: test split folder under cropped_dir",
        "dataset.segmentations_dir": "str: folder name containing segmentation masks",
        "dataset.images_dir": "str: image root for datasets with split files (e.g., Stanford Dogs)",
        "dataset.split_dir": "str: folder containing split files (e.g., train_list.mat)",
        "dataset.train_list": "str: split filename for training (e.g., train_list.mat)",
        "dataset.val_list": "str: optional split filename for validation",
        "dataset.test_list": "str: split filename for testing (e.g., test_list.mat)",
        "dataset.input_size": "tuple[int,int]: model input size",
        "dataset.normalize": "bool: apply ImageNet normalization",
    }
