"""Stanford Dogs dataset helpers.

The dataset ships as ImageNet-style class folders under `Images/` but the
official train/test split is provided via MATLAB files in `train_test_split/`.
This module reads those split files so we can keep the canonical split without
restructuring the data on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset


def _mat_to_list(mat_array: np.ndarray) -> list[str]:
    """Flatten MATLAB string arrays to a plain Python list of strings."""

    flattened = mat_array.squeeze()
    items: list[str] = []
    for entry in flattened:
        if isinstance(entry, np.ndarray):
            entry = entry[0]
        items.append(str(entry))
    return items


class StanfordDogsDataset(Dataset):
    """Dataset wrapper that respects the official Stanford Dogs split files."""

    def __init__(
        self,
        images_root: str | Path,
        split_file: str | Path,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *,
        return_index: bool = True,
    ) -> None:
        self.images_root = Path(images_root)
        self.split_file = Path(split_file)
        self.transform = transforms
        self.target_transform = target_transform
        self.return_index = return_index

        if not self.images_root.exists():
            raise FileNotFoundError(f"Images root not found: {self.images_root}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        split_data = sio.loadmat(self.split_file.as_posix())
        file_list = _mat_to_list(split_data["file_list"])
        labels = split_data["labels"].squeeze().astype(int) - 1

        if len(file_list) != len(labels):
            raise ValueError(
                f"Split file mismatch: {len(file_list)} paths vs {len(labels)} labels."
            )

        num_classes = int(labels.max()) + 1
        classes: list[Optional[str]] = [None] * num_classes
        samples: list[tuple[Path, int]] = []

        for rel_path, label in zip(file_list, labels):
            rel_path = Path(rel_path)
            class_name = rel_path.parts[0]
            if classes[label] is None:
                classes[label] = class_name
            elif classes[label] != class_name:
                raise ValueError(
                    f"Label {label} maps to multiple classes: "
                    f"{classes[label]} vs {class_name}."
                )

            full_path = self.images_root / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"Image file not found: {full_path}")
            samples.append((full_path, int(label)))

        # Fill any missing class names defensively.
        self.classes = [
            name if name is not None else f"class_{idx:03d}"
            for idx, name in enumerate(classes)
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = samples
        self.targets = [label for _, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_index:
            return image, label, index
        return image, label


__all__ = ["StanfordDogsDataset"]
