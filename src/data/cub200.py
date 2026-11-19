"""CUB-200-2011 datasets and helpers.

This module centralizes dataset loading for training and evaluation. All paths
are derived from the provided root so the repo never relies on hard-coded
locations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import os
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.augmentations.crops.functional import crop_keypoint_by_coords
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import to_tensor

from src.utils.imagenet import IMAGENET_MEAN, IMAGENET_STD


def build_default_transforms(
    input_size: Tuple[int, int] = (224, 224), *, normalize: bool = True
) -> Compose:
    """Return the standard resize pipeline with optional ImageNet normalization."""

    ops = [Resize(input_size), ToTensor()]
    if normalize:
        ops.append(Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return Compose(ops)


class CUBDataset(ImageFolder):
    """Simple ImageFolder-backed dataset for CUB cropped splits."""

    def __init__(
        self,
        root: str | Path,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *,
        return_index: bool = True,
    ) -> None:
        super().__init__(
            root=str(root), transform=transforms, target_transform=target_transform
        )
        self.return_index = return_index

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = Image.open(im_path).convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_index:
            return im, label, index
        return im, label


class CUBEvalDataset(ImageFolder):
    """Evaluation dataset that returns part keypoints and attributes.

    This dataset is used by some interpretability metrics that require part
    annotations and class-level attributes.
    """

    def __init__(
        self,
        images_root: str | Path,
        annotations_root: Optional[str | Path] = None,
        *,
        normalization: bool = True,
        input_size: int = 224,
    ) -> None:
        transforms = [A.Resize(width=input_size, height=input_size)]
        if normalization:
            transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

        super().__init__(
            root=str(images_root),
            transform=A.Compose(
                transforms,
                keypoint_params=A.KeypointParams(
                    format="xy",
                    label_fields=None,
                    remove_invisible=True,
                    angle_in_degrees=True,
                ),
            ),
            target_transform=None,
        )
        self.input_size = input_size
        annotations_root = (
            Path(annotations_root)
            if annotations_root
            else Path(images_root).parents[2] / "CUB_200_2011"
        )

        path_df = pd.read_csv(
            annotations_root / "images.txt",
            header=None,
            names=["image_id", "image_path"],
            sep=" ",
        )
        bbox_df = pd.read_csv(
            annotations_root / "bounding_boxes.txt",
            header=None,
            names=["image_id", "x", "y", "w", "h"],
            sep=" ",
        )
        self.bbox_df = path_df.merge(bbox_df, on="image_id")
        self.part_loc_df = pd.read_csv(
            annotations_root / "parts" / "part_locs.txt",
            header=None,
            names=["image_id", "part_id", "kp_x", "kp_y", "visible"],
            sep=" ",
        )

        attributes_np = np.loadtxt(
            annotations_root / "attributes" / "class_attribute_labels_continuous.txt"
        )
        self.attributes = F.normalize(
            torch.tensor(attributes_np, dtype=torch.float32), p=2, dim=-1
        )

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = np.array(Image.open(im_path).convert("RGB"))

        row = self.bbox_df[
            self.bbox_df["image_path"] == "/".join(Path(im_path).parts[-2:])
        ].iloc[0]
        image_id = row["image_id"]
        bbox_coords = row[["x", "y", "w", "h"]].values.flatten()

        mask = self.part_loc_df["image_id"] == image_id
        keypoints = self.part_loc_df[mask][["kp_x", "kp_y"]].values

        keypoints_cropped = [
            crop_keypoint_by_coords(
                keypoint=tuple(kp) + (None, None), crop_coords=bbox_coords[:2]
            )
            for kp in keypoints
        ]
        keypoints_cropped = [
            (np.clip(x, 0, self.input_size), np.clip(y, 0, self.input_size))
            for x, y, _, _ in keypoints_cropped
        ]

        transformed = self.transform(image=im, keypoints=keypoints_cropped)
        transformed_im, transformed_keypoints = (
            transformed["image"],
            transformed["keypoints"],
        )

        return (
            to_tensor(transformed_im),
            torch.tensor(transformed_keypoints, dtype=torch.float32),
            label,
            self.attributes[label, :],
            index,
        )
