#!/usr/bin/env python3
"""Offline augmentation for CUB cropped images using Augmentor."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import Augmentor
from Augmentor.Operations import RotateRange
from PIL import Image, ImageOps

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9 compatibility
    RESAMPLE_BICUBIC = Image.BICUBIC  # type: ignore[attr-defined]


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _set_short_class_label(pipeline: Augmentor.Pipeline, label: str) -> None:
    """Force a short class label to avoid overly long output filenames."""

    for augmentor_image in pipeline.augmentor_images:
        augmentor_image.class_label = label


def _safe_rotate_range_perform_operation(self, images):
    """Rotate images and center-crop back to the original size safely."""

    augmented_images = []
    for image in images:
        random_left = random.randint(self.max_left_rotation, 0)
        random_right = random.randint(0, self.max_right_rotation)
        rotation = random_left if random.randint(0, 1) == 0 else random_right

        rotated = image.rotate(rotation, expand=True, resample=RESAMPLE_BICUBIC)
        restored = ImageOps.fit(
            rotated, image.size, method=RESAMPLE_BICUBIC, centering=(0.5, 0.5)
        )
        augmented_images.append(restored)

    return augmented_images


RotateRange.perform_operation = _safe_rotate_range_perform_operation


def parse_args() -> argparse.Namespace:
    """Parse CLI args for offline augmentation of cropped images."""

    parser = argparse.ArgumentParser(description="Offline augmentation for CUB crops.")
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--cropped-dir", type=str, default="cub200_cropped")
    parser.add_argument("--source-dir", type=str, default="train_cropped")
    parser.add_argument("--target-dir", type=str, default="train_cropped_augmented")
    parser.add_argument("--passes", type=int, default=15, help="Augmentor pipeline passes per class.")
    return parser.parse_args()


def main() -> None:
    """Run class-wise Augmentor pipelines and write augmented images to disk."""

    args = parse_args()

    dataset_root = Path(args.data_root) / args.cropped_dir
    source_root = dataset_root / args.source_dir
    target_root = dataset_root / args.target_dir

    _mkdir(target_root)
    class_folders = sorted([folder for folder in source_root.iterdir() if folder.is_dir()])

    for class_folder in class_folders:
        target_folder = target_root / class_folder.name
        _mkdir(target_folder)

        short_label = class_folder.name.split(".", 1)[0]

        # Rotation augmentation
        pipeline = Augmentor.Pipeline(
            source_directory=str(class_folder.resolve()),
            output_directory=str(target_folder.resolve()),
        )
        _set_short_class_label(pipeline, short_label)
        pipeline.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        pipeline.flip_left_right(probability=0.5)
        for _ in range(args.passes):
            pipeline.process()
        del pipeline

        # Skew augmentation
        pipeline = Augmentor.Pipeline(
            source_directory=str(class_folder.resolve()),
            output_directory=str(target_folder.resolve()),
        )
        _set_short_class_label(pipeline, short_label)
        pipeline.skew(probability=1, magnitude=0.2)
        pipeline.flip_left_right(probability=0.5)
        for _ in range(args.passes):
            pipeline.process()
        del pipeline

        # Shear augmentation
        pipeline = Augmentor.Pipeline(
            source_directory=str(class_folder.resolve()),
            output_directory=str(target_folder.resolve()),
        )
        _set_short_class_label(pipeline, short_label)
        pipeline.shear(probability=1, max_shear_left=10, max_shear_right=10)
        pipeline.flip_left_right(probability=0.5)
        for _ in range(args.passes):
            pipeline.process()
        del pipeline


if __name__ == "__main__":
    main()
