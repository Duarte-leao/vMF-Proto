#!/usr/bin/env python3
"""Crop CUB images and create train/test split folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for CUB cropping and split export."""

    parser = argparse.ArgumentParser(description="Crop CUB images using bounding boxes.")
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--cub-dir", type=str, default="CUB_200_2011")
    parser.add_argument("--cropped-dir", type=str, default="cub200_cropped")
    parser.add_argument("--train-dir", type=str, default="train_cropped")
    parser.add_argument("--test-dir", type=str, default="test_cropped")
    return parser.parse_args()


def main() -> None:
    """Write cropped train/test splits without modifying original images."""

    args = parse_args()

    data_root = Path(args.data_root)
    cub_root = data_root / args.cub_dir
    images_root = cub_root / "images"

    cropped_root = data_root / args.cropped_dir
    train_root = cropped_root / args.train_dir
    test_root = cropped_root / args.test_dir

    # Read metadata
    names = pd.read_table(
        cub_root / "images.txt", delimiter=" ", names=["id", "name"]
    ).to_numpy()
    boxes = pd.read_table(
        cub_root / "bounding_boxes.txt",
        delimiter=" ",
        names=["id", "x", "y", "width", "height"],
    ).to_numpy()
    labels = pd.read_table(
        cub_root / "train_test_split.txt", delimiter=" ", names=["id", "label"]
    ).to_numpy()

    # Create class folders
    classes = pd.read_table(
        cub_root / "classes.txt", delimiter=" ", names=["id", "folder"]
    ).to_numpy()
    for _, folder in classes:
        _mkdir(train_root / folder)
        _mkdir(test_root / folder)

    # Split images into train/test folders
    for idx in range(len(names)):
        src = images_root / names[idx][1]
        dst_root = train_root if labels[idx][1] == 1 else test_root
        x, y, w, h = boxes[idx][1:]
        with Image.open(src) as image:
            image.crop((x, y, x+w, y+h)).convert("RGB").save(dst_root / names[idx][1], quality=95)
        if (idx + 1) % 1000 == 0:
            print(f"[split] {idx + 1} / {len(names)} images copied")
    print("[split] Done.")


if __name__ == "__main__":
    main()
