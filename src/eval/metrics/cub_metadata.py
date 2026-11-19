"""CUB-200-2011 metadata helpers for interpretability metrics.

These helpers are only used by CUB-based prototype metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from src.utils.imagenet import IMAGENET_MEAN as mean
from src.utils.imagenet import IMAGENET_STD as std


@dataclass(frozen=True)
class CUBMetadata:
    """Parsed CUB metadata used by interpretability metrics."""

    id_to_path: Dict[int, Tuple[str, str]]
    id_to_bbox: Dict[int, Tuple[int, int, int, int]]
    cls_to_id: Dict[int, List[int]]
    id_to_train: Dict[int, int]
    part_id_to_part: Dict[str, str]
    part_num: int
    id_to_part_loc: Dict[int, List[List[int]]]


def in_bbox(loc: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> bool:
    """Return True when loc lies inside bbox (x1, x2, y1, y2)."""

    return (
        loc[0] >= bbox[0]
        and loc[0] <= bbox[1]
        and loc[1] >= bbox[2]
        and loc[1] <= bbox[3]
    )


@lru_cache(maxsize=None)
def load_cub_metadata(data_root: str | Path) -> CUBMetadata:
    """Load CUB part/bbox metadata once per data root."""

    data_root = Path(data_root)
    cub_root = data_root / "CUB_200_2011"

    img_txt = cub_root / "images.txt"
    cls_txt = cub_root / "image_class_labels.txt"
    bbox_txt = cub_root / "bounding_boxes.txt"
    train_txt = cub_root / "train_test_split.txt"
    part_cls_txt = cub_root / "parts" / "parts.txt"
    part_loc_txt = cub_root / "parts" / "part_locs.txt"

    # id_to_path: image id -> (class folder, filename)
    id_to_path: Dict[int, Tuple[str, str]] = {}
    for line in img_txt.read_text().strip().splitlines():
        img_id_str, img_path = line.split(" ")
        img_id = int(img_id_str)
        img_folder, img_name = img_path.split("/")
        id_to_path[img_id] = (img_folder, img_name)

    # id_to_bbox: image id -> (x1, y1, x2, y2)
    id_to_bbox: Dict[int, Tuple[int, int, int, int]] = {}
    for line in bbox_txt.read_text().strip().splitlines():
        img_id_str, x, y, w, h = line.split(" ")
        img_id = int(img_id_str)
        bbox_x = int(float(x))
        bbox_y = int(float(y))
        bbox_w = int(float(w))
        bbox_h = int(float(h))
        id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h)

    # cls_to_id: class id -> list of image ids
    cls_to_id: Dict[int, List[int]] = {}
    for line in cls_txt.read_text().strip().splitlines():
        img_id_str, cls_id_str = line.split(" ")
        img_id = int(img_id_str)
        cls_id = int(cls_id_str) - 1
        cls_to_id.setdefault(cls_id, []).append(img_id)

    # id_to_train: image id -> 1 (train) | 0 (test)
    id_to_train: Dict[int, int] = {}
    for line in train_txt.read_text().strip().splitlines():
        img_id_str, is_train_str = line.split(" ")
        id_to_train[int(img_id_str)] = int(is_train_str)

    # part_id_to_part: part id -> part name
    part_id_to_part: Dict[str, str] = {}
    for line in part_cls_txt.read_text().strip().splitlines():
        parts = line.split(" ")
        part_id = parts[0]
        part_name = " ".join(parts[1:])
        part_id_to_part[part_id] = part_name
    part_num = len(part_id_to_part)

    # id_to_part_loc: image id -> list[part_id, x, y] for visible parts only
    id_to_part_loc: Dict[int, List[List[int]]] = {}
    for line in part_loc_txt.read_text().strip().splitlines():
        img_id_str, part_id_str, x, y, visible = line.split(" ")
        if int(visible) != 1:
            continue
        img_id = int(img_id_str)
        part_id = int(part_id_str)
        loc_x = int(float(x))
        loc_y = int(float(y))
        id_to_part_loc.setdefault(img_id, []).append([part_id, loc_x, loc_y])

    return CUBMetadata(
        id_to_path=id_to_path,
        id_to_bbox=id_to_bbox,
        cls_to_id=cls_to_id,
        id_to_train=id_to_train,
        part_id_to_part=part_id_to_part,
        part_num=part_num,
        id_to_part_loc=id_to_part_loc,
    )


class Cub2011Eval(Dataset):
    """Dataset wrapper for part-based evaluation on the CUB test split."""

    base_folder = "test_cropped"

    def __init__(self, root: str | Path, train: bool = True, transform=None) -> None:
        self.root = Path(root).expanduser()
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

    def _load_metadata(self) -> None:
        images = pd.read_csv(
            self.root / "CUB_200_2011" / "images.txt",
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            self.root / "CUB_200_2011" / "image_class_labels.txt",
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            self.root / "CUB_200_2011" / "train_test_split.txt",
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")
        self.data = self.data[self.data.is_training_img == (1 if self.train else 0)]

    def _check_integrity(self) -> bool:
        try:
            self._load_metadata()
        except Exception:
            return False

        for _, row in self.data.iterrows():
            filepath = self.root / "cub200_cropped" / self.base_folder / row.filepath
            if not filepath.is_file():
                return False
        return True

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        path = self.root / "cub200_cropped" / self.base_folder / sample.filepath
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0.
        img = self.loader(str(path))
        img_id = sample.img_id

        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_id
