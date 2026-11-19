"""Dataset/dataloader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .cub200 import CUBDataset, build_default_transforms
from .stanford_dogs import StanfordDogsDataset


def build_dataset(cfg: Any, split: str) -> Any:
    """Build and return the dataset for the requested split."""

    validate_data_config(cfg)

    dataset_name = str(cfg.dataset.name).lower()
    root = Path(str(cfg.dataset.root))
    input_size = tuple(cfg.dataset.input_size)
    transforms = build_default_transforms(
        input_size=input_size, normalize=bool(cfg.dataset.normalize)
    )

    if dataset_name == "cub200":
        cropped_dir = root / str(cfg.dataset.cropped_dir)
        split_dir = _resolve_split_dir(cfg, split)
        split_path = cropped_dir / split_dir
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        return CUBDataset(split_path, transforms=transforms)

    if dataset_name == "ap10k":
        from .ap10k import AP10KDataset
        return AP10KDataset(root, split, transforms=transforms,
            split_id=int(getattr(cfg.dataset, "split_id", 1)),
            annotation_dir=str(getattr(cfg.dataset, "annotation_dir", "annotations")),
            images_dir=str(getattr(cfg.dataset, "images_dir", "data")),
            num_classes=int(cfg.dataset.num_classes))

    if dataset_name == "stanford_cars":
        split_dir = _resolve_split_dir(cfg, split)
        split_path = root / split_dir
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        # Stanford Cars already follows ImageFolder-style split directories.
        return ImageFolder(root=str(split_path), transform=transforms)

    if dataset_name == "stanford_dogs":
        images_root = root / str(cfg.dataset.images_dir)
        split_root = root / str(cfg.dataset.split_dir)
        split_file = split_root / _resolve_split_list(cfg, split)
        # The Dogs split file points to paths under Images/.
        return StanfordDogsDataset(
            images_root=images_root,
            split_file=split_file,
            transforms=transforms,
        )

    raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")


def build_dataloader(
    cfg: Any,
    split: str,
    *,
    dataset: Optional[Any] = None,
    sampler: Optional[Any] = None,
) -> Any:
    """Build and return a PyTorch DataLoader for the requested split."""

    if dataset is None:
        dataset = build_dataset(cfg, split)

    if split == "train":
        batch_size = int(cfg.train.batch_size)
        shuffle = sampler is None
    else:
        batch_size = int(getattr(cfg.eval, "batch_size", cfg.train.batch_size))
        shuffle = sampler is None

    num_workers = int(cfg.train.num_workers)
    prefetch_factor = (
        int(getattr(cfg.train, "prefetch_factor", 2)) if num_workers > 0 else None
    )
    persistent_workers = bool(cfg.train.persistent_workers) if num_workers > 0 else False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def validate_data_config(cfg: Any) -> None:
    """
    Validate required dataset-related keys exist in the config.

    This is intentionally lightweight and does not touch disk.
    """

    dataset_name = str(getattr(cfg.dataset, "name", "")).lower()
    required_paths = [
        "dataset.name",
        "dataset.root",
        "dataset.input_size",
        "dataset.normalize",
        "train.batch_size",
        "train.num_workers",
    ]

    if dataset_name == "cub200":
        required_paths.extend(
            [
                "dataset.cub_raw_dir",
                "dataset.cropped_dir",
                "dataset.train_dir",
                "dataset.test_dir",
                "dataset.segmentations_dir",
            ]
        )
    elif dataset_name == "stanford_cars":
        required_paths.extend(
            [
                "dataset.train_dir",
                "dataset.test_dir",
            ]
        )
    elif dataset_name == "stanford_dogs":
        required_paths.extend(
            [
                "dataset.images_dir",
                "dataset.split_dir",
                "dataset.train_list",
                "dataset.test_list",
            ]
        )
    else:
        required_paths.append("dataset.name")
    missing = _missing_cfg_paths(cfg, required_paths)
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")


def _missing_cfg_paths(cfg: Any, paths: Any) -> Dict[str, str]:
    """
    Return a dict of missing keys -> hint.

    TODO: Keep this helper consistent with `scripts/*` validation behavior if you
    factor it out into `src/utils/validate.py` in the derived project.
    """

    missing: Dict[str, str] = {}
    for path in paths:
        if _get_cfg(cfg, path) is None:
            missing[path] = "Add this key to the appropriate YAML in configs/."
    return missing


def _get_cfg(cfg: Any, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _resolve_split_dir(cfg: Any, split: str) -> str:
    """Return the folder name for the requested split."""

    split = split.lower()
    if split == "train":
        return str(cfg.dataset.train_dir)
    if split == "val":
        val_dir = getattr(cfg.dataset, "val_dir", None)
        if val_dir:
            return str(val_dir)
        # Default to test split if no validation is configured.
        return str(cfg.dataset.test_dir)
    if split == "test":
        return str(cfg.dataset.test_dir)
    raise ValueError(f"Unknown split: {split}")


def _resolve_split_list(cfg: Any, split: str) -> str:
    """Return the split list filename for datasets that use split files."""

    split = split.lower()
    if split == "train":
        return str(cfg.dataset.train_list)
    if split == "val":
        val_list = getattr(cfg.dataset, "val_list", None)
        if val_list:
            return str(val_list)
        return str(cfg.dataset.test_list)
    if split == "test":
        return str(cfg.dataset.test_list)
    raise ValueError(f"Unknown split: {split}")
