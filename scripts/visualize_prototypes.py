#!/usr/bin/env python3
"""Visualization utility for prototype activation panels.

This script renders publication-ready grids containing the network input image
and prototype heatmaps overlaid on top. It mirrors the training/evaluation
preprocessing pipeline (Resize 224 -> ToTensor -> ImageNet normalization) and
reuses the repo's dataset builder so it can visualize runs trained on:

* CUB-200-2011 (cropped ImageFolder splits under ``<root>/cub200_cropped``),
* Stanford Dogs (official ``train_list.mat`` / ``test_list.mat`` splits),
* Stanford Cars (ImageFolder splits under ``<root>/stanford_cars``).

Prototype identifiers accept three flavours:

* Scoped index ``k`` (0..K-1) interpreted relative to the current focus class;
* Explicit ``c:k`` or ``bg:k`` forms for class-specific selection;
* Flat global indices ``g`` in [0, (C+1)*K - 1].

Raw heatmaps share a per-panel color scale by default (min/max across the
displayed maps) to aid visual comparison; ``--no-shared-scale`` switches to
per-tile scaling. Paste-to-original is intentionally omitted until crop metadata
is tracked during preprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from matplotlib import cm
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader


def _add_repo_to_path() -> None:
    """Ensure both repo root and src/ are importable for tool scripts."""

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    for path in (repo_root, src_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_add_repo_to_path()

from data.dataloader import build_dataset
from src.utils.imagenet import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Define CLI arguments for prototype visualization."""

    parser = argparse.ArgumentParser("Prototype evidence visualization")

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "cub200", "ap10k", "stanford_dogs", "stanford_cars"],
        help="Dataset to visualize. 'auto' reads cfg.dataset.name from the checkpoint.",
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for forward passes"
    )

    parser.add_argument(
        "--train-dir-name",
        type=str,
        default=None,
        help="Optional override for cfg.dataset.train_dir (cub200/stanford_cars).",
    )
    parser.add_argument(
        "--test-dir-name",
        type=str,
        default=None,
        help="Optional override for cfg.dataset.test_dir (cub200/stanford_cars).",
    )
    parser.add_argument(
        "--val-dir-name",
        type=str,
        default=None,
        help="Optional override for cfg.dataset.val_dir (cub200/stanford_cars).",
    )

    parser.add_argument("--focus", type=str, choices=["pred", "gt"], default="pred")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--proto-ids",
        type=str,
        default=None,
        help="Comma-separated prototype ids (k, c:k, bg:k, or global index)",
    )
    parser.add_argument("--include-bg", action="store_true")

    parser.add_argument(
        "--map-source", type=str, choices=["raw", "norm", "resp"], default="resp"
    )
    parser.add_argument(
        "--norm-mode", type=str, choices=["minmax", "maxonly"], default="maxonly"
    )
    parser.add_argument("--resp-temp", type=float, default=1.5)
    parser.add_argument("--display-temperature", type=float, default=None,
                        help="Display maps: spatial softmax of T*cosine. Display only; no foreground gating by default.")
    parser.add_argument("--blur-kernel", type=int, choices=[0, 1, 3, 5], default=0)
    parser.add_argument(
        "--resize-mode", type=str, choices=["nearest", "bilinear"], default="bilinear"
    )

    parser.add_argument(
        "--fg-gate",
        action="store_true",
        help="Multiply maps by foreground mask when available",
    )

    parser.add_argument(
        "--fg-mask-alpha",
        type=float,
        default=0.5,
        help="Overlay alpha for the *_fg_mask.png output.",
    )
    parser.add_argument(
        "--fg-mask-color",
        type=str,
        default="255,0,0",
        help="Comma separated RGB values (0-255) used for the fg mask overlay.",
    )

    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--cmap",
        type=str,
        choices=["turbo", "jet", "magma", "viridis"],
        default="jet",
    )
    parser.add_argument(
        "--grid", type=str, choices=["2x2", "3x2", "custom"], default="custom"
    )
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--show-image-first", action="store_true")
    parser.add_argument(
        "--shared-scale", dest="shared_scale", action="store_true", default=True
    )
    parser.add_argument("--no-shared-scale", dest="shared_scale", action="store_false")

    parser.add_argument("--annotate", type=str, default="proto_id,score")
    parser.add_argument(
        "--score-type", type=str, choices=["max", "mean", "sum"], default="max"
    )
    parser.add_argument("--mark-peak", action="store_true")
    parser.add_argument(
        "--title", type=str, choices=["pred", "gt", "both", "none"], default="both"
    )
    parser.add_argument("--colorbar", action="store_true")

    parser.add_argument("--out-dir", type=str, default="runs/vis")
    parser.add_argument("--file-prefix", type=str, default="panel")
    parser.add_argument("--save-maps", action="store_true")

    parser.add_argument("--wandb", type=str, choices=["on", "off"], default="off")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument(
        "--indices",
        type=str,
        nargs="*",
        default=None,
        help="Explicit indices or file path containing indices",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--only-misclassified", action="store_true")

    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figure-width", type=float, default=10.0)
    parser.add_argument("--figure-height", type=float, default=6.0)

    return parser


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


@dataclass
class PrototypeSelection:
    """A (class, prototype) selection used for panel generation."""

    class_idx: int
    proto_idx: int
    label: str  # textual identifier for metadata/titles


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, object]:
    """Instantiate a model from checkpoint + saved hparams.

    This script instantiates the model defined by the checkpoint config by
    delegating construction to `src/models/api.py::build_model`.
    """

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "hparams" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing 'hparams' and/or 'state_dict'")

    hparams = ckpt["hparams"]
    state_dict = ckpt["state_dict"]

    from omegaconf import OmegaConf

    from models.api import build_model

    cfg = OmegaConf.create(hparams)
    net = build_model(cfg)
    if any(key.startswith("prototype_mu_init") for key in state_dict):
        enable_parametric = getattr(net, "enable_parametric_prototypes", None)
        if callable(enable_parametric):
            enable_parametric()

    missing, unexpected = net.load_state_dict(state_dict, strict=str(getattr(cfg.model, "recipe", "legacy")) == "four_loss")
    if missing:
        print(f"[visualize] Warning: {len(missing)} missing keys", file=sys.stderr)
    if unexpected:
        print(
            f"[visualize] Warning: {len(unexpected)} unexpected keys", file=sys.stderr
        )

    if hasattr(net, "maps_norm_mode"):
        net.maps_norm_mode = args.norm_mode
    net.eval()
    return net, cfg


def _cfg_get(cfg: object, path: str, default: object | None = None) -> object | None:
    cur: object = cfg
    for part in path.split("."):
        if isinstance(cur, argparse.Namespace):
            cur = getattr(cur, part, None)
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def load_dataset(args: argparse.Namespace, cfg: object) -> tuple[Any, bool]:
    """Load the dataset split described by the checkpoint config.

    The training pipeline supports multiple datasets (CUB-200, Stanford Dogs,
    Stanford Cars). This visualization script mirrors that behavior by reusing
    `data.dataloader.build_dataset` instead of hard-coding CUB split folders.
    """

    from omegaconf import OmegaConf

    dataset_name = str(args.dataset).lower()
    if dataset_name == "auto":
        dataset_name = str(_cfg_get(cfg, "dataset.name", "cub200")).lower()

    # Clone the config so we can apply CLI overrides without mutating the model config.
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg_copy, False)

    cfg_copy.dataset.name = dataset_name
    cfg_copy.dataset.root = str(args.data_dir)

    # Optional split-dir overrides (mainly for cub200 / stanford_cars).
    if args.train_dir_name is not None and hasattr(cfg_copy.dataset, "train_dir"):
        cfg_copy.dataset.train_dir = str(args.train_dir_name)
    if args.test_dir_name is not None and hasattr(cfg_copy.dataset, "test_dir"):
        cfg_copy.dataset.test_dir = str(args.test_dir_name)
    if args.val_dir_name is not None:
        cfg_copy.dataset.val_dir = str(args.val_dir_name)

    # Satisfy minimal training keys required by validate_data_config().
    if not hasattr(cfg_copy, "train") or cfg_copy.train is None:
        cfg_copy.train = {}
    cfg_copy.train.batch_size = int(
        getattr(cfg_copy.train, "batch_size", args.batch_size)
    )
    cfg_copy.train.num_workers = int(getattr(cfg_copy.train, "num_workers", 0))

    normalize = bool(getattr(cfg_copy.dataset, "normalize", True))
    dataset = build_dataset(cfg_copy, args.split)
    return dataset, normalize


def to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to a device with non-blocking transfer when possible."""

    return batch.to(device=device, non_blocking=True)


def unpack_dataset_item(
    item: object, fallback_index: int
) -> tuple[torch.Tensor, int, int]:
    """Normalize dataset items to (image, label, index)."""

    if not isinstance(item, (tuple, list)):
        raise ValueError(f"Unsupported dataset item type: {type(item)}")

    if len(item) < 2:
        raise ValueError(
            f"Dataset item must contain at least (image, label). Got: {item}"
        )

    image = item[0]
    label = int(item[1])
    if len(item) >= 3:
        try:
            index = int(item[2])
        except Exception:
            index = int(fallback_index)
    else:
        index = int(fallback_index)
    return image, label, index


def denormalize_image(tensor: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Invert ImageNet normalization for visualization."""

    if not normalize:
        return tensor
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(
        3, 1, 1
    )
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(
        3, 1, 1
    )
    return tensor * std + mean


def parse_rgb_triplet(color: str) -> np.ndarray:
    """Parse an 'R,G,B' string into a float RGB triplet in [0, 1]."""

    parts = [p.strip() for p in str(color).split(",")]
    if len(parts) != 3:
        raise ValueError("Color must be provided as R,G,B.")
    rgb = np.array([float(p) for p in parts], dtype=np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _as_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Normalize mask layout to (B, 1, H, W) float tensor."""

    mask_tensor = mask
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor[None, None, ...]
    elif mask_tensor.dim() == 3:
        # Accept (B, H, W) or (1, H, W).
        mask_tensor = mask_tensor[:, None, ...]
    elif mask_tensor.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported fg_mask shape: {tuple(mask_tensor.shape)}")
    return mask_tensor.to(dtype=torch.float32)


def resolve_fg_mask(
    *,
    model: torch.nn.Module,
    outputs: dict,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    """Return an fg mask on the token grid as (B, 1, H, W) in [0, 1].

    We prefer the mask returned by the model (when available). Otherwise we
    compute it using the model's foreground extractor (mirrors debug_foreground.py).
    """

    fg_mask = outputs.get("fg_mask")
    if torch.is_tensor(fg_mask):
        return _as_mask_tensor(fg_mask).clamp(0.0, 1.0)

    backbone = getattr(model, "backbone", None)
    fg_extractor = getattr(model, "fg_extractor", None)
    if backbone is None or fg_extractor is None:
        return None

    backbone_out = backbone(images)
    if not isinstance(backbone_out, tuple):
        raise ValueError("Backbone must return a tuple of tensors.")
    if len(backbone_out) == 3:
        patch_tokens, raw_patch_tokens, cls_tokens = backbone_out
    elif len(backbone_out) == 2:
        patch_tokens, cls_tokens = backbone_out
        raw_patch_tokens = patch_tokens
    else:
        raise ValueError("Unexpected number of outputs from backbone.")

    fg_tokens = raw_patch_tokens if raw_patch_tokens is not None else patch_tokens
    fg_tokens_norm = F.normalize(fg_tokens, p=2, dim=-1)

    fg_kwargs: dict[str, torch.Tensor | None] = {"cls_tokens": cls_tokens}
    if bool(getattr(model, "fg_requires_images", False)):
        fg_kwargs["images"] = images

    pseudo_patch_labels = fg_extractor(fg_tokens_norm, labels, **fg_kwargs)

    gate = getattr(fg_extractor, "_last_gate", None)
    if torch.is_tensor(gate):
        return _as_mask_tensor(gate).clamp(0.0, 1.0)

    bg_class = getattr(fg_extractor, "bg_class", None)
    if bg_class is None:
        return None
    return (pseudo_patch_labels != int(bg_class)).to(dtype=torch.float32).unsqueeze(1)


def save_fg_mask_overlay(
    out_path: Path,
    *,
    base_image: np.ndarray,
    fg_mask: torch.Tensor,
    alpha: float,
    color_rgb: np.ndarray,
) -> None:
    """Save an RGB image with the foreground mask overlaid (debug_foreground style)."""

    img_h, img_w = int(base_image.shape[0]), int(base_image.shape[1])
    mask_up = F.interpolate(
        fg_mask,
        size=(img_h, img_w),
        mode="nearest",
    )[0, 0]
    mask_np = mask_up.detach().cpu().numpy().astype(np.float32)
    mask_np = np.clip(mask_np, 0.0, 1.0)

    overlay = base_image * (1.0 - alpha * mask_np[..., None])
    overlay = overlay + color_rgb[None, None, :] * (alpha * mask_np[..., None])
    overlay = np.clip(overlay, 0.0, 1.0)

    Image.fromarray((overlay * 255.0).round().astype(np.uint8)).save(out_path)


def parse_indices_argument(
    spec: Sequence[str] | None, dataset_len: int
) -> list[int] | None:
    """Parse explicit dataset indices from CLI args.

    Accepts either:
      - a list of tokens (e.g. `--indices 1 2 3`), or
      - a single file path containing indices (whitespace/comma separated).
    """

    if not spec:
        return None
    if len(spec) == 1 and Path(spec[0]).exists():
        content = Path(spec[0]).read_text().replace("\n", " ")
        tokens = content.replace(",", " ").split()
    else:
        tokens: list[str] = []
        for part in spec:
            tokens.extend(part.replace(",", " ").split())
    indices: list[int] = []
    for token in tokens:
        if token.strip() == "":
            continue
        idx = int(token)
        if idx < 0 or idx >= dataset_len:
            raise ValueError(f"Index {idx} out of range (0..{dataset_len-1})")
        indices.append(idx)
    return indices


def sample_indices(
    dataset: Any,
    args: argparse.Namespace,
    predictions: list[int] | None,
    labels: list[int] | None,
) -> list[int]:
    """Select dataset indices based on CLI sampling constraints."""

    rng = random.Random(args.seed)
    available = list(range(len(dataset)))

    if args.only_misclassified:
        if predictions is None or labels is None:
            raise ValueError("Predictions required for --only-misclassified")
        available = [i for i, (p, y) in enumerate(zip(predictions, labels)) if p != y]

    if args.class_id is not None:
        available = [i for i in available if dataset.samples[i][1] == args.class_id]

    if not available:
        raise RuntimeError("No indices available after filtering")

    if args.per_class is not None and args.class_id is not None:
        rng.shuffle(available)
        return available[: args.per_class]

    if args.num_samples is not None:
        rng.shuffle(available)
        return available[: args.num_samples]

    return available


def compute_predictions(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], list[int]]:
    """Run a forward pass over a dataset and return (predictions, labels)."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    preds: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            images, targets = batch[:2]
            images = to_device(images, device)
            outputs = model(images)
            logits = outputs["class_logits"]
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(targets.tolist())
    return preds, labels


def gaussian_kernel(
    kernel_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a 1D Gaussian kernel (normalized to sum to 1)."""

    if kernel_size <= 1:
        return torch.tensor([1.0], device=device, dtype=dtype)
    sigma = kernel_size / 3.0
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d


def apply_token_blur(maps: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply separable 1D Gaussian smoothing over a token grid tensor."""

    if kernel_size <= 1:
        return maps
    kernel_1d = gaussian_kernel(kernel_size, maps.device, maps.dtype)
    kernel = kernel_1d.view(1, 1, -1)
    maps = F.conv1d(
        maps.view(-1, 1, maps.size(-1)),
        kernel,
        padding=kernel.size(-1) // 2,
        groups=1,
    ).view_as(maps)
    maps = maps.transpose(-1, -2)
    maps = F.conv1d(
        maps.contiguous().view(-1, 1, maps.size(-1)),
        kernel,
        padding=kernel.size(-1) // 2,
        groups=1,
    ).view_as(maps)
    return maps.transpose(-1, -2)


def softmax_with_temp(logits: torch.Tensor, dim: int, temp: float) -> torch.Tensor:
    """Compute a temperature-scaled softmax along `dim`."""

    if temp <= 0:
        raise ValueError("Temperature must be positive")
    scaled = logits / temp
    return F.softmax(scaled, dim=dim)


def parse_annotate(arg: str) -> list[str]:
    """Parse the `--annotate` argument into a list of supported entries."""

    entries = [item.strip().lower() for item in arg.split(",") if item.strip()]
    if "none" in entries:
        return []
    allowed = {"proto_id", "score", "peak"}
    for entry in entries:
        if entry not in allowed:
            raise ValueError(f"Unsupported annotation entry '{entry}'")
    return entries


def resolve_proto_ids(
    proto_spec: str | None,
    focus_class: int,
    num_classes: int,
    num_prototypes: int,
) -> list[PrototypeSelection] | None:
    """Resolve a `--proto-ids` specification into `(class, prototype)` selections."""

    if proto_spec is None:
        return None
    tokens = [
        tok.strip() for tok in proto_spec.replace("\n", " ").split(",") if tok.strip()
    ]
    explicit_tokens = [
        tok
        for tok in tokens
        if ":" in tok or tok.lower().startswith("bg") or "p" in tok.lower()
    ]

    def _make_selection(class_idx: int, proto_idx: int) -> PrototypeSelection:
        if class_idx < 0 or class_idx > num_classes:
            raise ValueError(f"Class index {class_idx} invalid")
        if proto_idx < 0 or proto_idx >= num_prototypes:
            raise ValueError(
                f"Prototype index {proto_idx} invalid for class {class_idx}"
            )
        label = (
            f"c={class_idx}:p={proto_idx}"
            if class_idx != num_classes
            else f"bg:p={proto_idx}"
        )
        return PrototypeSelection(class_idx, proto_idx, label)

    if explicit_tokens:
        selections: list[PrototypeSelection] = []
        for token in tokens:
            lower = token.lower().strip()
            class_idx: int | None = None
            proto_idx: int | None = None

            if ":" in lower:
                left, right = lower.split(":", 1)
                cls_part = left.strip()
                proto_idx = int(right.strip().lstrip("p"))
                if cls_part.startswith("bg"):
                    class_idx = num_classes
                else:
                    for alias in ("class", "cls", "c"):
                        if cls_part.startswith(alias):
                            cls_part = cls_part[len(alias) :]
                            break
                    cls_part = cls_part if cls_part else str(focus_class)
                    class_idx = int(cls_part)
            elif lower.startswith("bg"):
                class_idx = num_classes
                proto_idx = int(lower[2:].lstrip(":p"))
            elif lower.startswith("c") and "p" in lower:
                cls_part, proto_part = lower.split("p", 1)
                class_idx = int(cls_part[1:] or focus_class)
                proto_idx = int(proto_part)
            else:
                global_idx = int(lower)
                class_idx = global_idx // num_prototypes
                proto_idx = global_idx % num_prototypes

            selections.append(_make_selection(class_idx, proto_idx))
        return selections

    scoped: list[PrototypeSelection] = []
    for token in tokens:
        value = int(token)
        if 0 <= value < num_prototypes:
            scoped.append(_make_selection(focus_class, value))
        else:
            global_idx = value
            class_idx = global_idx // num_prototypes
            proto_idx = global_idx % num_prototypes
            scoped.append(_make_selection(class_idx, proto_idx))
    return scoped


def compute_scores(maps: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce per-prototype maps to a scalar score for ranking."""

    if mode == "max":
        return maps.amax(dim=(-2, -1))
    if mode == "mean":
        return maps.mean(dim=(-2, -1))
    if mode == "sum":
        return maps.sum(dim=(-2, -1))
    raise ValueError(f"Unsupported score mode '{mode}'")


def upsample_maps(maps: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    """Resize token-grid maps to image resolution."""

    return F.interpolate(
        maps.unsqueeze(0), size=size, mode=mode, align_corners=False
    ).squeeze(0)


def apply_foreground_gate(
    maps: torch.Tensor, gate: torch.Tensor | None
) -> torch.Tensor:
    """Multiply maps by a foreground gate (resizing gate if needed)."""

    if gate is None:
        return maps
    gate_tensor = gate

    if gate_tensor.dim() == 4:
        gate_tensor = gate_tensor[0]
    if gate_tensor.dim() == 3:
        gate_tensor = gate_tensor[0]
    if gate_tensor.shape != maps.shape[-2:]:
        gate_tensor = F.interpolate(
            gate_tensor.unsqueeze(0).unsqueeze(0),
            size=maps.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    return maps * gate_tensor


def peak_coordinates(map_tensor: torch.Tensor) -> tuple[int, int]:
    """Return (row, col) of the maximum activation in a 2D map."""

    flat_idx = int(map_tensor.flatten().argmax().item())
    h, w = map_tensor.shape
    return flat_idx // w, flat_idx % w


def map_to_image_coords(
    token_hw: tuple[int, int], peak: tuple[int, int], image_hw: tuple[int, int]
) -> tuple[float, float]:
    """Convert token-grid coordinates to pixel coordinates in image space."""

    h_tokens, w_tokens = token_hw
    y_token, x_token = peak
    scale_y = image_hw[0] / h_tokens
    scale_x = image_hw[1] / w_tokens
    y = (y_token + 0.5) * scale_y
    x = (x_token + 0.5) * scale_x
    return y, x


def ensure_colormap(name: str):
    """Return a Matplotlib colormap, raising a friendly error if missing."""

    try:
        return plt.get_cmap(name)
    except ValueError as exc:
        raise ValueError(f"Unsupported colormap '{name}'") from exc


def make_figure_layout(grid: str, num_tiles: int, cols: int) -> tuple[int, int]:
    """Compute subplot grid layout (rows, cols) for the requested style."""

    if grid == "2x2":
        return 2, 2
    if grid == "3x2":
        return 2, 3
    cols = max(1, cols)
    rows = math.ceil(num_tiles / cols)
    return rows, cols


def summarize_proto_ids(selections: list[PrototypeSelection]) -> str:
    """Create a short string summary of selected prototype ids for filenames."""

    if not selections:
        return "none"
    tokens: list[str] = []
    for sel in selections:
        if sel.class_idx == selections[0].class_idx:
            tokens.append(f"p{sel.proto_idx}")
        else:
            tokens.append(sel.label.replace("=", ""))
    return "-".join(tokens)


def ensure_out_dir(path: Path) -> None:
    """Create an output directory if it doesn't exist."""

    path.mkdir(parents=True, exist_ok=True)


def load_wandb(args: argparse.Namespace):
    """Initialize a Weights & Biases run if requested."""

    if args.wandb == "off":
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "wandb not installed; install it or pass --wandb off"
        ) from exc

    run_name = args.wandb_run_name or args.wandb_name
    if run_name is None:
        ckpt_stem = Path(args.ckpt).stem
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{ckpt_stem}-{args.split}-{timestamp}"
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config={
            "ckpt": args.ckpt,
            "split": args.split,
            "map_source": args.map_source,
            "focus": args.focus,
        },
    )
    return wandb


def close_wandb(wandb_module):
    """Close a Weights & Biases run (if open)."""

    if wandb_module is not None:
        wandb_module.finish()


# ---------------------------------------------------------------------------
# Panel generation
# ---------------------------------------------------------------------------


def build_maps(
    outputs: dict,
    args: argparse.Namespace,
    focus_class: int,
    include_bg: bool,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Extract class/prototype maps from model outputs for visualization."""

    maps_raw = outputs["maps_raw"][0].detach()
    maps_norm = outputs["maps_norm"][0].detach()
    patch_logits = outputs["patch_prototype_logits"][0].detach()
    n_classes_plus_bg, n_prototypes, h, w = maps_raw.shape
    num_classes = n_classes_plus_bg - 1

    if args.map_source == "raw":
        source = maps_raw
    elif args.map_source == "norm":
        source = maps_norm
    else:
        source = torch.empty_like(maps_raw)
        for c in range(n_classes_plus_bg):
            logits = patch_logits[:, c, :]
            # resp = softmax_with_temp(logits, dim=0, temp=args.resp_temp)
            resp = softmax_with_temp(logits, dim=0, temp=1.0)
            resp_maps = resp.transpose(0, 1).reshape(n_prototypes, h, w)
            source[c] = resp_maps

    selected_classes = [focus_class]
    if include_bg and focus_class != num_classes:
        selected_classes.append(num_classes)

    class_to_maps: dict[int, torch.Tensor] = {c: source[c] for c in selected_classes}
    return source, class_to_maps


def select_prototypes(
    maps_by_class: dict[int, torch.Tensor],
    args: argparse.Namespace,
    focus_class: int,
    num_classes: int,
) -> list[PrototypeSelection]:
    """Choose which prototypes to visualize for a given image."""

    num_prototypes = next(iter(maps_by_class.values())).shape[0]
    explicit = resolve_proto_ids(
        args.proto_ids, focus_class, num_classes, num_prototypes
    )
    if explicit is not None:
        return explicit

    focus_maps = maps_by_class[focus_class]
    scores = compute_scores(focus_maps, args.score_type)
    topk = min(args.topk, scores.numel())
    indices = scores.topk(topk).indices.tolist()
    return [
        PrototypeSelection(focus_class, idx, f"c={focus_class}:p={idx}")
        for idx in indices
    ]


def prepare_heatmaps(
    selections: list[PrototypeSelection],
    source_maps: torch.Tensor,
    outputs: dict,
    args: argparse.Namespace,
    focus_class: int,
) -> list[dict]:
    """Build display-ready heatmap entries for each selected prototype."""

    grid_hw = outputs["grid_hw"]
    image_hw = (224, 224)
    fg_mask = outputs.get("fg_mask") if args.fg_gate else None
    prepared: list[dict] = []
    for sel in selections:
        base_map = source_maps[sel.class_idx, sel.proto_idx]
        processed = base_map.clone()
        if args.blur_kernel > 1:
            processed = apply_token_blur(
                processed.unsqueeze(0), args.blur_kernel
            ).squeeze(0)
        if fg_mask is not None:
            processed = apply_foreground_gate(processed, fg_mask)

        peak_token = peak_coordinates(processed)
        peak_image = map_to_image_coords(grid_hw, peak_token, image_hw)

        upsampled = upsample_maps(
            processed.unsqueeze(0), image_hw, args.resize_mode
        ).squeeze(0)
        stats = {
            "min": float(processed.min().item()),
            "max": float(processed.max().item()),
            "mean": float(processed.mean().item()),
        }

        prepared.append(
            {
                "selection": sel,
                "map": processed,
                "upsampled": upsampled,
                "peak_token": peak_token,
                "peak_image": peak_image,
                "peak_value_raw": float(base_map.max().item()),
                "peak_value_display": float(processed.max().item()),
                "stats": stats,
            }
        )
    return prepared


def color_scale_limits(heatmaps: list[dict], shared: bool) -> list[tuple[float, float]]:
    """Compute (vmin, vmax) pairs used to normalize colormap overlays."""

    if not heatmaps:
        return []
    if shared:
        lows = [item["stats"]["min"] for item in heatmaps]
        highs = [item["stats"]["max"] for item in heatmaps]
        vmin = min(lows)
        vmax = max(highs)
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        return [(vmin, vmax)] * len(heatmaps)

    limits: list[tuple[float, float]] = []
    for item in heatmaps:
        vmin, vmax = item["stats"]["min"], item["stats"]["max"]
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        limits.append((vmin, vmax))
    return limits


def make_figure(
    base_image: torch.Tensor,
    heatmaps: list[dict],
    args: argparse.Namespace,
    title_text: str,
) -> tuple[plt.Figure, list[np.ndarray]]:
    """Render a panel figure with input image + prototype overlays."""

    colormap = ensure_colormap(args.cmap)
    annotations = parse_annotate(args.annotate)

    num_tiles = len(heatmaps) + (1 if args.show_image_first else 0)
    rows, cols = make_figure_layout(args.grid, num_tiles, args.cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(args.figure_width, args.figure_height), dpi=args.dpi
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    idx = 0
    if args.show_image_first:
        axes[idx].imshow(base_image)
        axes[idx].axis("off")
        axes[idx].set_title("input")
        idx += 1

    limits = color_scale_limits(
        heatmaps, shared=args.shared_scale and args.map_source == "raw"
    )
    overlays: list[np.ndarray] = []

    for heatmap, lims in zip(heatmaps, limits):
        ax = axes[idx]
        idx += 1
        ax.imshow(base_image)
        data = heatmap["upsampled"].cpu().numpy()
        mapped = Normalize(vmin=lims[0], vmax=lims[1])(data)
        overlay = colormap(mapped)
        ax.imshow(overlay, alpha=args.alpha)

        title_tokens = [heatmap["selection"].label]
        if "score" in annotations:
            title_tokens.append(f"{args.score_type}={heatmap['stats']['max']:.3f}")
        if "peak" in annotations:
            y, x = heatmap["peak_image"]
            title_tokens.append(f"peak=({int(x)}, {int(y)})")
        ax.set_title(" | ".join(title_tokens))
        ax.axis("off")

        if args.mark_peak:
            y, x = heatmap["peak_image"]
            ax.scatter([x], [y], c="white", s=16, marker="x")

        if args.colorbar:
            fig.colorbar(
                cm.ScalarMappable(
                    norm=Normalize(vmin=lims[0], vmax=lims[1]), cmap=colormap
                ),
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
        overlays.append(overlay)

    for ax in axes[idx:]:
        ax.axis("off")

    if title_text:
        fig.suptitle(title_text)
    fig.tight_layout()
    return fig, overlays


def format_header(
    args: argparse.Namespace,
    dataset: Any,
    focus_class: int,
    pred_class: int,
    pred_prob: float,
    gt_class: int,
) -> str:
    """Build the panel title string based on `--title` and `--focus`."""

    title_mode = args.title
    if title_mode == "none":
        return ""
    tokens: list[str] = []
    if title_mode in {"pred", "both"}:
        tokens.append(f"pred: {dataset.classes[pred_class]} (p={pred_prob:.2f})")
    if title_mode in {"gt", "both"}:
        tokens.append(f"gt: {dataset.classes[gt_class]}")
    tokens.append(f"focus={args.focus}")
    tokens.append(f"source={args.map_source}")
    return " | ".join(tokens)


def save_panel(fig: plt.Figure, out_path: Path) -> None:
    """Save a Matplotlib figure and close it to free resources."""

    fig.savefig(out_path.as_posix(), bbox_inches="tight")
    plt.close(fig)


def save_metadata(
    json_path: Path,
    *,
    dataset: Any,
    index: int,
    focus_class: int,
    pred_class: int,
    pred_prob: float,
    heatmaps: list[dict],
    args: argparse.Namespace,
    source_limits: list[tuple[float, float]],
) -> None:
    """Write a JSON sidecar describing the saved panel/heatmaps."""

    proto_entries = []
    for item, lims in zip(heatmaps, source_limits):
        proto_entries.append(
            {
                "proto_id": item["selection"].label,
                "class_idx": item["selection"].class_idx,
                "proto_idx": item["selection"].proto_idx,
                "score": float(
                    item["map"].amax().item()
                    if args.score_type == "max"
                    else compute_scores(item["map"].unsqueeze(0), args.score_type)[
                        0
                    ].item()
                ),
                "peak_token": item["peak_token"],
                "peak_image": item["peak_image"],
                "peak_value_raw": item["peak_value_raw"],
                "peak_value_display": item["peak_value_display"],
                "map_stats": item["stats"],
                "color_range": {"min": lims[0], "max": lims[1]},
            }
        )

    payload = {
        "index": index,
        "focus_class": {"id": focus_class, "name": dataset.classes[focus_class]},
        "predicted": {
            "id": pred_class,
            "name": dataset.classes[pred_class],
            "prob": pred_prob,
        },
        "map_source": args.map_source,
        "norm_mode": args.norm_mode,
        "proto_annotations": proto_entries,
        "display": {
            "alpha": args.alpha,
            "cmap": args.cmap,
            "shared_scale": args.shared_scale,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main():
    """CLI entrypoint."""

    parser = build_argparser()
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args)
    dataset, normalize_inputs = load_dataset(args, cfg)
    model.to(device)
    if str(getattr(cfg.model, "recipe", "legacy")) == "four_loss":
        from src.utils.visualization import visualize_fitted
        return visualize_fitted(model, dataset, args, normalize_inputs)
    proto_init = getattr(model, "prototype_mu_init", None)
    if torch.is_tensor(proto_init) and proto_init.device != device:
        model.prototype_mu_init = proto_init.to(device)

    explicit_indices = parse_indices_argument(args.indices, len(dataset))

    predictions = labels = None
    if args.only_misclassified and explicit_indices is None:
        print(
            "[visualize] Computing predictions for misclassification filter...",
            file=sys.stderr,
        )
        predictions, labels = compute_predictions(
            model, dataset, device, args.batch_size
        )

    if explicit_indices is not None:
        indices = explicit_indices
    else:
        indices = sample_indices(dataset, args, predictions, labels)

    wandb_module = load_wandb(args)
    wandb_table = None
    if wandb_module is not None:
        wandb_table = wandb_module.Table(
            columns=[
                "file",
                "focus_class",
                "proto_id",
                "score",
                "peak",
                "pred_class",
                "prob",
                "map_source",
            ]
        )

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    with torch.inference_mode():
        for idx in indices:
            item = dataset[idx]
            image, label, index = unpack_dataset_item(item, idx)
            batch = image.unsqueeze(0).to(device)
            outputs = model(batch)

            class_logits = outputs["class_logits"]
            probs = torch.softmax(class_logits, dim=-1)
            pred_prob, pred_class = probs.max(dim=-1)
            pred_prob = float(pred_prob.item())
            pred_class = int(pred_class.item())
            gt_class = int(label)

            focus_class = pred_class if args.focus == "pred" else gt_class
            source_maps, class_maps = build_maps(
                outputs, args, focus_class, args.include_bg
            )
            selections = select_prototypes(
                class_maps, args, focus_class, model.n_classes
            )
            heatmaps = prepare_heatmaps(
                selections, source_maps, outputs, args, focus_class
            )

            base_image = (
                denormalize_image(image, normalize=normalize_inputs)
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            limits = color_scale_limits(
                heatmaps, shared=args.shared_scale and args.map_source == "raw"
            )
            title = format_header(
                args, dataset, focus_class, pred_class, pred_prob, gt_class
            )
            fig, _ = make_figure(base_image, heatmaps, args, title)

            proto_suffix = ""
            if args.proto_ids:
                proto_suffix = f"_ids-{summarize_proto_ids(selections)}"
            png_name = f"{args.file_prefix}_{args.split}_{index:05d}_c{focus_class}_topK{args.topk}{proto_suffix}.png"
            png_path = out_dir / png_name
            save_panel(fig, png_path)

            json_path = png_path.with_suffix(".json")
            save_metadata(
                json_path,
                dataset=dataset,
                index=index,
                focus_class=focus_class,
                pred_class=pred_class,
                pred_prob=pred_prob,
                heatmaps=heatmaps,
                args=args,
                source_limits=limits,
            )

            label_tensor = torch.as_tensor([gt_class], device=device, dtype=torch.long)
            fg_mask = resolve_fg_mask(
                model=model, outputs=outputs, images=batch, labels=label_tensor
            )
            if fg_mask is not None:
                fg_path = png_path.with_name(png_path.stem + "_fg_mask.png")
                save_fg_mask_overlay(
                    fg_path,
                    base_image=base_image,
                    fg_mask=fg_mask,
                    alpha=float(args.fg_mask_alpha),
                    color_rgb=parse_rgb_triplet(args.fg_mask_color),
                )

            if args.save_maps:
                for item in heatmaps:
                    map_name = (
                        png_path.stem
                        + f"_{item['selection'].label.replace(':', '-')}.pt"
                    )
                    torch.save(item["map"], out_dir / map_name)

            if wandb_module is not None:
                wandb_module.log({"panel": wandb_module.Image(str(png_path))})
                if wandb_table is not None:
                    for item in heatmaps:
                        wandb_table.add_data(
                            png_path.name,
                            item["selection"].class_idx,
                            item["selection"].label,
                            float(item["map"].amax().item()),
                            item["peak_image"],
                            pred_class,
                            pred_prob,
                            args.map_source,
                        )

    if wandb_module is not None:
        if wandb_table is not None:
            wandb_module.log({"prototype_table": wandb_table})
        close_wandb(wandb_module)


if __name__ == "__main__":
    main()
