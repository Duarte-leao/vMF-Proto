#!/usr/bin/env python3
"""Visualize foreground masks produced by vMF-MP gating modules.

This script is a lightweight debugging/QA tool: it reuses the repo's backbone
implementations and foreground gate extractors (PCA, attention, attention→PCA
refinement), runs them over a dataset split, and saves overlay images showing
the computed binary foreground mask.

It intentionally *does not* depend on the training model class (VMFMP). The only
things we need here are:
  - a backbone that outputs patch tokens
  - an fg_extractor that turns patch tokens into pseudo patch labels
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import warnings
import sys

warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf


def _add_repo_to_path() -> None:
    """Ensure both repo root and src/ are importable for tool scripts."""

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    for path in (repo_root, src_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_add_repo_to_path()

from data.dataloader import build_dataset
from models.vmf_mp import backbone as backbone_module
from models.vmf_mp.backbone import (
    DINOv2Backbone,
    DINOv2BackboneExpanded,
    DINOv3Backbone,
    DINOv3BackboneExpanded,
)
from models.vmf_mp import (
    AttnThenPCAForegroundExtractor,
    AttentionForegroundExtractor,
    BackboneAttentionForegroundGater,
    PCA,
)
from src.utils.imagenet import IMAGENET_MEAN, IMAGENET_STD


class _FGDebugNet(torch.nn.Module):
    """Minimal wrapper exposing `backbone` + `fg_extractor`.

    The foreground-debug script only needs backbone tokens and the
    foreground extractor. Keeping this tiny wrapper lets us visualize masks
    from generative-vMF checkpoints without depending on the removed baseline
    reference implementation.
    """

    def __init__(self, *, backbone: torch.nn.Module, fg_extractor: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.fg_extractor = fg_extractor


IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
IMAGENET_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32)

DATASET_ALIASES = {
    "cub": "cub200",
    "cub200": "cub200",
    "stanford_dogs": "stanford_dogs",
    "stanford_cars": "stanford_cars",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for foreground-mask visualization."""

    parser = argparse.ArgumentParser(description="Visualize foreground masks.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ckpt-path", type=str, default=None)

    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cub200",
        choices=["auto", "cub200", "stanford_dogs", "stanford_cars", "CUB", "cub"],
        help=(
            "Dataset to visualize. Use 'auto' to read cfg.dataset.name from the "
            "checkpoint when --ckpt-path is provided."
        ),
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=[
            "dinov2_vitb14",
            "dinov2_vits14",
            "dinov3_vits16",
            "dinov3_vitb16",
        ],
    )
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument(
        "--fg-threshold",
        type=float,
        default=0.5,
        help="Threshold for PCA gating (fg-method=pca).",
    )
    parser.add_argument(
        "--fg-method",
        type=str,
        default="pca",
        choices=["pca", "attn", "attn_pca_refine"],
        help="Foreground gating method used for mask overlays.",
    )
    parser.add_argument(
        "--pca-keep-quantile",
        type=float,
        default=0.5,
        help="Fraction of ROI patches to keep in PCA refinement (ignored if --pca-threshold is set).",
    )
    parser.add_argument(
        "--pca-threshold",
        type=float,
        default=None,
        help="Threshold in [0,1] on PCA scores within the ROI; keep scores >= threshold.",
    )
    parser.add_argument(
        "--pca-min-patches",
        type=int,
        default=16,
        help="Minimum ROI patches required to run PCA refinement.",
    )

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--color",
        type=str,
        default="255,0,0",
        help="Comma separated RGB values used for mask overlay.",
    )
    parser.add_argument(
        "--enable-attn-gater",
        action="store_true",
        help="Enable attention-based foreground gater overlays.",
    )
    parser.add_argument(
        "--attn-dir",
        type=str,
        default=None,
        help="Optional directory to store attention gater overlays. Defaults to <output-dir>/attn_gater.",
    )
    parser.add_argument(
        "--attn-color",
        type=str,
        default="0,255,0",
        help="Overlay color for attention-based gate.",
    )
    parser.add_argument("--attn-alpha", type=float, default=0.4)
    parser.add_argument(
        "--attn-pool",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Pooling strategy across attention layers.",
    )
    parser.add_argument(
        "--attn-center",
        type=float,
        default=0.0,
        help="Mix factor for adding a center prior to the attention gate.",
    )
    parser.add_argument(
        "--attn-smooth",
        type=int,
        default=3,
        help="Smoothing kernel size for attention gate post-processing.",
    )
    parser.add_argument(
        "--attn-threshold",
        type=float,
        default=None,
        help="Optional threshold applied after normalization for the attention gate.",
    )
    parser.add_argument(
        "--attn-use-edges",
        action="store_true",
        help="Blend an edge prior derived from the RGB image into the attention gate.",
    )
    parser.add_argument(
        "--dinov3-vits16-url",
        type=str,
        default=None,
        help="Optional override URL for dinov3_vits16 weights.",
    )
    parser.add_argument(
        "--dinov3-vitb16-url",
        type=str,
        default=None,
        help="Optional override URL for dinov3_vitb16 weights.",
    )

    return parser.parse_args()


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset aliases to canonical config names (e.g. `cub` -> `cub200`)."""

    key = str(dataset_name).strip().lower()
    return DATASET_ALIASES.get(key, key)


def resolve_n_classes(dataset_name: str) -> int:
    """Return the number of foreground classes for a dataset (excluding background)."""

    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "cub200":
        return 200
    if dataset_name == "stanford_dogs":
        return 120
    if dataset_name == "stanford_cars":
        return 196
    raise NotImplementedError(f"Dataset {dataset_name} is not supported.")


def _get_nested(cfg: object, path: str, default: object):
    node = cfg
    for key in path.split("."):
        if isinstance(node, dict):
            if key not in node:
                return default
            node = node[key]
        else:
            if not hasattr(node, key):
                return default
            node = getattr(node, key)
    return node


def _override_dinov3_urls(
    backbone_name: str, vits_url: Optional[str], vitb_url: Optional[str]
) -> None:
    if "dinov3" not in backbone_name:
        return

    if vits_url:
        backbone_module.DINOV3_URL_DICT["dinov3_vits16"] = vits_url
    if vitb_url:
        backbone_module.DINOV3_URL_DICT["dinov3_vitb16"] = vitb_url

    if backbone_name in backbone_module.DINOV3_URL_DICT:
        url = backbone_module.DINOV3_URL_DICT[backbone_name]
        if url.startswith("TODO:"):
            raise ValueError(
                f"Missing weights URL for {backbone_name}. "
                "Provide --dinov3-vits16-url/--dinov3-vitb16-url or update DINOV3_URL_DICT."
            )


def build_backbone(
    name: str,
    num_splits: int,
    *,
    dinov3_vits16_url: Optional[str] = None,
    dinov3_vitb16_url: Optional[str] = None,
):
    """Build a backbone matching the training/eval code paths.

    Args:
        name: Backbone identifier (e.g. `dinov2_vits14`).
        num_splits: Number of extra "expanded" blocks (0 disables expansion).
        dinov3_vits16_url: Optional override for DINOv3 small weights URL.
        dinov3_vitb16_url: Optional override for DINOv3 base weights URL.
    """

    _override_dinov3_urls(name, dinov3_vits16_url, dinov3_vitb16_url)
    if "dinov3" in name:
        if num_splits and num_splits > 0:
            return DINOv3BackboneExpanded(
                name=name,
                n_splits=num_splits,
                mode="block_expansion",
                freeze_norm_layer=True,
            )
        return DINOv3Backbone(name=name)
    if "dinov2" in name:
        if num_splits and num_splits > 0:
            return DINOv2BackboneExpanded(
                name=name,
                n_splits=num_splits,
                mode="block_expansion",
                freeze_norm_layer=True,
            )
        return DINOv2Backbone(name=name)
    raise NotImplementedError(f"Backbone {name} is not implemented.")


def build_fg_debug_net(
    backbone_name: str,
    num_splits: int,
    n_classes: int,
    fg_threshold: float,
    fg_method: str,
    attn_pool: str,
    attn_center: float,
    attn_smooth: int,
    attn_threshold: float,
    attn_use_edges: bool,
    pca_keep_quantile: float,
    pca_threshold: float | None,
    pca_min_patches: int,
    dinov3_vits16_url: Optional[str],
    dinov3_vitb16_url: Optional[str],
) -> torch.nn.Module:
    """Create a minimal network that outputs tokens + pseudo-labels.

    This returns a tiny wrapper containing:
      - a backbone that emits patch tokens, and
      - an `fg_extractor` that produces pseudo patch labels (FG => class label,
        BG => `bg_class`).
    """

    backbone = build_backbone(
        backbone_name,
        num_splits=num_splits,
        dinov3_vits16_url=dinov3_vits16_url,
        dinov3_vitb16_url=dinov3_vitb16_url,
    )
    dim = backbone.dim
    fg_method = fg_method.lower()
    if fg_method == "pca":
        fg_extractor = PCA(
            bg_class=n_classes,
            compare_fn="le",
            threshold=fg_threshold,
        )
    else:
        postproc_cfg = {
            "smooth_k": attn_smooth,
            "normalize": True,
            "threshold": attn_threshold if fg_method == "attn" else None,
        }
        attn_gater = BackboneAttentionForegroundGater(
            backbone=backbone,
            backbone_name=backbone_name,
            attn_pool=attn_pool,
            combine_center=attn_center,
            postproc=postproc_cfg,
            device=None,
            use_edges=attn_use_edges,
        )
        if fg_method == "attn":
            fg_extractor = AttentionForegroundExtractor(
                gater=attn_gater,
                threshold=attn_threshold,
                bg_class=n_classes,
            )
        elif fg_method == "attn_pca_refine":
            fg_extractor = AttnThenPCAForegroundExtractor(
                gater=attn_gater,
                attn_threshold=attn_threshold,
                pca_threshold=pca_threshold,
                pca_keep_quantile=pca_keep_quantile,
                pca_min_patches=pca_min_patches,
                bg_class=n_classes,
            )
        else:
            raise ValueError(f"Unknown fg-method: {fg_method}")
    return _FGDebugNet(backbone=backbone, fg_extractor=fg_extractor)


def load_or_init_model(args: argparse.Namespace, device: torch.device):
    """Load a checkpoint for visualization, or build from CLI args.

    Returns:
        Tuple of (net, dataset_name, data_root).
    """

    if args.ckpt_path is None:
        dataset = normalize_dataset_name(args.dataset)
        if dataset == "auto":
            dataset = "cub200"
        data_root = Path(args.data_root)
        n_classes = resolve_n_classes(dataset)
        net = build_fg_debug_net(
            backbone_name=args.backbone,
            num_splits=args.num_splits,
            n_classes=n_classes,
            fg_threshold=args.fg_threshold,
            fg_method=args.fg_method,
            attn_pool=args.attn_pool,
            attn_center=args.attn_center,
            attn_smooth=args.attn_smooth,
            attn_threshold=float(args.attn_threshold or 0.5),
            attn_use_edges=args.attn_use_edges,
            pca_keep_quantile=args.pca_keep_quantile,
            pca_threshold=args.pca_threshold,
            pca_min_patches=args.pca_min_patches,
            dinov3_vits16_url=args.dinov3_vits16_url,
            dinov3_vitb16_url=args.dinov3_vitb16_url,
        )
        net = net.to(device).eval()
        gater = getattr(getattr(net, "fg_extractor", None), "gater", None)
        if gater is not None and hasattr(gater, "device"):
            gater.device = device
        return net, dataset, data_root

    ckpt_path = Path(args.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "hparams" not in ckpt or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain 'hparams' and 'state_dict'.")
    hparams = ckpt["hparams"]

    dataset = normalize_dataset_name(_get_nested(hparams, "dataset.name", args.dataset))
    if dataset == "auto":
        dataset = normalize_dataset_name(args.dataset)
    # Allow overriding the checkpoint dataset root by passing --data-root.
    ckpt_root = str(_get_nested(hparams, "dataset.root", args.data_root))
    data_root = (
        Path(args.data_root) if args.data_root != "./datasets" else Path(ckpt_root)
    )
    n_classes = resolve_n_classes(dataset)

    net = build_fg_debug_net(
        backbone_name=_get_nested(hparams, "model.backbone", args.backbone),
        num_splits=int(_get_nested(hparams, "model.num_splits", args.num_splits)),
        n_classes=n_classes,
        fg_threshold=float(
            _get_nested(hparams, "model.fg.pca_threshold", args.fg_threshold)
        ),
        fg_method=args.fg_method,
        attn_pool=str(_get_nested(hparams, "model.fg.attn.pool", args.attn_pool)),
        attn_center=float(
            _get_nested(hparams, "model.fg.attn.center", args.attn_center)
        ),
        attn_smooth=int(_get_nested(hparams, "model.fg.attn.smooth", args.attn_smooth)),
        attn_threshold=float(
            _get_nested(hparams, "model.fg.attn.threshold", args.attn_threshold or 0.5)
        ),
        attn_use_edges=bool(
            _get_nested(hparams, "model.fg.attn.use_edges", args.attn_use_edges)
        ),
        pca_keep_quantile=args.pca_keep_quantile,
        pca_threshold=args.pca_threshold,
        pca_min_patches=args.pca_min_patches,
        dinov3_vits16_url=args.dinov3_vits16_url,
        dinov3_vitb16_url=args.dinov3_vitb16_url,
    )
    missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        logging.warning("Missing keys when loading checkpoint: %s", ", ".join(missing))
    if unexpected:
        logging.warning(
            "Unexpected keys when loading checkpoint: %s", ", ".join(unexpected)
        )
    net = net.to(device).eval()
    gater = getattr(getattr(net, "fg_extractor", None), "gater", None)
    if gater is not None and hasattr(gater, "device"):
        gater.device = device
    return net, dataset, data_root


class _ReturnIndexDataset(torch.utils.data.Dataset):
    """Wrap a dataset to always return (image, label, index)."""

    def __init__(self, dataset: object) -> None:
        self._dataset = dataset

    def __len__(self) -> int:  # pragma: no cover
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        item = self._dataset[index]  # type: ignore[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError(f"Unexpected dataset item: {item}")
        if len(item) >= 3:
            return item[0], item[1], item[2]
        return item[0], item[1], index


def build_dataloader(
    dataset_name: str,
    split: str,
    data_root: Path,
    batch_size: int,
) -> DataLoader:
    """Build a deterministic dataloader for the chosen dataset/split."""

    dataset_name = normalize_dataset_name(dataset_name)
    dataset_cfg_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "dataset"
        / f"{dataset_name}.yaml"
    )
    if not dataset_cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_cfg_path}")

    dataset_cfg = OmegaConf.load(dataset_cfg_path.as_posix())
    cfg = OmegaConf.create(
        {
            "dataset": dataset_cfg,
            "train": {"batch_size": int(batch_size), "num_workers": 4},
            "eval": {"batch_size": int(batch_size)},
        }
    )
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.name = dataset_name
    cfg.dataset.root = str(data_root)

    dataset = build_dataset(cfg, split)
    dataset = _ReturnIndexDataset(dataset)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization on a batch of images."""

    mean = IMAGENET_MEAN_T.view(1, -1, 1, 1).to(images.device)
    std = IMAGENET_STD_T.view(1, -1, 1, 1).to(images.device)
    return (images * std) + mean


def parse_overlay_color(color_str: str, device: torch.device) -> torch.Tensor:
    """Parse a `R,G,B` string into a broadcastable float tensor in [0,1]."""

    parts = color_str.split(",")
    if len(parts) != 3:
        raise ValueError("Color must be provided as R,G,B.")
    rgb = [float(p.strip()) / 255.0 for p in parts]
    color = torch.tensor(rgb, dtype=torch.float32, device=device)
    return color.view(1, 3, 1, 1)


def save_overlays(
    net: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device,
    alpha: float,
    overlay_color: torch.Tensor,
    limit: Optional[int],
    *,
    attn_gater: Optional[BackboneAttentionForegroundGater] = None,
    attn_output_dir: Optional[Path] = None,
    attn_alpha: float = 0.4,
    attn_color: Optional[torch.Tensor] = None,
    attn_mask_threshold: Optional[float] = 0.5,
    enable_attn: bool = False,
):
    """Run FG extraction on a dataloader and save overlay images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    net.eval()
    fg_extractor = net.fg_extractor
    bg_class = fg_extractor.bg_class

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Generating overlays"):
            images, labels, indices = batch
            images = images.to(device)
            labels = labels.to(device)

            backbone_out = net.backbone(images)
            if not isinstance(backbone_out, tuple):
                raise ValueError("Backbone must return a tuple.")
            if len(backbone_out) == 3:
                patch_tokens, raw_patch_tokens, cls_tokens = backbone_out
            elif len(backbone_out) == 2:
                patch_tokens, cls_tokens = backbone_out
                raw_patch_tokens = patch_tokens
            else:
                raise ValueError("Unexpected number of outputs from backbone.")

            fg_tokens = (
                raw_patch_tokens if raw_patch_tokens is not None else patch_tokens
            )
            fg_tokens_norm = F.normalize(fg_tokens, p=2, dim=-1)

            fg_kwargs = {"cls_tokens": cls_tokens}
            if getattr(fg_extractor, "requires_images", False):
                fg_kwargs["images"] = images
            pseudo_patch_labels = fg_extractor(fg_tokens_norm, labels, **fg_kwargs)
            fg_mask = (pseudo_patch_labels != bg_class).float().unsqueeze(1)
            mask_upsampled = F.interpolate(
                fg_mask, size=images.shape[-2:], mode="nearest"
            )

            images_denorm = denormalize(images).clamp(0.0, 1.0)
            overlay = images_denorm * (1.0 - mask_upsampled * alpha)
            overlay = overlay + overlay_color * (mask_upsampled * alpha)

            attn_overlay = None
            if enable_attn and attn_gater is not None:
                if attn_output_dir is None:
                    raise ValueError("Attention output directory must be provided.")
                if attn_color is None:
                    raise ValueError("Attention overlay color tensor must be provided.")
                attn_masks = attn_gater.generate_from_batch(
                    images,
                    images_denorm,
                    out_hw=images.shape[-2:],
                )
                if attn_mask_threshold is not None:
                    attn_masks = (attn_masks >= float(attn_mask_threshold)).float()
                else:
                    attn_masks = attn_masks.clamp(0.0, 1.0)
                attn_masks = attn_masks.unsqueeze(1)
                attn_masks = F.interpolate(
                    attn_masks, size=images.shape[-2:], mode="nearest"
                )
                attn_overlay = images_denorm * (1.0 - attn_masks * attn_alpha)
                attn_overlay = attn_overlay + attn_color * (attn_masks * attn_alpha)
                attn_output_dir.mkdir(parents=True, exist_ok=True)

            batch_size = images.size(0)
            for i in range(batch_size):
                idx = indices[i].item()
                label = labels[i].item()
                filename = output_dir / f"{idx:05d}_label{label:03d}.png"
                vutils.save_image(
                    overlay[i].detach().cpu(),
                    filename.as_posix(),
                )
                if attn_overlay is not None:
                    attn_filename = (
                        attn_output_dir / f"{idx:05d}_label{label:03d}_attn.png"
                    )
                    vutils.save_image(
                        attn_overlay[i].detach().cpu(),
                        attn_filename.as_posix(),
                    )
                processed += 1
                if limit is not None and processed >= limit:
                    return


def main():
    """CLI entrypoint."""

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] - %(message)s",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, dataset_name, data_root = load_or_init_model(args, device)
    dataloader = build_dataloader(
        dataset_name=dataset_name,
        split=args.split,
        data_root=data_root,
        batch_size=args.batch_size,
    )

    overlay_color = parse_overlay_color(args.color, device=device)
    output_dir = Path(args.output_dir)
    attn_gater = None
    attn_output_dir = None
    attn_color_tensor = None
    attn_mask_threshold = None

    if args.enable_attn_gater:
        postproc_cfg = {
            "smooth_k": args.attn_smooth,
            "normalize": True,
            "threshold": args.attn_threshold,
        }
        attn_gater = BackboneAttentionForegroundGater(
            backbone=net.backbone,
            attn_pool=args.attn_pool,
            combine_center=args.attn_center,
            postproc=postproc_cfg,
            device=device,
            use_edges=args.attn_use_edges,
        )
        attn_output_dir = (
            Path(args.attn_dir)
            if args.attn_dir is not None
            else output_dir / "attn_gater"
        )
        attn_color_tensor = parse_overlay_color(args.attn_color, device=device)
        attn_mask_threshold = (
            float(args.attn_threshold) if args.attn_threshold is not None else 0.5
        )

    save_overlays(
        net=net,
        dataloader=dataloader,
        output_dir=output_dir,
        device=device,
        alpha=args.alpha,
        overlay_color=overlay_color,
        limit=args.limit,
        attn_gater=attn_gater,
        attn_output_dir=attn_output_dir,
        attn_alpha=args.attn_alpha,
        attn_color=attn_color_tensor,
        attn_mask_threshold=attn_mask_threshold,
        enable_attn=args.enable_attn_gater,
    )
    logging.info("Saved overlays to %s", output_dir)


if __name__ == "__main__":
    main()
