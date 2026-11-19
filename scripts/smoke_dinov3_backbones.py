#!/usr/bin/env python3
"""Smoke-check DINOv3 backbones and FG gating integration.

This script runs a single forward pass for each combination of:
  - backbone: dinov3_vits16, dinov3_vitb16
  - gating: pca, attn, attn_pca_refine
  - expansion: num_splits in {0, 1}

Note: DINOv3 weight URLs are placeholders in `src/models/vmf_mp/backbone.py`.
Provide real URLs via CLI flags or edit the URL dict directly before running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _add_repo_to_path() -> None:
    """Ensure both repo root and src/ are on sys.path."""

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    for path in (repo_root, src_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _override_urls(vits_url: str | None, vitb_url: str | None) -> None:
    """Override placeholder URL entries for DINOv3 weights."""

    from src.models.vmf_mp import backbone as backbone_module

    if vits_url:
        backbone_module.DINOV3_URL_DICT["dinov3_vits16"] = vits_url
    if vitb_url:
        backbone_module.DINOV3_URL_DICT["dinov3_vitb16"] = vitb_url

    missing = [
        name
        for name, url in backbone_module.DINOV3_URL_DICT.items()
        if url.startswith("TODO:")
    ]
    if missing:
        missing_fmt = ", ".join(missing)
        raise SystemExit(
            f"Missing DINOv3 URL(s): {missing_fmt}. "
            "Pass --vits16-url/--vitb16-url or update DINOV3_URL_DICT."
        )


def _build_fg_extractor(backbone, backbone_name: str, method: str, num_classes: int):
    """Create a foreground extractor for the requested method."""

    from src.models.vmf_mp import (
        AttnThenPCAForegroundExtractor,
        AttentionForegroundExtractor,
        BackboneAttentionForegroundGater,
        PCA,
    )

    method = method.lower()
    if method == "pca":
        return PCA(bg_class=num_classes, compare_fn="le", threshold=0.5)

    gater = BackboneAttentionForegroundGater(
        backbone=backbone,
        backbone_name=backbone_name,
        attn_pool="last",
        combine_center=0.0,
        postproc={"smooth_k": 3, "normalize": True, "threshold": None},
        device=next(backbone.parameters()).device,
        use_edges=False,
    )

    if method == "attn":
        return AttentionForegroundExtractor(
            gater=gater, threshold=0.1, bg_class=num_classes
        )

    if method == "attn_pca_refine":
        return AttnThenPCAForegroundExtractor(
            gater=gater,
            attn_threshold=0.01,
            pca_threshold=0.5,
            pca_min_patches=16,
            pca_niter=10,
            pca_center=True,
            bg_class=num_classes,
        )

    raise ValueError(f"Unknown fg method: {method}")


def _run_case(backbone_name: str, num_splits: int, fg_method: str, device: torch.device) -> None:
    """Run a single forward pass and validate output shapes."""

    from src.models.vmf_mp import DINOv3Backbone, DINOv3BackboneExpanded, VMFMP

    if num_splits > 0:
        backbone = DINOv3BackboneExpanded(
            name=backbone_name,
            n_splits=num_splits,
            mode="block_expansion",
            freeze_norm_layer=True,
        )
    else:
        backbone = DINOv3Backbone(name=backbone_name)

    backbone.to(device)
    fg_extractor = _build_fg_extractor(backbone, backbone_name, fg_method, num_classes=200)

    model = VMFMP(
        backbone=backbone,
        fg_extractor=fg_extractor,
        n_classes=200,
        n_prototypes=5,
        dim=backbone.dim,
    ).to(device)
    model.eval()

    with torch.inference_mode():
        x = torch.randn(1, 3, 224, 224, device=device)
        outputs = model(x)

    class_logits = outputs["class_logits"]
    maps_raw = outputs["maps_raw"]
    grid_hw = outputs["grid_hw"]

    assert class_logits.shape == (1, 200)
    assert maps_raw.shape[0] == 1
    assert maps_raw.shape[1] == 201
    assert maps_raw.shape[2] == 5
    assert maps_raw.shape[3:] == tuple(grid_hw)

    print(
        f"OK: {backbone_name} | splits={num_splits} | fg={fg_method} "
        f"| grid={grid_hw} | logits={tuple(class_logits.shape)}"
    )


def main() -> int:
    """CLI entrypoint for running DINOv3 + gating smoke tests."""

    parser = argparse.ArgumentParser(description="Smoke-check DINOv3 backbones + gating.")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda).")
    parser.add_argument("--vits16-url", default=None, help="Override dinov3_vits16 URL.")
    parser.add_argument("--vitb16-url", default=None, help="Override dinov3_vitb16 URL.")
    args = parser.parse_args()

    _add_repo_to_path()
    _override_urls(args.vits16_url, args.vitb16_url)

    device = torch.device(args.device)
    backbones = ["dinov3_vits16", "dinov3_vitb16"]
    fg_methods = ["pca", "attn", "attn_pca_refine"]
    splits = [0, 1]

    for backbone_name in backbones:
        for num_splits in splits:
            for fg_method in fg_methods:
                _run_case(backbone_name, num_splits, fg_method, device=device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
