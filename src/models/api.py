"""Model builder API for vMF-MP."""

from __future__ import annotations

from typing import Any

from torch import nn


def build_model(cfg: Any) -> nn.Module:
    """Build a vMF-MP model from config."""

    from .vmf_mp import (
        AttnThenPCAForegroundExtractor,
        AttentionForegroundExtractor,
        BackboneAttentionForegroundGater,
        DINOv2Backbone,
        DINOv2BackboneExpanded,
        DINOv3Backbone,
        DINOv3BackboneExpanded,
        PCA,
        VMFMP,
    )

    def _cfg_value(node: Any, key: str, default: Any) -> Any:
        if node is None:
            return default
        if hasattr(node, "get"):
            return node.get(key, default)
        return getattr(node, key, default)

    model_name = str(cfg.model.name).lower()
    if model_name not in {
        # Current names.
        "vmf_mp",
        "vmf_mp_dogs",
        # Supported model aliases.
        "generative_vmf_protos",
        "generative_vmf_protos_dogs",
    }:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    backbone_name = str(cfg.model.backbone)
    num_splits = int(getattr(cfg.model, "num_splits", 0))
    if "dinov3" in backbone_name:
        if num_splits > 0:
            backbone = DINOv3BackboneExpanded(
                name=backbone_name,
                n_splits=num_splits,
                mode="block_expansion",
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv3Backbone(name=backbone_name)
        dim = backbone.dim
    elif "dinov2" in backbone_name:
        if num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=backbone_name,
                n_splits=num_splits,
                mode="block_expansion",
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=backbone_name)
        dim = backbone.dim
    elif backbone_name.startswith("dino"):
        raise ValueError(
            f"Legacy DINO(v1) backbones are not supported: {backbone_name}. "
            "Use a DINOv2/DINOv3 backbone (dinov2_* or dinov3_*)."
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    fg_method = str(cfg.model.fg.method).lower()
    if fg_method == "pca":
        fg_extractor = PCA(
            bg_class=int(cfg.dataset.num_classes),
            compare_fn="le",
            threshold=float(getattr(cfg.model.fg, "pca_threshold", 0.5)),
        )
    elif fg_method in {"attn", "attn_pca_refine"}:
        attn_threshold = float(getattr(cfg.model.fg.attn, "threshold", 0.1))
        attn_postproc = {
            "smooth_k": int(getattr(cfg.model.fg.attn, "smooth", 3)),
            "normalize": True,
            # Keep the raw gate for refinement; thresholding happens downstream.
            "threshold": attn_threshold if fg_method == "attn" else None,
        }
        attn_gater = BackboneAttentionForegroundGater(
            backbone=backbone,
            attn_pool=str(getattr(cfg.model.fg.attn, "pool", "last")),
            combine_center=float(getattr(cfg.model.fg.attn, "center", 0.0)),
            postproc=attn_postproc,
            device=None,
            use_edges=bool(getattr(cfg.model.fg.attn, "use_edges", False)),
        )
        if fg_method == "attn":
            fg_extractor = AttentionForegroundExtractor(
                gater=attn_gater,
                threshold=attn_threshold,
                bg_class=int(cfg.dataset.num_classes),
            )
        else:
            refine_cfg = getattr(cfg.model.fg, "attn_pca_refine", None)
            fg_extractor = AttnThenPCAForegroundExtractor(
                gater=attn_gater,
                attn_threshold=float(_cfg_value(refine_cfg, "attn_threshold", 0.01)),
                pca_threshold=float(_cfg_value(refine_cfg, "pca_threshold", 0.5)),
                pca_min_patches=int(_cfg_value(refine_cfg, "min_patches", 16)),
                pca_niter=int(_cfg_value(refine_cfg, "niter", 10)),
                pca_center=bool(_cfg_value(refine_cfg, "center", True)),
                bg_class=int(cfg.dataset.num_classes),
            )
    else:
        raise ValueError(f"Unknown fg.method: {cfg.model.fg.method}")

    ot_cfg = getattr(cfg.model, "ot", None)
    pi_cfg = getattr(cfg.model, "pi", None)
    recipe = str(getattr(cfg.model, "recipe", "legacy"))
    extra = {}
    model_type = VMFMP
    if recipe == "four_loss":
        from .vmf_mp.four_loss import FourLossVMF
        model_type = FourLossVMF
        extra = dict(variant=str(getattr(cfg.model, "variant", "class_specific")),
                     shared_prototypes=int(getattr(cfg.model, "shared_prototypes", 1000)))
    elif recipe != "legacy":
        raise ValueError(f"Unknown model recipe: {recipe}")
    return model_type(
        **extra,
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=int(cfg.model.num_prototypes),
        n_classes=int(cfg.dataset.num_classes),
        gamma=float(cfg.model.gamma),
        temperature=float(cfg.model.temperature),
        kappa_init=float(cfg.model.kappa_init),
        kappa_learnable=bool(getattr(cfg.model, "kappa_learnable", True)),
        norm_prototypes=bool(getattr(cfg.model, "norm_prototypes", True)),
        cluster_use_norm_tokens=bool(
            getattr(cfg.model, "cluster_use_norm_tokens", True)
        ),
        maps_norm_mode=str(getattr(cfg.model, "maps_norm_mode", "minmax")),
        ot_mode=str(_cfg_value(ot_cfg, "mode", "balanced")),
        ot_epsilon=float(_cfg_value(ot_cfg, "epsilon", 0.05)),
        ot_iters=int(_cfg_value(ot_cfg, "iters", 50)),
        pi_decoupled=bool(_cfg_value(pi_cfg, "decoupled", False)),
        pi_init_uniform=bool(_cfg_value(pi_cfg, "init_uniform", True)),
    )
