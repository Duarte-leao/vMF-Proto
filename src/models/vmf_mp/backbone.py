"""Backbone wrappers for DINOv2/DINOv3 variants used by vMF-MP."""

from functools import partial
from math import sqrt
from pathlib import Path
import os
import sys
import warnings

# The gate needs probabilities; xFormers bypasses DINOv2's attention-drop hooks.
os.environ.setdefault("XFORMERS_DISABLED", "1")
warnings.filterwarnings("ignore")

import torch


def _ensure_dinov2_on_path() -> None:
    """Add the repo root to sys.path so the vendored `dinov2/` is importable."""

    repo_root = Path(__file__).resolve().parents[3]
    dinov2_root = Path(os.environ.get("DINOV2_SOURCE", str(repo_root / "dinov2")))
    if dinov2_root.exists() and str(dinov2_root) not in sys.path:
        sys.path.insert(0, str(dinov2_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _ensure_dinov3_on_path() -> None:
    """Add the repo root to sys.path so the vendored `dinov3/` is importable."""

    repo_root = Path(__file__).resolve().parents[3]
    dinov3_root = Path(os.environ.get("DINOV3_SOURCE", str(repo_root / "dinov3")))
    if dinov3_root.exists() and str(dinov3_root) not in sys.path:
        sys.path.insert(0, str(dinov3_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _reset_module_cache(prefix: str) -> None:
    """Remove cached modules so a subsequent import can use the updated sys.path."""

    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


try:
    from dinov2.layers.block import Block, MemEffAttention
    from dinov2.models.vision_transformer import (
        DinoVisionTransformer as Dinov2VisionTransformer,
    )
except ModuleNotFoundError:  # pragma: no cover - only hits when path is missing.
    _ensure_dinov2_on_path()
    _reset_module_cache("dinov2")
    from dinov2.layers.block import Block, MemEffAttention
    from dinov2.models.vision_transformer import (
        DinoVisionTransformer as Dinov2VisionTransformer,
    )

try:
    from dinov3.models.vision_transformer import (
        DinoVisionTransformer as Dinov3VisionTransformer,
    )
except ModuleNotFoundError:  # pragma: no cover - only hits when path is missing.
    _ensure_dinov3_on_path()
    _reset_module_cache("dinov3")
    from dinov3.models.vision_transformer import (
        DinoVisionTransformer as Dinov3VisionTransformer,
    )

from einops import rearrange
from torch import nn

from .utils import block_expansion_dino, append_blocks

dinov2_common_kwargs = dict(
    img_size=518,
    patch_size=14,
    mlp_ratio=4,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
    block_fn=partial(Block, attn_class=MemEffAttention),
)

vit_small_kwargs = dict(embed_dim=384, num_heads=6)
vit_base_kwargs = dict(embed_dim=768, num_heads=12)

MODEL_DICT = {
    "dinov2_vits14": partial(
        Dinov2VisionTransformer, **vit_small_kwargs, **dinov2_common_kwargs
    ),
    "dinov2_vitb14": partial(
        Dinov2VisionTransformer, **vit_base_kwargs, **dinov2_common_kwargs
    ),
}

URL_DICT = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
}

DIM_DICT = {"dinov2_vits14": 384, "dinov2_vitb14": 768}


dinov3_common_kwargs = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    pos_embed_rope_base=100,
    pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype="fp32",
    ffn_ratio=4,
    qkv_bias=True,
    drop_path_rate=0.0,
    layerscale_init=1.0e-5,
    norm_layer="layernormbf16",
    ffn_layer="mlp",
    ffn_bias=True,
    proj_bias=True,
    n_storage_tokens=4,
    mask_k_bias=True,
)

dinov3_vits_kwargs = dict(embed_dim=384, depth=12, num_heads=6)
dinov3_vitb_kwargs = dict(embed_dim=768, depth=12, num_heads=12)

DINOV3_MODEL_KWARGS = {
    "dinov3_vits16": dinov3_vits_kwargs,
    "dinov3_vitb16": dinov3_vitb_kwargs,
}

DINOV3_URL_DICT = {
    "dinov3_vits16": "INSERT REQUESTED URL HERE",
    "dinov3_vitb16": "INSERT REQUESTED URL HERE",
}

DINOV3_DIM_DICT = {"dinov3_vits16": 384, "dinov3_vitb16": 768}


def _load_pretrained(name):
    """Optional local licensed weights, without embedding signed download URLs."""
    local = os.environ.get("VMF_BACKBONE_WEIGHTS")
    if local:
        return torch.load(Path(local).expanduser(), map_location="cpu", weights_only=True)
    url = (DINOV3_URL_DICT if "dinov3" in name else URL_DICT)[name]
    if not url.startswith("https://"):
        raise ValueError("For DINOv3, download the official weights and set VMF_BACKBONE_WEIGHTS=/path/to/weights.pth")
    return torch.hub.load_state_dict_from_url(url, map_location="cpu")


def _build_dinov3_vit(
    name: str, *, depth_override: int | None = None
) -> Dinov3VisionTransformer:
    """Construct a DINOv3 ViT backbone matching the official hub configs."""

    name = name.lower()
    if name not in DINOV3_MODEL_KWARGS:
        raise ValueError(f"Unknown DINOv3 backbone: {name}")

    cfg = dict(DINOV3_MODEL_KWARGS[name])
    if depth_override is not None:
        cfg["depth"] = int(depth_override)

    cfg.update(dinov3_common_kwargs)
    return Dinov3VisionTransformer(**cfg)


class DINOv2Backbone(nn.Module):
    """Thin wrapper around a DINOv2 ViT with a stable forward signature."""

    def __init__(self, name: str = "dinov2_vitb14"):
        super().__init__()
        arch = MODEL_DICT[name]
        state_dict = _load_pretrained(name)
        self.dino = arch()
        self.dino.load_state_dict(state_dict)
        self.dim = DIM_DICT[name]

    def learnable_parameters(self):
        """Return parameters that should receive gradients (stage-2 fine-tuning)."""

        return self.dino.parameters()

    def set_requires_grad(self):
        """Enable gradients for the full DINOv2 backbone."""

        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        key: str = "x_norm_patchtokens",
        cls_key: str = "x_norm_clstoken",
        reshape: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Return normalized patch tokens and CLS token.

        Args:
            x: Normalized image batch (B, 3, H, W).
            key: Feature dict key for patch tokens.
            cls_key: Feature dict key for CLS token.
            reshape: If True, optionally reshape patch tokens to (B, D, H', W').

        Returns:
            (patch_tokens, cls_token)
        """

        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict[key]
        cls_token = feature_dict[cls_key]

        B, n_patches, dim = feature.shape

        if reshape and key == "x_norm_patch_tokens":
            H = W = int(sqrt(n_patches))
            feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)

        return feature, cls_token


class DINOv3Backbone(nn.Module):
    """Thin wrapper around a DINOv3 ViT with a stable forward signature."""

    def __init__(self, name: str = "dinov3_vitb16"):
        super().__init__()
        self.name = str(name).lower()
        self.dino = _build_dinov3_vit(self.name)
        state_dict = _load_pretrained(self.name)
        self.dino.load_state_dict(state_dict, strict=True)
        self.dim = DINOV3_DIM_DICT[self.name]

    def learnable_parameters(self):
        """Return parameters that should receive gradients in stage-2."""

        return self.dino.parameters()

    def set_requires_grad(self):
        """Enable gradients for the full DINOv3 backbone."""

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return normalized patch tokens and CLS token."""

        feature_dict = self.dino.forward_features(x)
        patch_tokens = feature_dict["x_norm_patchtokens"]
        cls_token = feature_dict["x_norm_clstoken"]
        return patch_tokens, cls_token


class DINOv2BackboneExpanded(nn.Module):
    """DINOv2 backbone with block expansion or appended blocks for fine-tuning."""

    def __init__(
        self,
        name: str = "dinov2_vitb14",
        n_splits: int = 0,
        mode: str = "block_expansion",
        freeze_norm_layer: bool = True,
    ):
        super().__init__()
        self.dim = DIM_DICT[name]
        assert mode in ["block_expansion", "append"]
        expand_state_dict = (
            block_expansion_dino if mode == "block_expansion" else append_blocks
        )
        if n_splits > 0:
            arch = MODEL_DICT[name]
            state_dict = _load_pretrained(name)
            expanded_state_dict, n_blocks, learnable_param_names, zero_param_names = (
                expand_state_dict(
                    state_dict=state_dict,
                    n_splits=n_splits,
                    freeze_layer_norm=freeze_norm_layer,
                )
            )
            self.dino = arch(depth=n_blocks)
            self.dino.load_state_dict(expanded_state_dict)
            self.learnable_param_names = learnable_param_names
        else:
            arch = MODEL_DICT[name]
            state_dict = _load_pretrained(name)
            self.dino = arch()
            self.dino.load_state_dict(state_dict)
            self.learnable_param_names = []

    def learnable_parameters(self):
        """Return only the parameters of the newly inserted blocks (stage-2)."""

        return list(
            param
            for name, param in self.dino.named_parameters()
            if name in self.learnable_param_names
        )

    def set_requires_grad(self):
        """Enable gradients only for the newly inserted blocks (stage-2)."""

        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return (patch_tokens, raw_patch_tokens, cls_tokens).

        For expanded backbones, we also return an "original feature" taken from
        the second-to-last block. This is useful when the final expanded block
        is being fine-tuned while preceding blocks remain frozen.
        """
        x = self.dino.prepare_tokens_with_masks(x)

        original_feature = None
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if i == len(self.dino.blocks) - 2:
                original_feature = self.dino.norm(x)
        x = self.dino.norm(x)
        return (
            x[:, self.dino.num_register_tokens + 1 :, :],
            original_feature[:, self.dino.num_register_tokens + 1 :, :],
            x[:, 0, :],
        )


class DINOv3BackboneExpanded(nn.Module):
    """DINOv3 backbone with block expansion or appended blocks for fine-tuning."""

    def __init__(
        self,
        name: str = "dinov3_vitb16",
        n_splits: int = 0,
        mode: str = "block_expansion",
        freeze_norm_layer: bool = True,
    ):
        super().__init__()
        self.name = str(name).lower()
        self.dim = DINOV3_DIM_DICT[self.name]
        assert mode in ["block_expansion", "append"]

        expand_state_dict = (
            block_expansion_dino if mode == "block_expansion" else append_blocks
        )
        state_dict = _load_pretrained(self.name)

        if n_splits > 0:
            expanded_state_dict, n_blocks, learnable_param_names, _ = expand_state_dict(
                state_dict=state_dict,
                n_splits=n_splits,
                freeze_layer_norm=freeze_norm_layer,
            )
            self.dino = _build_dinov3_vit(self.name, depth_override=n_blocks)
            self.dino.load_state_dict(expanded_state_dict, strict=True)
            self.learnable_param_names = learnable_param_names
        else:
            self.dino = _build_dinov3_vit(self.name)
            self.dino.load_state_dict(state_dict, strict=True)
            self.learnable_param_names = []

        self.num_prefix_tokens = 1 + int(getattr(self.dino, "n_storage_tokens", 0))

    def learnable_parameters(self):
        """Return parameters that should receive gradients in stage-2."""

        return list(
            param
            for name, param in self.dino.named_parameters()
            if name in self.learnable_param_names
        )

    def set_requires_grad(self):
        """Enable gradients only for expanded blocks (stage-2)."""

        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DINOv3's norm logic, honoring any untied norm configuration."""

        if (
            self.dino.untie_cls_and_patch_norms
            or self.dino.untie_global_and_local_cls_norm
        ):
            if self.dino.untie_global_and_local_cls_norm and self.dino.training:
                cls_norm = self.dino.local_cls_norm(x[:, : self.num_prefix_tokens])
            elif self.dino.untie_cls_and_patch_norms:
                cls_norm = self.dino.cls_norm(x[:, : self.num_prefix_tokens])
            else:
                cls_norm = self.dino.norm(x[:, : self.num_prefix_tokens])
            patch_norm = self.dino.norm(x[:, self.num_prefix_tokens :])
            return torch.cat((cls_norm, patch_norm), dim=1)
        return self.dino.norm(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return (patch_tokens, raw_patch_tokens, cls_tokens)."""

        x, (H, W) = self.dino.prepare_tokens_with_masks(x)

        original_feature = None
        for i, blk in enumerate(self.dino.blocks):
            rope = self.dino.rope_embed(H=H, W=W) if self.dino.rope_embed else None
            x = blk(x, rope)
            if i == len(self.dino.blocks) - 2:
                original_feature = self._apply_norm(x)

        x = self._apply_norm(x)
        if original_feature is None:
            original_feature = x

        return (
            x[:, self.num_prefix_tokens :, :],
            original_feature[:, self.num_prefix_tokens :, :],
            x[:, 0, :],
        )
