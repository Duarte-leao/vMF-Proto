"""Foreground gating implementations for vMF-MP.

This module provides label-free foreground masks (PCA, attention, attention→PCA
refinement) used both during training (pseudo labels) and at inference time.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from src.utils.imagenet import IMAGENET_MEAN, IMAGENET_STD

from .backbone import (
    DINOv2Backbone,
    DINOv2BackboneExpanded,
    DINOv3Backbone,
    DINOv3BackboneExpanded,
)


class BackboneAttentionForegroundGater:
    """
    Foreground gate powered by backbone attention maps.

    The gate operates in [0, 1] and can optionally blend in classic image priors
    such as center bias or edge maps. Attention maps are drawn from the last block
    of the provided backbone (pretrained weights are used; no fine-tuned checkpoint
    is required).
    """

    def __init__(
        self,
        backbone: Optional[torch.nn.Module] = None,
        *,
        backbone_name: str = "dinov2_vitb14",
        attn_pool: str = "last",
        combine_center: float = 0.0,
        postproc: Optional[Dict[str, float | bool]] = None,
        device: Optional[torch.device] = None,
        use_edges: bool = False,
    ) -> None:
        if backbone is None:
            backbone = _build_backbone(backbone_name)
        self.backbone = backbone
        self.backbone_name = str(backbone_name).lower()
        self.device = device or next(backbone.parameters()).device
        self.attn_pool = attn_pool.lower()
        if self.attn_pool not in {"last", "mean"}:
            raise ValueError(f"Unknown attention pooling strategy: {attn_pool}")
        self.combine_center = float(combine_center)
        self.use_edges = use_edges
        self.postproc = postproc or {
            "smooth_k": 3,
            "normalize": True,
            "threshold": None,
        }

        self._raw_model = getattr(backbone, "dino", backbone)
        raw_module = self._raw_model.__class__.__module__.lower()
        self._is_dinov3 = ("dinov3" in self.backbone_name) or ("dinov3" in raw_module)
        if not hasattr(self._raw_model, "blocks"):
            raise ValueError("Provided backbone does not expose transformer blocks.")

        patch_size = getattr(self._raw_model.patch_embed, "patch_size", 14)
        if isinstance(patch_size, Sequence):
            self.patch_h, self.patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            self.patch_h = self.patch_w = int(patch_size)
        self.num_register_tokens = int(
            getattr(self._raw_model, "num_register_tokens", 0)
        )
        self.num_storage_tokens = int(getattr(self._raw_model, "n_storage_tokens", 0))
        if self._is_dinov3:
            self.num_prefix_tokens = 1 + self.num_storage_tokens
        else:
            self.num_prefix_tokens = 1 + self.num_register_tokens

        self.requires_images = False

    def generate_from_batch(
        self,
        images: torch.Tensor,
        original_images: Optional[torch.Tensor] = None,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Generate foreground gate for a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W) already normalized for the backbone.
            original_images: Optional tensor with unnormalized images, used for edge priors.
            out_hw: Optional tuple defining the spatial resolution of the returned gate.

        Returns:
            Tensor of shape (B, H_out, W_out) with values in [0, 1].
        """

        attn_layers, patch_tokens = self._collect_attention(images)
        if not attn_layers:
            raise RuntimeError("Failed to capture attention maps from backbone.")

        if self.attn_pool == "mean":
            attn_tensor = torch.stack(attn_layers, dim=0).mean(dim=0)
        else:
            # # attn_tensor = attn_layers[-1]
            # # attn_tensor = torch.stack(attn_layers[1:2], dim=0).mean(dim=0)
            # if len(attn_layers) <= 12:
            #     # attn_tensor = torch.stack(attn_layers[-2:], dim=0).max(dim=0).values
            #     attn_tensor = torch.stack(attn_layers[10:12], dim=0).max(dim=0).values
            # else:
            attn_tensor = torch.stack(attn_layers[10:12], dim=0).max(dim=0).values

        attn_tensor = attn_tensor.mean(dim=1)  # (B, tokens, tokens)
        cls_attn = attn_tensor[:, 0, self.num_prefix_tokens :]
        b, n_tokens = cls_attn.shape

        grid_h = int(math.isqrt(n_tokens))
        grid_w = n_tokens // grid_h if grid_h > 0 else 0
        if grid_h * grid_w != n_tokens:
            grid_h = images.shape[-2] // self.patch_h
            grid_w = images.shape[-1] // self.patch_w
        if grid_h * grid_w != n_tokens:
            raise RuntimeError(
                f"Cannot reshape attention map of length {n_tokens} into 2D grid."
            )

        gate = cls_attn.view(b, grid_h, grid_w)

        if self.combine_center > 0.0:
            center = self._center_prior(grid_h, grid_w, gate.device)
            gate = (1.0 - self.combine_center) * gate + self.combine_center * center

        if self.use_edges and original_images is not None:
            edge_maps = self._edge_prior_batch(original_images, (grid_h, grid_w))
            gate = 0.7 * gate + 0.3 * edge_maps

        gate = self._apply_postproc(gate)

        if out_hw is not None:
            gate = F.interpolate(
                gate[:, None, ...], size=out_hw, mode="bilinear", align_corners=False
            )[:, 0, ...]
        return gate

    def generate(
        self,
        image: Image.Image,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate foreground gate for a single PIL image.
        """

        image_tensor = self._preprocess_image(image).to(self.device)
        attn_layers, _ = self._collect_attention(image_tensor)
        if not attn_layers:
            raise RuntimeError("Failed to capture attention maps from backbone.")

        if self.attn_pool == "mean":
            attn_tensor = torch.stack(attn_layers, dim=0).mean(dim=0)
        else:
            # attn_tensor = attn_layers[-1]
            # attn_tensor = torch.stack(attn_layers[-2:], dim=0).max(dim=0).values
            attn_tensor = torch.stack(attn_layers[10:12], dim=0).max(dim=0).values

        attn_tensor = attn_tensor.mean(dim=1)
        cls_attn = attn_tensor[:, 0, self.num_prefix_tokens :]
        n_tokens = cls_attn.shape[-1]
        grid_h = int(math.isqrt(n_tokens))
        grid_w = n_tokens // grid_h if grid_h > 0 else 0
        if grid_h * grid_w != n_tokens:
            raise RuntimeError(
                f"Cannot reshape attention map of length {n_tokens} into 2D grid."
            )

        gate = cls_attn.view(1, grid_h, grid_w)
        gate = self._apply_postproc(gate)
        if self.combine_center > 0.0:
            center = self._center_prior(grid_h, grid_w, gate.device)
            gate = (1.0 - self.combine_center) * gate + self.combine_center * center

        if out_hw is not None:
            gate = F.interpolate(
                gate[:, None, ...], size=out_hw, mode="bilinear", align_corners=False
            )[:, 0, ...]

        return gate.squeeze(0).cpu().numpy()

    def _collect_attention(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if self._is_dinov3:
            return self._collect_attention_dinov3(images)

        raw = self._raw_model
        num_blocks = len(getattr(raw, "blocks", []))
        buffers: List[Optional[torch.Tensor]] = [None] * num_blocks
        handles = []

        def make_hook(idx: int):
            def hook(module, inputs, output):
                attn = inputs[0]
                if isinstance(attn, tuple):
                    attn = attn[0]
                if attn is None:
                    return
                buffers[idx] = attn.detach()

            return hook

        for idx, block in enumerate(raw.blocks):
            attn_module = getattr(block, "attn", None)
            attn_drop = getattr(attn_module, "attn_drop", None) if attn_module else None
            if attn_drop is None:
                continue
            handles.append(attn_drop.register_forward_hook(make_hook(idx)))

        with torch.inference_mode():
            outputs = self.backbone(images)

        for handle in handles:
            handle.remove()

        attn_layers = [buf for buf in buffers if buf is not None]
        if isinstance(outputs, tuple):
            patch_tokens = outputs[0]
        else:
            patch_tokens = outputs
        return attn_layers, patch_tokens

    def _collect_attention_dinov3(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Collect attention maps for DINOv3 by recomputing QK attention."""

        raw = self._raw_model
        if not hasattr(raw, "rope_embed"):
            raise ValueError(
                "DINOv3 attention path requires rope_embed on the raw model."
            )

        was_training = raw.training
        raw.eval()

        attn_layers: List[torch.Tensor] = []
        with torch.inference_mode():
            x, (H, W) = raw.prepare_tokens_with_masks(images)
            for blk in raw.blocks:
                rope = raw.rope_embed(H=H, W=W) if raw.rope_embed else None
                attn_layers.append(self._compute_dinov3_attention(blk, x, rope))
                x = blk(x, rope)

            x_norm = self._apply_dinov3_norm(raw, x)
            patch_tokens = x_norm[:, self.num_prefix_tokens :, :]

        if was_training:
            raw.train()

        return attn_layers, patch_tokens

    def _compute_dinov3_attention(
        self,
        block: torch.nn.Module,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Return attention weights (B, heads, tokens, tokens) for a DINOv3 block."""

        x_norm = block.norm1(x)
        qkv = block.attn.qkv(x_norm)
        B, N, _ = qkv.shape
        C = block.attn.qkv.in_features
        qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
        q, k, _ = torch.unbind(qkv, dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if rope is not None:
            q, k = block.attn.apply_rope(q, k, rope)

        q = q.float()
        k = k.float()
        head_dim = q.shape[-1]
        scale = head_dim**-0.5
        attn_scores = torch.matmul(q * scale, k.transpose(-2, -1))
        return attn_scores.softmax(dim=-1)

    def _apply_dinov3_norm(self, raw: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply DINOv3's norm logic for both CLS/register and patch tokens."""

        if raw.untie_cls_and_patch_norms or raw.untie_global_and_local_cls_norm:
            if raw.untie_global_and_local_cls_norm and raw.training:
                cls_norm = raw.local_cls_norm(x[:, : self.num_prefix_tokens])
            elif raw.untie_cls_and_patch_norms:
                cls_norm = raw.cls_norm(x[:, : self.num_prefix_tokens])
            else:
                cls_norm = raw.norm(x[:, : self.num_prefix_tokens])
            patch_norm = raw.norm(x[:, self.num_prefix_tokens :])
            return torch.cat((cls_norm, patch_norm), dim=1)
        return raw.norm(x)

    def _apply_postproc(self, tensor: torch.Tensor) -> torch.Tensor:
        cfg = self.postproc or {}
        smooth_k = int(cfg.get("smooth_k", 1) or 1)
        if smooth_k > 1:
            pad = smooth_k // 2
            tensor = F.avg_pool2d(
                tensor[:, None, ...],
                kernel_size=smooth_k,
                stride=1,
                padding=pad,
            )[:, 0, ...]

        if cfg.get("normalize", True):
            tensor = _minmax(tensor)

        threshold = cfg.get("threshold")
        if threshold is not None:
            tensor = (tensor >= float(threshold)).to(tensor.dtype)
        return tensor

    def _center_prior(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        dy = (yy - cy) / max(1e-6, 0.5 * h)
        dx = (xx - cx) / max(1e-6, 0.5 * w)
        r2 = dx * dx + dy * dy
        prior = torch.exp(-3.0 * r2)
        return prior

    def _edge_prior_batch(
        self, images: torch.Tensor, out_hw: Tuple[int, int]
    ) -> torch.Tensor:
        imgs = images.detach().cpu()
        b = imgs.shape[0]
        edges: List[torch.Tensor] = []
        for i in range(b):
            pil = self._tensor_to_pil(imgs[i])
            edge_arr = self._edges(pil, out_hw=out_hw)
            edges.append(torch.from_numpy(edge_arr))
        stacked = torch.stack(edges, dim=0).to(dtype=torch.float32, device=self.device)
        return stacked

    def _edges(
        self, img: Image.Image, out_hw: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        blurred = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
        edges = blurred.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(edges, dtype="float32")
        arr = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
        arr = np.power(arr, 0.5)
        if out_hw is not None:
            pil = Image.fromarray((arr * 255).astype("uint8"))
            pil = pil.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
            arr = np.asarray(pil, dtype="float32") / 255.0
        return arr

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize(
            (self.patch_w * 16, self.patch_h * 16), Image.BILINEAR
        )
        arr = torch.from_numpy(np.array(img).astype("float32") / 255.0).permute(2, 0, 1)
        arr = (arr - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(
            IMAGENET_STD
        ).view(3, 1, 1)
        return arr.unsqueeze(0)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = tensor.clamp(0.0, 1.0)
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(arr)


def _minmax(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.view(tensor.shape[0], -1)
    min_vals = flat.min(dim=1, keepdim=True).values
    max_vals = flat.max(dim=1, keepdim=True).values
    norm = (flat - min_vals) / (max_vals - min_vals + 1e-6)
    return norm.view_as(tensor)


def _build_backbone(name: str) -> torch.nn.Module:
    name = name.lower()
    if name.startswith("dinov3"):
        if "expand" in name:
            return DINOv3BackboneExpanded(name=name, n_splits=0, mode="block_expansion")
        return DINOv3Backbone(name=name)
    if name.startswith("dinov2"):
        if "expand" in name:
            return DINOv2BackboneExpanded(name=name, n_splits=0, mode="block_expansion")
        return DINOv2Backbone(name=name)
    raise ValueError(f"Unsupported backbone name '{name}'.")


class PCA(torch.nn.Module):
    """Patch-level pseudo-labeling via PCA thresholding.

    Assigns foreground patches to the image label and background patches to
    `bg_class` based on a PCA projection threshold.

    The public attributes `compare_fn`, `threshold`, and `bg_class` are used by
    other components (e.g., inference-time PCA gating in the generative model).
    """

    requires_images = False

    def __init__(
        self,
        *,
        compare_fn: str = "le",
        threshold: float = 0.5,
        n_components: int = 1,
        bg_class: int = 200,
    ) -> None:
        super().__init__()
        self.compare_fn = torch.ge if str(compare_fn) == "ge" else torch.le
        self.threshold = float(threshold)
        self.n_components = int(n_components)
        self.bg_class = int(bg_class)

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        *,
        images: torch.Tensor | None = None,
        cls_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pseudo patch labels via PCA thresholding.

        Foreground patches are assigned the image label; background patches are
        assigned `self.bg_class`. This is used during training as a label-free
        proxy for routing FG/BG patches.
        """

        _ = images
        _ = cls_tokens
        B, N, dim = tokens.shape
        H = W = int(math.isqrt(N))
        if H * W != N:
            raise ValueError(
                f"Token count {N} is not a square number; cannot form a grid."
            )

        U, _, _ = torch.pca_lowrank(
            tokens.reshape(-1, dim),
            q=self.n_components,
            center=True,
            niter=10,
        )
        # Scale to [0,1] so the threshold is interpretable/stable across images.
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()
        U_scaled = U_scaled.reshape(B, H, W)

        mask = self.compare_fn(U_scaled, other=self.threshold).to(dtype=torch.bool)
        labels_expanded = labels.view(B, 1, 1).expand(-1, H, W)
        bg = torch.full_like(labels_expanded, fill_value=self.bg_class)
        pseudo = torch.where(mask, labels_expanded, bg)
        return pseudo.to(dtype=torch.long)

    def infer_mask(
        self,
        tokens: torch.Tensor,
        *,
        images: torch.Tensor | None = None,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Infer a label-free foreground mask via PCA thresholding.

        Returns a float mask in {0,1} with the same dtype/device as `tokens`.
        """

        _ = images
        B, N, dim = tokens.shape
        H, W = grid_hw
        if H * W != N:
            raise ValueError(f"Token count {N} does not match grid {grid_hw}.")

        tokens_f = tokens.to(dtype=torch.float32)
        U, _, _ = torch.pca_lowrank(
            tokens_f.reshape(-1, dim),
            q=self.n_components,
            center=True,
            niter=10,
        )
        u_min = U.min()
        u_max = U.max()
        u_scaled = (U - u_min) / (u_max - u_min + 1.0e-6)
        u_scaled = u_scaled.reshape(B, H, W)

        mask = self.compare_fn(u_scaled, other=self.threshold)
        return mask.to(device=tokens.device, dtype=tokens.dtype)


class AttentionForegroundExtractor(torch.nn.Module):
    """
    Wrapper that turns the attention gate into pseudo patch labels compatible with vMF-MP training.
    """

    requires_images = True

    def __init__(
        self,
        gater: BackboneAttentionForegroundGater,
        *,
        threshold: float = 0.5,
        bg_class: int = 200,
    ) -> None:
        super().__init__()
        self.gater = gater
        self.threshold = float(threshold)
        self.bg_class = int(bg_class)
        self._last_gate: torch.Tensor | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        *,
        images: torch.Tensor,
        cls_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pseudo patch labels from an attention-based foreground gate."""

        B, N, _ = tokens.shape
        H = W = int(math.isqrt(N))
        if H * W != N:
            raise ValueError(
                f"Token count {N} is not a square number; cannot form a grid."
            )

        with torch.inference_mode():
            gate = self.gater.generate_from_batch(images, out_hw=(H, W))

        # Cache the soft gate so downstream modules (e.g., spatial losses)
        # can reuse it without recomputing attention.
        self._last_gate = gate

        if gate.dim() != 3:
            raise ValueError(
                f"Attention gate must be (B, H, W); got {tuple(gate.shape)}."
            )

        mask = (gate >= self.threshold).to(dtype=torch.bool)
        labels_expanded = labels.view(B, 1, 1).expand(-1, H, W)
        bg = torch.full_like(labels_expanded, fill_value=self.bg_class)
        pseudo = torch.where(mask, labels_expanded, bg)
        return pseudo.to(dtype=torch.long)

    def infer_mask(
        self,
        tokens: torch.Tensor,
        *,
        images: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Infer a soft foreground mask without labels (label-free)."""

        H, W = grid_hw
        with torch.inference_mode():
            gate = self.gater.generate_from_batch(images, out_hw=(H, W))
        self._last_gate = gate
        return gate.to(device=tokens.device, dtype=tokens.dtype)


class AttnThenPCAForegroundExtractor(torch.nn.Module):
    """
    Two-stage foreground extractor: attention ROI followed by PCA refinement.

    Stage 1: use attention to produce a hard ROI mask.
    Stage 2: run per-image PCA on ROI patch tokens and keep the subset that
    best aligns with the attention mass (removes likely background patches).
    """

    requires_images = True

    def __init__(
        self,
        gater: BackboneAttentionForegroundGater,
        *,
        attn_threshold: float = 0.5,
        pca_threshold: float | None = None,
        pca_keep_quantile: float = 0.5,
        pca_min_patches: int = 16,
        pca_niter: int = 10,
        pca_center: bool = True,
        bg_class: int = 200,
    ) -> None:
        super().__init__()
        self.gater = gater
        self.attn_threshold = float(attn_threshold)
        self.pca_threshold = None if pca_threshold is None else float(pca_threshold)
        self.pca_keep_quantile = float(pca_keep_quantile)
        self.pca_min_patches = int(pca_min_patches)
        self.pca_niter = int(pca_niter)
        self.pca_center = bool(pca_center)
        self.bg_class = int(bg_class)
        self._last_gate: torch.Tensor | None = None

    def _refine_mask(
        self,
        tokens: torch.Tensor,
        attn_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Return a refined binary mask for a single image."""

        flat_gate = attn_gate.reshape(-1)
        hard = flat_gate >= self.attn_threshold
        num_selected = int(hard.sum().item())
        if num_selected < self.pca_min_patches:
            return hard.reshape_as(attn_gate)

        selected = tokens[hard].to(dtype=torch.float32)
        if self.pca_center:
            selected = selected - selected.mean(dim=0, keepdim=True)

        _, _, v = torch.pca_lowrank(selected, q=1, center=False, niter=self.pca_niter)
        scores = selected @ v[:, 0]
        if scores.numel() == 0:
            return hard.reshape_as(attn_gate)

        s_min = scores.min()
        s_max = scores.max()
        denom = (s_max - s_min).clamp_min(1.0e-6)
        scores_norm = (scores - s_min) / denom  # [0, 1] within ROI

        # PCA direction has an arbitrary sign. Orient scores so "high" aligns
        # with the attention mass inside the ROI.
        cut = torch.quantile(scores_norm, 0.5)
        high = scores_norm >= cut
        if high.all() or (~high).all():
            return hard.reshape_as(attn_gate)

        attn_selected = flat_gate[hard]
        mean_high = attn_selected[high].mean()
        mean_low = attn_selected[~high].mean()
        if mean_high < mean_low:
            scores_norm = 1.0 - scores_norm

        if self.pca_threshold is not None:
            keep_mask = scores_norm >= self.pca_threshold
        else:
            keep_q = min(max(self.pca_keep_quantile, 1.0e-3), 1.0 - 1.0e-3)
            thr = torch.quantile(scores_norm, 1.0 - keep_q)
            keep_mask = scores_norm >= thr

        if keep_mask.all() or (~keep_mask).all():
            return hard.reshape_as(attn_gate)
        if keep_mask.sum().item() == 0:
            return hard.reshape_as(attn_gate)

        refined = hard.clone()
        refined[hard] = keep_mask
        return refined.reshape_as(attn_gate)

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        *,
        images: torch.Tensor,
        cls_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pseudo patch labels using attention ROI + PCA refinement."""

        B, N, _ = tokens.shape
        H = W = int(math.isqrt(N))
        if H * W != N:
            raise ValueError(
                f"Token count {N} is not a square number; cannot form a grid."
            )

        with torch.inference_mode():
            gate = self.gater.generate_from_batch(images, out_hw=(H, W))

        if gate.dim() != 3:
            raise ValueError(
                f"Attention gate must be (B, H, W); got {tuple(gate.shape)}."
            )

        refined_masks = []
        for b in range(B):
            refined = self._refine_mask(tokens[b], gate[b])
            refined_masks.append(refined)
        refined_mask = torch.stack(refined_masks, dim=0)

        self._last_gate = refined_mask.to(dtype=tokens.dtype)
        labels_expanded = labels.view(B, 1, 1).expand(-1, H, W)
        bg = torch.full_like(labels_expanded, fill_value=self.bg_class)
        pseudo = torch.where(refined_mask, labels_expanded, bg)
        return pseudo.to(dtype=torch.long)

    def infer_mask(
        self,
        tokens: torch.Tensor,
        *,
        images: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Infer a refined foreground mask without labels (label-free)."""

        B, N, _ = tokens.shape
        H, W = grid_hw
        if H * W != N:
            raise ValueError(f"Token count {N} does not match grid {grid_hw}.")

        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        with torch.inference_mode():
            gate = self.gater.generate_from_batch(images, out_hw=(H, W))

        refined_masks = []
        for b in range(B):
            refined = self._refine_mask(tokens_norm[b], gate[b])
            refined_masks.append(refined)
        refined_mask = torch.stack(refined_masks, dim=0).to(
            device=tokens.device, dtype=tokens.dtype
        )
        self._last_gate = refined_mask
        return refined_mask


__all__ = [
    "BackboneAttentionForegroundGater",
    "AttentionForegroundExtractor",
    "AttnThenPCAForegroundExtractor",
]
