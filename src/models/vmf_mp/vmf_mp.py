"""vMF-MP: class-conditional vMF mixture prototypes over patch tokens.

This module mirrors the backbone + FG gating wiring used throughout the repo,
but replaces classification with a generative mixture score and uses OT-based
assignments.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn

from .utils import (
    momentum_update,
    sinkhorn_transport,
)
from .vmf_normalizer import logCp_vmf


class VMFMP(nn.Module):
    """Class-conditional vMF mixture model over patch tokens.

    The model computes per-class log-likelihoods by mixing K directional
    components (prototypes) per class and uses OT-based assignments to update
    the prototype bank during stage-1 training.
    """

    EPS = 1.0e-6

    def __init__(
        self,
        backbone: nn.Module,
        fg_extractor: nn.Module,
        *,
        gamma: float = 0.999,
        n_prototypes: int = 5,
        n_classes: int = 200,
        norm_prototypes: bool = True,
        cluster_use_norm_tokens: bool = True,
        temperature: float = 1.0,
        dim: int = 768,
        kappa_init: float = 5.0,
        kappa_learnable: bool = True,
        maps_norm_mode: str = "minmax",
        ot_mode: str = "balanced",
        ot_epsilon: float = 0.05,
        ot_iters: int = 50,
        pi_decoupled: bool = False,
        pi_init_uniform: bool = True,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.n_prototypes = int(n_prototypes)
        self.n_classes = int(n_classes)
        self.C = self.n_classes + 1
        self.bg_class = self.n_classes
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.norm_prototypes = bool(norm_prototypes)
        self.cluster_use_norm_tokens = bool(cluster_use_norm_tokens)

        maps_norm_mode = str(maps_norm_mode).lower()
        if maps_norm_mode not in {"minmax", "maxonly"}:
            raise ValueError(f"Unknown maps_norm_mode: {maps_norm_mode}")
        self.maps_norm_mode = maps_norm_mode

        self.backbone = backbone
        self.fg_extractor = fg_extractor
        self.fg_requires_images = getattr(fg_extractor, "requires_images", False)

        ot_mode = str(ot_mode).lower()
        if ot_mode not in {"balanced", "marginal"}:
            raise ValueError(f"Unknown OT mode: {ot_mode}")
        self.ot_mode = ot_mode

        self.pi_decoupled = bool(pi_decoupled)

        self.ot_epsilon = float(ot_epsilon)
        self.ot_iters = int(ot_iters)

        prototype_shape = (self.C, self.n_prototypes, self.dim)
        self.register_buffer(
            "prototype_mu", torch.randn(prototype_shape, dtype=torch.float32)
        )
        nn.init.trunc_normal_(self.prototype_mu, std=0.02)

        # Concentrations κ. When kappa_learnable=False, κ is fixed (no gradients,
        # not included in the stage-2 optimizer) but still participates in logits.
        kappa_raw_init = torch.full(
            (self.C, self.n_prototypes),
            float(kappa_init),
            dtype=torch.float32,
        )
        if bool(kappa_learnable):
            self.kappa_raw = nn.Parameter(kappa_raw_init)
        else:
            self.register_buffer("kappa_raw", kappa_raw_init)

        # Learnable per-class mixture weights π = softmax(pi_logits).
        self.pi_logits = nn.Parameter(
            torch.zeros(self.C, self.n_prototypes, dtype=torch.float32)
        )
        if not bool(pi_init_uniform):
            # Randomize π initialization while keeping it on the simplex for each
            # class (including the background class). This affects only the initial
            # condition: checkpoints overwrite `pi_logits` when loaded.
            with torch.no_grad():
                init_pi = torch.rand_like(self.pi_logits).clamp_min(self.EPS)
                init_pi = init_pi / init_pi.sum(dim=-1, keepdim=True).clamp_min(
                    self.EPS
                )
                self.pi_logits.copy_(torch.log(init_pi))

        self.optimizing_prototypes = True
        self.parametric_prototypes = False

    def enable_parametric_prototypes(self) -> None:
        """Switch prototypes from a buffer to a learnable parameter."""

        if self.parametric_prototypes:
            return

        proto = self.prototype_mu.detach().clone()
        if "prototype_mu" in self._buffers:
            del self._buffers["prototype_mu"]
        self.prototype_mu_init = proto.clone()
        self.prototype_mu = nn.Parameter(proto)
        self.parametric_prototypes = True

    def _normalize_maps(self, maps: torch.Tensor) -> torch.Tensor:
        eps = self.EPS
        if self.maps_norm_mode == "maxonly":
            denom = maps.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
            normalized = maps / denom
        else:
            flat = maps.reshape(*maps.shape[:-2], -1)
            min_vals = flat.min(dim=-1, keepdim=True).values
            max_vals = flat.max(dim=-1, keepdim=True).values
            normalized = (flat - min_vals) / (max_vals - min_vals + eps)
            normalized = normalized.reshape_as(maps)
        return normalized.clamp(0.0, 1.0)

    def _get_pi(self) -> torch.Tensor:
        if self.ot_mode == "balanced":
            return torch.full(
                (self.C, self.n_prototypes),
                1.0 / max(self.n_prototypes, 1),
                device=self.prototype_mu.device,
                dtype=self.prototype_mu.dtype,
            )
        return F.softmax(self.pi_logits, dim=-1)

    def _get_pi_for_scoring(self) -> torch.Tensor:
        """Return the mixture weights used for *scoring* (classification).

        By default, scoring uses the same π as OT.
        When `pi_decoupled=True`, scoring uses a learnable π even if OT uses
        balanced (uniform) column marginals.
        """

        if not self.pi_decoupled:
            return self._get_pi()
        return F.softmax(self.pi_logits, dim=-1)

    def _infer_fg_mask(
        self,
        tokens: torch.Tensor,
        *,
        images: Optional[torch.Tensor],
        grid_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        infer_mask = getattr(self.fg_extractor, "infer_mask", None)
        if callable(infer_mask):
            if images is None and self.fg_requires_images:
                return None
            return infer_mask(tokens, images=images, grid_hw=grid_hw)

        gater = getattr(self.fg_extractor, "gater", None)
        if gater is None or images is None:
            return None

        with torch.inference_mode():
            gate = gater.generate_from_batch(images, out_hw=grid_hw)
        return gate.to(device=tokens.device, dtype=tokens.dtype)

    def _score_fg_only(
        self, log_p_fg: torch.Tensor, fg_weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if fg_weights is None:
            return log_p_fg.sum(dim=1)
        # Use a per-image average to keep scores comparable across FG coverage.
        weight_sum = fg_weights.sum(dim=1, keepdim=True).clamp_min(self.EPS)
        return (log_p_fg * fg_weights.unsqueeze(-1)).sum(dim=1) / weight_sum

    def _online_clustering(
        self,
        prototypes: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_prototype_logits: torch.Tensor,
        patch_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run OT assignments and return updated prototypes/weights."""

        B, H, W = patch_labels.shape
        K = self.n_prototypes
        patch_labels_flat = patch_labels.reshape(-1)
        patches_flat = rearrange(patch_tokens, "B N D -> (B N) D").to(
            dtype=torch.float32
        )
        patches_norm = F.normalize(patches_flat, p=2, dim=-1)
        logits_flat = rearrange(patch_prototype_logits, "B N C K -> (B N) C K").to(
            dtype=torch.float32
        )

        part_assignment_maps = torch.full(
            (patch_labels_flat.numel(),),
            fill_value=self.bg_class * K,
            device=patch_labels_flat.device,
            dtype=torch.long,
        )

        P_old = prototypes.clone()
        P_new = prototypes.clone()
        pi_for_ot = self._get_pi().detach()

        unique_labels = patch_labels_flat.unique().tolist()
        for c in unique_labels:
            c_int = int(c)
            class_mask = patch_labels_flat == c_int
            if not class_mask.any():
                continue

            I_c = patches_flat[class_mask]
            if I_c.numel() == 0:
                continue

            scores = logits_flat[class_mask, c_int, :]
            num_patches = int(class_mask.sum())

            row_marginals = torch.ones(
                num_patches, device=scores.device, dtype=scores.dtype
            )
            if self.ot_mode == "balanced":
                col_marginals = torch.full(
                    (K,),
                    row_marginals.sum() / max(K, 1),
                    device=scores.device,
                    dtype=scores.dtype,
                )
            else:
                pi_c = pi_for_ot[c_int]
                pi_c = pi_c / pi_c.sum().clamp_min(self.EPS)
                col_marginals = pi_c * row_marginals.sum()

            Q = sinkhorn_transport(
                scores,
                row_marginals=row_marginals,
                col_marginals=col_marginals,
                n_iters=self.ot_iters,
                epsilon=self.ot_epsilon,
            )

            hard_indices = Q.argmax(dim=1)
            # We always use hard OT assignments for prototype updates. The soft plan
            # `Q` is still computed (for stable transport) but is only used to choose
            # the argmax prototype for each patch.
            assignments = F.one_hot(hard_indices, num_classes=K).to(dtype=Q.dtype)

            P_c_new = assignments.t() @ I_c
            P_c_old = P_old[c_int]
            P_new[c_int, ...] = momentum_update(P_c_old, P_c_new, momentum=self.gamma)

            part_assignment_maps[class_mask] = hard_indices + c_int * K

        part_assignment_maps = rearrange(
            part_assignment_maps, "(B H W) -> B (H W)", B=B, H=H, W=W
        )

        return part_assignment_maps, P_new

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        use_gumbel: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass returning generative class logits and prototype maps."""

        _ = use_gumbel  # gumbel is not used in OT mode (kept for call parity).
        assert (not self.training) or (labels is not None)

        backbone_out = self.backbone(x)
        if not isinstance(backbone_out, tuple):
            raise ValueError("Backbone must return a tuple of tensors.")
        if len(backbone_out) == 3:
            patch_tokens, raw_patch_tokens, cls_tokens = backbone_out
        elif len(backbone_out) == 2:
            patch_tokens, cls_tokens = backbone_out
            raw_patch_tokens = patch_tokens
        else:
            raise ValueError("Unexpected number of outputs from backbone.")

        tokens_for_kernel = patch_tokens
        fg_tokens = raw_patch_tokens if raw_patch_tokens is not None else patch_tokens

        tokens_norm = F.normalize(tokens_for_kernel, p=2, dim=-1)
        mu_hat = F.normalize(self.prototype_mu, p=2, dim=-1)
        cos_logits = einsum(tokens_norm, mu_hat, "B N D, C K D -> B N C K")
        kappa_values = F.softplus(self.kappa_raw)
        kappa = kappa_values.unsqueeze(0).unsqueeze(0)
        # vMF component log-density (up to a constant):
        #   log p(t | μ, κ) = log C_d(κ) + κ μᵀt
        #
        # We use the *centered* normalizer log C_d(κ) - log C_d(0) because the
        # -log C_d(0) shift is a dimension-only constant (same for every class,
        # prototype, and patch) and therefore cancels in classification while
        # keeping values numerically well-scaled in high-D.
        logC = logCp_vmf(self.dim, kappa_values)
        patch_prototype_logits = kappa * cos_logits + logC.unsqueeze(0).unsqueeze(0)

        if True:
            max_indices = torch.argmax(patch_prototype_logits, dim=-1, keepdim=True)
            # zero the entries not corresponding to max indices
            patch_prototype_max = torch.where(
                torch.arange(
                    self.n_prototypes, device=patch_prototype_logits.device
                ).view(1, 1, 1, -1)
                == max_indices,
                patch_prototype_logits,
                torch.zeros_like(patch_prototype_logits),
            )
        else:
            patch_prototype_max = patch_prototype_logits

        pi = self._get_pi()
        log_pi = torch.log(pi.clamp_min(self.EPS))
        log_p = torch.logsumexp(
            patch_prototype_max + log_pi.unsqueeze(0).unsqueeze(0), dim=-1
        )
        log_p = torch.exp(log_p)
        log_p_fg = log_p[:, :, :-1]

        batch_size, n_patches, _, _ = patch_prototype_logits.shape
        H = W = int(sqrt(n_patches))

        fg_mask: Optional[torch.Tensor] = None
        pseudo_patch_labels: Optional[torch.Tensor] = None

        if labels is not None:
            fg_tokens_norm = F.normalize(fg_tokens, p=2, dim=-1)
            fg_kwargs = {"cls_tokens": cls_tokens}
            if self.fg_requires_images:
                fg_kwargs["images"] = x
            pseudo_patch_labels = self.fg_extractor(
                fg_tokens_norm.detach(), labels, **fg_kwargs
            ).detach()

            gate = getattr(self.fg_extractor, "_last_gate", None)
            if gate is not None:
                fg_mask = gate.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            else:
                fg_mask = (pseudo_patch_labels != self.bg_class).to(
                    dtype=patch_tokens.dtype
                )
        else:
            fg_mask = self._infer_fg_mask(
                fg_tokens,
                images=x if self.fg_requires_images else None,
                grid_hw=(H, W),
            )

        fg_weights = None
        if fg_mask is not None:
            fg_weights = fg_mask.reshape(batch_size, n_patches).clamp(0.0, 1.0)
            weight_sum = fg_weights.sum(dim=1, keepdim=True)
            if (weight_sum <= 0).any():
                fallback = torch.ones_like(fg_weights)
                fg_weights = torch.where(weight_sum > 0, fg_weights, fallback)

        class_logits = self._score_fg_only(log_p_fg, fg_weights)
        class_logits = class_logits / self.temperature

        maps_raw = rearrange(
            patch_prototype_logits, "B (H W) C K -> B C K H W", H=H, W=W
        )
        maps_norm = self._normalize_maps(maps_raw)

        outputs: dict[str, torch.Tensor | None] = {
            "patch_prototype_logits": patch_prototype_logits,
            "patch_prototype_evidence": patch_prototype_max,
            "image_prototype_logits": patch_prototype_logits.sum(dim=1),
            "class_logits": class_logits,
            "grid_hw": (H, W),
            "maps_raw": maps_raw,
            "maps_norm": maps_norm,
            # Expose parameters needed by external losses (do not detach here).
            "prototype_mu": self.prototype_mu,
            "kappa_values": kappa_values,
            "fg_mask": None,
        }

        if self.parametric_prototypes:
            proto_norm = self.prototype_mu.norm(p=2, dim=-1).mean()
            outputs["_metric_proto_norm_mean"] = proto_norm.detach()

        if labels is not None and pseudo_patch_labels is not None:
            tokens_for_update = (
                raw_patch_tokens if raw_patch_tokens is not None else patch_tokens
            )
            if self.cluster_use_norm_tokens:
                tokens_for_update = F.normalize(tokens_for_update, p=2, dim=-1)

            (
                part_assignment_maps,
                new_mu,
            ) = self._online_clustering(
                prototypes=self.prototype_mu,
                patch_tokens=tokens_for_update.detach(),
                patch_prototype_logits=patch_prototype_logits.detach(),
                patch_labels=pseudo_patch_labels,
            )

            if (
                self.training
                and self.optimizing_prototypes
                and not self.parametric_prototypes
            ):
                updated = (
                    F.normalize(new_mu, p=2, dim=-1) if self.norm_prototypes else new_mu
                )
                self.prototype_mu.copy_(updated.detach())

            outputs.update(
                {
                    "patches": F.normalize(fg_tokens, p=2, dim=-1),
                    "part_assignment_maps": part_assignment_maps,
                    "pseudo_patch_labels": pseudo_patch_labels,
                    "fg_mask": (
                        fg_mask.to(device=maps_raw.device, dtype=maps_raw.dtype)
                        .unsqueeze(1)
                        .clone()
                        if fg_mask is not None
                        else None
                    ),
                    "topk_proto_indices": None,
                }
            )
        else:
            outputs["fg_mask"] = (
                fg_mask.to(device=maps_raw.device, dtype=maps_raw.dtype)
                .unsqueeze(1)
                .clone()
                if fg_mask is not None
                else None
            )
            outputs["topk_proto_indices"] = None

        outputs.setdefault("fg_mask", None)
        outputs.setdefault("topk_proto_indices", None)
        outputs.setdefault("peaks_xy", None)
        outputs.setdefault("peaks_feat", None)

        return outputs

    def push_forward(self, x: torch.Tensor):
        """Return prototype activation maps in the format expected by metrics.

        Returns:
            Tuple `(None, maps)` where `maps` has shape `(B, C*K, H, W)` and
            contains per-(class,prototype) patch scores reshaped to the token grid.
            The background class is excluded from the returned tensor.
        """

        backbone_out = self.backbone(x)
        if not isinstance(backbone_out, tuple):
            raise ValueError("Backbone must return a tuple of tensors.")
        if len(backbone_out) == 3:
            patch_tokens, raw_patch_tokens, _ = backbone_out
        elif len(backbone_out) == 2:
            patch_tokens, _ = backbone_out
            raw_patch_tokens = patch_tokens
        else:
            raise ValueError("Unexpected number of outputs from backbone.")

        tokens = raw_patch_tokens if raw_patch_tokens is not None else patch_tokens
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        mu_hat = F.normalize(self.prototype_mu, p=2, dim=-1)
        cos_logits = einsum(tokens_norm, mu_hat, "B N D, C K D -> B N C K")
        kappa_values = F.softplus(self.kappa_raw)
        kappa = kappa_values.unsqueeze(0).unsqueeze(0)
        logC = logCp_vmf(self.dim, kappa_values)
        patch_prototype_logits = kappa * cos_logits + logC.unsqueeze(0).unsqueeze(0)

        patch_prototype_logits = torch.exp(
            patch_prototype_logits
        )  # convert log-densities to densities
        batch_size, n_patches, _, _ = patch_prototype_logits.shape
        H = W = int(sqrt(n_patches))
        prototype_logits = rearrange(
            patch_prototype_logits[:, :, :-1, :],
            "B (H W) C K -> B (C K) H W",
            H=H,
            W=W,
        )
        return None, prototype_logits


__all__ = ["VMFMP"]
