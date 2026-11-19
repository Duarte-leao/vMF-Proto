"""Class-specific and shared distributional prototype models."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .fitting import log_class_evidence, raw_scores
from .vmf_mp import VMFMP
from .vmf_normalizer import logCp_vmf


class FourLossVMF(VMFMP):
    def __init__(self, *, variant="class_specific", shared_prototypes=1000, **kwargs):
        super().__init__(**kwargs)
        if variant not in {"class_specific", "shared"}:
            raise ValueError("variant must be class_specific or shared")
        self.variant = variant
        self.kappa_learnable = bool(kwargs.get("kappa_learnable", True))
        self.enable_parametric_prototypes()
        means, kappa = self.prototype_mu.detach(), self.kappa_raw.detach()
        if variant == "shared":
            if (
                shared_prototypes < 2
                or shared_prototypes > self.n_classes * self.n_prototypes
            ):
                raise ValueError("Shared budget must be between 2 and C*J")
            self.foreground_prototypes = int(shared_prototypes)
            means = torch.cat((means[:-1].flatten(0, 1)[:shared_prototypes], means[-1]))
            kappa = torch.cat((kappa[:-1].flatten()[:shared_prototypes], kappa[-1]))
            self.background_pi_logits = nn.Parameter(
                self.pi_logits[-1].detach().clone()
            )
            logits = torch.zeros(self.n_classes, shared_prototypes)
            if not bool(kwargs.get("pi_init_uniform", True)):
                weights = torch.rand_like(logits).clamp_min(self.EPS)
                logits = (weights / weights.sum(-1, keepdim=True)).log()
            self.pi_logits = nn.Parameter(logits)
        else:
            self.foreground_prototypes = self.n_classes * self.n_prototypes
            means, kappa = means.flatten(0, 1), kappa.flatten()
        self.prototype_mu = nn.Parameter(means.clone())
        # The configured value is actual kappa, not its softplus preimage.
        initial = torch.full_like(kappa, float(kwargs.get("kappa_init", 5.0)))
        self.kappa_raw = nn.Parameter(
            initial + torch.log(-torch.expm1(-initial)),
            requires_grad=self.kappa_learnable,
        )
        self.optimizing_prototypes = False

    @property
    def shared(self):
        return self.variant == "shared"

    def bank(self):
        means = F.normalize(self.prototype_mu.double(), dim=-1)
        kappa = F.softplus(self.kappa_raw.double())
        pi = (
            self.pi_logits.double().softmax(-1)
            if self.shared
            else self.pi_logits.softmax(-1).double()
        )
        return dict(
            means=means,
            kappa=kappa,
            kappa_learnable=self.kappa_learnable,
            logc=logCp_vmf(self.dim, kappa),
            pi=pi if self.shared else pi[:-1],
            bg_pi=self.background_pi_logits.double().softmax(-1)
            if self.shared
            else pi[-1],
        )

    def class_evidence(self, scores, foreground, bank):
        raw = scores[..., : self.foreground_prototypes]
        if not self.shared:
            raw = raw.reshape(*raw.shape[:2], self.n_classes, self.n_prototypes)
        return log_class_evidence(raw, foreground, bank["pi"], shared=self.shared)

    def class_prototype_indices(self, labels, topk=5):
        if self.shared:
            return (
                self.pi_logits[labels]
                .topk(min(topk, self.foreground_prototypes), -1)
                .indices
            )
        return labels[:, None] * self.n_prototypes + torch.arange(
            self.n_prototypes, device=labels.device
        )

    @torch.no_grad()
    def frozen_prefix(self, images):
        """Cache only frozen blocks; no trained feature cache/donor is required."""
        if hasattr(self.backbone, "frozen_prefix"):
            return self.backbone.frozen_prefix(images)
        dino = self.backbone.dino
        prepared = dino.prepare_tokens_with_masks(images)
        if isinstance(prepared, tuple):
            value, grid = prepared
            rope = dino.rope_embed(H=grid[0], W=grid[1]) if dino.rope_embed else None
            for block in dino.blocks[:-1]:
                value = block(value, rope)
        else:
            value = prepared
            for block in dino.blocks[:-1]:
                value = block(value)
        return value

    def adapt_prefix(self, prefix, grid):
        if hasattr(self.backbone, "adapt_prefix"):
            return F.normalize(self.backbone.adapt_prefix(prefix, grid), dim=-1)
        dino = self.backbone.dino
        if hasattr(dino, "rope_embed"):
            rope = dino.rope_embed(H=grid[0], W=grid[1]) if dino.rope_embed else None
            value = self.backbone._apply_norm(dino.blocks[-1](prefix, rope))
            offset = self.backbone.num_prefix_tokens
        else:
            value = dino.norm(dino.blocks[-1](prefix))
            offset = 1 + dino.num_register_tokens
        return F.normalize(value[:, offset:], dim=-1)

    def frozen_patches(self, prefix):
        if hasattr(self.backbone, "frozen_patches"):
            return self.backbone.frozen_patches(prefix)
        if hasattr(self.backbone, "_apply_norm"):
            return self.backbone._apply_norm(prefix)[
                :, self.backbone.num_prefix_tokens :
            ]
        return self.backbone.dino.norm(prefix)[
            :, 1 + self.backbone.dino.num_register_tokens :
        ]

    def foreground(self, frozen, images, grid):
        gate = self._infer_fg_mask(frozen, images=images, grid_hw=grid)
        if gate is None:
            raise ValueError("A foreground extractor with infer_mask is required")
        gate = gate.reshape(frozen.shape[:2]) >= 0.5
        return torch.where(gate.any(-1, keepdim=True), gate, True)

    def forward(self, images, labels=None, **unused):
        adapted, frozen, *_ = self.backbone(images)
        side = math.isqrt(adapted.shape[1])
        grid = (side, side)
        if side * side != adapted.shape[1]:
            raise ValueError("Expected a square patch grid")
        gate = self.foreground(frozen, images, grid)
        bank = self.bank()
        raw = raw_scores(adapted, bank)
        logits = self.class_evidence(raw, gate, bank)
        maps = raw.transpose(1, 2).reshape(len(images), -1, *grid)
        return dict(
            class_logits=logits,
            log_class_evidence=logits,
            component_scores=raw,
            maps_raw=maps,
            maps_norm=self._normalize_maps(maps),
            fg_mask=gate.reshape(len(images), 1, *grid),
            grid_hw=grid,
        )

    def push_forward(self, images):
        # Use adapted features for prototype activation maps.
        raw = self(images)["maps_raw"][:, : self.foreground_prototypes]
        density = (raw - raw.amax((-2, -1), keepdim=True)).exp().float()
        return None, density

    @torch.no_grad()
    def install_bank(self, bank):
        self.prototype_mu.copy_(bank["means"].to(self.prototype_mu))
        k = bank["kappa"].double()
        self.kappa_raw.copy_((k + torch.log(-torch.expm1(-k))).to(self.kappa_raw))
