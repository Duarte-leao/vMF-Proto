"""Losses for vMF-MP training.

This module contains small, modular loss components plus a thin criterion that
combines them based on config coefficients.

Design goals:
- Losses are computed *outside* the model forward pass.
- Each loss is an isolated module with a consistent interface.
- The training loop calls a single criterion (as before).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.models.vmf_mp.utils import prototype_logdet_vmf_overlap


@dataclass(frozen=True)
class LossTermOutput:
    """Standard return type for individual loss terms."""

    value: torch.Tensor
    unscaled: torch.Tensor | None = None


class ClassificationCrossEntropy(nn.Module):
    """Classification loss on `outputs["class_logits"]`."""

    def __init__(self) -> None:
        super().__init__()
        self._xe = nn.CrossEntropyLoss()

    def forward(
        self, outputs: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> LossTermOutput:
        """Compute cross-entropy between predicted class logits and labels."""

        loss = self._xe(outputs["class_logits"], labels)
        return LossTermOutput(value=loss, unscaled=loss.detach())


class PatchPrototypeDistillation(nn.Module):
    """PPD loss: distill patch-prototype logits to OT hard labels."""

    def __init__(self, *, class_weight: torch.Tensor, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)
        # Stored as buffer so it moves with `.to(device)` if desired.
        self.register_buffer("class_weight", class_weight.to(dtype=torch.float32))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        assignments: torch.Tensor,
    ) -> LossTermOutput:
        """Distill patch-prototype logits to discrete OT assignments."""

        logits = outputs["patch_prototype_logits"]
        logits = rearrange(logits, "B N C K -> B (C K) N") / self.temperature
        # `nn.Module` instances are not automatically moved to the training device
        # in this repo, so keep the weight device-safe here.
        weight = self.class_weight.to(device=logits.device, dtype=torch.float32)
        loss = F.cross_entropy(logits, target=assignments, weight=weight)
        return LossTermOutput(value=loss, unscaled=loss.detach())


class ResponsibilityPurity(nn.Module):
    """Background-aware responsibility entropy penalty.

    Encourages *high-entropy* prototype responsibilities on background patches
    of the correct class (to avoid a foreground prototype specializing on BG).
    """

    def __init__(self, *, bg_class: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.bg_class = int(bg_class)
        self.temperature = float(temperature)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        labels: torch.Tensor,
        pseudo_patch_labels: torch.Tensor,
    ) -> LossTermOutput:
        """Penalize background patches that assign high mass to a single prototype."""

        bg_mask = (pseudo_patch_labels == self.bg_class).reshape(labels.size(0), -1)
        if not bg_mask.any():
            zero = outputs["class_logits"].new_zeros(())
            return LossTermOutput(value=zero, unscaled=zero.detach())

        patch_logits = outputs["patch_prototype_logits"]
        class_logits = patch_logits[
            torch.arange(labels.size(0), device=labels.device),
            :,
            labels,
            :,
        ]
        log_resp = F.log_softmax(class_logits / self.temperature, dim=-1)
        resp = log_resp.exp()
        entropy = -(resp * log_resp).sum(dim=-1)
        entropy_bg = entropy[bg_mask]
        # Maximize entropy => minimize negative entropy.
        loss = -entropy_bg.mean()
        return LossTermOutput(value=loss, unscaled=loss.detach())


class LogDetVMFDiversity(nn.Module):
    """vMF-aware prototype diversity penalty via log-det of an overlap kernel."""

    def __init__(self, *, eps: float = 1.0e-4, include_background: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.include_background = bool(include_background)

    def forward(
        self,
        *,
        prototypes: torch.Tensor,
        kappa_values: torch.Tensor,
    ) -> LossTermOutput:
        """Encourage diverse prototypes by penalizing overlap in vMF space."""

        # Only meaningful as a training signal when μ is learnable.
        if not prototypes.requires_grad:
            zero = prototypes.new_zeros((), dtype=torch.float32)
            return LossTermOutput(value=zero, unscaled=zero.detach())

        dim = int(prototypes.shape[-1])
        penalty = prototype_logdet_vmf_overlap(
            prototypes,
            kappa_values,
            dim=dim,
            eps=self.eps,
            include_background=self.include_background,
        )
        return LossTermOutput(value=penalty, unscaled=penalty.detach())


class VMFMPCriterion(nn.Module):
    """Aggregate criterion used by the training loop."""

    def __init__(
        self,
        *,
        num_classes: int,
        n_prototypes: int,
        bg_class_weight: float,
        l_ppd_coef: float,
        l_ppd_temp: float,
        l_resp_purity_coef: float,
        l_resp_purity_temp: float,
        l_resp_purity_start_epoch: int,
        l_logdet_vmf_coef: float,
        l_logdet_eps: float,
        l_logdet_include_bg: bool,
    ) -> None:
        super().__init__()

        self.current_epoch = 0
        self.current_step: int | None = None

        self.bg_class = int(num_classes)
        self.C = int(num_classes)
        self.K = int(n_prototypes)

        self.l_ppd_coef = float(l_ppd_coef)
        self.l_resp_purity_coef = float(l_resp_purity_coef)
        self.l_resp_purity_start_epoch = int(l_resp_purity_start_epoch)
        self.l_logdet_vmf_coef = float(l_logdet_vmf_coef)

        self.loss_y = ClassificationCrossEntropy()

        class_weights = torch.tensor(
            [1.0] * (self.C * self.K) + [float(bg_class_weight)] * self.K,
            dtype=torch.float32,
        )
        self.loss_ppd = PatchPrototypeDistillation(
            class_weight=class_weights, temperature=float(l_ppd_temp)
        )
        self.loss_resp_purity = ResponsibilityPurity(
            bg_class=self.bg_class, temperature=float(l_resp_purity_temp)
        )
        self.loss_logdet_vmf = LogDetVMFDiversity(
            eps=float(l_logdet_eps), include_background=bool(l_logdet_include_bg)
        )

    def set_epoch(self, epoch: int, step: int | None = None) -> None:
        """Update internal epoch/step state used for loss scheduling."""

        self.current_epoch = int(epoch)
        self.current_step = None if step is None else int(step)

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]):
        """Compute enabled loss terms and return a flat scalar loss dict."""

        labels = batch[1]
        loss_dict: dict[str, torch.Tensor] = {}

        l_y = self.loss_y(outputs, labels)
        loss_dict["l_y"] = l_y.value

        if self.l_ppd_coef != 0 and "part_assignment_maps" in outputs:
            l_ppd = self.loss_ppd(outputs, assignments=outputs["part_assignment_maps"])
            loss_dict["l_ppd"] = self.l_ppd_coef * l_ppd.value
            loss_dict["_l_ppd_unadjusted"] = l_ppd.unscaled if l_ppd.unscaled is not None else l_ppd.value.detach()

        if (
            self.l_resp_purity_coef != 0
            and self.current_epoch >= self.l_resp_purity_start_epoch
            and outputs.get("pseudo_patch_labels") is not None
        ):
            l_resp = self.loss_resp_purity(
                outputs,
                labels=labels,
                pseudo_patch_labels=outputs["pseudo_patch_labels"],
            )
            loss_dict["l_resp_purity"] = self.l_resp_purity_coef * l_resp.value
            loss_dict["_l_resp_purity_unadjusted"] = (
                l_resp.unscaled if l_resp.unscaled is not None else l_resp.value.detach()
            )

        if self.l_logdet_vmf_coef != 0:
            prototypes = outputs.get("prototype_mu")
            kappa_values = outputs.get("kappa_values")
            if (
                prototypes is not None
                and kappa_values is not None
                and bool(getattr(prototypes, "requires_grad", False))
            ):
                l_logdet = self.loss_logdet_vmf(
                    prototypes=prototypes, kappa_values=kappa_values
                )
                loss_dict["l_logdet_vmf"] = self.l_logdet_vmf_coef * l_logdet.value
                loss_dict["_l_logdet_vmf_unadjusted"] = (
                    l_logdet.unscaled
                    if l_logdet.unscaled is not None
                    else l_logdet.value.detach()
                )

        return loss_dict


__all__ = [
    "ClassificationCrossEntropy",
    "LogDetVMFDiversity",
    "LossTermOutput",
    "PatchPrototypeDistillation",
    "ResponsibilityPurity",
    "VMFMPCriterion",
]
