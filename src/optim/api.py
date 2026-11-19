"""Optimizer and scheduler construction for vMF-MP."""

from __future__ import annotations

from typing import Any, Tuple


def build_optimizer_and_scheduler(cfg: Any, model: Any) -> Tuple[Any, Any]:
    """Build the stage-2 optimizer and (optional) scheduler.

    Stage-1 prototype updates are handled inside the model and do not require
    a gradient-based optimizer.
    """

    import torch

    weight_decay = float(getattr(cfg.train, "weight_decay", 0.0))

    param_groups = []

    backbone_params = list(model.backbone.learnable_parameters())
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(cfg.train.backbone_lr),
                "weight_decay": weight_decay,
            }
        )

    kappa_param = getattr(model, "kappa_raw", None)
    if isinstance(kappa_param, torch.nn.Parameter) and kappa_param.requires_grad:
        param_groups.append(
            {
                "params": [kappa_param],
                "lr": float(getattr(cfg.train, "kappa_lr", cfg.train.classifier_lr)),
                "weight_decay": 0.0,
            }
        )

    pi_param = getattr(model, "pi_logits", None)
    if isinstance(pi_param, torch.nn.Parameter) and pi_param.requires_grad:
        param_groups.append(
            {
                "params": [pi_param],
                "lr": float(getattr(cfg.train, "pi_lr", cfg.train.classifier_lr)),
                "weight_decay": 0.0,
            }
        )

    if getattr(model, "parametric_prototypes", False):
        param_groups.append(
            {
                "params": [model.prototype_mu],
                "lr": float(cfg.train.proto_lr),
                "weight_decay": 0.0,
            }
        )

    optimizer = torch.optim.Adam(param_groups)
    scheduler = None
    return optimizer, scheduler
