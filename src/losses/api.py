"""Loss builder API for vMF-MP."""

from __future__ import annotations

from typing import Any

from torch import nn


def build_loss(cfg: Any) -> nn.Module:
    """Build the vMF-MP criterion from config."""

    from .vmf_mp import VMFMPCriterion

    loss_cfg = cfg.train.loss
    return VMFMPCriterion(
        l_ppd_coef=float(loss_cfg.l_ppd_coef),
        l_ppd_temp=float(loss_cfg.l_ppd_temp),
        l_resp_purity_coef=float(getattr(loss_cfg, "l_resp_purity_coef", 0.0)),
        l_resp_purity_temp=float(getattr(loss_cfg, "l_resp_purity_temp", 1.0)),
        l_resp_purity_start_epoch=int(
            getattr(loss_cfg, "l_resp_purity_start_epoch", 0)
        ),
        l_logdet_vmf_coef=float(getattr(loss_cfg, "l_logdet_vmf_coef", 0.0)),
        l_logdet_eps=float(getattr(loss_cfg, "l_logdet_eps", 1.0e-4)),
        l_logdet_include_bg=bool(getattr(loss_cfg, "l_logdet_include_bg", False)),
        num_classes=int(cfg.dataset.num_classes),
        n_prototypes=int(cfg.model.num_prototypes),
        bg_class_weight=float(loss_cfg.bg_class_weight),
    )
