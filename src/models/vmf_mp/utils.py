"""vMF-MP helper utilities.

This module groups together a few utility functions used by the model:
- backbone block expansion helpers (for expanded DINO backbones)
- OT/Sinkhorn transport used during online clustering
- small math utilities (EMA-style updates, prototype diversity kernel)

The functions here are intentionally framework-light and reusable.
"""

from __future__ import annotations

import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def block_expansion_dino(
    state_dict: dict[str, torch.Tensor],
    n_splits: int = 3,
    freeze_layer_norm: bool = True,
):
    """Expand a ViT state dict via Block Expansion.

    This implements the idea of "Block Expansion" (see https://arxiv.org/abs/2404.17245):
    blocks are duplicated to create a deeper backbone, and newly inserted blocks
    get their final projections (MLP fc2 + attn proj) zero-initialized so the
    expanded model initially behaves like the original network.

    Returns:
        expanded_state_dict: remapped/expanded weights.
        new_depth: number of transformer blocks after expansion.
        learnable_param_names: parameter names intended to be fine-tuned.
        zero_param_names: parameter names that were zero-initialized.
    """
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)

    block_indices = np.arange(0, n_blocks).reshape((n_splits, -1,))
    block_indices = np.concatenate([block_indices, block_indices[:, -1:]], axis=-1)

    n_splits, n_block_per_split = block_indices.shape
    new_block_indices = list((i + 1) * n_block_per_split - 1 for i in range(n_splits))

    expanded_state_dict = dict()
    learnable_param_names, zero_param_names = [], []

    for dst_idx, src_idx in enumerate(block_indices.flatten()):
        src_keys = [k for k in state_dict if f"blocks.{src_idx}" in k]
        dst_keys = [k.replace(f"blocks.{src_idx}", f"blocks.{dst_idx}") for k in src_keys]

        block_state_dict = dict()

        for src_k, dst_k in zip(src_keys, dst_keys):
            if ("mlp.fc2" in dst_k or "attn.proj" in dst_k) and (dst_idx in new_block_indices):
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
                zero_param_names.append(dst_k)
            else:
                block_state_dict[dst_k] = state_dict[src_k]

        expanded_state_dict.update(block_state_dict)

        if dst_idx in new_block_indices:
            learnable_param_names += dst_keys

    expanded_state_dict.update({k: v for k, v in state_dict.items() if "block" not in k})

    if not freeze_layer_norm:
        learnable_param_names += ["norm.weight", "norm.bias"]

    return expanded_state_dict, len(block_indices.flatten()), learnable_param_names, zero_param_names


def append_blocks(
    state_dict: dict[str, torch.Tensor],
    n_splits: int = 1,
    freeze_layer_norm: bool = True,
):
    """Append extra transformer blocks to a ViT state dict.

    New blocks are initialized by copying the last existing block, then
    zero-initializing the final projections (MLP fc2 + attn proj) so the
    appended blocks act as near-identities at init.
    """
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)

    src_block_idx = n_blocks - 1
    src_keys = [k for k in state_dict if f"blocks.{src_block_idx}" in k]  # keys of parameters to copy from

    expanded_state_dict = deepcopy(state_dict)
    learnable_param_names, zero_param_names = [], []
    for i in range(n_splits):
        dst_block_idx = n_blocks + i
        dst_keys = [k.replace(f"blocks.{src_block_idx}", f"blocks.{dst_block_idx}") for k in src_keys]

        block_state_dict = dict()
        for src_k, dst_k in zip(src_keys, dst_keys):
            if "mlp.fc2" in dst_k or "attn.proj" in dst_k:
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
                zero_param_names.append(dst_k)
            else:
                block_state_dict[dst_k] = state_dict[src_k]
        expanded_state_dict.update(block_state_dict)
        learnable_param_names += dst_keys

    if not freeze_layer_norm:
        learnable_param_names += ["norm.weight", "norm.bias"]

    return expanded_state_dict, n_blocks + n_splits, learnable_param_names, zero_param_names


#
# The following functions are adapted from https://github.com/tfzhou/ProtoSeg
#

def momentum_update(old_value, new_value, momentum, debug=False):
    """Exponential moving average (EMA) update used for non-parametric prototypes."""

    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def sinkhorn_transport(
    scores: torch.Tensor,
    *,
    row_marginals: torch.Tensor | None = None,
    col_marginals: torch.Tensor | None = None,
    n_iters: int = 50,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Compute an entropic OT plan with prescribed row/column marginals.

    Args:
        scores: Tensor of shape [N, K] with higher = more similar.
        row_marginals: Optional tensor of shape [N] whose sum is total mass.
        col_marginals: Optional tensor of shape [K] whose sum is total mass.
        n_iters: Number of Sinkhorn iterations.
        epsilon: Entropic regularization strength.

    Returns:
        Transport plan Q of shape [N, K] with the requested marginals.
    """

    if scores.dim() != 2:
        raise ValueError(f"scores must be 2D (N, K); got {scores.shape}")

    n_rows, n_cols = scores.shape
    device = scores.device
    dtype = scores.dtype

    row_marginals = (
        row_marginals
        if row_marginals is not None
        else torch.ones(n_rows, device=device, dtype=dtype)
    )
    col_marginals = (
        col_marginals
        if col_marginals is not None
        else torch.full(
            (n_cols,), row_marginals.sum() / max(n_cols, 1), device=device, dtype=dtype
        )
    )

    total_mass = row_marginals.sum().clamp_min(1.0e-6)
    col_marginals = col_marginals * (total_mass / col_marginals.sum().clamp_min(1.0e-6))

    log_r = torch.log(row_marginals.clamp_min(1.0e-6))
    log_c = torch.log(col_marginals.clamp_min(1.0e-6))

    log_K = scores / epsilon
    log_K = log_K - log_K.max()

    u = torch.zeros(n_rows, device=device, dtype=dtype)
    v = torch.zeros(n_cols, device=device, dtype=dtype)
    for _ in range(int(n_iters)):
        u = log_r - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_c - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

    log_Q = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    return torch.exp(log_Q)


def prototype_logdet_vmf_overlap(
    prototypes: torch.Tensor,
    kappa: torch.Tensor,
    *,
    dim: int,
    eps: float = 1.0e-4,
    include_background: bool = False,
) -> torch.Tensor:
    """Return a vMF-aware per-class prototype diversity penalty via L2 overlaps.

    Instead of building a cosine Gram matrix over mean directions, we build a
    *distributional* similarity kernel using the closed-form overlap between
    two von Mises–Fisher components on S^{dim-1}:

        p_i(t) = C_d(κ_i) · exp(κ_i μ̂_iᵀ t),   with ||μ̂_i|| = 1

    The L2 inner product (overlap) between two components has the form:

        S_ij = ∫ p_i(t) p_j(t) dt
            = C_d(κ_i) C_d(κ_j) / C_d(|| κ_i μ̂_i + κ_j μ̂_j ||)

    We then normalize to a correlation-like kernel:

        K_ij = S_ij / sqrt(S_ii S_jj)  (so K_ii = 1)

    and penalize collapse by maximizing log det(K + eps I) per class:

        L = - mean_c log det(K_c + eps I)

    Notes:
      - This kernel is PSD (it's a Gram matrix in L2), so the logdet story is
        mathematically coherent.
      - The kernel depends on both μ and κ, so it can discourage "broad"
        prototypes from overlapping heavily with others.

    Args:
        prototypes: Tensor shaped (C, K, D).
        kappa: Tensor shaped (C, K) with κ >= 0 (e.g., softplus(kappa_raw)).
        dim: Token dimension D (vMF on S^{D-1}).
        eps: Diagonal jitter for numerical stability.
        include_background: Whether to include the last class in the average.

    Returns:
        Scalar penalty (float32) suitable for adding to a training loss.
    """

    if prototypes.dim() != 3:
        raise ValueError(f"prototypes must be (C, K, D); got {prototypes.shape}")
    if kappa.dim() != 2:
        raise ValueError(f"kappa must be (C, K); got {kappa.shape}")
    if prototypes.size(0) != kappa.size(0) or prototypes.size(1) != kappa.size(1):
        raise ValueError(
            "prototypes and kappa must agree on (C, K); "
            f"got {prototypes.shape[:2]} vs {kappa.shape}"
        )
    if int(dim) <= 0:
        raise ValueError("dim must be a positive integer.")

    protos = prototypes
    kappas = kappa
    if (not include_background) and protos.size(0) > 1:
        protos = protos[:-1]
        kappas = kappas[:-1]

    num_classes, num_prototypes, _ = protos.shape
    if num_classes == 0 or num_prototypes <= 1:
        return prototypes.new_zeros((), dtype=torch.float32)

    from .vmf_normalizer import logCp_vmf  # local import to keep utils lightweight

    mu = F.normalize(protos.to(dtype=torch.float32), p=2, dim=-1)  # (C, K, D)
    k = kappas.to(dtype=torch.float32).clamp_min(0.0)  # (C, K)

    # Cosine similarities between mean directions.
    cos = torch.matmul(mu, mu.transpose(-1, -2))  # (C, K, K)

    # || κ_i μ_i + κ_j μ_j || = sqrt(κ_i^2 + κ_j^2 + 2 κ_i κ_j cos(μ_i, μ_j))
    ki = k.unsqueeze(-1)
    kj = k.unsqueeze(-2)
    radicand = ki * ki + kj * kj + 2.0 * ki * kj * cos
    v = torch.sqrt(radicand.clamp_min(0.0))

    logC_k = logCp_vmf(int(dim), k)  # (C, K) centered log normalizer
    logC_v = logCp_vmf(int(dim), v)  # (C, K, K)

    logS = logC_k.unsqueeze(-1) + logC_k.unsqueeze(-2) - logC_v  # (C, K, K)
    logS_diag = torch.diagonal(logS, dim1=-2, dim2=-1)  # (C, K)
    logK = logS - 0.5 * (logS_diag.unsqueeze(-1) + logS_diag.unsqueeze(-2))
    kernel = torch.exp(logK).to(dtype=torch.float32)  # (C, K, K), kernel_ii = 1
    kernel = 0.5 * (kernel + kernel.transpose(-1, -2))

    eye = (
        torch.eye(num_prototypes, device=kernel.device, dtype=kernel.dtype)
        .expand_as(kernel)
        .contiguous()
    )
    kernel = kernel + float(eps) * eye

    chol, info = torch.linalg.cholesky_ex(kernel)
    diag = torch.diagonal(chol, dim1=-2, dim2=-1).clamp_min(1.0e-12)
    logdet = 2.0 * torch.log(diag).sum(dim=-1)

    if info.any():
        _, logabsdet = torch.linalg.slogdet(kernel)
        logdet = torch.where(info == 0, logdet, logabsdet)

    return (-logdet.mean()).to(dtype=torch.float32)
