"""vMF normalizing constant utilities.

The vMF-MP model needs the normalizer term to make the story coherent:

    log p(t | μ, κ) = log C_d(κ) + κ μᵀt

Without log C_d(κ), κ can behave like an arbitrary logit scale.
"""

from __future__ import annotations

import math
from typing import Literal

import torch


def logCp_vmf(
    p: int | float,
    kappa: torch.Tensor,
    *,
    method: Literal["amos"] = "amos",
    center: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return log C_p(kappa) for the vMF on S^{p-1}.

    Definition:
        C_p(kappa) = kappa^nu / ((2*pi)^(p/2) * I_nu(kappa)),
        nu = p/2 - 1.

    Notes:
      - Pure torch -> GPU-friendly and differentiable.
      - Uses a uniform asymptotic approximation (Amos / Debye-style) for
        log I_nu(kappa) because torch does not provide I_nu for general nu.
      - Handles kappa == 0 via the exact limit:
            C_p(0) = 1 / Area(S^{p-1})
            Area(S^{p-1}) = 2 * pi^(p/2) / Gamma(p/2)
      - By default returns a centered value:
            log C_p(kappa) - log C_p(0),
        which is a numerically stable log-density ratio to the uniform
        distribution on S^{p-1}. The subtraction is a constant in p and
        cancels in classification.
    """

    p = float(p)
    if p <= 0:
        raise ValueError("p must be > 0.")
    if not torch.is_tensor(kappa):
        kappa = torch.as_tensor(kappa)
    if (kappa < 0).any():
        raise ValueError("kappa must be >= 0 for the standard vMF.")

    if method != "amos":
        raise ValueError(f"Unknown method: {method}")

    # Compute in float32 for stability (especially important under autocast).
    dtype = (
        torch.float32 if kappa.dtype in {torch.float16, torch.bfloat16} else kappa.dtype
    )
    kappa_f = kappa.to(dtype=dtype)

    device = kappa_f.device
    half_p = 0.5 * p
    nu = half_p - 1.0

    # Exact kappa==0 limit:
    # log C_p(0) = log Gamma(p/2) - log 2 - (p/2) log pi
    logC0 = (
        torch.lgamma(torch.tensor(half_p, device=device, dtype=dtype))
        - math.log(2.0)
        - half_p * math.log(math.pi)
    )

    # Amos-style uniform approximation for log I_nu(kappa):
    # I_nu(k) ≈ exp(xi) / sqrt(2π xi) * (k / (nu + xi))^nu,   xi = sqrt(nu^2 + k^2)
    # => log C_p(k) = nu log k - (p/2)log(2π) - log I_nu(k)
    #              ≈ -(p/2)log(2π) - xi + nu log(nu+xi) + 0.5 log(2π xi)
    nu_t = torch.tensor(nu, device=device, dtype=dtype)
    xi = torch.sqrt(nu_t * nu_t + kappa_f * kappa_f)
    log_two_pi = math.log(2.0 * math.pi)
    logC_amos = (
        -half_p * log_two_pi
        - xi
        + nu_t * torch.log((nu_t + xi).clamp_min(eps))
        + 0.5 * (log_two_pi + torch.log(xi.clamp_min(eps)))
    )

    # Use the exact limit at kappa==0 to avoid log(0) and keep behavior correct.
    is_zero = kappa_f == 0
    out = torch.where(is_zero, logC0, logC_amos)
    if center:
        out = out - logC0
    return out.to(dtype=kappa.dtype)
