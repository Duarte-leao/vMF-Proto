"""Exact moment fitting and stable exponential evidence.

The statistical concentration solve is exact; scoring uses the centred Amos
normalizer implemented by logCp_vmf.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from scipy.special import gammaln, ive
from torch.nn import functional as F

from .vmf_normalizer import logCp_vmf


def exact_vmf_terms(dim, kappa):
    k = np.asarray(kappa, dtype=np.float64)
    if dim < 3 or np.any(~np.isfinite(k)) or np.any(k < 0):
        raise ValueError("Require D>=3 and finite nonnegative concentrations")
    flat, nu = k.reshape(-1), dim / 2 - 1
    logc, align = np.zeros_like(flat), np.zeros_like(flat)
    positive = flat > 0
    kp = flat[positive]
    base, nxt = ive(nu, kp), ive(nu + 1, kp)
    good = (base > 0) & (nxt > 0) & np.isfinite(base) & np.isfinite(nxt)
    lc, a = np.zeros_like(kp), np.zeros_like(kp)
    lc[good] = (
        nu * np.log(kp[good] / 2) - gammaln(nu + 1) - np.log(base[good]) - kp[good]
    )
    a[good] = nxt[good] / base[good]
    for index in np.flatnonzero(~good):
        value, totals = kp[index], []
        for b in (dim / 2, dim / 2 + 1):
            term = total = 0.0
            for m in range(1, 10000):
                term += 2 * math.log(value / 2) - math.log(m) - math.log(b + m - 1)
                total = np.logaddexp(total, term)
                if term < total - 40 and m > value:
                    break
            else:
                raise ArithmeticError("Bessel series failed to converge")
            totals.append(total)
        lc[index] = -totals[0]
        a[index] = value / dim * math.exp(totals[1] - totals[0])
    logc[positive], align[positive] = lc, a
    return logc.reshape(k.shape), align.reshape(k.shape)


def exact_concentration(dim, resultant, upper=100000.0):
    target = np.asarray(resultant, dtype=np.float64)
    if np.any(~np.isfinite(target)) or np.any(target < 0) or np.any(target > 1 + 1e-10):
        raise ValueError("Invalid resultant lengths")
    target = np.clip(target, 0, 1)
    maximum = exact_vmf_terms(dim, np.full_like(target, upper))[1]
    clipped = target >= maximum
    fitted_target = np.minimum(target, maximum)
    lo, hi = np.zeros_like(target), np.full_like(target, upper)
    k = np.minimum(target * (dim - target**2) / np.maximum(1 - target**2, 1e-16), upper)
    for _ in range(40):
        _, a = exact_vmf_terms(dim, k)
        error = a - fitted_target
        if np.max(np.abs(error), initial=0.0) < 1e-11:
            break
        lo, hi = np.where(error < 0, k, lo), np.where(error >= 0, k, hi)
        derivative = np.where(
            k > 1e-8, 1 - a * a - (dim - 1) * a / np.maximum(k, 1e-30), 1 / dim
        )
        candidate = k - error / np.maximum(derivative, 1e-30)
        k = np.where((candidate > lo) & (candidate < hi), candidate, 0.5 * (lo + hi))
    k = np.where(target == 0, 0.0, np.where(clipped, upper, k))
    np.testing.assert_allclose(
        exact_vmf_terms(dim, k)[1], fitted_target, atol=2e-10, rtol=0
    )
    return k, clipped


class ExactConcentration(torch.autograd.Function):
    """Implicit derivative through A_D(kappa)=resultant, zero at the cap."""

    @staticmethod
    def forward(ctx, resultant, dim, upper=100000.0):
        k, clipped = exact_concentration(dim, resultant.detach().cpu().numpy(), upper)
        a = torch.as_tensor(
            exact_vmf_terms(dim, k)[1], device=resultant.device, dtype=resultant.dtype
        )
        k = torch.as_tensor(k, device=resultant.device, dtype=resultant.dtype)
        clipped = torch.as_tensor(clipped, device=resultant.device)
        derivative = torch.where(
            k > 1e-8,
            1 - a.square() - (dim - 1) * a / k.clamp_min(1e-30),
            torch.full_like(k, 1 / dim),
        )
        if ((derivative <= 0) & ~clipped).any():
            raise FloatingPointError("Invalid exact vMF derivative")
        ctx.save_for_backward(
            torch.where(
                clipped, torch.zeros_like(k), derivative.clamp_min(1e-30).reciprocal()
            )
        )
        return k

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.saved_tensors[0], None, None


def effective_bank(free, moment, mass, fraction, minimum_images=5):
    if not 0 <= fraction <= 1:
        raise ValueError("Fit fraction must lie in [0,1]")
    if fraction == 0:
        return dict(free)
    fit_concentration = free.get("kappa_learnable", True)
    valid = (mass >= minimum_images - 1e-8) & (moment.norm(dim=-1) > 1e-10)
    safe = torch.where(valid[..., None], moment, free["means"].detach() * 0.01)
    k = (
        ExactConcentration.apply(safe.norm(dim=-1), moment.shape[-1], 100000.0)
        if fit_concentration
        else free["kappa"].detach()
    )
    mu = F.normalize(safe, dim=-1)
    fallback_mu = free["means"].detach() if fraction == 1 else free["means"]
    fallback_k = free["kappa"].detach() if fraction == 1 else free["kappa"]
    mu, k = (
        torch.where(valid[..., None], mu, fallback_mu),
        torch.where(valid, k, fallback_k),
    )
    if fraction < 1:
        eta = (1 - fraction) * free["kappa"][..., None] * free["means"] + fraction * k[
            ..., None
        ] * mu
        k = eta.norm(dim=-1)
        mu = torch.where(
            (k > 1e-12)[..., None], eta / k.clamp_min(1e-12)[..., None], free["means"]
        )
    if not fit_concentration:
        # Normalizing the interpolated directions must not shrink fixed kappa.
        k = free["kappa"].detach()
    return dict(
        free, means=mu, kappa=k, logc=logCp_vmf(moment.shape[-1], k), supported=valid
    )


def raw_scores(features, bank):
    z = F.normalize(features.double(), dim=-1)
    return (z @ bank["means"].T) * bank["kappa"] + bank["logc"]


def log_class_evidence(raw, foreground, pi, *, shared=False):
    """Log mean_FG[1+pi_w*(exp(score_w)-1)], winner chosen BEFORE pi.

    Taking the log preserves predictions, not softmax calibration. Discovery
    trains CE(A); fitting trains CE(g*log A).
    """
    if not foreground.any(-1).all() or (pi <= 0).any() or not torch.isfinite(pi).all():
        raise ValueError(
            "Nonempty foreground and finite positive mixture weights required"
        )
    if shared:
        maximum, winner = raw.double().max(-1)
        selected = pi.double().T[winner]
    else:
        maximum, winner = raw.double().max(-1)
        selected = (
            pi.double()[None]
            .expand(len(raw), -1, -1)
            .gather(2, winner.transpose(1, 2))
            .transpose(1, 2)
        )
    if shared:
        maximum = maximum[..., None]
    # Clamping only the baseline handles the exactly-one weight of a K=1
    # control without log1p(-1) backward NaNs. Real mixtures remain unchanged.
    baseline = (1 - selected).clamp_min(torch.finfo(torch.float64).tiny).log()
    patch = torch.logaddexp(baseline, selected.log() + maximum)
    return (
        torch.logsumexp(patch.masked_fill(~foreground[..., None], -torch.inf), 1)
        - foreground.sum(-1).double().log()[:, None]
    )


@torch.no_grad()
def selected_weights(cosine, ids, components, importance=80.0, floor=0.01):
    """One unit per image/present prototype; every assigned patch stays positive."""
    if not 0 < floor <= 1 or importance < 0 or cosine.shape != ids.shape:
        raise ValueError("Invalid image-balanced fitting weights")
    count = cosine.new_zeros(len(ids), components).scatter_add(
        1, ids, torch.ones_like(cosine)
    )
    value = importance * cosine.double()
    maximum = value.new_full((len(ids), components), -torch.inf).scatter_reduce(
        1, ids, value, reduce="amax", include_self=True
    )
    exponential = (value - maximum.gather(1, ids)).exp()
    denominator = value.new_zeros(len(ids), components).scatter_add(1, ids, exponential)
    return floor / count.gather(1, ids) + (
        1 - floor
    ) * exponential / denominator.gather(1, ids)


def moment_sum(features, ids, weights, components):
    z = F.normalize(features.double(), dim=-1)
    return z.new_zeros(components, z.shape[-1]).index_add(
        0, ids.flatten(), (z * weights[..., None]).flatten(0, 1)
    )


def support_vjp(features, ids, weights, mass, gradient):
    z = F.normalize(features.double(), dim=-1)
    return (
        z * weights[..., None] * (gradient / mass.clamp_min(1e-30)[:, None])[ids]
    ).sum()


def fit_fraction(epoch, step, steps, discovery_until=12, fitted_by=13):
    if not 1 <= discovery_until < fitted_by or not 1 <= step <= steps:
        raise ValueError("Invalid fitting schedule")
    progress = min(
        1.0,
        max(
            0.0,
            ((epoch - discovery_until - 1) * steps + step)
            / ((fitted_by - discovery_until) * steps),
        ),
    )
    return 0.5 - 0.5 * math.cos(math.pi * progress)
