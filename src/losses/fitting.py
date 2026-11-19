"""Four fitting auxiliaries; no part annotations are used."""

import math
import torch
from torch.nn import functional as F


def spatial_features(features, strength=0.85):
    side = math.isqrt(features.shape[-2])
    if side * side != features.shape[-2] or not 0 <= strength <= 1:
        raise ValueError("Expected square patch grids and a blend in [0,1]")
    z = F.normalize(features, dim=-1)
    grid = z.transpose(-1, -2).reshape(-1, z.shape[-1], side, side)
    mean = (
        F.avg_pool2d(grid, 3, stride=1, padding=1, count_include_pad=False)
        .flatten(-2)
        .transpose(-1, -2)
        .reshape(z.shape)
    )
    return F.normalize((1 - strength) * z + strength * mean, dim=-1)


def feature_anchor(features, reference, foreground):
    cosine = (
        F.normalize(features.double(), dim=-1)
        * F.normalize(reference.detach().double(), dim=-1)
    ).sum(-1)
    return (1 - cosine.clamp(-1, 1))[foreground.bool()].mean()


def prediction_kl(logits, reference_probabilities):
    q = reference_probabilities.detach().double()
    return (q * (q.clamp_min(1e-300).log() - logits.log_softmax(-1))).sum(-1).mean()


def peak_loss(
    features, means, kappa, reference_features, reference_means, presence, margin=0.02
):
    """Fixed reference peak on BOTH views, averaged per image/present prototype.

    For the separate noisy-retention loss, pass detached CURRENT clean features
    and CURRENT directions instead of the fixed discovery reference.
    """
    if not 0 <= margin < 1:
        raise ValueError("Cosine margin must lie in [0,1)")
    current = (F.normalize(features.double(), dim=-1) @ means.T).transpose(1, 2)
    with torch.no_grad():
        target = (
            (
                F.normalize(reference_features.double(), dim=-1)
                @ reference_means.detach().T
            )
            .transpose(1, 2)
            .argmax(-1)
        )
    image, proto = presence.nonzero(as_tuple=True)
    if not len(image):
        return current.sum() * 0
    scores, positions = current[image, proto], target[image, proto]
    at = scores.gather(1, positions[:, None]).squeeze(1)
    rivals = (scores + margin).scatter(1, positions[:, None], at[:, None])
    k = kappa[proto]
    penalty = 20 * (torch.logsumexp(k[:, None] * rivals, 1) / k - at)
    return (penalty / presence.sum(-1)[image].clamp_min(1)).sum() / len(features)


def prediction_weight(
    epoch, discovery_until=12, epochs=25, initial=0.5, final=0.1, hold=6
):
    if not 0 <= final <= initial or epochs <= discovery_until:
        raise ValueError("Invalid KL schedule")
    hold = min(hold, epochs - discovery_until - 1)
    index, total = epoch - discovery_until - 1, epochs - discovery_until
    progress = min(1.0, max(0.0, (index - hold) / max(1, total - hold - 1)))
    return final + 0.5 * (initial - final) * (1 + math.cos(math.pi * progress))
