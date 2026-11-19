"""Prototype class selectivity metrics (label-based, dataset-agnostic)."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _auc_from_scores(scores: torch.Tensor, pos_mask: torch.Tensor) -> float:
    """Compute AUC via rank statistics (Mann–Whitney U)."""

    scores = scores.to(dtype=torch.float32)
    pos_mask = pos_mask.to(dtype=torch.bool)
    n_pos = int(pos_mask.sum().item())
    n_total = int(scores.numel())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, n_total + 1, device=scores.device, dtype=torch.float32)
    rank_sum_pos = ranks[pos_mask].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


@torch.no_grad()
def evaluate_pcs_auc(
    net: nn.Module,
    dataloader: DataLoader,
    *,
    topk: int = 1,
    num_classes: int,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute Prototype Class Selectivity (PCS) as mean AUC across prototypes.

    Response for each prototype is defined as the mean of the top-K activations
    in its map. K=1 reduces to the peak response.
    """

    net.to(device)
    net.eval()

    responses: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for batch in dataloader:
        batch = tuple(item.to(device=device) for item in batch)
        images, targets = batch[:2]
        _, all_maps = net.push_forward(images)
        if all_maps is None:
            raise ValueError("Model push_forward() returned no maps.")

        batch_size, ck, h, w = all_maps.shape
        if ck % num_classes != 0:
            raise ValueError(
                f"Expected channel count divisible by num_classes ({num_classes}), got {ck}."
            )
        k_proto = ck // num_classes
        topk_eff = min(int(topk), h * w)

        maps = all_maps.reshape(batch_size, num_classes, k_proto, -1)
        topk_vals = torch.topk(maps, k=topk_eff, dim=-1).values
        resp = topk_vals.mean(dim=-1)  # (B, C, K)

        responses.append(resp.cpu())
        labels.append(targets.cpu())

    if not responses:
        return float("nan")

    resp_all = torch.cat(responses, dim=0)  # (N, C, K)
    labels_all = torch.cat(labels, dim=0).to(dtype=torch.long)

    auc_values: list[float] = []
    for c in range(num_classes):
        pos_mask = labels_all == c
        if pos_mask.sum() == 0 or pos_mask.sum() == labels_all.numel():
            continue
        for k in range(resp_all.shape[2]):
            scores = resp_all[:, c, k]
            auc_values.append(_auc_from_scores(scores, pos_mask))

    if not auc_values:
        return float("nan")
    return float(torch.tensor(auc_values, dtype=torch.float32).mean().item())


__all__ = ["evaluate_pcs_auc"]
