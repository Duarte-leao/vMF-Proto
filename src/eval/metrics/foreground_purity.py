"""Foreground purity metrics for prototype maps (CUB segmentation required)."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader

from .comprehensiveness import CUBEvalDataset
from .cub_metadata import mean, std


@torch.no_grad()
def evaluate_pfr_topk(
    net: nn.Module,
    data_root: str,
    *,
    topk: int = 1,
    num_classes: int = 200,
    device: torch.device = torch.device("cpu"),
    input_size: tuple[int, int] = (224, 224),
    segmentations_dir: str = "segmentations",
) -> float:
    """Compute Peak-Foreground Rate (PFR) with a strict top-K criterion.

    For each image and each prototype of the ground-truth class:
      - take the top-K activation locations in the prototype map,
      - score 1 if *all* top-K locations fall inside the GT foreground mask,
        else 0.

    The final score is the mean over all (image, prototype) pairs.
    """

    normalize = T.Normalize(mean=mean, std=std)
    transform = T.Compose([T.Resize(input_size), T.ToTensor(), normalize])
    eval_dataset = CUBEvalDataset(
        root=data_root,
        input_size=input_size[0],
        transform=transform,
        segmentations_dir=segmentations_dir,
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset, batch_size=64, num_workers=2, shuffle=False
    )

    net.to(device)
    net.eval()

    total_hits = 0.0
    total_count = 0.0

    for batch in eval_dataloader:
        images, targets, seg_map_cropped, _ = tuple(
            item.to(device=device) for item in batch
        )
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

        proto_indices = (targets * k_proto).unsqueeze(dim=-1).repeat(1, k_proto)
        proto_indices += torch.arange(k_proto, device=device)
        proto_indices = proto_indices[:, :, None, None].repeat(1, 1, h, w)
        gt_maps = torch.gather(all_maps, 1, proto_indices)  # (B, K, H, W)

        # Downsample the GT mask to the token grid using "any-pixel" pooling:
        # a patch is considered foreground if *any* pixel within its corresponding
        # image region is foreground. This is implemented as adaptive max pooling.
        seg_down = F.adaptive_max_pool2d(seg_map_cropped.float(), output_size=(h, w)).squeeze(1)
        seg_flat = seg_down.reshape(batch_size, -1)

        maps_flat = gt_maps.reshape(batch_size, k_proto, -1)
        topk_idx = torch.topk(maps_flat, k=topk_eff, dim=-1).indices
        seg_expanded = seg_flat[:, None, :].expand(-1, k_proto, -1)
        topk_masks = torch.gather(seg_expanded, dim=-1, index=topk_idx)
        all_inside = topk_masks.min(dim=-1).values  # 1 if all top-K are FG

        total_hits += float(all_inside.sum().item())
        total_count += float(all_inside.numel())

    if total_count == 0:
        return float("nan")
    return total_hits / total_count


__all__ = ["evaluate_pfr_topk"]
