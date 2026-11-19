"""Accuracy and unmasked part metrics for class-specific and shared models."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataset
from src.models.api import build_model
from src.eval.types import EvalResult


def peak_positions(maps, size):
    positions = np.zeros((*maps.shape[:2], 2), dtype=np.int64)
    for b in range(len(maps)):
        for p, values in enumerate(maps[b]):
            resized = cv2.resize(values, (size, size), interpolation=cv2.INTER_CUBIC)
            y, x = np.unravel_index(resized.argmax(), resized.shape)
            positions[b, p] = x, y
    return positions


def point_hits(peaks, locations, count, size=224, half=36):
    hits = np.zeros((len(peaks), peaks.shape[1], count), dtype=bool)
    visible = np.zeros((len(peaks), count), dtype=bool)
    for b, points in enumerate(locations):
        for part, x, y in points:
            visible[b, part] = True
            px, py = peaks[b].T
            hits[b, :, part] |= (
                (x >= np.maximum(0, px - half))
                & (x <= np.minimum(size, px + half))
                & (y >= np.maximum(0, py - half))
                & (y <= np.minimum(size, py + half))
            )
    return hits, visible


def distinctiveness(maps, size=224, box_size=72):
    if maps.shape[1] < 2:
        raise ValueError("Distinctiveness requires at least two prototypes")
    # Use 2x2 average-pool / bilinear geometry for the peak windows.
    selected = F.interpolate(
        F.avg_pool2d(maps, 2, stride=2),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    indices = selected.flatten(2).argmax(-1).cpu().numpy()
    y, x = np.divmod(indices, size)
    intersection = np.maximum(
        0, box_size - np.abs(x[:, :, None] - x[:, None, :])
    ) * np.maximum(0, box_size - np.abs(y[:, :, None] - y[:, None, :]))
    iou = intersection / (2 * box_size * box_size - intersection)
    a, b = np.triu_indices(maps.shape[1], 1)
    return 1 - iou[:, a, b].mean(-1)


class PartAnnotations:
    def __init__(self, dataset, cfg):
        self.dataset, self.size, self.name = (
            dataset,
            int(cfg.dataset.input_size[0]),
            str(cfg.dataset.name),
        )
        self.regions = None
        if self.name == "cub200":
            from .cub_metadata import load_cub_metadata

            self.metadata = load_cub_metadata(cfg.dataset.root)
            lookup = {tuple(path): i for i, path in self.metadata.id_to_path.items()}
            self.ids = [
                lookup[(Path(path).parent.name, Path(path).name)]
                for path, _ in dataset.samples
            ]
            self.count, self.protocol = (
                self.metadata.part_num,
                "CUB visible landmarks; 72px peak window",
            )
        elif self.name == "ap10k":
            self.ids = dataset.image_ids
            self.images = {r["id"]: r for r in dataset.annotations["images"]}
            self.rows = defaultdict(list)
            for row in dataset.annotations["annotations"]:
                self.rows[row["image_id"]].append(row)
            keypoints = dataset.categories[0]["keypoints"]
            self.names = list(
                dict.fromkeys(
                    n.removeprefix("left_").removeprefix("right_") for n in keypoints
                )
            )
            self.groups = [
                self.names.index(n.removeprefix("left_").removeprefix("right_"))
                for n in keypoints
            ]
            self.keypoint_names = keypoints
            self.count, self.protocol = (
                len(self.names),
                "AP-10K visible keypoints, left/right merged; 72px peak window (keypoint-only diagnostic)",
            )
            region_path = getattr(cfg.eval, "ap10k_regions", None)
            if region_path:
                with np.load(str(region_path), allow_pickle=False) as data:
                    self.region_names = data["group_names"].astype(str).tolist()
                    self.regions = data["masks"].astype(bool)
                    self.region_lookup = {
                        int(i): row for row, i in enumerate(data["image_ids"])
                    }
                if self.regions.ndim != 4 or self.regions.shape[1] != len(
                    self.region_names
                ):
                    raise ValueError(
                        "Region NPZ requires masks[N,G,H,W] and group_names[G]"
                    )
                if (
                    len(self.region_lookup) != len(self.regions)
                    or not set(self.ids) <= self.region_lookup.keys()
                ):
                    raise ValueError(
                        "Region pack has duplicate IDs or is missing evaluation images"
                    )
                self.count, self.protocol = (
                    len(self.region_names),
                    "AP-10K evaluation-only coarse masks with visible-leg-keypoint windows",
                )
        else:
            raise ValueError("Part annotations are available only for CUB and AP-10K")

    def hits(self, maps, indices):
        peaks = peak_positions(maps, self.size)
        locations, image_ids = [], [self.ids[int(i)] for i in indices]
        if self.name == "cub200":
            for image_id in image_ids:
                x1, y1, x2, y2 = self.metadata.id_to_bbox[image_id]
                locations.append(
                    [
                        (
                            p - 1,
                            int(self.size * (x - x1) / (x2 - x1)),
                            int(self.size * (y - y1) / (y2 - y1)),
                        )
                        for p, x, y in self.metadata.id_to_part_loc.get(image_id, [])
                    ]
                )
        else:
            for image_id in image_ids:
                info, points = self.images[image_id], []
                for row in self.rows[image_id]:
                    for part, (x, y, visibility) in zip(
                        self.groups, np.asarray(row["keypoints"]).reshape(-1, 3)
                    ):
                        if (
                            visibility == 2
                            and 0 <= x < info["width"]
                            and 0 <= y < info["height"]
                        ):
                            points.append(
                                (
                                    part,
                                    int(self.size * x / info["width"]),
                                    int(self.size * y / info["height"]),
                                )
                            )
                locations.append(points)
        hits, visible = point_hits(
            peaks,
            locations,
            len(self.names) if self.name == "ap10k" else self.count,
            self.size,
        )
        if self.regions is None:
            return hits, visible
        region_hits = np.zeros((*peaks.shape[:2], self.count), dtype=bool)
        region_present = np.zeros((len(peaks), self.count), dtype=bool)
        for b, image_id in enumerate(image_ids):
            masks = self.regions[self.region_lookup[image_id]]
            height, width = masks.shape[-2:]
            for g, name in enumerate(self.region_names):
                if name == "legs":
                    leg_ids = [
                        i
                        for i, n in enumerate(self.names)
                        if any(
                            w in n for w in ("shoulder", "elbow", "paw", "hip", "knee")
                        )
                    ]
                    region_hits[b, :, g] = hits[b][:, leg_ids].any(-1)
                    region_present[b, g] = visible[b, leg_ids].any()
                else:
                    region_present[b, g] = masks[g].any()
                    for p, (x, y) in enumerate(peaks[b]):
                        left, right = (
                            max(0, x - 36) * width // self.size,
                            min(self.size, x + 36) * width // self.size,
                        )
                        top, bottom = (
                            max(0, y - 36) * height // self.size,
                            min(self.size, y + 36) * height // self.size,
                        )
                        region_hits[b, p, g] = masks[g, top:bottom, left:right].any()
        return region_hits, region_present


def evaluate_model(model, dataset, cfg):
    model.eval()
    device = model.prototype_mu.device
    metrics = set(map(str, cfg.eval.metrics))
    if (
        not len(dataset)
        or int(getattr(cfg.eval, "limit", 0)) < 0
        or int(cfg.eval.topk) < 1
    ):
        raise ValueError(
            "Evaluation requires nonempty data, positive topk, and nonnegative limit"
        )
    unknown = metrics - {"accuracy", "consistency", "stability", "distinctiveness"}
    if unknown:
        raise ValueError(f"Four-loss evaluation does not implement {sorted(unknown)}")
    annotation = (
        PartAnnotations(dataset, cfg)
        if metrics & {"consistency", "stability"}
        else None
    )
    classes = model.n_classes
    k = (
        min(int(cfg.eval.topk), model.foreground_prototypes)
        if model.shared
        else model.n_prototypes
    )
    image_counts, hit_counts, visible_counts, agreements = (
        np.zeros(classes),
        None,
        None,
        np.zeros((classes, k)),
    )
    if annotation is not None:
        hit_counts = np.zeros((classes, k, annotation.count))
        visible_counts = np.zeros((classes, annotation.count))
    correct = total = offset = 0
    distinct = []
    generator = torch.Generator(device=device).manual_seed(int(cfg.train.seed))
    limit = int(getattr(cfg.eval, "limit", 0))
    from torch.utils.data import Subset

    loader = DataLoader(
        Subset(dataset, range(min(limit, len(dataset)))) if limit else dataset,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            correct += int((outputs["class_logits"].argmax(-1) == labels).sum())
            total += len(images)
            selected = model.class_prototype_indices(labels, k)
            raw = outputs["maps_raw"].gather(
                1, selected[..., None, None].expand(-1, -1, *outputs["grid_hw"])
            )
            maps = (raw - raw.amax((-2, -1), keepdim=True)).exp().float()
            if "distinctiveness" in metrics:
                distinct.extend(distinctiveness(maps, int(cfg.dataset.input_size[0])))
            y = labels.cpu().numpy()
            np.add.at(image_counts, y, 1)
            if annotation is not None:
                indices = np.arange(offset, offset + len(images))
                hits, visible = annotation.hits(maps.cpu().numpy(), indices)
                np.add.at(hit_counts, y, hits)
                np.add.at(visible_counts, y, visible)
                if "stability" in metrics:
                    delta = torch.randn(
                        images.shape, generator=generator, device=device
                    ) * float(cfg.train.fitting.noise_std)
                    noisy = model(
                        images
                        + delta.clamp(
                            -float(cfg.train.fitting.noise_clip),
                            float(cfg.train.fitting.noise_clip),
                        )
                    )
                    raw_noise = noisy["maps_raw"].gather(
                        1, selected[..., None, None].expand_as(raw)
                    )
                    noise_maps = (
                        (raw_noise - raw_noise.amax((-2, -1), keepdim=True))
                        .exp()
                        .float()
                    )
                    noisy_hits, _ = annotation.hits(noise_maps.cpu().numpy(), indices)
                    np.add.at(agreements, y, (hits == noisy_hits).all(-1))
            offset += len(images)
    report = dict(
        images=total,
        partial_evaluation=bool(limit),
        prototype_selection="top class weights" if model.shared else "own class bank",
        prototypes_per_class=k,
        part_protocol=None if annotation is None else annotation.protocol,
    )
    if "accuracy" in metrics:
        report["accuracy_pct"] = 100 * correct / total
    represented = image_counts > 0
    if "consistency" in metrics:
        rates = hit_counts / np.maximum(visible_counts[:, None, :], 1)
        report["consistency_pct"] = 100 * float(
            (rates[represented] >= 0.8).any(-1).mean()
        )
    if "stability" in metrics:
        report["stability_pct"] = 100 * float(
            (agreements / np.maximum(image_counts[:, None], 1))[represented].mean()
        )
    if "distinctiveness" in metrics:
        report["distinctiveness_pct"] = 100 * float(np.mean(distinct))
    return report


def evaluate_fitted(cfg, run_dir, checkpoint_path, saved):
    eval_cfg = OmegaConf.create(saved["hparams"])
    eval_cfg.train.device = str(cfg.train.device)
    eval_cfg.train.num_workers = int(cfg.train.num_workers)
    for key in ("metrics", "batch_size", "topk", "split", "ap10k_regions", "limit"):
        value = OmegaConf.select(cfg, f"eval.{key}")
        if value is not None:
            OmegaConf.update(eval_cfg, f"eval.{key}", value, merge=False)
    if getattr(cfg.eval, "data_root", None):
        eval_cfg.dataset.root = str(cfg.eval.data_root)
    model = build_model(eval_cfg).to(str(eval_cfg.train.device)).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    dataset = build_dataset(eval_cfg, str(eval_cfg.eval.split))
    report = evaluate_model(model, dataset, eval_cfg)
    report.update(
        checkpoint=str(checkpoint_path),
        partial_checkpoint=bool(saved.get("partial", False)),
        epoch=saved["epoch"],
        checkpoint_monitor=saved["monitor"],
    )
    output = (
        Path(run_dir) / f"eval_{Path(checkpoint_path).stem}_{eval_cfg.eval.split}.json"
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return EvalResult(summary=report, sample_paths=None)
