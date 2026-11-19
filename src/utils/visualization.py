"""Spatial-softmax galleries for class-specific and shared checkpoints."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .imagenet import IMAGENET_MEAN, IMAGENET_STD


def display_map(cosine, inverse_temperature):
    if inverse_temperature <= 0:
        raise ValueError("Display inverse-temperature must be positive")
    value = (
        (inverse_temperature * cosine.double())
        .flatten(-2)
        .softmax(-1)
        .reshape_as(cosine)
    )
    return value / value.amax((-2, -1), keepdim=True).clamp_min(1e-300)


@torch.no_grad()
def visualize_fitted(model, dataset, args, normalize_inputs=True):
    if args.topk < 1 or (args.num_samples is not None and args.num_samples < 1):
        raise ValueError("topk and num-samples must be positive")
    # Explicit prototype IDs and background panels are not supported here.
    if args.include_bg or args.proto_ids is not None:
        raise ValueError(
            "The fitted gallery shows foreground class prototypes; use --topk to select the count"
        )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = model.prototype_mu.device
    if args.indices:
        indices = [
            int(i) for item in args.indices for i in item.replace(",", " ").split()
        ]
    else:
        indices = list(range(len(dataset)))
        np.random.default_rng(args.seed).shuffle(indices)
    temperature = args.display_temperature
    if temperature is None:
        temperature = (
            (175 if getattr(dataset, "image_ids", None) is not None else 400)
            if model.shared
            else 650
        )
    shown, per_class, records = 0, {}, []
    for index in indices:
        image, label, *_ = dataset[index]
        label = int(label)
        if args.class_id is not None and label != args.class_id:
            continue
        if args.per_class is not None and per_class.get(label, 0) >= args.per_class:
            continue
        result = model(image[None].to(device))
        prediction = int(result["class_logits"].argmax(-1))
        if args.only_misclassified and prediction == label:
            continue
        focus = prediction if args.focus == "pred" else label
        selected = model.class_prototype_indices(
            torch.tensor([focus], device=device), args.topk
        )[0]
        if not model.shared:
            weights = model.bank()["pi"][focus]
            selected = selected[weights.argsort(descending=True)[: args.topk]]
        bank = model.bank()
        raw = result["maps_raw"][0, selected]
        cosine = (raw - bank["logc"][selected, None, None]) / bank["kappa"][
            selected, None, None
        ]
        heat = display_map(cosine, temperature)
        heat = F.interpolate(
            heat[:, None], size=image.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        if args.fg_gate:
            heat *= F.interpolate(
                result["fg_mask"].double(), size=image.shape[-2:], mode="nearest"
            )[0]
        base = image.cpu()
        if normalize_inputs:
            base = (
                base * torch.tensor(IMAGENET_STD)[:, None, None]
                + torch.tensor(IMAGENET_MEAN)[:, None, None]
            )
        base = base.clamp(0, 1).permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(
            1,
            len(selected) + 1,
            figsize=(2.1 * (len(selected) + 1), 2.5),
            squeeze=False,
            dpi=args.dpi,
        )
        axes[0, 0].imshow(base)
        axes[0, 0].set_title(dataset.classes[label], fontsize=10)
        for ax, values in zip(axes[0, 1:], heat.cpu().numpy()):
            overlay = plt.get_cmap(args.cmap)(values)[..., :3]
            ax.imshow((1 - args.alpha) * base + args.alpha * overlay)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout(pad=0.3)
        path = output / f"{index:06d}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        records.append(
            dict(
                index=index,
                label=label,
                prediction=prediction,
                prototype_ids=selected.cpu().tolist(),
                file=path.name,
            )
        )
        per_class[label] = per_class.get(label, 0) + 1
        shown += 1
        if shown >= (args.num_samples if args.num_samples is not None else 5):
            break
    (output / "gallery.json").write_text(
        json.dumps(
            dict(
                variant=model.variant,
                display_inverse_temperature=temperature,
                normalization="spatial softmax(T*cosine), then per-map maximum scaling for display",
                gated=args.fg_gate,
                examples=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {shown} galleries to {output}")
