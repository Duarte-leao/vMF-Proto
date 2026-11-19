"""Self-contained four-loss learner, shared by both prototype variants.

Frozen DINO prefixes are cached on CPU. Each query batch is excluded from its
fitting support. The exact support gradient is replayed in microbatches, so the
full training-set graph never has to fit on the GPU. No experiment wrappers,
trained donor checkpoints, test annotations, or cluster paths are required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import minimize_scalar
from torch.nn import functional as F

from src.losses.fitting import (
    feature_anchor,
    peak_loss,
    prediction_kl,
    prediction_weight,
    spatial_features,
)
from src.models.api import build_model
from src.models.vmf_mp.fitting import (
    effective_bank,
    fit_fraction,
    moment_sum,
    raw_scores,
    selected_weights,
    support_vjp,
)
from src.models.vmf_mp.utils import prototype_logdet_vmf_overlap, sinkhorn_transport

LOG = logging.getLogger(__name__)
FORMAT = "vmfproto/early_four_loss/v1"


def detached(bank, device=None):
    return {
        k: (v.detach().to(device) if device is not None else v.detach())
        if torch.is_tensor(v)
        else v
        for k, v in bank.items()
    }


def extract(model, cache, rows=None, batch_size=64):
    rows = torch.arange(len(cache["labels"])) if rows is None else rows
    device = model.prototype_mu.device
    return torch.cat(
        [
            model.adapt_prefix(cache["prefix"][q].to(device), cache["grid"])
            .detach()
            .cpu()
            for q in rows.split(batch_size)
        ]
    )


@torch.no_grad()
def cache_split(model, loader, cfg, path, noisy=False):
    """Only image inputs/labels enter this cache; no anatomical annotations."""
    settings = dict(
        model=OmegaConf.to_container(cfg.model, resolve=True),
        dataset=OmegaConf.to_container(cfg.dataset, resolve=True),
        seed=cfg.train.seed,
        noise_std=cfg.train.fitting.noise_std,
        noise_clip=cfg.train.fitting.noise_clip,
        smoke_limit=int(getattr(cfg.run, "smoke_limit", 0)),
    )
    dataset = loader.dataset
    rows = getattr(dataset, "indices", range(len(dataset)))
    dataset = getattr(dataset, "dataset", dataset)
    if hasattr(dataset, "samples"):
        manifest = []
        for index in rows:
            image, label = dataset.samples[index]
            stat = Path(image).stat()
            manifest.append((str(image), int(label), stat.st_size, stat.st_mtime_ns))
        settings["input_manifest"] = manifest
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if saved.get("cache_key") != key or len(saved["labels"]) != len(loader.dataset):
            raise ValueError(f"Cache/config mismatch: {path}; use a new run directory")
        return saved
    chunks, labels, gates, noise = [], [], [], [[], []]
    device = model.prototype_mu.device
    # A local RNG makes resume independent of whether the cache already exists.
    generator = torch.Generator(device=device).manual_seed(int(cfg.train.seed) + 2718)
    for batch in loader:
        images = batch[0].to(device)
        prefix = model.frozen_prefix(images)
        frozen = model.frozen_patches(prefix)
        side = math.isqrt(frozen.shape[1])
        grid = (side, side)
        chunks.append(prefix.cpu())
        labels.append(batch[1].cpu())
        gates.append(model.foreground(frozen, images, grid).cpu())
        if noisy:
            for view in noise:
                delta = torch.randn(
                    images.shape, device=device, generator=generator
                ) * float(cfg.train.fitting.noise_std)
                view.append(
                    model.frozen_prefix(
                        images
                        + delta.clamp(
                            -float(cfg.train.fitting.noise_clip),
                            float(cfg.train.fitting.noise_clip),
                        )
                    ).cpu()
                )
    if not chunks:
        raise ValueError("Empty dataset split")
    saved = dict(
        prefix=torch.cat(chunks),
        labels=torch.cat(labels),
        foreground=torch.cat(gates),
        grid=grid,
        cache_key=key,
    )
    if noisy:
        saved["noise_prefix"] = torch.stack([torch.cat(view) for view in noise])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    return saved


@torch.no_grad()
def assignments(model, features, foreground, labels, bank):
    raw = raw_scores(features, bank)
    r, k = model.foreground_prototypes, model.n_prototypes
    if model.shared:
        winner = raw[..., :r].argmax(-1)
    else:
        own = raw[..., :r].reshape(*raw.shape[:2], model.n_classes, k)
        winner = (
            own[torch.arange(len(raw), device=raw.device), :, labels].argmax(-1)
            + labels[:, None] * k
        )
    return torch.where(foreground, winner, raw[..., r:].argmax(-1) + r)


@torch.no_grad()
def membership(
    model, features, cache, bank, reference_features, reference_means, options
):
    ids, weights = [], []
    device = model.prototype_mu.device
    for rows in torch.arange(len(features)).split(int(options.support_microbatch)):
        current = features[rows].to(device)
        group = assignments(
            model,
            current,
            cache["foreground"][rows].to(device),
            cache["labels"][rows].to(device),
            bank,
        )
        reference = F.normalize(reference_features[rows].to(device).double(), dim=-1)
        cosine = (reference * reference_means[group]).sum(-1)
        weight = selected_weights(
            cosine,
            group,
            len(bank["kappa"]),
            float(options.importance),
            float(options.uniform_floor),
        )
        ids.append(group.cpu())
        weights.append(weight.cpu())
    return dict(ids=torch.cat(ids), weights=torch.cat(weights))


def moments_from_features(features, groups, components, device, batch_size):
    sums = torch.zeros(
        components, features.shape[-1], dtype=torch.float64, device=device
    )
    mass = torch.zeros(components, dtype=torch.float64, device=device)
    for rows in torch.arange(len(features)).split(batch_size):
        ids, weights = (
            groups["ids"][rows].to(device),
            groups["weights"][rows].to(device),
        )
        sums += moment_sum(features[rows].to(device), ids, weights, components)
        mass.index_add_(0, ids.flatten(), weights.flatten())
    return sums / mass.clamp_min(1e-30)[:, None], mass


@torch.no_grad()
def refit(
    model, features, cache, initial, reference_features, reference_means, options
):
    bank = detached(initial)
    converged = False
    for iteration in range(int(options.refit_steps)):
        groups = membership(
            model, features, cache, bank, reference_features, reference_means, options
        )
        moment, mass = moments_from_features(
            features,
            groups,
            len(bank["kappa"]),
            model.prototype_mu.device,
            int(options.support_microbatch),
        )
        updated = effective_bank(
            bank, moment, mass, 1.0, int(options.minimum_support_images)
        )
        # Export an FP32-representable bank for deterministic checkpoint reloads.
        means = updated["means"].float().double()
        k = updated["kappa"]
        raw_k = (k + torch.log(-torch.expm1(-k))).float().double()
        from src.models.vmf_mp.vmf_normalizer import logCp_vmf

        updated = dict(
            updated, means=F.normalize(means, dim=-1), kappa=F.softplus(raw_k)
        )
        updated["logc"] = logCp_vmf(model.dim, updated["kappa"])
        delta = float((bank["means"] - updated["means"]).abs().max())
        relative = float(
            (
                (bank["kappa"] - updated["kappa"]).abs() / updated["kappa"].clamp_min(1)
            ).max()
        )
        bank = detached(updated)
        if delta < 1e-5 and relative < 1e-4:
            converged = True
            break
    if not converged:
        LOG.warning(
            "Refit reached %d iterations without the convergence tolerance",
            options.refit_steps,
        )
    return bank, dict(
        iterations=iteration + 1,
        converged=converged,
        supported_foreground=int(
            bank["supported"][: model.foreground_prototypes].sum()
        ),
    )


@torch.no_grad()
def ot_targets(model, raw, foreground, labels, bank, scale):
    r, k = model.foreground_prototypes, model.n_prototypes
    result = torch.empty_like(foreground, dtype=torch.long)
    if model.shared:
        counts = foreground.sum(-1).double()
        marginal = (bank["pi"][labels] * counts[:, None]).sum(0) / counts.sum()
        if model.ot_mode == "balanced":
            marginal = torch.full_like(marginal, 1 / r)
        populations = [
            (foreground, raw[..., :r], marginal, 0),
            (~foreground, raw[..., r:], bank["bg_pi"], r),
        ]
    else:
        patch_classes = torch.where(foreground, labels[:, None], model.n_classes)
        populations = []
        for c in patch_classes.unique().tolist():
            pi = bank["bg_pi"] if c == model.n_classes else bank["pi"][c]
            if model.ot_mode == "balanced":
                pi = torch.full_like(pi, 1 / k)
            populations.append(
                (patch_classes == c, raw[..., c * k : (c + 1) * k], pi, c * k)
            )
    for mask, scores, marginal, offset in populations:
        if mask.any():
            plan = sinkhorn_transport(
                scores[mask].double() / scale,
                col_marginals=marginal.detach() * int(mask.sum()),
                epsilon=model.ot_epsilon,
                n_iters=model.ot_iters,
            )
            result[mask] = plan.argmax(-1) + offset
    return result


def discovery_diversity(model, bank):
    r, k = model.foreground_prototypes, model.n_prototypes
    if model.shared:
        pi = bank["pi"].detach()
        active = pi.max(-1).values > pi.min(-1).values
        if not active.any():
            return bank["means"].sum() * 0
        selected = pi[active].topk(min(5, r), -1).indices
        mu, concentration = bank["means"][selected], bank["kappa"][selected]
        multiplier = active.double().mean()
    else:
        mu = bank["means"][:r].reshape(model.n_classes, k, model.dim)
        concentration = bank["kappa"][:r].reshape(model.n_classes, k)
        multiplier = 1.0
    return multiplier * prototype_logdet_vmf_overlap(
        mu, concentration, dim=model.dim, eps=1e-4, include_background=True
    )


@torch.no_grad()
def warmup(model, cache, cfg):
    device = model.prototype_mu.device
    order = torch.randperm(len(cache["labels"]))
    for query in order.split(int(cfg.train.batch_size)):
        z = extract(model, cache, query, int(cfg.train.fitting.micro_batch_size)).to(
            device
        )
        bank = model.bank()
        target = ot_targets(
            model,
            raw_scores(z, bank),
            cache["foreground"][query].to(device),
            cache["labels"][query].to(device),
            bank,
            max(float(bank["kappa"].median()), 1e-6),
        )
        sums = torch.zeros_like(model.prototype_mu).index_add(
            0, target.flatten(), z.flatten(0, 1).float()
        )
        updated = F.normalize(
            model.gamma * model.prototype_mu + (1 - model.gamma) * sums, dim=-1
        )
        if model.shared:
            model.prototype_mu.copy_(updated)
        else:
            # Update all J components of each class present in the batch.
            present = (target // model.n_prototypes).unique()
            active = (
                torch.arange(len(model.kappa_raw), device=device) // model.n_prototypes
            )[:, None] == present[None]
            model.prototype_mu.copy_(
                torch.where(active.any(-1, keepdim=True), updated, model.prototype_mu)
            )


class Learner:
    def __init__(self, model, cfg):
        self.model, self.cfg, self.reference, self.fixed_gain = model, cfg, None, None
        model.requires_grad_(False)
        model.backbone.set_requires_grad()
        for name in ("prototype_mu", "kappa_raw", "pi_logits", "background_pi_logits"):
            if hasattr(model, name):
                getattr(model, name).requires_grad_(
                    name != "kappa_raw" or model.kappa_learnable
                )
        options = cfg.train.fitting
        pi = [model.pi_logits] + ([model.background_pi_logits] if model.shared else [])
        self.optimizer = torch.optim.Adam(
            [
                dict(
                    params=model.backbone.learnable_parameters(),
                    lr=float(options.feature_lr),
                    name="features",
                ),
                dict(params=pi, lr=float(options.pi_lr), name="pi"),
                dict(
                    params=[model.prototype_mu], lr=float(options.mu_lr), name="means"
                ),
                dict(
                    params=[model.kappa_raw], lr=float(options.kappa_lr), name="kappa"
                ),
            ],
            weight_decay=0.0,
        )

    @torch.no_grad()
    def initialize_reference(self, cache):
        model, options = self.model, self.cfg.train.fitting
        device = model.prototype_mu.device
        features = extract(model, cache, batch_size=int(options.support_microbatch))
        bank = detached(model.bank())
        groups = membership(
            model, features, cache, bank, features, bank["means"], options
        )
        probabilities = []
        gain = model.foreground_prototypes / 7 if model.shared else 1.0
        for rows in torch.arange(len(features)).split(int(options.micro_batch_size)):
            raw = raw_scores(features[rows].to(device), bank)
            loga = model.class_evidence(raw, cache["foreground"][rows].to(device), bank)
            probabilities.append((gain * loga.exp()).softmax(-1).cpu())
        fitted, report = refit(
            model, features, cache, bank, features, bank["means"], options
        )
        self.reference = dict(
            features=features,
            bank=detached(bank, "cpu"),
            groups=groups,
            probabilities=torch.cat(probabilities),
            fitted_means=fitted["means"].cpu(),
            fit_report=report,
        )
        self.optimizer.state.clear()

    def step(self, cache, query, support, fraction, epoch):
        if torch.isin(query, support).any():
            raise ValueError("Query images must be excluded from fitting support")
        model, cfg, optimizer = self.model, self.cfg, self.optimizer
        options, device = cfg.train.fitting, model.prototype_mu.device
        if fraction and self.reference is None:
            self.initialize_reference(cache)
        model.prototype_mu.requires_grad_(fraction < 1)
        model.kappa_raw.requires_grad_(model.kappa_learnable and fraction < 1)
        optimizer.param_groups[0]["lr"] = float(
            options.fitting_feature_lr if fraction else options.feature_lr
        )
        optimizer.param_groups[1]["lr"] = float(
            options.fitting_pi_lr if fraction else options.pi_lr
        )
        with torch.no_grad():
            if fraction:
                features = extract(
                    model, cache, support, int(options.support_microbatch)
                )
                groups = {k: v[support] for k, v in self.reference["groups"].items()}
                moment, mass = moments_from_features(
                    features,
                    groups,
                    len(model.kappa_raw),
                    device,
                    int(options.support_microbatch),
                )
            else:
                moment = torch.zeros_like(model.prototype_mu, dtype=torch.float64)
                mass = torch.zeros_like(model.kappa_raw, dtype=torch.float64)
        leaf = moment.detach().requires_grad_(bool(fraction))
        with torch.no_grad():
            free = model.bank()
            bank = effective_bank(
                free, leaf, mass, fraction, int(options.minimum_support_images)
            )
            fitted = (
                effective_bank(
                    free, leaf, mass, 1.0, int(options.minimum_support_images)
                )
                if fraction
                else free
            )
            r = model.foreground_prototypes
            beta = (
                float(
                    (
                        fraction
                        * fitted["kappa"][:r]
                        / (
                            (1 - fraction) * free["kappa"][:r]
                            + fraction * fitted["kappa"][:r]
                        )
                    ).median()
                )
                if fraction
                else 0.0
            )
            raw_parts, log_parts = [], []
            for q in query.split(int(options.micro_batch_size)):
                z = model.adapt_prefix(cache["prefix"][q].to(device), cache["grid"])
                raw = raw_scores(z, bank)
                raw_parts.append(raw)
                log_parts.append(
                    model.class_evidence(raw, cache["foreground"][q].to(device), bank)
                )
            raw_all, log_all = torch.cat(raw_parts), torch.cat(log_parts)
            scale = max(float(bank["kappa"].median()), 1e-6) if fraction else 1.0
            target = ot_targets(
                model,
                raw_all,
                cache["foreground"][query].to(device),
                cache["labels"][query].to(device),
                bank,
                scale,
            )
            gain = model.foreground_prototypes / 7 if model.shared else 1.0
            if fraction:
                if self.fixed_gain is None:
                    centered = log_all - log_all.amax(-1, keepdim=True)

                    def objective(log_gain):
                        return float(
                            F.cross_entropy(
                                math.exp(log_gain) * centered,
                                cache["labels"][query].to(device),
                            )
                        )

                    solution = minimize_scalar(
                        objective,
                        bounds=(-9.210340371976184, 9.210340371976184),
                        method="bounded",
                        options={"xatol": 1e-6, "maxiter": 100},
                    )
                    if not solution.success:
                        raise RuntimeError("TRAIN confidence calibration failed")
                    gain = math.exp(solution.x)
                    if fraction == 1 and len(query) >= int(cfg.train.batch_size):
                        self.fixed_gain = gain
                else:
                    gain = self.fixed_gain
        optimizer.zero_grad(set_to_none=True)
        fg_all = cache["foreground"][query]
        denominator = float(fg_all.sum() + 0.1 * (~fg_all).sum())
        kl_weight = (
            prediction_weight(
                epoch,
                int(options.discovery_until),
                int(cfg.train.epochs),
                float(options.prediction_kl_initial),
                float(options.prediction_kl_final),
                int(options.prediction_kl_hold),
            )
            if fraction
            else 0.0
        )
        totals = dict(
            CE=0.0,
            PPD=0.0,
            prediction_KL=0.0,
            feature_anchor=0.0,
            reference_peak=0.0,
            noisy_retention=0.0,
            LogDet=0.0,
        )
        offset = 0
        for q in query.split(int(options.micro_batch_size)):
            fg, labels = (
                cache["foreground"][q].to(device),
                cache["labels"][q].to(device),
            )
            targets = target[offset : offset + len(q)]
            offset += len(q)
            views = [cache["prefix"][q]]
            if fraction:
                views.append(cache["noise_prefix"][(epoch - 2) % 2, q])
                reference = spatial_features(
                    self.reference["features"][q].to(device),
                    float(options.target_blend),
                )
                reference_means = self.reference["fitted_means"].to(device)[:r]
                if model.shared:
                    presence = torch.zeros(len(q), r, dtype=torch.bool, device=device)
                    image, patch = fg.nonzero(as_tuple=True)
                    presence[
                        image,
                        self.reference["groups"]["ids"][q].to(device)[image, patch],
                    ] = True
                else:
                    presence = torch.zeros(len(q), r, dtype=torch.bool, device=device)
                    presence.scatter_(1, model.class_prototype_indices(labels), True)
            clean = None
            for view, prefix in enumerate(views):
                bank = effective_bank(
                    model.bank(),
                    leaf,
                    mass,
                    fraction,
                    int(options.minimum_support_images),
                )
                z = model.adapt_prefix(prefix.to(device), cache["grid"])
                raw = raw_scores(z, bank)
                loga = model.class_evidence(raw, fg, bank)
                logits = gain * (loga if fraction else loga.exp())
                ce = F.cross_entropy(logits, labels)
                class_weight = raw.new_ones(raw.shape[-1])
                class_weight[r:] = float(cfg.train.loss.bg_class_weight)
                ppd = (
                    F.cross_entropy(
                        raw.transpose(1, 2)
                        / (scale * float(cfg.train.loss.l_ppd_temp)),
                        targets,
                        weight=class_weight,
                        reduction="sum",
                    )
                    / denominator
                )
                extra, values = raw.new_zeros(()), dict(CE=ce, PPD=ppd)
                if fraction:
                    kl = prediction_kl(
                        logits, self.reference["probabilities"][q].to(device)
                    )
                    anchor = feature_anchor(z, reference, fg)
                    peak = peak_loss(
                        z,
                        bank["means"][:r],
                        bank["kappa"][:r],
                        reference,
                        reference_means,
                        presence,
                        float(options.peak_margin),
                    )
                    extra = fraction * kl_weight * kl + beta * (
                        float(options.feature_anchor) * anchor
                        + float(options.reference_peak) * peak
                    )
                    values.update(
                        prediction_KL=kl, feature_anchor=anchor, reference_peak=peak
                    )
                    if view == 0:
                        clean = z.detach()
                    else:
                        pair = peak_loss(
                            z,
                            bank["means"][:r],
                            bank["kappa"][:r],
                            clean,
                            bank["means"][:r].detach(),
                            presence,
                            float(options.peak_margin),
                        )
                        extra = extra + 2 * beta * float(options.noisy_retention) * pair
                        totals["noisy_retention"] += (
                            float(pair.detach()) * len(q) / len(query)
                        )
                loss = (
                    ce * len(q) / len(query)
                    + float(cfg.train.loss.l_ppd_coef) * (1 - beta) * ppd
                    + extra * len(q) / len(query)
                ) / len(views)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite four-loss training objective")
                loss.backward()
                for key, value in values.items():
                    totals[key] += (
                        float(value.detach())
                        * (1.0 if key == "PPD" else len(q) / len(query))
                        / len(views)
                    )
        if fraction == 0:
            diversity = discovery_diversity(model, model.bank())
            (float(cfg.train.loss.l_logdet_vmf_coef) * diversity).backward()
            totals["LogDet"] = float(diversity.detach())
        if fraction:
            if leaf.grad is None or not torch.isfinite(leaf.grad).all():
                raise FloatingPointError("Missing or invalid exact support derivative")
            for rows in support.split(int(options.support_microbatch)):
                z = model.adapt_prefix(cache["prefix"][rows].to(device), cache["grid"])
                ids, weights = (
                    self.reference["groups"][key][rows].to(device)
                    for key in ("ids", "weights")
                )
                support_vjp(z, ids, weights, mass, leaf.grad.detach()).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
        torch.nn.utils.clip_grad_norm_(model.backbone.learnable_parameters(), 5.0)
        geometry = [model.prototype_mu, model.kappa_raw, model.pi_logits]
        if model.shared:
            geometry.append(model.background_pi_logits)
        torch.nn.utils.clip_grad_norm_(geometry, 5.0)
        optimizer.step()
        return dict(
            totals,
            fit_fraction=fraction,
            fitted_natural_share=beta,
            TRAIN_confidence_gain=gain,
            effective_PPD_weight=float(cfg.train.loss.l_ppd_coef) * (1 - beta),
            query_images=len(query),
            support_images=len(support),
        )


@torch.no_grad()
def cache_accuracy(model, cache, bank, batch_size=64):
    correct = total = 0
    for rows in torch.arange(len(cache["labels"])).split(batch_size):
        features = extract(model, cache, rows, batch_size).to(model.prototype_mu.device)
        logits = model.class_evidence(
            raw_scores(features, bank),
            cache["foreground"][rows].to(features.device),
            bank,
        )
        correct += int((logits.argmax(-1).cpu() == cache["labels"][rows]).sum())
        total += len(rows)
    return 100 * correct / total


def train_fitted(cfg, run_dir, *, model=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")],
        force=True,
    )
    options = cfg.train.fitting
    if int(cfg.model.num_splits) != 1:
        raise ValueError(
            "The fitted recipe requires exactly one appended/expanded block"
        )
    if (
        not 1
        <= int(options.discovery_until)
        < int(options.fitted_by)
        <= int(cfg.train.epochs)
    ):
        raise ValueError("Require 1 <= discovery_until < fitted_by <= epochs")
    if (
        int(options.refit_steps) < 1
        or min(
            int(cfg.train.batch_size),
            int(options.micro_batch_size),
            int(options.support_microbatch),
        )
        < 1
    ):
        raise ValueError("Batch sizes and refit_steps must be positive")
    random.seed(int(cfg.train.seed))
    np.random.seed(int(cfg.train.seed))
    torch.manual_seed(int(cfg.train.seed))
    torch.set_float32_matmul_precision("highest")
    device = torch.device(str(cfg.train.device))
    model = build_model(cfg) if model is None else model
    model.eval().to(device)
    learner = Learner(model, cfg)
    last_path = checkpoint_dir / "last.pth"
    start, best = 1, -math.inf
    resume = bool(getattr(cfg.run, "resume_dir", ""))
    if last_path.is_file():
        if not resume:
            raise FileExistsError(
                "A checkpoint already exists; set run.resume_dir to resume"
            )
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("format") != FORMAT:
            raise ValueError("Unsupported checkpoint format")
        previous = OmegaConf.create(saved["hparams"])
        for key in (
            "model",
            "dataset",
            "train.loss",
            "train.fitting",
            "train.batch_size",
            "train.epochs",
            "train.seed",
        ):
            if OmegaConf.select(previous, key) != OmegaConf.select(cfg, key):
                raise ValueError(f"Resume configuration differs at {key}")
        model.load_state_dict(saved["training_state_dict"], strict=True)
        learner.optimizer.load_state_dict(saved["optimizer"])
        learner.reference, learner.fixed_gain = saved["reference"], saved["fixed_gain"]
        start, best = saved["epoch"] + 1, saved["best_monitor_accuracy"]
    elif resume:
        raise FileNotFoundError(last_path)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    # Cache loaders MUST be sequential: memberships and targets index these rows.
    from torch.utils.data import DataLoader
    from src.data.dataloader import build_dataset

    monitor = str(options.monitor_split)
    if monitor not in {"val", "test"}:
        raise ValueError("monitor_split must be val or test")
    caches = {}
    for split in dict.fromkeys(("train", monitor)):
        dataset = build_dataset(cfg, split)
        if len(dataset.classes) != model.n_classes:
            raise ValueError(
                f"{split} has {len(dataset.classes)} classes, expected {model.n_classes}"
            )
        smoke_limit = int(getattr(cfg.run, "smoke_limit", 0))
        if smoke_limit:
            from torch.utils.data import Subset

            classes = dataset.classes
            dataset = Subset(dataset, range(min(smoke_limit, len(dataset))))
            dataset.classes = classes
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.eval.batch_size),
            shuffle=False,
            num_workers=int(cfg.train.num_workers),
        )
        caches[split] = cache_split(
            model,
            loader,
            cfg,
            run_dir / "cache" / f"{split}.pt",
            noisy=split == "train",
        )
    if resume:
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(start, int(cfg.train.epochs) + 1):
        train = caches["train"]
        if epoch == 1:
            warmup(model, train, cfg)
        else:
            order = torch.randperm(
                len(train["labels"]),
                generator=torch.Generator().manual_seed(int(cfg.train.seed) + epoch),
            )
            batches = list(order.split(int(cfg.train.batch_size)))
            for step, query in enumerate(batches, 1):
                keep = torch.ones(len(train["labels"]), dtype=torch.bool)
                keep[query] = False
                support = keep.nonzero(as_tuple=True)[0]
                if not len(support):
                    raise ValueError(
                        "Query batch leaves no fitting support; reduce train.batch_size"
                    )
                fraction = fit_fraction(
                    epoch,
                    step,
                    len(batches),
                    int(options.discovery_until),
                    int(options.fitted_by),
                )
                record = learner.step(train, query, support, fraction, epoch)
                LOG.info("epoch %d step %d/%d %s", epoch, step, len(batches), record)
        bank = detached(model.bank())
        fit_report = None
        if learner.reference is not None:
            features = extract(model, train, batch_size=int(options.support_microbatch))
            bank, fit_report = refit(
                model,
                features,
                train,
                bank,
                learner.reference["features"],
                learner.reference["bank"]["means"].to(device),
                options,
            )
        accuracy = cache_accuracy(
            model, caches[monitor], bank, int(cfg.eval.batch_size)
        )
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        exported = dict(state)
        exported["prototype_mu"] = bank["means"].cpu().float()
        k = bank["kappa"].cpu()
        exported["kappa_raw"] = (k + torch.log(-torch.expm1(-k))).float()
        improved = accuracy > best
        best = max(best, accuracy)
        payload = dict(
            format=FORMAT,
            epoch=epoch,
            state_dict=exported,
            training_state_dict=state,
            optimizer=learner.optimizer.state_dict(),
            hparams=OmegaConf.to_container(cfg, resolve=True),
            reference=learner.reference,
            fixed_gain=learner.fixed_gain,
            best_monitor_accuracy=best,
            monitor=dict(
                split=monitor, accuracy_pct=accuracy, test_selected=monitor == "test"
            ),
            refit=fit_report,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            inference="Winner-before-pi all-foreground exponential evidence; stable log arithmetic",
        )
        payload["partial"] = bool(getattr(cfg.run, "smoke_limit", 0))
        temporary = last_path.with_suffix(".pth.tmp")
        torch.save(payload, temporary)
        temporary.replace(last_path)
        if improved:
            temporary = checkpoint_dir / "best.pth.tmp"
            torch.save(payload, temporary)
            temporary.replace(checkpoint_dir / "best.pth")
        with (run_dir / "history.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        epoch=epoch,
                        monitor_split=monitor,
                        accuracy_pct=accuracy,
                        refit=fit_report,
                    )
                )
                + "\n"
            )
        LOG.info(
            "epoch %d %s accuracy %.3f%%; last.pth saved", epoch, monitor, accuracy
        )
    return last_path
