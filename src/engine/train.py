"""Training loop for vMF-MP experiments."""

from __future__ import annotations

import logging
import json
import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from omegaconf import OmegaConf

from data.dataloader import build_dataloader, build_dataset
from losses.api import build_loss
from models.api import build_model
from optim.api import build_optimizer_and_scheduler
from utils import io as run_io
from utils.logging import Logger, init_logger


def _configure_python_logger(run_dir: Path) -> logging.Logger:
    """Configure a basic file+stdout logger."""

    log_dir = run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("train")


def _configure_torch_runtime(cfg: Any, logger: logging.Logger) -> None:
    """Apply runtime knobs for performance tuning."""

    precision = str(getattr(cfg.train, "matmul_precision", "high"))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)
        logger.info("Matmul precision: %s", precision)

    torch.backends.cudnn.benchmark = bool(getattr(cfg.train, "cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(
        getattr(cfg.train, "deterministic", False)
    )
    logger.info(
        "cuDNN benchmark=%s deterministic=%s",
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )


def _resolve_amp(
    cfg: Any, device: torch.device
) -> tuple[bool, torch.dtype, Optional[GradScaler]]:
    """Resolve autocast settings and return an optional GradScaler."""

    amp_enabled = bool(getattr(cfg.train.amp, "enabled", False))
    dtype_name = str(getattr(cfg.train.amp, "dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    use_scaler = amp_enabled and device.type == "cuda" and amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)
    return amp_enabled, amp_dtype, scaler


def _autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype):
    if not amp_enabled:
        return nullcontext()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, dtype=amp_dtype)


def _log_attention_backend(logger: logging.Logger) -> None:
    """Log whether fast attention backends are available."""

    try:
        import xformers  # noqa: F401

        logger.info("xFormers detected: fast attention kernels enabled.")
    except Exception:
        logger.info("xFormers not detected: using standard attention kernels.")


def _freedman_diaconis_bins(values) -> int:
    """Return a reasonable histogram bin count via the Freedman–Diaconis rule.

    Matches the "paper_hist" style reference used in this project.
    """

    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return 10
    q25, q75 = np.percentile(x, [25.0, 75.0])
    iqr = float(q75 - q25)
    if iqr <= 0:
        return int(np.ceil(np.sqrt(n)))
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if bin_width <= 0 or not np.isfinite(bin_width):
        return int(np.ceil(np.sqrt(n)))
    return max(5, int(np.ceil((float(x.max()) - float(x.min())) / bin_width)))


def _save_paper_histogram(
    values: torch.Tensor,
    *,
    out_base: Path,
    xlabel: str,
    color: str,
    uniform_line: float | None = None,
) -> tuple[Path, Path]:
    """Save a histogram with the project’s paper style as (PDF, PNG)."""

    import logging
    import numpy as np
    import matplotlib as mpl
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    out_base.parent.mkdir(parents=True, exist_ok=True)

    x = values.detach().to(dtype=torch.float32, device="cpu").flatten().numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Histogram values are empty or non-finite.")

    bins = _freedman_diaconis_bins(x)
    mean = float(np.mean(x))
    median = float(np.median(x))
    q05, q95 = np.percentile(x, [5.0, 95.0])

    rcparams = {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with mpl.rc_context(rcparams):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

        # Build a histogram explicitly so bars are *centered* at their bin centers
        # (i.e., the middle of each bar corresponds to the bin's representative x-value).
        counts, edges = np.histogram(x, bins=bins, density=False)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        ax.bar(
            centers,
            counts,
            width=widths,
            align="center",
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )

        ax.axvspan(float(q05), float(q95), color="black", alpha=0.07, label="5–95%")
        ax.axvline(mean, color="black", linewidth=2.0, linestyle="-", label="mean")
        ax.axvline(median, color="black", linewidth=2.0, linestyle="--", label="median")
        if uniform_line is not None:
            ax.axvline(
                float(uniform_line),
                color="black",
                linewidth=2.0,
                linestyle=":",
                label=r"uniform $1/K$",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.grid(False, axis="x")
        ax.margins(x=0.02)

        # Avoid scientific notation / axis offsets like "1e-5 + 2e-1".
        formatter = mticker.ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.get_offset_text().set_visible(False)

        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

        out_pdf = out_base.with_suffix(".pdf")
        out_png = out_base.with_suffix(".png")

        # Matplotlib may emit very verbose INFO logs when font subsetting PDF
        # outputs (e.g., `fontTools.subset: ... pruned`). These are not useful
        # for experiment logs, so we silence them for this save only.
        #
        # NOTE: Some environments log via both `fontTools.subset` and the parent
        # `fontTools` logger; we silence both to be robust.
        fonttools_subset_logger = logging.getLogger("fontTools.subset")
        fonttools_logger = logging.getLogger("fontTools")
        prev_subset_level = fonttools_subset_logger.level
        prev_fonttools_level = fonttools_logger.level
        try:
            fonttools_subset_logger.setLevel(logging.WARNING)
            fonttools_logger.setLevel(logging.WARNING)
            fig.savefig(out_pdf, bbox_inches="tight")
            fig.savefig(out_png, bbox_inches="tight")
        finally:
            fonttools_subset_logger.setLevel(prev_subset_level)
            fonttools_logger.setLevel(prev_fonttools_level)
        plt.close(fig)

    return out_pdf, out_png


def _select_tracked_classes(
    *,
    run_dir: Path,
    class_names: list[str],
    seed: int,
    num_classes_to_track: int = 10,
) -> list[int]:
    """Pick a small fixed set of classes to track across epochs.

    This file is written once per run so the same classes are used at every epoch,
    enabling clean "π_c,k vs epoch" and "κ_c,k vs epoch" plots.
    """

    mech_dir = run_dir / "vis" / "mechanism"
    mech_dir.mkdir(parents=True, exist_ok=True)
    path = mech_dir / "tracked_classes.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [int(x) for x in payload.get("class_indices", [])]

    num_classes = len(class_names)
    count = min(int(num_classes_to_track), num_classes)
    generator = torch.Generator().manual_seed(int(seed))
    class_indices = torch.randperm(num_classes, generator=generator)[:count].tolist()
    payload = {
        "seed": int(seed),
        "class_indices": class_indices,
        "class_names": [class_names[i] for i in class_indices],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [int(x) for x in class_indices]


@torch.inference_mode()
def _log_vmf_mechanism_stats(
    *,
    model: nn.Module,
    dataloader_ref: torch.utils.data.DataLoader,
    tracked_classes: list[int],
    epoch: int,
    logger: logging.Logger,
    metric_logger: Logger,
    run_dir: Path,
    device: torch.device,
) -> None:
    """Log appendix-worthy mechanism stats for π and κ.

    We keep these logs lightweight and reproducible:
    - Scalars are logged every epoch.
    - Histogram PNGs are logged every epoch for easy inspection.
    - The "κ is not just a scale" sanity check uses a fixed reference batch from
      the (non-shuffled) test dataloader, so values are comparable across epochs.
    """

    vmf = model  # naming clarity
    if not hasattr(vmf, "pi_logits") or not hasattr(vmf, "kappa_raw") or not hasattr(vmf, "prototype_mu"):
        logger.warning("Skipping mechanism logs: model missing expected vMF-MP parameters.")
        return

    from models.vmf_mp.vmf_normalizer import logCp_vmf

    pi = F.softmax(vmf.pi_logits, dim=-1)
    kappa = F.softplus(vmf.kappa_raw)

    # Exclude the background class (the last one) for analysis plots.
    if hasattr(vmf, "bg_class"):
        bg_class = int(vmf.bg_class)
    else:
        bg_class = pi.shape[0] - 1

    keep_mask = torch.ones(pi.shape[0], device=pi.device, dtype=torch.bool)
    if 0 <= bg_class < keep_mask.numel():
        keep_mask[bg_class] = False

    pi_fg = pi[keep_mask]
    kappa_fg = kappa[keep_mask]

    # π entropy and effective number of prototypes (per class).
    eps = 1.0e-8
    ent = -(pi_fg * (pi_fg + eps).log()).sum(dim=-1)  # (C,)
    k_eff = ent.exp()

    def _q(x: torch.Tensor, q: float) -> float:
        return float(torch.quantile(x.to(dtype=torch.float32), q).item())

    metric_logger.log_scalar("pi/entropy_mean", float(ent.mean().item()), step=epoch, split="mech")
    metric_logger.log_scalar("pi/entropy_median", float(ent.median().item()), step=epoch, split="mech")
    metric_logger.log_scalar("pi/entropy_p10", _q(ent, 0.10), step=epoch, split="mech")
    metric_logger.log_scalar("pi/entropy_p90", _q(ent, 0.90), step=epoch, split="mech")

    metric_logger.log_scalar("pi/k_eff_mean", float(k_eff.mean().item()), step=epoch, split="mech")
    metric_logger.log_scalar("pi/k_eff_median", float(k_eff.median().item()), step=epoch, split="mech")
    metric_logger.log_scalar("pi/k_eff_p10", _q(k_eff, 0.10), step=epoch, split="mech")
    metric_logger.log_scalar("pi/k_eff_p90", _q(k_eff, 0.90), step=epoch, split="mech")

    # κ summary stats.
    metric_logger.log_scalar("kappa/mean", float(kappa_fg.mean().item()), step=epoch, split="mech")
    metric_logger.log_scalar("kappa/p05", _q(kappa_fg, 0.05), step=epoch, split="mech")
    metric_logger.log_scalar("kappa/p50", _q(kappa_fg, 0.50), step=epoch, split="mech")
    metric_logger.log_scalar("kappa/p95", _q(kappa_fg, 0.95), step=epoch, split="mech")

    # Track π_c and κ_c for a small fixed set of classes (line plots in W&B).
    for c in tracked_classes:
        if c < 0 or c >= pi_fg.shape[0]:
            continue
        for k in range(pi_fg.shape[1]):
            metric_logger.log_scalar(f"pi/class_{c}/k{k}", float(pi_fg[c, k].item()), step=epoch, split="mech")
            metric_logger.log_scalar(
                f"kappa/class_{c}/k{k}", float(kappa_fg[c, k].item()), step=epoch, split="mech"
            )

    # "Is κ just a scale?" sanity check using a fixed reference batch.
    ref_batch = next(iter(dataloader_ref))
    ref_batch = tuple(item.to(device, non_blocking=True) for item in ref_batch)
    images, labels = ref_batch[:2]

    backbone_out = vmf.backbone(images)
    if not isinstance(backbone_out, tuple):
        return
    if len(backbone_out) == 3:
        patch_tokens, _, _ = backbone_out
    elif len(backbone_out) == 2:
        patch_tokens, _ = backbone_out
    else:
        return

    mu_hat = F.normalize(vmf.prototype_mu, p=2, dim=-1)
    tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
    cos_logits = torch.einsum("bnd,ckd->bnck", tokens_norm, mu_hat)

    # Only measure magnitudes for the *correct class* per image (makes the numbers interpretable).
    c_idx = labels.to(dtype=torch.long).clamp_min(0).clamp_max(mu_hat.shape[0] - 2)  # exclude bg
    cos_c = cos_logits[torch.arange(c_idx.numel(), device=device), :, c_idx, :]  # (B, N, K)
    kappa_c = kappa_fg[c_idx]  # (B, K)
    logC_c = logCp_vmf(vmf.dim, kappa_c)  # (B, K) centered

    term1 = (kappa_c.unsqueeze(1) * cos_c).abs().mean()
    term2 = logC_c.abs().mean()
    ratio = (term1 / (term2 + 1.0e-8)).item()

    metric_logger.log_scalar("vmf_terms/abs_kappa_cos_mean", float(term1.item()), step=epoch, split="mech")
    metric_logger.log_scalar("vmf_terms/abs_logC_mean", float(term2.item()), step=epoch, split="mech")
    metric_logger.log_scalar("vmf_terms/ratio", float(ratio), step=epoch, split="mech")

    mech_dir = run_dir / "vis" / "mechanism"
    k_proto = int(pi_fg.shape[1])
    _, pi_png = _save_paper_histogram(
        pi_fg,
        out_base=mech_dir / f"hist_pi_epoch{epoch:03d}",
        xlabel=r"Mixing weight $\pi_i$",
        color="#4C78A8",
        uniform_line=1.0 / max(k_proto, 1),
    )
    _, kappa_png = _save_paper_histogram(
        kappa_fg,
        out_base=mech_dir / f"hist_kappa_epoch{epoch:03d}",
        xlabel=r"Concentration $\kappa_i$",
        color="#F58518",
        uniform_line=None,
    )
    metric_logger.log_image(f"pi_hist_epoch{epoch:03d}", pi_png, step=epoch)
    metric_logger.log_image(f"kappa_hist_epoch{epoch:03d}", kappa_png, step=epoch)


def _load_prototypes_if_available(
    model: nn.Module, cfg: Any, logger: logging.Logger
) -> None:
    """Optionally load precomputed prototypes into the model."""

    path = getattr(cfg.train, "prototypes_init_path", "")
    if not path:
        return

    ckpt = torch.load(path, map_location="cpu")
    candidate_keys = ["prototype_mu", "prototypes", "prototype_mu_init"]
    proto = None
    for key in candidate_keys:
        if key in ckpt:
            proto = ckpt[key]
            break
    if proto is None:
        logger.warning("No prototype tensor found in %s; skipping init.", path)
        return

    with torch.no_grad():
        model.prototype_mu.copy_(proto)
    logger.info("Loaded prototype initialization from %s", path)


def _save_checkpoint(model: nn.Module, cfg: Any, path: Path, *, epoch: int) -> None:
    """Serialize the current model state to the given path."""

    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "hparams": OmegaConf.to_container(cfg, resolve=True),
        # Epoch index of the checkpoint (0-based). This enables robust resume
        # without relying on log parsing.
        "epoch": int(epoch),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _infer_last_epoch_from_history(run_dir: Path) -> Optional[int]:
    """Best-effort: return max `step` in history.csv (epochs are logged as step)."""

    path = run_dir / "history.csv"
    if not path.exists():
        return None
    max_step: Optional[int] = None
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = row.get("step")
                if step in (None, "", "None"):
                    continue
                try:
                    value = int(float(step))
                except Exception:
                    continue
                max_step = value if max_step is None else max(max_step, value)
    except Exception:
        return None
    return max_step


def _load_resume_epoch(run_dir: Path) -> Optional[int]:
    """Return the last completed epoch index from checkpoints/history (or None)."""

    ckpt_path = run_dir / "checkpoints" / "last.pth"
    if not ckpt_path.exists():
        return None

    try:
        payload = torch.load(ckpt_path, map_location="cpu")
        epoch = payload.get("epoch")
        if epoch is not None:
            return int(epoch)
    except Exception:
        pass

    # Backward compatibility: older checkpoints didn't store epoch, so fall back
    # to the on-disk history file.
    return _infer_last_epoch_from_history(run_dir)


def _train_one_epoch(
    model: nn.Module,
    base_model: nn.Module,
    criterion: Optional[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    optimizer: Optional[optim.Optimizer],
    logger: logging.Logger,
    metric_logger: Logger,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: Optional[GradScaler],
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """Run a single training epoch and return aggregated metrics."""

    model.train()
    running_losses: Dict[str, float] = {}
    correct = torch.zeros(1, device=device, dtype=torch.float64)
    total = torch.zeros(1, device=device, dtype=torch.float64)

    grad_accum_steps = max(1, int(grad_accum_steps))
    use_scaler = scaler is not None and scaler.is_enabled()
    autocast_ctx = _autocast_context(device, amp_enabled, amp_dtype)
    grad_enabled = optimizer is not None
    num_batches = len(dataloader)
    last_step = max(0, num_batches - 1)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc=f"train {epoch}")):
        batch = tuple(item.to(device, non_blocking=True) for item in batch)
        images, labels = batch[:2]
        if criterion is not None:
            criterion.set_epoch(epoch, step=step)

        loss_dict: Optional[Dict[str, torch.Tensor]] = None
        outputs: Dict[str, Any]

        is_last_micro = ((step + 1) % grad_accum_steps == 0) or (step == last_step)
        with torch.set_grad_enabled(grad_enabled):
            with autocast_ctx:
                outputs = model(images, labels=labels)
                if criterion is not None and optimizer is not None:
                    loss_dict = criterion(outputs, batch)
                    loss = sum(
                        val for key, val in loss_dict.items() if not key.startswith("_")
                    )
                else:
                    loss = None

        if loss is not None and optimizer is not None:
            loss = loss / grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if is_last_micro:
                if use_scaler:
                    scaler.unscale_(optimizer)
                if getattr(base_model, "parametric_prototypes", False):
                    proto_param = getattr(base_model, "prototype_mu", None)
                    if (
                        isinstance(proto_param, nn.Parameter)
                        and proto_param.requires_grad
                    ):
                        clip_grad_norm_([proto_param], max_norm=1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if loss_dict is not None:
            batch_size = labels.size(0)
            for key, val in loss_dict.items():
                running_losses[key] = (
                    running_losses.get(key, 0.0) + val.item() * batch_size
                )

        preds = outputs["class_logits"].argmax(dim=1)
        correct += (preds == labels).sum().to(dtype=torch.float64)
        total += torch.tensor(labels.numel(), device=device, dtype=torch.float64)

    metrics: Dict[str, float] = {}
    if total.item() > 0:
        for key, total_loss in running_losses.items():
            loss_tensor = torch.tensor(total_loss, device=device, dtype=torch.float64)
            metrics[key] = (loss_tensor / total).item()
            logger.info("EPOCH %d train %s: %.4f", epoch, key, metrics[key])
            if not key.startswith("_"):
                metric_logger.log_scalar(key, metrics[key], step=epoch, split="train")

        train_acc = (correct / total).item()
    else:
        train_acc = 0.0

    metrics["acc"] = train_acc
    logger.info("EPOCH %d train acc: %.4f", epoch, train_acc)
    metric_logger.log_scalar("acc", train_acc, step=epoch, split="train")
    return metrics


@torch.inference_mode()
def _evaluate_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    logger: logging.Logger,
    metric_logger: Logger,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Evaluate classification accuracy on the test split."""

    model.eval()
    correct = torch.zeros(1, device=device, dtype=torch.float64)
    total = torch.zeros(1, device=device, dtype=torch.float64)
    autocast_ctx = _autocast_context(device, amp_enabled, amp_dtype)

    for _, batch in enumerate(tqdm(dataloader, desc=f"test {epoch}")):
        batch = tuple(item.to(device, non_blocking=True) for item in batch)
        images, labels = batch[:2]
        with autocast_ctx:
            outputs = model(images)
        preds = outputs["class_logits"].argmax(dim=1)
        correct += (preds == labels).sum().to(dtype=torch.float64)
        total += torch.tensor(labels.numel(), device=device, dtype=torch.float64)

    test_acc = (correct / total).item() if total.item() > 0 else 0.0
    logger.info("EPOCH %d test acc: %.4f", epoch, test_acc)
    metric_logger.log_scalar("acc", test_acc, step=epoch, split="test")
    return test_acc


def train_and_eval(cfg: Any, run_dir: Path) -> Path:
    """Train the model, write artifacts, and return the final checkpoint path."""
    if str(getattr(cfg.model, "recipe", "legacy")) == "four_loss":
        from .fitted import train_fitted
        return train_fitted(cfg, run_dir)

    device = torch.device(str(cfg.train.device))
    py_logger = _configure_python_logger(run_dir)
    metric_logger = init_logger(cfg, run_dir)

    _configure_torch_runtime(cfg, py_logger)
    _log_attention_backend(py_logger)

    last_ckpt_path = run_dir / "checkpoints" / "last.pth"
    resume_epoch = _load_resume_epoch(run_dir)
    start_epoch = int(resume_epoch + 1) if resume_epoch is not None else 0
    if start_epoch > 0:
        py_logger.info(
            "Resuming training from run_dir=%s (last completed epoch=%d -> next=%d).",
            str(run_dir),
            int(resume_epoch),
            int(start_epoch),
        )

    seed = int(cfg.train.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_train = build_dataset(cfg, split="train")
    dataset_test = build_dataset(cfg, split="test")
    dataloader_train = build_dataloader(cfg, split="train", dataset=dataset_train)
    dataloader_test = build_dataloader(cfg, split="test", dataset=dataset_test)

    tracked_classes = _select_tracked_classes(
        run_dir=run_dir,
        class_names=list(dataset_train.classes),
        seed=int(cfg.train.seed),
        num_classes_to_track=10,
    )

    py_logger.info(
        "Data: train=%d test=%d batch=%d workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        len(dataset_train),
        len(dataset_test),
        int(cfg.train.batch_size),
        int(cfg.train.num_workers),
        bool(getattr(cfg.train, "pin_memory", True)),
        bool(getattr(cfg.train, "persistent_workers", False)),
        int(getattr(cfg.train, "prefetch_factor", 2)),
    )
    py_logger.info("Dataset root: %s", str(cfg.dataset.root))

    model = build_model(cfg)
    model.to(device)

    # Load weights when resuming from an existing run directory.
    if resume_epoch is not None and last_ckpt_path.exists():
        try:
            payload = torch.load(last_ckpt_path, map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                py_logger.warning("Resume checkpoint missing keys: %s", missing)
            if unexpected:
                py_logger.warning("Resume checkpoint unexpected keys: %s", unexpected)
            py_logger.info("Loaded checkpoint weights from %s", str(last_ckpt_path))
        except Exception as exc:
            py_logger.warning(
                "Failed to load resume checkpoint %s; restarting from scratch. %s",
                str(last_ckpt_path),
                exc,
            )
            resume_epoch = None
            start_epoch = 0

    amp_enabled, amp_dtype, scaler = _resolve_amp(cfg, device)
    py_logger.info(
        "AMP enabled=%s dtype=%s scaler=%s",
        amp_enabled,
        amp_dtype,
        bool(scaler is not None and scaler.is_enabled()),
    )

    compile_enabled = bool(getattr(cfg.train.compile, "enabled", False))
    if compile_enabled and bool(getattr(cfg.train, "stage_2_proto_learn", False)):
        py_logger.info(
            "torch.compile skipped because stage_2_proto_learn updates parameters at runtime."
        )
        compile_enabled = False
    fg_method = str(getattr(cfg.model.fg, "method", "")).lower()
    if compile_enabled and fg_method == "attn":
        py_logger.info(
            "torch.compile skipped because attention-based foreground extraction uses hooks."
        )
        compile_enabled = False

    if compile_enabled:
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode=str(cfg.train.compile.mode))
                py_logger.info(
                    "torch.compile enabled (mode=%s).", str(cfg.train.compile.mode)
                )
            except Exception as exc:
                py_logger.warning("torch.compile failed; skipping. %s", exc)
        else:
            py_logger.warning("torch.compile is not available in this PyTorch build.")

    base_model = model
    if start_epoch == 0:
        _load_prototypes_if_available(base_model, cfg, py_logger)

    criterion = build_loss(cfg)
    if criterion is not None:
        criterion.to(device)

    grad_accum_steps = int(getattr(cfg.train, "grad_accum_steps", 1))
    if grad_accum_steps > 1:
        py_logger.info("Gradient accumulation steps: %d", grad_accum_steps)
    effective_batch = int(cfg.train.batch_size) * max(1, grad_accum_steps)
    py_logger.info("Effective global batch size: %d", effective_batch)

    best_acc = float("-inf")
    best_epoch = -1
    if start_epoch > 0:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                best_acc = float(metrics.get("summary", {}).get("best/acc", best_acc))
                best_epoch = int(metrics.get("best", {}).get("step", best_epoch) or best_epoch)
            except Exception:
                pass
    epochs = int(cfg.train.epochs)
    last_epoch = max(0, epochs - 1)
    eval_every = int(getattr(cfg.train, "eval_every_n_epochs", 1))
    ckpt_every = int(getattr(cfg.train, "checkpoint_every_n_epochs", 1))

    for epoch in range(start_epoch, epochs):
        is_fine_tuning = epoch >= int(cfg.train.fine_tuning_start_epoch)

        if is_fine_tuning:
            if cfg.train.stage_2_proto_learn and not base_model.parametric_prototypes:
                base_model.enable_parametric_prototypes()

            for name, param in base_model.named_parameters():
                param.requires_grad = ("backbone" not in name) and (
                    "fg_extractor" not in name
                )
            base_model.backbone.set_requires_grad()

            optimizer, _ = build_optimizer_and_scheduler(cfg, base_model)
            base_model.optimizing_prototypes = not base_model.parametric_prototypes
        else:
            for param in base_model.parameters():
                param.requires_grad = False
            optimizer = None
            base_model.optimizing_prototypes = True

        if criterion is not None:
            criterion.set_epoch(epoch)

        _train_one_epoch(
            model=model,
            base_model=base_model,
            criterion=criterion if is_fine_tuning else None,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer if is_fine_tuning else None,
            logger=py_logger,
            metric_logger=metric_logger,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
        )

        run_eval = eval_every > 0 and (
            (epoch + 1) % eval_every == 0 or epoch == last_epoch
        )
        test_acc = None
        if run_eval:
            test_acc = _evaluate_accuracy(
                model=model,
                dataloader=dataloader_test,
                epoch=epoch,
                logger=py_logger,
                metric_logger=metric_logger,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        run_ckpt = ckpt_every > 0 and (
            (epoch + 1) % ckpt_every == 0 or epoch == last_epoch
        )

        is_best = run_eval and test_acc is not None and test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch

        if run_ckpt or is_best:
            _save_checkpoint(model, cfg, last_ckpt_path, epoch=epoch)
            metric_logger.log_checkpoint(last_ckpt_path, alias="last")

            if is_best:
                metric_logger.log_checkpoint(last_ckpt_path, alias="best")

        if run_eval and test_acc is not None:
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                metrics = run_io.default_metrics()
            primary_metric = str(getattr(cfg.eval, "primary_metric", "test/acc"))
            metrics["primary_metric"] = primary_metric
            metrics.setdefault("summary", {})
            metrics["summary"].update({"test/acc": test_acc, "best/acc": best_acc})
            best_metrics = metrics.get("best", {}).get("metrics", {})
            best_value = best_metrics.get(primary_metric, float("-inf"))
            if test_acc >= best_value:
                metrics["best"] = {
                    "step": epoch,
                    "metrics": {primary_metric: test_acc},
                    "checkpoint_path": str(
                        (run_dir / "checkpoints" / "best.pth").resolve()
                    ),
                }
            metrics_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metric_logger.log_metrics_summary(
                summary={"test/acc": test_acc, "best/acc": best_acc},
                step=epoch,
            )

        _log_vmf_mechanism_stats(
            model=base_model,
            dataloader_ref=dataloader_test,
            tracked_classes=tracked_classes,
            epoch=epoch,
            logger=py_logger,
            metric_logger=metric_logger,
            run_dir=run_dir,
            device=device,
        )

    py_logger.info("DONE! Best epoch %d with acc %.4f", best_epoch, best_acc)
    metric_logger.finish(status="completed")

    return last_ckpt_path
