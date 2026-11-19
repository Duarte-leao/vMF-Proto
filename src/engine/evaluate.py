"""Evaluation loop for vMF-MP experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data.dataloader import build_dataloader
from eval.types import EvalResult
from eval.metrics import (
    evaluate_comprehensiveness,
    evaluate_consistency,
    evaluate_distinctiveness,
    evaluate_pcs_auc,
    evaluate_pfr_topk,
    evaluate_stability,
)
from models.api import build_model
from utils.logging import init_logger


def _configure_python_logger(run_dir: Path) -> logging.Logger:
    """Configure a file+stdout logger for eval metrics."""

    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((run_dir / "eval.log").as_posix()),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("evaluate")


def _load_checkpoint_path(cfg: Any, run_dir: Path) -> Path:
    """Resolve which checkpoint path should be evaluated."""

    ckpt_override = getattr(cfg.eval, "checkpoint_path", None)
    if ckpt_override:
        return Path(str(ckpt_override))

    alias = str(getattr(cfg.eval, "checkpoint_alias", "best"))
    candidate = run_dir / "checkpoints" / f"{alias}.pth"
    if candidate.exists():
        return candidate
    return run_dir / "checkpoints" / "last.pth"


def _load_eval_cfg(cfg: Any, ckpt: Dict[str, Any]) -> Any:
    """Optionally rebuild config from the checkpoint payload."""

    if bool(getattr(cfg.eval, "use_checkpoint_config", True)) and "hparams" in ckpt:
        ckpt_cfg = OmegaConf.create(ckpt["hparams"])
        metrics_override = OmegaConf.select(cfg, "eval.metrics")
        if metrics_override is not None:
            OmegaConf.update(ckpt_cfg, "eval.metrics", metrics_override, merge=False)
        for key in ("eval.pfr_topk", "eval.pcs_topk"):
            value = OmegaConf.select(cfg, key)
            if value is not None:
                OmegaConf.update(ckpt_cfg, key, value, merge=False)
        return ckpt_cfg
    return cfg


@torch.inference_mode()
def _eval_accuracy(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    """Compute classification accuracy for the test split."""

    model.eval()
    mca = MulticlassAccuracy(
        num_classes=len(dataloader.dataset.classes), average="micro"
    ).to(device)

    for batch in tqdm(dataloader, desc="eval acc"):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]
        outputs = model(images)
        mca(outputs["class_logits"], labels)

    return mca.compute().item()


def evaluate_run(cfg: Any, run_dir: Path) -> EvalResult:
    """Evaluate a trained model and update run artifacts."""

    run_dir = Path(run_dir)
    # Inspect the saved recipe, not the caller's default dataset/model config.
    checkpoint_path = _load_checkpoint_path(cfg, run_dir)
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("format") == "vmfproto/early_four_loss/v1":
            from src.eval.metrics.fitted import evaluate_fitted
            return evaluate_fitted(cfg, run_dir, checkpoint_path, saved)
    py_logger = _configure_python_logger(run_dir)
    ckpt_path = _load_checkpoint_path(cfg, run_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    eval_cfg = _load_eval_cfg(cfg, ckpt)

    device = torch.device(str(eval_cfg.train.device))
    model = build_model(eval_cfg)

    state_dict = ckpt.get("state_dict", ckpt)
    if any(key.startswith("prototype_mu_init") for key in state_dict):
        model.enable_parametric_prototypes()
    model.load_state_dict(state_dict, strict=False)
    model.optimizing_prototypes = False
    model.initializing = False  # type: ignore[attr-defined]
    model.to(device)

    dataloader_test = build_dataloader(eval_cfg, split="test")
    logger = init_logger(eval_cfg, run_dir)

    summary: Dict[str, float] = {}
    # summary["test/acc"] = _eval_accuracy(model, dataloader_test, device)
    # logger.log_scalar("acc", summary["test/acc"], split="test")
    # py_logger.info("Accuracy: %.4f", summary["test/acc"])

    metrics = set(str(m).lower() for m in eval_cfg.eval.metrics)
    dataset_name = str(getattr(eval_cfg.dataset, "name", "")).lower()
    if "consistency" in metrics or "stability" in metrics:
        model.img_size = int(eval_cfg.dataset.input_size[0])
        model.num_prototypes_per_class = model.n_prototypes

    if "consistency" in metrics:
        summary["consistency"] = float(
            evaluate_consistency(
                model,
                OmegaConf.create(
                    {
                        "nb_classes": eval_cfg.dataset.num_classes,
                        "data_path": eval_cfg.dataset.root,
                        "test_batch_size": eval_cfg.eval.batch_size,
                    }
                ),
                save_dir=run_dir,
                device=device,
            )
        )
        logger.log_scalar("consistency", summary["consistency"], split="eval")
        py_logger.info("Consistency: %.4f", summary["consistency"])
    if "stability" in metrics:
        summary["stability"] = float(
            evaluate_stability(
                model,
                OmegaConf.create(
                    {
                        "nb_classes": eval_cfg.dataset.num_classes,
                        "data_path": eval_cfg.dataset.root,
                        "test_batch_size": eval_cfg.eval.batch_size,
                    }
                ),
                device=device,
            )
        )
        logger.log_scalar("stability", summary["stability"], split="eval")
        py_logger.info("Stability: %.4f", summary["stability"])
    if "distinctiveness" in metrics:
        summary["distinctiveness"] = float(
            evaluate_distinctiveness(
                model,
                eval_cfg.dataset.root,
                device=device,
            )
        )
        logger.log_scalar("distinctiveness", summary["distinctiveness"], split="eval")
        py_logger.info("Distinctiveness: %.4f", summary["distinctiveness"])
    if "comprehensiveness" in metrics:
        summary["comprehensiveness"] = float(
            evaluate_comprehensiveness(
                model,
                eval_cfg.dataset.root,
                device=device,
                segmentations_dir=eval_cfg.dataset.segmentations_dir,
            )
        )
        logger.log_scalar(
            "comprehensiveness", summary["comprehensiveness"], split="eval"
        )
        py_logger.info("Comprehensiveness: %.4f", summary["comprehensiveness"])

    if "pfr_topk" in metrics:
        if dataset_name != "cub200":
            py_logger.warning(
                "PFR-TopK requires CUB segmentations; skipping for dataset=%s.",
                dataset_name or "unknown",
            )
        else:
            topk = int(getattr(eval_cfg.eval, "pfr_topk", 1))
            summary["pfr_topk"] = float(
                evaluate_pfr_topk(
                    model,
                    eval_cfg.dataset.root,
                    topk=topk,
                    num_classes=eval_cfg.dataset.num_classes,
                    device=device,
                    segmentations_dir=eval_cfg.dataset.segmentations_dir,
                )
            )
            logger.log_scalar("pfr_topk", summary["pfr_topk"], split="eval")
            py_logger.info("PFR-TopK (k=%d): %.4f", topk, summary["pfr_topk"])

    if "pcs_auc" in metrics:
        topk = int(getattr(eval_cfg.eval, "pcs_topk", 1))
        summary["pcs_auc"] = float(
            evaluate_pcs_auc(
                model,
                dataloader_test,
                topk=topk,
                num_classes=eval_cfg.dataset.num_classes,
                device=device,
            )
        )
        logger.log_scalar("pcs_auc", summary["pcs_auc"], split="eval")
        py_logger.info("PCS AUC (k=%d): %.4f", topk, summary["pcs_auc"])

    logger.log_metrics_summary(summary)
    logger.finish(status="completed")
    return EvalResult(summary=summary, sample_paths=None)
