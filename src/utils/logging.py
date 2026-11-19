"""Project logging utilities and W&B integration (optional).

This module implements the required logging contract defined in the template
and provides a small `Logger` wrapper used by the training/eval engines.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from . import io as run_io


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _none_if_blank(value: Any) -> Optional[str]:
    """Return `None` for empty/whitespace strings, else a stripped string."""

    text = str(value).strip()
    return text or None


def _safe_run(cmd: list[str]) -> str:
    """Run a small git command and return stdout (or 'unknown' on failure)."""

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Collect git commit/branch/dirty status for provenance."""

    commit = _safe_run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    branch = _safe_run(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_flag = _safe_run(["git", "-C", str(repo_root), "status", "--porcelain"])
    dirty = bool(dirty_flag and dirty_flag != "unknown")
    return {"commit": commit, "dirty": dirty, "branch": branch}


def _env_info() -> Dict[str, Any]:
    """Collect a lightweight environment snapshot."""

    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"

    return {
        "python": sys.version.split(" ")[0],
        "pytorch": torch_version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


def _dataset_snapshot(cfg: Any) -> Dict[str, Any]:
    """Capture dataset metadata for `meta.json`."""

    dataset_name = str(getattr(cfg.dataset, "name", "")).lower()
    train_split = getattr(cfg.dataset, "train_dir", None)
    test_split = getattr(cfg.dataset, "test_dir", None)
    val_split = getattr(cfg.dataset, "val_dir", None)
    if train_split is None:
        train_split = getattr(cfg.dataset, "train_list", "")
    if test_split is None:
        test_split = getattr(cfg.dataset, "test_list", "")
    if val_split is None:
        val_split = getattr(cfg.dataset, "val_list", "")

    return {
        "name": str(getattr(cfg.dataset, "name", "")),
        "root": str(getattr(cfg.dataset, "root", "")),
        "cropped_dir": str(getattr(cfg.dataset, "cropped_dir", "")),
        "segmentations_dir": str(getattr(cfg.dataset, "segmentations_dir", "")),
        "images_dir": str(getattr(cfg.dataset, "images_dir", "")),
        "split_dir": str(getattr(cfg.dataset, "split_dir", "")),
        "splits": {
            "train": str(train_split or ""),
            "val": str(val_split or ""),
            "test": str(test_split or ""),
        },
        "preprocessing": {
            "input_size": list(getattr(cfg.dataset, "input_size", [])),
            "normalize": bool(getattr(cfg.dataset, "normalize", True)),
        },
        "type": dataset_name,
    }


def _config_hash(resolved_yaml: str) -> str:
    """Return a short hash of the resolved config for provenance."""

    return hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()


def _write_history_row(path: Path, row: Dict[str, Any]) -> None:
    """Append a single row to `history.csv`, creating it when missing."""

    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_io.HISTORY_CSV_REQUIRED_COLUMNS))
        if needs_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in run_io.HISTORY_CSV_REQUIRED_COLUMNS})


@dataclass
class Logger:
    """Thin logger wrapper that writes local artifacts and optionally logs to W&B."""

    cfg: Any
    run_dir: Path
    wandb_run: Optional[Any] = None

    def log_scalar(self, name: str, value: Any, step: Optional[int] = None, *, split: str = "train") -> None:
        """Log a scalar to history.csv and W&B (if enabled)."""

        _write_history_row(
            self.run_dir / "history.csv",
            {
                "step": step,
                "split": split,
                "metric": name,
                "value": float(value),
                "timestamp": _now_iso(),
            },
        )
        if self.wandb_run is not None:
            key = f"{split}/{name}"
            self.wandb_run.log({key: value}, step=step)

    def log_metrics_summary(self, summary: Dict[str, Any], *, step: Optional[int] = None) -> None:
        """Update metrics.json summary and W&B summary."""

        metrics_path = self.run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = run_io.default_metrics()

        metrics.setdefault("summary", {})
        metrics["summary"].update(summary)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if self.wandb_run is not None:
            for key, value in summary.items():
                self.wandb_run.summary[key] = value
            if step is not None:
                self.wandb_run.summary["step"] = step

    def log_image(self, name: str, path: Path, *, step: Optional[int] = None) -> None:
        """Copy an image into the samples folder and optionally log to W&B."""

        target = self.run_dir / "samples" / f"{name}{path.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log({name: wandb.Image(str(target))}, step=step)

    def log_checkpoint(self, path: Path, *, alias: str) -> None:
        """Copy a checkpoint into the run directory under a stable alias."""

        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / f"{alias}.pth"
        target.write_bytes(path.read_bytes())

        if self.wandb_run is not None:
            import wandb

            # Use a run-scoped artifact name to avoid cross-run collisions.
            artifact = wandb.Artifact(f"{self.wandb_run.id}-checkpoint-{alias}", type="model")
            artifact.add_file(str(target))
            self.wandb_run.log_artifact(artifact, aliases=[alias])

    def finish(self, status: str) -> None:
        """Finalize a run by updating meta.json and closing W&B."""

        meta_path = self.run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = run_io.default_meta(run_dir=self.run_dir)

        meta["status"] = status
        meta["updated_at"] = _now_iso()
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if self.wandb_run is not None:
            self.wandb_run.finish()


def init_logger(cfg: Any, run_dir: Path) -> Logger:
    """Initialize logging, create artifacts, and (optionally) start W&B."""

    run_io.ensure_run_dir_structure(run_dir)

    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    (run_dir / "config.yaml").write_text(resolved_yaml, encoding="utf-8")

    py_logger = logging.getLogger("utils.logging")

    meta_path = run_dir / "meta.json"
    existing_meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}

    existing_wandb = existing_meta.get("wandb")
    existing_wandb_run_id: Optional[str] = None
    if isinstance(existing_wandb, dict):
        existing_wandb_run_id = existing_wandb.get("run_id")

    repo_root = Path(__file__).resolve().parents[2]
    meta = {
        "run_id": run_dir.name.split("_")[-1],
        "run_name": "_".join(run_dir.name.split("_")[:-1]) or run_dir.name,
        "date": run_dir.parent.name,
        "created_at": existing_meta.get("created_at", _now_iso()),
        "git": _git_info(repo_root),
        "env": _env_info(),
        "wandb": {
            "mode": str(cfg.wandb.mode),
            "entity": str(cfg.wandb.entity),
            "project": str(cfg.wandb.project),
            # Preserve the W&B run id for this run directory so eval can resume
            # the same run instead of creating a new one.
            "run_id": existing_wandb_run_id,
        },
        "config_hash": _config_hash(resolved_yaml),
        "dataset": _dataset_snapshot(cfg),
        "status": "running",
        "notes": str(cfg.run.note),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        run_io.write_json(metrics_path, run_io.default_metrics())
    # Ensure the primary metric is explicit for downstream tooling.
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    primary_metric = getattr(cfg.eval, "primary_metric", None)
    if primary_metric:
        metrics["primary_metric"] = str(primary_metric)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not (run_dir / "history.csv").exists():
        run_io.write_history_csv(run_dir / "history.csv", rows=[])

    wandb_run = None
    if str(cfg.wandb.mode) != "disabled":
        try:
            import wandb

            entity = _none_if_blank(getattr(cfg.wandb, "entity", ""))
            project = _none_if_blank(getattr(cfg.wandb, "project", "")) or _none_if_blank(
                getattr(cfg.project, "name", "")
            )
            if project is None:
                project = "vmf_mp"
            name = _none_if_blank(getattr(cfg.wandb, "name", "")) or str(cfg.run.name)
            group = _none_if_blank(getattr(cfg.wandb, "group", ""))
            tags = list(cfg.wandb.tags) if getattr(cfg.wandb, "tags", None) else []
            wandb_id = meta.get("wandb", {}).get("run_id")

            init_kwargs: Dict[str, Any] = {
                "entity": entity,
                "project": project,
                "mode": str(cfg.wandb.mode),
                "name": name,
                "group": group,
                "tags": tags,
                # Keep W&B artifacts within the run directory (avoids cluttering the repo root).
                "dir": str(run_dir),
            }
            if wandb_id:
                init_kwargs.update({"id": str(wandb_id), "resume": "allow"})
            else:
                init_kwargs.update({"config": OmegaConf.to_container(cfg, resolve=True)})

            wandb_run = wandb.init(**init_kwargs)
            if wandb_run is not None:
                meta["wandb"]["run_id"] = wandb_run.id
                meta_path.write_text(
                    json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        except Exception as exc:
            # W&B is optional; continue with local logging only.
            py_logger.warning(
                "W&B init failed (mode=%s). Falling back to local logging only. %s",
                cfg.wandb.mode,
                exc,
            )
            wandb_run = None

    return Logger(cfg=cfg, run_dir=run_dir, wandb_run=wandb_run)


def wandb_mode_help() -> str:
    """Return a short help string for W&B modes."""

    return (
        "W&B modes: online (log to server), offline (log locally; sync later), "
        "disabled (no W&B init; still write local artifacts)."
    )
