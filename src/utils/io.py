"""Run directory IO utilities.

This module defines the on-disk *contract* used by training/eval runs:

- run directory layout (which files/folders exist)
- helpers to create a new run directory
- helpers to read/write the small JSON/YAML/CSV artifacts stored in runs/

Important: this module should stay side-effect free and must not compute model
metrics; it only reads/writes artifacts produced by the engines.
"""

from __future__ import annotations

import csv
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Run directory schema (spec)
# -----------------------------------------------------------------------------

RUN_DIR_RELATIVE_LAYOUT: Tuple[str, ...] = (
    "config.yaml",
    "meta.json",
    "metrics.json",
    "history.csv",
    "samples/",
    "checkpoints/",
)

META_JSON_REQUIRED_FIELDS: Tuple[str, ...] = (
    # Identity
    "run_id",
    "run_name",
    "date",
    "created_at",
    # Provenance
    "git",
    "env",
    # Config & tracking
    "wandb",
    "config_hash",
    # Dataset/preprocessing metadata
    "dataset",
    # Bookkeeping
    "status",
    "notes",
)

METRICS_JSON_REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "primary_metric",
    "summary",
    "best",
    "notes",
)

HISTORY_CSV_REQUIRED_COLUMNS: Tuple[str, ...] = (
    # Long format: one row per (step, split, metric)
    "step",
    "split",
    "metric",
    "value",
    "timestamp",
)


def today_yyyymmdd() -> str:
    """Return the current UTC date as `YYYYMMDD`."""

    return datetime.now(timezone.utc).strftime("%Y%m%d")


def generate_shortid(length: int = 8) -> str:
    """
    Generate a short, filesystem-safe run identifier.

    NOTE: This is used for directory naming only; it is not cryptographic security.
    """

    # token_hex produces 2 chars per byte; choose a conservative fallback.
    needed_bytes = max(4, (length + 1) // 2)
    return secrets.token_hex(needed_bytes)[:length]


def build_run_dir(runs_root: Path, date: str, run_name: str, shortid: str) -> Path:
    """
    Construct `runs/<YYYYMMDD>/<run_name>_<shortid>/` under `runs_root`.
    """

    safe_run_name = _sanitize_run_name(run_name)
    return runs_root / date / f"{safe_run_name}_{shortid}"


def ensure_run_dir_structure(run_dir: Path) -> None:
    """
    Create required subdirectories for a run directory (idempotent).
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def write_placeholder_run_files(
    run_dir: Path,
    *,
    resolved_config_yaml: str,
    meta: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write placeholder `config.yaml`, `meta.json`, `metrics.json`, `history.csv`.

    This is used by `scripts/*.py --dry_run` to produce a correctly-shaped run
    directory without executing training/eval/sampling.
    """

    ensure_run_dir_structure(run_dir)

    (run_dir / "config.yaml").write_text(resolved_config_yaml, encoding="utf-8")
    write_json(run_dir / "meta.json", meta or default_meta(run_dir=run_dir))
    write_json(run_dir / "metrics.json", metrics or default_metrics())
    write_history_csv(run_dir / "history.csv", rows=[])


def default_meta(*, run_dir: Path) -> Dict[str, Any]:
    """
    Produce a placeholder `meta.json` that matches the required schema.

    Required fields are listed in `META_JSON_REQUIRED_FIELDS`.
    """

    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_dir.name.split("_")[-1],
        "run_name": "_".join(run_dir.name.split("_")[:-1]) or run_dir.name,
        "date": run_dir.parent.name,
        "created_at": now,
        "git": {
            "commit": "TODO_GIT_COMMIT",
            "dirty": "TODO_BOOL",
            "branch": "TODO_GIT_BRANCH",
        },
        "env": {
            "python": "TODO_PYTHON_VERSION",
            "pytorch": "TODO_TORCH_VERSION",
            "platform": "TODO_PLATFORM",
            "hostname": "TODO_HOSTNAME",
        },
        "wandb": {
            "mode": "TODO: online|offline|disabled",
            "entity": "TODO_WANDB_ENTITY",
            "project": "TODO_WANDB_PROJECT",
            "run_id": "TODO_WANDB_RUN_ID_OR_null",
        },
        "config_hash": "TODO_HASH_OF_RESOLVED_CONFIG",
        "dataset": {
            "name": "TODO_DATASET_NAME",
            "root": "TODO_DATASET_ROOT",
            "splits": {"train": "TODO", "val": "TODO", "test": "TODO"},
            "preprocessing": "TODO: declarative preprocessing config snapshot",
        },
        "status": "created",  # created|running|completed|failed
        "notes": "TODO: free-form note about the purpose of this run",
    }


def default_metrics() -> Dict[str, Any]:
    """
    Produce a placeholder `metrics.json` that matches the required schema.

    Required fields are listed in `METRICS_JSON_REQUIRED_FIELDS`.
    """

    return {
        "schema_version": 1,
        "primary_metric": "TODO_PRIMARY_METRIC_NAME",
        "summary": {
            # Example placeholder (values must be numbers once implemented):
            # "val/TODO_METRIC": null
        },
        "best": {
            "step": None,  # int once implemented
            "metrics": {},  # metric_name -> value
            "checkpoint_path": None,
        },
        "notes": "TODO: how these metrics were computed and any caveats",
    }


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a dict-like object as pretty-printed JSON with a trailing newline."""

    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_history_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write a long-format history CSV.

    Schema:
    - Columns in `HISTORY_CSV_REQUIRED_COLUMNS`
    - Each row corresponds to a (step, split, metric) observation.
    """

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HISTORY_CSV_REQUIRED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in HISTORY_CSV_REQUIRED_COLUMNS})


def validate_run_dir(run_dir: Path) -> List[str]:
    """
    Validate that the required run files/directories exist.

    Returns a list of human-readable problems. Empty means "looks OK".
    """

    problems: List[str] = []
    for rel in RUN_DIR_RELATIVE_LAYOUT:
        p = run_dir / rel.rstrip("/")
        if rel.endswith("/"):
            if not p.exists() or not p.is_dir():
                problems.append(f"Missing directory: {p}")
        else:
            if not p.exists() or not p.is_file():
                problems.append(f"Missing file: {p}")
    return problems


def _sanitize_run_name(name: str) -> str:
    """
    Minimal sanitization for filesystem friendliness.

    TODO: Extend to your naming rules (and keep consistent with `README.md`).
    """

    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_") or "run"
