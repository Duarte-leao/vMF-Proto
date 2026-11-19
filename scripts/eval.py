#!/usr/bin/env python3
"""Hydra entrypoint for evaluation."""

from __future__ import annotations

import os
import warnings
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

import hydra


def _add_repo_to_path() -> None:
    """Ensure both repo root and src/ are on sys.path."""

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    # Keep deterministic precedence: `src/` then repo root.
    # This avoids collisions with sibling modules under `scripts/`
    # (e.g. `scripts/eval.py` shadowing package `src/eval`).
    for path in (src_root, repo_root):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(src_root))


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg) -> int:
    """Evaluate a saved run directory using the configured metrics."""

    _add_repo_to_path()

    from engine.evaluate import evaluate_run

    run_dir_value = str(getattr(cfg.eval, "run_dir", "")).strip()
    if not run_dir_value:
        raise ValueError("cfg.eval.run_dir must be set for evaluation.")
    run_dir = Path(run_dir_value).expanduser()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    evaluate_run(cfg, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
