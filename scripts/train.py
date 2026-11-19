#!/usr/bin/env python3
"""Hydra entrypoint for training."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

import hydra
from omegaconf import OmegaConf


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
    """Train the model and write a run directory under `cfg.paths.runs_root`."""

    _add_repo_to_path()

    from utils import io as run_io
    from engine.train import train_and_eval

    resume_dir = getattr(cfg.run, "resume_dir", "")
    if resume_dir:
        # Resume training in an existing run directory. This is primarily used
        # by the ablation runner to continue partially-completed runs without
        # creating new <run_name>_<shortid> folders.
        run_dir = Path(str(resume_dir)).expanduser()
    else:
        runs_root = Path(str(cfg.paths.runs_root))
        run_dir = run_io.build_run_dir(
            runs_root=runs_root,
            date=run_io.today_yyyymmdd(),
            run_name=str(cfg.run.name),
            shortid=run_io.generate_shortid(),
        )

    if bool(getattr(cfg.run, "dry_run", False)):
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        run_io.write_placeholder_run_files(run_dir, resolved_config_yaml=resolved_yaml)
        print(f"[dry_run] Created run skeleton: {run_dir}")
        return 0

    train_and_eval(cfg, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
