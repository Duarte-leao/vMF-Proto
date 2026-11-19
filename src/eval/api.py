"""Evaluation API wrapper."""

from __future__ import annotations

from typing import Any

from .types import EvalResult


def evaluate(cfg: Any, run_dir: Any) -> EvalResult:
    """Evaluate a trained model and write artifacts into `run_dir`."""

    from engine.evaluate import evaluate_run

    return evaluate_run(cfg, run_dir)
