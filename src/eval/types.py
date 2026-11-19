"""Shared evaluation result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """Standard evaluation result container."""

    summary: Dict[str, Any]
    sample_paths: Optional[List[str]] = None

