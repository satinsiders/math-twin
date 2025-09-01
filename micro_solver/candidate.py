from __future__ import annotations

"""Candidate model for anytime micro‑solver results."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class Candidate:
    """Represents a candidate solution with metadata."""

    value: Any
    residuals: Dict[str, float] = field(default_factory=dict)
    verified: bool = False
    error_bounds: Optional[Tuple[float, float]] = None
