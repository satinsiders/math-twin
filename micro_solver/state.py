from __future__ import annotations

"""State model for the micro-solver.

This refactors the previous flat :class:`MicroState` into a hierarchy of five
dictionaries representing ``R`` (representations), ``C`` (constraints), ``V``
(variables and working memory), ``A`` (answers) and ``M`` (metrics).  Callers
should use these nested dictionaries directly, e.g. ``state.R['symbolic']`` or
``state.C['symbolic']``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MicroState:
    """Blackboard state for the micro‑solver.

    The state captures recognition artifacts (tokens, variables, relations),
    reasoning artifacts (candidate schemas, strategies, plan), and calculation
    artifacts (intermediate expressions, numeric evaluations, final answer).

    State data is grouped into five top level buckets following the R/C/V/A/M
    plan:

    ``R`` – representations for symbolic/numeric/alternative views
    ``C`` – constraints/relations for each representation
    ``V`` – variables, environment and derived data
    ``A`` – candidate and final answers
    ``M`` – solver metrics

    ``problem_text`` and a handful of orchestration hints remain at the top
    level for convenience.
    """

    # ------------------------------------------------------------------
    # raw inputs
    problem_text: str = ""

    # ------------------------------------------------------------------
    # R/C/V/A/M containers with dual (symbolic/numeric/alt) representations
    R: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"symbolic": {}, "numeric": {}, "alt": {}}
    )
    C: Dict[str, List[str]] = field(
        default_factory=lambda: {"symbolic": [], "numeric": [], "alt": []}
    )
    V: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            rep: {
                "variables": [],
                "constants": [],
                "identifiers": [],
                "points": [],
                "functions": [],
                "parameters": [],
                "quantities": [],
                "env": {},
                "derived": {},
            }
            for rep in ("symbolic", "numeric", "alt")
        }
    )
    A: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            rep: {
                "candidates": [],
                "final": None,
                "explanation": None,
                "intermediate": [],
                "certificate": None,
            }
            for rep in ("symbolic", "numeric", "alt")
        }
    )
    M: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Solver control and reasoning artifacts (unchanged layout)
    representations: List[str] = field(default_factory=lambda: ["symbolic", "numeric"])
    representation: str = "symbolic"
    numeric_seed: float = 0.0
    case_splits: List[List[str]] = field(default_factory=list)
    active_case: int = 0

    goal: Optional[str] = None  # e.g. "solve for x"
    problem_type: Optional[str] = None  # e.g. "linear", "quadratic"

    schemas: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    chosen_strategy: Optional[str] = None
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step_idx: int = 0

    # Domain knowledge extracted from constraints
    domain: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    qual: dict[str, set[str]] = field(default_factory=dict)

    # Control / diagnostics
    qa_feedback: Optional[str] = None
    error: Optional[str] = None
    skip_qa: bool = False
    next_steps: Optional[List] = None

