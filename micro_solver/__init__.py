from __future__ import annotations

"""Micro‑solver package: ultra‑granular multi‑agent math solving.

This module provides a recognition → reasoning → calculation pipeline broken
down into micro‑steps, each handled by a single specialized agent. It is
designed to minimize per‑agent cognitive load by enforcing very small,
verifiable actions with strict pre/post‑conditions.

The package is additive and does not modify the twin generator. See
``docs/micro_solver.md`` for an overview and usage.
"""

from .state import MicroState  # noqa: F401
from .orchestrator import MicroGraph, MicroRunner  # noqa: F401
from . import agents as micro_agents  # noqa: F401
from . import steps as micro_steps  # noqa: F401
from .candidate import Candidate  # noqa: F401
from .operators import (
    Operator,
    InitCandidatesOperator,
    SimplifyOperator,
    SubstituteOperator,
    FeasibleSampleOperator,
    SolveOperator,
    VerifyOperator,
    DEFAULT_OPERATORS,
)  # noqa: F401
from .scheduler import solve, solve_with_defaults  # noqa: F401
from .constraint_analysis import (  # noqa: F401
    mark_redundant_constraints,
    attempt_rank_repair,
)

__all__ = [
    "MicroState",
    "MicroGraph",
    "MicroRunner",
    "micro_agents",
    "micro_steps",
    "Candidate",
    "Operator",
    "InitCandidatesOperator",
    "SimplifyOperator",
    "SubstituteOperator",
    "FeasibleSampleOperator",
    "SolveOperator",
    "VerifyOperator",
    "DEFAULT_OPERATORS",
    "solve",
    "solve_with_defaults",
    "mark_redundant_constraints",
    "attempt_rank_repair",
]

