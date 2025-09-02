from __future__ import annotations

"""Operator interface and basic implementations for the micro‑solver rebuild.

Operators perform small state transitions and return a progress signal.  They
are intentionally lightweight so they can be scheduled dynamically based on
observed progress rather than a fixed strategy tree.
"""

from dataclasses import dataclass
from typing import Tuple

from .state import MicroState
from .sym_utils import (
    rewrite_relations,
    simplify_expr,
    verify_candidate,
    solve_for,
    solve_any,
)


class Operator:
    """Protocol for reasoning or calculation operators."""

    name: str

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - interface
        return True

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class SimplifyOperator(Operator):
    """Canonicalize all relations using :func:`simplify_expr`."""

    name: str = "simplify"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.relations)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        before = sum(len(r) for r in state.relations)
        state.relations = [simplify_expr(r) for r in state.relations]
        after = sum(len(r) for r in state.relations)
        delta = float(before - after)
        return state, delta


@dataclass
class SubstituteOperator(Operator):
    """Perform deterministic substitutions on all relations."""

    replacements: dict[str, str]
    name: str = "substitute"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(self.replacements) and bool(state.relations)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        step = {"action": "substitute", "args": {"replacements": self.replacements}}
        new_rel = rewrite_relations(state.relations, step)
        delta = float(len(state.relations) - len(new_rel))
        state.relations = new_rel
        return state, delta


@dataclass
class FeasibleSampleOperator(Operator):
    """Toy numeric sampler that records a random point for free variables."""

    name: str = "feasible_sample"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.variables)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        import random

        sample = {v: random.random() for v in state.variables}
        state.derived["sample"] = sample
        return state, 0.0


@dataclass
class SolveOperator(Operator):
    """Solve relations for a target symbol when system is determined."""

    name: str = "solve"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return (
            state.degrees_of_freedom == 0
            and bool(state.relations)
            and not state.candidate_answers
        )

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        target = state.variables[0] if state.variables else None
        sols = solve_for(state.relations, target)
        if not sols:
            sols = solve_any(state.relations)
        if sols:
            state.candidate_answers.extend(sols)
            return state, 1.0
        return state, 0.0


@dataclass
class VerifyOperator(Operator):
    """Verify the latest candidate against original relations."""

    name: str = "verify"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return (
            state.degrees_of_freedom == 0
            and bool(state.candidate_answers)
            and state.final_answer is None
        )

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            candidate = str(state.candidate_answers[-1])
        except Exception:
            return state, 0.0
        var = state.variables[0] if state.variables else None
        if verify_candidate(state.relations, candidate, varname=var):
            state.final_answer = candidate
            return state, 1.0
        return state, 0.0


# Default operator pool used by the high-level scheduler entrypoint.
#
# The set is intentionally small; it demonstrates the operator protocol with a
# mix of symbolic and validation steps while keeping the scheduling loop
# lightweight.  Additional operators can be appended by callers as needed.
DEFAULT_OPERATORS: list[Operator] = [
    SolveOperator(),
    VerifyOperator(),
    SimplifyOperator(),
]

