from __future__ import annotations

"""Progress‑driven operator scheduler for the micro‑solver rebuild."""

from typing import Sequence

from .state import MicroState
from .operators import Operator
from .steps_meta import _micro_monitor_dof


def update_metrics(state: MicroState) -> MicroState:
    """Refresh solver metrics like degrees of freedom and progress score."""

    state = _micro_monitor_dof(state)
    state.progress_score = float(-abs(state.degrees_of_freedom))
    return state


def goal_satisfied(state: MicroState) -> bool:
    return state.final_answer is not None


def select_operator(state: MicroState, operators: Sequence[Operator]) -> Operator | None:
    """Pick the first applicable operator."""

    for op in operators:
        try:
            if op.applicable(state):
                return op
        except Exception:
            continue
    return None


def replan(state: MicroState) -> MicroState:
    """Simple replan heuristic: rotate relations to explore different forms."""

    state.relations = list(reversed(state.relations))
    state.needs_replan = False
    return state


def solve(state: MicroState, operators: Sequence[Operator], *, max_iters: int = 10) -> MicroState:
    """Iteratively apply operators chosen by progress signals."""

    for _ in range(max_iters):
        state = update_metrics(state)
        if goal_satisfied(state):
            break
        if state.needs_replan or state.stalls > 3:
            state = replan(state)
            state.stalls = 0
            continue
        op = select_operator(state, operators)
        if op is None:
            break
        before = state.progress_score
        state, _delta = op.apply(state)
        state = update_metrics(state)
        if state.progress_score <= before:
            state.stalls += 1
        else:
            state.stalls = 0
    return state
