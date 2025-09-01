from __future__ import annotations

from typing import Optional

from .state import MicroState

# Re-export step functions from refactored modules
from .steps_recognition import (
    _micro_normalize,
    _micro_tokenize,
    _micro_entities,
    _micro_relations,
    _micro_goal,
    _micro_classify,
    _micro_repr,
)
from .steps_reasoning import (
    _micro_schema,
    _micro_strategies,
    _micro_choose_strategy,
)
from .steps_execution import _micro_execute_plan
from .steps_candidate import (
    _micro_solve_sympy,
    _micro_extract_candidate,
    _micro_simplify_candidate_sympy,
    _micro_verify_sympy,
)
from .steps_meta import _micro_monitor_dof


# Convenience top‑level graph for a simple end‑to‑end solve pass
# Reintroduce entities/relations for richer downstream context.
DEFAULT_MICRO_STEPS = [
    _micro_normalize,
    _micro_tokenize,
    _micro_entities,
    _micro_relations,
    _micro_goal,
    _micro_classify,
    _micro_repr,
    _micro_schema,
    _micro_strategies,
    _micro_choose_strategy,
    _micro_execute_plan,
    _micro_monitor_dof,
]


def build_steps(*, max_iters: Optional[int] = None) -> list:
    """Return the default micro-steps with a configurable execute-plan budget."""

    def _exec(state: MicroState) -> MicroState:
        return _micro_execute_plan(state, max_iters=max_iters)

    _exec.__name__ = _micro_execute_plan.__name__

    return [
        _micro_normalize,
        _micro_tokenize,
        _micro_entities,
        _micro_relations,
        _micro_goal,
        _micro_classify,
        _micro_repr,
        _micro_schema,
        _micro_strategies,
        _micro_choose_strategy,
        _exec,
        _micro_monitor_dof,
        _micro_solve_sympy,
        _micro_extract_candidate,
        _micro_simplify_candidate_sympy,
        _micro_verify_sympy,
    ]
