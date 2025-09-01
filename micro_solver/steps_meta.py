from __future__ import annotations

"""Meta reasoning steps for the micro‑solver.

Currently this module only exposes a degrees‑of‑freedom monitor which counts
unknown variables and constraints.  The step annotates the state so that later
stages can decide whether a replan is required.
"""

from .state import MicroState
from .sym_utils import estimate_jacobian_rank


def _micro_monitor_dof(state: MicroState) -> MicroState:
    """Track a rough degrees‑of‑freedom estimate.

    The count uses a Jacobian rank estimator on equality relations. Inequalities
    are tracked separately as they help prune branches but do not reduce degrees
    of freedom.  The result is stored on the state for downstream heuristics.
    """

    unknowns = [v for v in state.variables + state.parameters if v not in state.env]
    eq_relations = [r for r in state.relations if "=" in r] + list(state.equations)
    eq_count = len(eq_relations)
    ineq_count = sum(
        1 for r in state.relations if any(op in r for op in ("<", ">", "≤", "≥")) and "=" not in r
    )

    rank = estimate_jacobian_rank(eq_relations, unknowns)
    state.eq_count = eq_count
    state.ineq_count = ineq_count
    state.jacobian_rank = rank
    state.degrees_of_freedom = len(unknowns) - rank
    state.needs_replan = state.degrees_of_freedom != 0
    return state
