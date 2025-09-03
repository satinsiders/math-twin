from __future__ import annotations

"""Meta reasoning steps for the micro‑solver.

Currently this module only exposes a degrees‑of‑freedom monitor which counts
unknown variables and constraints.  The step annotates the state so that later
stages can decide whether a replan is required.
"""

from .state import MicroState
from .sym_utils import estimate_jacobian_rank, parse_relation_sides
from .constraint_analysis import build_independence_graph


def _micro_monitor_dof(state: MicroState) -> MicroState:
    """Track a rough degrees‑of‑freedom estimate.

    The count uses a Jacobian rank estimator on equality relations. Inequalities
    are tracked separately as they help prune branches but do not reduce degrees
    of freedom.  The result is stored on the state for downstream heuristics.
    """

    sym_vars = state.V["symbolic"].get("variables", [])
    sym_params = state.V["symbolic"].get("parameters", [])
    env = state.V["symbolic"].get("env", {})
    unknowns = [v for v in sym_vars + sym_params if v not in env]
    eq_relations = [r for r in state.C["symbolic"] if "=" in r]
    eq_count = len(eq_relations)
    ineq_count = sum(
        1 for r in state.C["symbolic"] if parse_relation_sides(r)[0] in ("<", "<=", ">", ">=")
    )

    independence = build_independence_graph(state.C["symbolic"], unknowns)
    redundant_idx = independence.get("redundant", [])
    state.M["redundant_constraints_idx"] = redundant_idx
    state.M["redundant_constraints"] = [state.C["symbolic"][i] for i in redundant_idx]
    state.M["independence_graph"] = independence.get("graph", {})

    rank = estimate_jacobian_rank(eq_relations, unknowns)
    state.M["eq_count"] = eq_count
    state.M["ineq_count"] = ineq_count
    state.M["jacobian_rank"] = rank
    state.M["degrees_of_freedom"] = len(unknowns) - rank
    # ``needs_replan`` is controlled externally and should not be
    # overwritten simply because degrees of freedom remain non-zero.
    return state
