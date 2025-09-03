from micro_solver.scheduler import (
    solve_with_defaults,
    select_operator,
    update_metrics,
)
from micro_solver.state import MicroState
from micro_solver.operators import DEFAULT_OPERATORS


def test_default_operator_pool_solves_linear_equation() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x + 2 = 5"]
    result = solve_with_defaults(state, max_iters=4)
    assert result.A["symbolic"].get("final") == "3"


def test_sampler_inapplicable_once_sample_exists() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    # Initial selection should sample since no sample exists
    op = select_operator(update_metrics(state), DEFAULT_OPERATORS)
    assert op and op.name == "feasible_sample"
    state, _ = op.apply(state)
    # After sampling, pruning should run before any resampling
    op = select_operator(update_metrics(state), DEFAULT_OPERATORS)
    assert op and op.name == "domain_prune"
