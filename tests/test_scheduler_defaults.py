from micro_solver.scheduler import solve_with_defaults
from micro_solver.state import MicroState


def test_default_operator_pool_solves_linear_equation() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x + 2 = 5"]
    result = solve_with_defaults(state, max_iters=4)
    assert result.A["symbolic"].get("final") == "3"
