from micro_solver.scheduler import solve
from micro_solver.state import MicroState
from micro_solver.state import MicroState
from micro_solver.scheduler import solve
from micro_solver.operators import SolveOperator, VerifyOperator


def test_solve_operator_gated_by_dof() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x + 2 = 5"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=4)
    assert result.A["symbolic"].get("candidate") == "3"
    assert result.A["symbolic"].get("final") == "3"


def test_solve_operator_skipped_when_underdetermined() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x", "y"]
    state.C["symbolic"] = ["x + y = 5"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=2)
    assert result.A["symbolic"].get("final") is None
    assert result.A["symbolic"].get("candidate") is None


def test_solve_operator_respects_env_bindings() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x", "y"]
    state.V["symbolic"]["env"] = {"x": 1}
    state.C["symbolic"] = ["x + y = 3"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=4)
    assert result.A["symbolic"].get("candidate") == "2"
    assert result.A["symbolic"].get("final") == "2"


def test_solve_operator_surfaces_bound_values() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.V["symbolic"]["env"] = {"x": 2}
    state.C["symbolic"] = ["x = 2"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=4)
    assert result.A["symbolic"].get("candidate") == "2"
    assert result.A["symbolic"].get("final") == "2"
