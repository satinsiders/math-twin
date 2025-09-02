from micro_solver.scheduler import solve
from micro_solver.state import MicroState
from micro_solver.operators import SolveOperator, VerifyOperator


def test_solve_operator_gated_by_dof() -> None:
    state = MicroState()
    state.variables = ["x"]
    state.relations = ["x + 2 = 5"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=4)
    assert result.final_answer == "3"


def test_solve_operator_skipped_when_underdetermined() -> None:
    state = MicroState()
    state.variables = ["x", "y"]
    state.relations = ["x + y = 5"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=2)
    assert result.final_answer is None
    assert result.candidate_answers == []


def test_solve_operator_respects_env_bindings() -> None:
    state = MicroState()
    state.variables = ["x", "y"]
    state.env = {"x": 1}
    state.relations = ["x + y = 3"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=4)
    assert result.candidate_answers == ["2"]
    assert result.final_answer == "2"
