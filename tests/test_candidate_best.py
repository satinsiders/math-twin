from micro_solver.state import MicroState
from micro_solver.steps_candidate import _micro_verify_sympy


def test_best_candidate_updates() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x + 2 = 5"]

    state.A["symbolic"]["candidates"].append("2")
    _micro_verify_sympy(state)
    assert str(state.A["symbolic"]["best"]) == "2"
    assert state.A["symbolic"].get("final") is None

    state.A["symbolic"]["candidates"].append("3")
    _micro_verify_sympy(state)
    assert str(state.A["symbolic"]["best"]) == "3"
    assert str(state.A["symbolic"].get("final")) == "3"
