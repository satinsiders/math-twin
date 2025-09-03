from micro_solver.state import MicroState
from micro_solver.scheduler import solve
from micro_solver.operators import SolveOperator, VerifyOperator


def test_strict_dof_zero_path() -> None:
    state = MicroState()
    state.M["verification_policy"] = "strict"
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x + 2 = 5"]
    result = solve(state, [SolveOperator(), VerifyOperator()], max_iters=4)
    assert result.A["symbolic"].get("final") == "3"
    assert result.A["symbolic"].get("final_confidence") == "verified"
    assert result.M["verification_context"]["via"] == "VerifyOperator"


def test_opportunistic_verification_with_dof() -> None:
    state = MicroState()
    state.M["verification_policy"] = "opportunistic"
    state.R["symbolic"]["canonical_repr"] = {"target": "2"}
    state.V["symbolic"]["variables"] = ["x"]  # DOF > 0
    result = solve(state, [VerifyOperator()], max_iters=1)
    assert result.A["symbolic"].get("final") == 2
    assert result.M["verification_context"]["via"] == "VerifyOperator"


def test_strict_epilogue_promotes_candidate() -> None:
    state = MicroState()
    state.M["verification_policy"] = "strict+epilogue"
    state.R["symbolic"]["canonical_repr"] = {"target": "2"}
    state.V["symbolic"]["variables"] = ["x"]  # DOF > 0
    result = solve(state, [VerifyOperator()], max_iters=1)
    assert result.A["symbolic"].get("final") == 2
    assert result.M["verification_context"]["via"] == "scheduler_epilogue"


def test_wrong_candidate_fails_verification() -> None:
    state = MicroState()
    state.M["verification_policy"] = "opportunistic"
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x = 3"]
    state.add_candidate("2")
    result = solve(state, [VerifyOperator()], max_iters=1)
    assert result.A["symbolic"].get("final") is None
    assert result.A["symbolic"].get("candidate") == "2"


def test_final_idempotent() -> None:
    state = MicroState()
    state.M["verification_policy"] = "opportunistic"
    state.R["symbolic"]["canonical_repr"] = {"target": "2"}
    state.V["symbolic"]["variables"] = ["x"]
    result = solve(state, [VerifyOperator()], max_iters=1)
    assert result.A["symbolic"].get("final") == 2
    # Subsequent verify attempts do nothing
    assert VerifyOperator().applicable(result) is False
    result2 = solve(result, [VerifyOperator()], max_iters=1)
    assert result2.A["symbolic"].get("final") == 2
    assert result2.A["symbolic"].get("final_confidence") == "verified"
