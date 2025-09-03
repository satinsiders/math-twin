from micro_solver.state import MicroState
from micro_solver.steps_numeric import _micro_numeric
from micro_solver.steps_alt import _micro_alt
from micro_solver.steps import build_steps
from micro_solver.scheduler import replan


def test_view_population() -> None:
    state = MicroState()
    state.problem_text = "x = 1"
    state.R["symbolic"]["normalized_text"] = "x = 1"
    state.C["symbolic"] = ["x = 1"]
    state.V["symbolic"]["variables"] = ["x"]

    state = _micro_numeric(state)
    state = _micro_alt(state)

    assert state.C["numeric"] == ["x = 1"]
    assert state.V["numeric"]["variables"] == ["x"]
    assert state.C["alt"] == ["x = 1"]
    assert state.V["alt"]["variables"] == ["x"]


def test_build_steps_includes_numeric_alt_before_reasoning() -> None:
    steps = build_steps()
    names = [s.__name__ for s in steps]
    n_idx = names.index("_micro_numeric")
    a_idx = names.index("_micro_alt")
    schema_idx = names.index("_micro_schema")
    assert n_idx < schema_idx and a_idx < schema_idx


def test_replan_switches_active_view_constraints_and_vars() -> None:
    state = MicroState()
    state.representations = ["symbolic", "numeric"]
    state.representation = "symbolic"
    state.C["symbolic"] = ["x = 1"]
    state.C["numeric"] = ["x = 2"]
    state.V["symbolic"]["variables"] = ["x"]
    state.V["numeric"]["variables"] = ["y"]

    new_state = replan(state)

    assert new_state.representation == "numeric"
    assert new_state.C["symbolic"] == ["x = 2"]
    assert new_state.V["symbolic"]["variables"] == ["y"]

