from micro_solver.scheduler import replan
from micro_solver.state import MicroState


def test_replan_switches_representation() -> None:
    state = MicroState()
    state.representations = ["symbolic", "numeric"]
    state.representation = "symbolic"
    new_state = replan(state)
    assert new_state.representation == "numeric"


def test_replan_reseeds_numeric_solver() -> None:
    state = MicroState()
    state.numeric_seed = 0.0
    new_state = replan(state)
    assert new_state.numeric_seed != 0.0


def test_replan_rotates_case_splits() -> None:
    state = MicroState()
    state.case_splits = [["x > 0"], ["x < 0"]]
    state.relations = ["x > 0"]
    state.active_case = 0
    new_state = replan(state)
    assert new_state.relations == ["x < 0"]
    assert new_state.active_case == 1
