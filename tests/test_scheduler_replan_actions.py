from micro_solver.scheduler import replan
from micro_solver.state import MicroState


def test_replan_switches_representation_cycle() -> None:
    state = MicroState()
    state.representations = ["symbolic", "alt", "numeric"]
    state.representation = "symbolic"
    state.relations = []
    state.derived = {}
    state = replan(state)
    assert state.representation == "alt"
    state = replan(state)
    assert state.representation == "numeric"
    state = replan(state)
    assert state.representation == "symbolic"


def test_replan_adjusts_numeric_grid_and_reseeds() -> None:
    state = MicroState()
    state.numeric_seed = 0.0
    state.relations = []
    state.derived = {}
    state.V["numeric"]["derived"]["grid"] = 1.0
    first = replan(state)
    first_seed = first.numeric_seed
    assert first_seed != 0.0
    assert first.V["numeric"]["derived"]["grid"] == 0.5
    second = replan(first)
    assert second.V["numeric"]["derived"]["grid"] == 1.0
    assert second.numeric_seed != first_seed


def test_replan_rotates_case_splits() -> None:
    state = MicroState()
    state.case_splits = [["x > 0"], ["x < 0"]]
    state.relations = ["x > 0"]
    state.active_case = 0
    state.derived = {}
    new_state = replan(state)
    assert new_state.relations == ["x < 0"]
    assert new_state.active_case == 1


def test_replan_decomposes_compound_goals() -> None:
    state = MicroState()
    state.goal = "find x and y"
    state.relations = []
    state.derived = {}
    first = replan(state)
    assert first.goal == "find x"
    assert first.derived["pending_goals"] == ["find y"]
    second = replan(first)
    assert second.goal == "find y"
    assert second.derived["pending_goals"] == ["find x"]
