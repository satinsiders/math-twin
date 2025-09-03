from micro_solver.scheduler import replan
from micro_solver.state import MicroState


def test_replan_decomposes_multi_goal() -> None:
    state = MicroState(goal="solve for x and y")
    state.representations = ["symbolic"]
    new_state = replan(state)
    assert new_state.goal == ["solve for x", "solve for y"]
    assert new_state.plan_steps == [
        {"action": "subgoal", "goal": "solve for x"},
        {"action": "subgoal", "goal": "solve for y"},
    ]
