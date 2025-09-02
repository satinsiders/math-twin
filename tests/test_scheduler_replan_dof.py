from micro_solver.scheduler import update_metrics, solve
from micro_solver.state import MicroState


def test_under_determined_triggers_replan(monkeypatch) -> None:
    monkeypatch.setattr("micro_solver.scheduler.random.random", lambda: 0.5)
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]

    state = update_metrics(state)
    assert not state.M.get("needs_replan")

    state = update_metrics(state)
    assert state.M.get("needs_replan") is True

    state.numeric_seed = 0.0
    result = solve(state, [], max_iters=1)
    assert result.numeric_seed == 0.5


def test_over_determined_triggers_replan(monkeypatch) -> None:
    monkeypatch.setattr("micro_solver.scheduler.random.random", lambda: 0.5)
    monkeypatch.setattr(
        "micro_solver.steps_meta.estimate_jacobian_rank",
        lambda relations, variables: len(list(variables)) + 1,
    )
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x = 1", "x = 2"]

    state = update_metrics(state)
    assert state.M["degrees_of_freedom"] == -1
    assert state.M.get("needs_replan") is True

    state.numeric_seed = 0.0
    result = solve(state, [], max_iters=1)
    assert result.numeric_seed == 0.5
