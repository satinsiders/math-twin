import pathlib
import sys
from typing import Any

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import micro_solver.agents as A  # noqa: E402
from micro_solver.state import MicroState  # noqa: E402
import pytest  # noqa: E402

try:
    from micro_solver.steps_execution import _micro_execute_plan
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("steps_execution removed", allow_module_level=True)


def test_skip_repeated_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"planner": 0, "executor": 0}

    def fake_invoke(
        agent: Any,
        payload: dict,
        qa_feedback: str | None = None,
    ) -> tuple[dict, str | None]:
        if agent is A.AtomicPlannerAgent:
            calls["planner"] += 1
            return ({"steps": [{"action": "simplify", "args": {}}]}, None)
        if agent is A.StepExecutorAgent:
            calls["executor"] += 1
            return ({}, None)
        # Force other agents to be skipped
        return ({}, "err")

    monkeypatch.setattr("micro_solver.steps_execution._invoke", fake_invoke)

    state = MicroState(goal="test")
    state.C["symbolic"] = ["x = x"]
    state.V["symbolic"]["env"] = {}
    _micro_execute_plan(state, max_iters=10)

    assert calls["executor"] == 1
    assert state.C["symbolic"] == ["x = x"]
