from micro_solver.state import MicroState
import micro_solver.steps_reasoning as SR


def test_micro_schema_handles_list_goal(monkeypatch) -> None:
    state = MicroState(goal=["g1", "g2"])

    def fake_invoke(agent, payload, qa_feedback=None, **kwargs):
        return {"schemas": [payload["target"] + "_schema"]}, None

    monkeypatch.setattr(SR, "_invoke", fake_invoke)
    new_state = SR._micro_schema(state)
    assert new_state.schemas == ["g1_schema", "g2_schema"]


def test_micro_strategies_handles_list_goal(monkeypatch) -> None:
    state = MicroState(goal=["g1", "g2"], schemas=["s"])

    def fake_invoke(agent, payload, qa_feedback=None, **kwargs):
        return {"strategies": [payload["target"] + "_strategy"]}, None

    monkeypatch.setattr(SR, "_invoke", fake_invoke)
    new_state = SR._micro_strategies(state)
    assert new_state.strategies == ["g1_strategy", "g2_strategy"]
