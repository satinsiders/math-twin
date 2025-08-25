from types import SimpleNamespace

import pytest

from twin_generator import pipeline_helpers


def test_invoke_agent_stops_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def mock_run_sync(agent, input, tools=None):
        nonlocal calls
        calls += 1
        return SimpleNamespace(final_output="not json")

    monkeypatch.setattr(pipeline_helpers.AgentsRunner, "run_sync", mock_run_sync)

    out, err = pipeline_helpers.invoke_agent(SimpleNamespace(name="A"), "payload", max_retries=3)
    assert out is None
    assert err and err.startswith("A failed")
    assert calls == 3


def test_invoke_agent_success_returns_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def mock_run_sync(agent, input, tools=None):
        nonlocal calls
        calls += 1
        return SimpleNamespace(final_output='{"x": 1}')

    monkeypatch.setattr(pipeline_helpers.AgentsRunner, "run_sync", mock_run_sync)

    out, err = pipeline_helpers.invoke_agent(SimpleNamespace(name="A"), "payload", max_retries=5)
    assert out == {"x": 1}
    assert err is None
    assert calls == 1


def test_invoke_agent_stops_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["not json", '{"ok": true}', "should not reach"])
    calls = 0

    def mock_run_sync(agent, input, tools=None):
        nonlocal calls
        calls += 1
        return SimpleNamespace(final_output=next(outputs))

    monkeypatch.setattr(pipeline_helpers.AgentsRunner, "run_sync", mock_run_sync)

    out, err = pipeline_helpers.invoke_agent(SimpleNamespace(name="A"), "payload", max_retries=5)
    assert out == {"ok": True}
    assert err is None
    assert calls == 2
