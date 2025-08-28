from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import twin_generator.pipeline as pipeline  # noqa: E402


def _qa_response(out: str, tools: Any | None) -> SimpleNamespace:
    tool_map = {t["name"]: t for t in (tools or [])}
    func = tool_map.get("sanitize_params_tool", {}).get("_func")
    if func:
        func("{}")
    return SimpleNamespace(final_output=out)


@pytest.mark.parametrize("always_fail", [False, True])
def test_runner_retries_on_qa_exception(
    monkeypatch: pytest.MonkeyPatch, always_fail: bool
) -> None:
    """_Runner should retry steps when QA agent raises an exception."""

    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output=1)
        if name == "QAAgent":
            if always_fail or call_counts[name] == 1:
                raise RuntimeError("boom")
            return _qa_response("pass", tools)
        raise AssertionError(f"unexpected agent {name}")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    def _step_good(state: pipeline.PipelineState) -> pipeline.PipelineState:
        res = pipeline.AgentsRunner.run_sync(pipeline.ParserAgent, input="x")
        state.extras["out"] = res.final_output
        return state

    runner = pipeline._Runner(pipeline._Graph([_step_good]), qa_max_retries=2)
    out = runner.run(pipeline.PipelineState())

    if always_fail:
        assert out.error == "QA failed for good: boom"
        assert call_counts.get("ParserAgent") == 2
        assert call_counts.get("QAAgent") == 2
    else:
        assert out.error is None
        assert out.extras["out"] == 1
        assert call_counts.get("ParserAgent") == 2
        assert call_counts.get("QAAgent") == 2

