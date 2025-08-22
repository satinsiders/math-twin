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
    func = tool_map.get("_sanitize_params_tool", {}).get("_func")
    if func:
        func("{}")
    return SimpleNamespace(final_output=out)


@pytest.mark.parametrize("always_fail", [False, True])
def test_runner_handles_non_serializable(monkeypatch: pytest.MonkeyPatch, always_fail: bool) -> None:
    """_Runner should retry steps producing non-serializable data and surface errors."""

    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            if always_fail or call_counts[name] == 1:
                return SimpleNamespace(final_output={1})  # not JSON serializable
            return SimpleNamespace(final_output={"ok": 1})
        if name == "QAAgent":
            return _qa_response("pass", tools)
        raise AssertionError(f"unexpected agent {name}")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    def _step_bad(state: pipeline.PipelineState) -> pipeline.PipelineState:
        res = pipeline.AgentsRunner.run_sync(pipeline.ParserAgent, input="x")
        state.extras["bad"] = res.final_output
        return state

    monkeypatch.setattr(pipeline, "_JSON_STEPS", set(pipeline._JSON_STEPS | {"bad"}))

    runner = pipeline._Runner(pipeline._Graph([_step_bad]), qa_max_retries=2)
    out = runner.run(pipeline.PipelineState())
    if always_fail:
        assert out.error is not None and out.error.startswith(
            "QA failed for bad: non-serializable"
        )
        assert call_counts.get("ParserAgent") == 2
        assert call_counts.get("QAAgent") is None
    else:
        assert out.error is None
        assert out.extras["bad"] == {"ok": 1}
        assert call_counts.get("ParserAgent") == 2
        assert call_counts.get("QAAgent") == 1
