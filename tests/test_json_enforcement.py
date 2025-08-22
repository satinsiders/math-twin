from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import twin_generator.pipeline as pipeline  # noqa: E402


def _qa_response(out: str, tools: Any | None) -> SimpleNamespace:
    """Simulate QAAgent using tools before returning *out*."""
    tool_map = {t["name"]: t for t in (tools or [])}
    func = tool_map.get("_sanitize_params_tool", {}).get("_func")
    if func:
        func("{}")
    return SimpleNamespace(final_output=out)


@pytest.mark.parametrize(
    "bad_agent",
    ["SampleAgent", "OperationsAgent", "StemChoiceAgent", "FormatterAgent"],
)
def test_generate_twin_invalid_json(monkeypatch: pytest.MonkeyPatch, bad_agent: str) -> None:
    """Pipeline should surface clear errors when downstream agents return invalid JSON."""

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        if name == "QAAgent":
            return _qa_response("pass", tools)
        if name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            return SimpleNamespace(
                final_output=(
                    '{"visual": {"type": "none"}, "answer_expression": "x", '
                    '"operations": [{"expr": "1", "output": "run_agent"}]}'
                )
            )
        if name == "SampleAgent":
            if bad_agent == "SampleAgent":
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym_solved")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="sym_simplified")
        if name == "OperationsAgent":
            if bad_agent == "OperationsAgent":
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(final_output='{"run_agent": 1}')
        if name == "StemChoiceAgent":
            if bad_agent == "StemChoiceAgent":
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(
                final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}'
            )
        if name == "FormatterAgent":
            if bad_agent == "FormatterAgent":
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "Q", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        raise AssertionError(f"unexpected agent {name}")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.error is not None and out.error.startswith(f"{bad_agent} failed:")
