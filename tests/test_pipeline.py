from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import twin_generator.pipeline as pipeline


def test_generate_twin_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def mock_run_sync(agent: Any, input: Any) -> SimpleNamespace:
        calls.append(agent.name)
        name = agent.name
        if name == "ParserAgent":
            return SimpleNamespace(final_output="parsed")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            return SimpleNamespace(final_output='{"visual": {"type": "none"}, "answer_expression": "x"}')
        if name == "SampleAgent":
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "What is 1?", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "What is 1?", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out["twin_stem"] == "What is 1?"
    assert out.get("errors") == []
    assert calls == [
        "ParserAgent",
        "ConceptAgent",
        "TemplateAgent",
        "SampleAgent",
        "StemChoiceAgent",
        "FormatterAgent",
    ]


def test_generate_twin_agent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order = []

    def mock_run_sync(agent: Any, input: Any) -> SimpleNamespace:
        call_order.append(agent.name)
        if len(call_order) == 3:
            raise RuntimeError("boom")
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.get("error") == "TemplateAgent failed: boom"
    # ensure pipeline stopped early
    assert call_order == ["ParserAgent", "ConceptAgent", "TemplateAgent"]
    assert "params" not in out

