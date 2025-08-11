from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import twin_generator.pipeline as pipeline  # noqa: E402


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
            return SimpleNamespace(
                final_output=(
                    '{"visual": {"type": "none"}, "answer_expression": "x", '
                    '"operations": ['
                    '{"kind": "sympy", "expr": "1", "output": "run_agent"}, '
                    '{"kind": "agent", "agent": "SampleAgent", '
                    '"input_key": "run_agent", "output": "extra", '
                    '"condition": "run_agent"}'
                    ']}'
                )
            )
        if name == "SampleAgent":
            if isinstance(input, int):
                return SimpleNamespace(final_output="extra_done")
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym_solved")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="sym_simplified")
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "What is 1?", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "What is 1?", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        if name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out["twin_stem"] == "What is 1?"
    assert out.get("errors") == []
    assert calls == [
        "ParserAgent",
        "QAAgent",
        "ConceptAgent",
        "QAAgent",
        "TemplateAgent",
        "QAAgent",
        "SampleAgent",
        "QAAgent",
        "SymbolicSolveAgent",
        "SymbolicSimplifyAgent",
        "QAAgent",
        "QAAgent",
        "QAAgent",
        "SampleAgent",
        "QAAgent",
        "QAAgent",
        "QAAgent",
        "StemChoiceAgent",
        "QAAgent",
        "FormatterAgent",
        "QAAgent",
    ]


def test_generate_twin_agent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order = []

    def mock_run_sync(agent: Any, input: Any) -> SimpleNamespace:
        call_order.append(agent.name)
        if agent.name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        if agent.name == "TemplateAgent":
            raise RuntimeError("boom")
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.get("error") == "TemplateAgent failed: boom"
    # ensure pipeline stopped early
    assert call_order == [
        "ParserAgent",
        "QAAgent",
        "ConceptAgent",
        "QAAgent",
        "TemplateAgent",
    ]
    assert "params" not in out


def test_generate_twin_qa_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output="parsed")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            return SimpleNamespace(final_output='{"visual": {"type": "none"}, "answer_expression": "0"}')
        if name == "SampleAgent":
            return SimpleNamespace(final_output='{}')
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "answer_index": 0, "answer_value": 1, "rationale": "r"}')
        if name == "QAAgent":
            # First QA check fails; subsequent ones pass
            return SimpleNamespace(final_output="fail" if call_counts[name] == 1 else "pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.get("error") is None
    # Parser step should have been retried due to QA failure
    assert call_counts.get("ParserAgent") == 2
    # QAAgent called once per step plus the extra retry (total 10)
    assert call_counts.get("QAAgent") == 10

