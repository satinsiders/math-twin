from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import json

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import twin_generator.pipeline as pipeline  # noqa: E402
from twin_generator.pipeline import PipelineState  # noqa: E402


def test_generate_twin_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        calls.append(agent.name)
        name = agent.name
        if name == "ParserAgent":
            return SimpleNamespace(final_output='{"parsed": true}')
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
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym_solved")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="sym_simplified")
        if name == "OperationsAgent":
            return SimpleNamespace(final_output='{"run_agent": 1}')
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
    assert out.twin_stem == "What is 1?"
    assert out.errors == []
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
        "OperationsAgent",
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

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        call_order.append(agent.name)
        if agent.name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        if agent.name == "TemplateAgent":
            raise RuntimeError("boom")
        if agent.name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.error == "TemplateAgent failed: boom"
    # ensure pipeline stopped early
    assert call_order == [
        "ParserAgent",
        "QAAgent",
        "ConceptAgent",
        "QAAgent",
        "TemplateAgent",
    ]
    assert out.params is None


def test_generate_twin_operations_non_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order = []

    def mock_run_sync(
        agent: Any, input: Any, tools: Any | None = None
    ) -> SimpleNamespace:
        call_order.append(agent.name)
        name = agent.name
        if name == "ParserAgent":
            return SimpleNamespace(final_output='{"parsed": true}')
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
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym_solved")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="sym_simplified")
        if name == "OperationsAgent":
            return SimpleNamespace(final_output="[]")
        if name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.error == "OperationsAgent produced non-dict output"
    assert call_order == [
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
        "OperationsAgent",
    ]


def test_generate_twin_parser_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order = []

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        call_order.append(agent.name)
        if agent.name == "ParserAgent":
            return SimpleNamespace(final_output="not json")
        if agent.name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert (
        out.error
        == (
            "ParserAgent failed: Agent output was not valid JSON even after repair. "
            "Original snippet: not json... Repaired snippet: not json..."
        )
    )
    assert call_order == ["ParserAgent"]


def test_generate_twin_template_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """TemplateAgent returning empty output should surface a clear error."""

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        if agent.name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        if agent.name == "TemplateAgent":
            return SimpleNamespace(final_output="")
        if agent.name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.error == "TemplateAgent failed: Agent output was empty"


def test_generate_twin_template_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            if call_counts[name] == 1:
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(final_output='{"visual": {"type": "none"}, "answer_expression": "0"}')
        if name == "SampleAgent":
            return SimpleNamespace(final_output='{}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="0")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="0")
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "Q", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        if name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.error is None
    assert call_counts.get("TemplateAgent") == 2


def test_generate_twin_qa_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            return SimpleNamespace(final_output='{"visual": {"type": "none"}, "answer_expression": "0"}')
        if name == "SampleAgent":
            return SimpleNamespace(final_output='{}')
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "Q", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        if name == "QAAgent":
            # First QA check fails; subsequent ones pass
            return SimpleNamespace(final_output="fail" if call_counts[name] == 1 else "pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.error is None
    # Parser step should have been retried due to QA failure
    assert call_counts.get("ParserAgent") == 2
    # QAAgent called once per step plus the extra retry (total 10)
    assert call_counts.get("QAAgent") == 10


def test_generate_twin_qa_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_twin should surface an error after exceeding QA retry limit."""
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        if name == "QAAgent":
            return SimpleNamespace(final_output="fail")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    assert out.error == "QA failed for parse: fail"
    assert call_counts.get("ParserAgent") == 5
    assert call_counts.get("QAAgent") == 5


def test_step_visual_handles_non_dict() -> None:
    state = PipelineState(template={"visual": "not-a-dict"})
    out = pipeline._step_visual(state)
    assert out.template == state.template


def test_step_sample_rejects_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        assert agent.name == "SampleAgent"
        return SimpleNamespace(final_output='{"x": "oops"}')

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    state = PipelineState(template={})
    out = pipeline._step_sample(state)
    assert out.error == "SampleAgent produced non-numeric params: x='oops'"


def test_step_operations_rejects_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        assert agent.name == "OperationsAgent"
        return SimpleNamespace(final_output='{"params": {"x": "oops"}}')

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    state = PipelineState(template={"operations": [1]}, params={})
    out = pipeline._step_operations(state)
    assert out.error == "OperationsAgent produced non-numeric params: x='oops'"


def test_step_operations_trims_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        assert agent.name == "OperationsAgent"
        captured["payload"] = input
        return SimpleNamespace(final_output="{}")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    ops = [{"expr": "1", "output": "x"}]
    state = PipelineState(
        template={"operations": ops},
        params={"a": 1},
        parsed={"keep": "no"},
    )
    pipeline._step_operations(state)
    sent = json.loads(captured["payload"])
    assert set(sent["data"].keys()) == {"template", "params"}


def test_step_operations_includes_referenced_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        assert agent.name == "OperationsAgent"
        captured["payload"] = input
        return SimpleNamespace(final_output="{}")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    ops = [{"expr": "1", "output": "x", "input": "symbolic_solution"}]
    state = PipelineState(
        template={"operations": ops},
        params={},
        symbolic_solution="sol",
    )
    pipeline._step_operations(state)
    sent = json.loads(captured["payload"])
    assert sent["data"].get("symbolic_solution") == "sol"
    assert set(sent["data"].keys()) == {"template", "params", "symbolic_solution"}


def test_step_operations_ignores_unexpected_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        assert agent.name == "OperationsAgent"
        return SimpleNamespace(final_output='{"foo": 1, "junk": 2}')

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    ops = [{"expr": "1", "output": "foo"}]
    state = PipelineState(template={"operations": ops}, params={})
    out = pipeline._step_operations(state)
    assert out.extras == {"foo": 1}


@pytest.mark.parametrize("always_fail", [False, True])
def test_sample_agent_json_retry(monkeypatch: pytest.MonkeyPatch, always_fail: bool) -> None:
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            return SimpleNamespace(final_output='{"visual": {"type": "none"}, "answer_expression": "x"}')
        if name == "SampleAgent":
            if always_fail or call_counts[name] == 1:
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="simp")
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "Q", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        if name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    if always_fail:
        assert out.error is not None and out.error.startswith("SampleAgent failed")
        assert call_counts.get("SampleAgent") == pipeline._JSON_MAX_RETRIES
    else:
        assert out.error is None
        assert call_counts.get("SampleAgent") == 2


@pytest.mark.parametrize("always_fail", [False, True])
def test_operations_agent_json_retry(monkeypatch: pytest.MonkeyPatch, always_fail: bool) -> None:
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
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
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="simp")
        if name == "OperationsAgent":
            if always_fail or call_counts[name] == 1:
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(final_output='{"run_agent": 1}')
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "Q", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        if name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    if always_fail:
        assert out.error is not None and out.error.startswith("OperationsAgent failed")
        assert call_counts.get("OperationsAgent") == pipeline._JSON_MAX_RETRIES
    else:
        assert out.error is None
        assert call_counts.get("OperationsAgent") == 2


@pytest.mark.parametrize("always_fail", [False, True])
def test_stem_choice_agent_json_retry(monkeypatch: pytest.MonkeyPatch, always_fail: bool) -> None:
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            return SimpleNamespace(final_output='{"visual": {"type": "none"}, "answer_expression": "x"}')
        if name == "SampleAgent":
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="simp")
        if name == "StemChoiceAgent":
            if always_fail or call_counts[name] == 1:
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "Q", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        if name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    if always_fail:
        assert out.error is not None and out.error.startswith("StemChoiceAgent failed")
        assert call_counts.get("StemChoiceAgent") == pipeline._JSON_MAX_RETRIES
    else:
        assert out.error is None
        assert call_counts.get("StemChoiceAgent") == 2


@pytest.mark.parametrize("always_fail", [False, True])
def test_formatter_agent_json_retry(monkeypatch: pytest.MonkeyPatch, always_fail: bool) -> None:
    call_counts: dict[str, int] = {}

    def mock_run_sync(agent: Any, input: Any, tools: Any | None = None) -> SimpleNamespace:
        name = agent.name
        call_counts[name] = call_counts.get(name, 0) + 1
        if name == "ParserAgent":
            return SimpleNamespace(final_output="{}")
        if name == "ConceptAgent":
            return SimpleNamespace(final_output="concept")
        if name == "TemplateAgent":
            return SimpleNamespace(final_output='{"visual": {"type": "none"}, "answer_expression": "x"}')
        if name == "SampleAgent":
            return SimpleNamespace(final_output='{"x": 1}')
        if name == "SymbolicSolveAgent":
            return SimpleNamespace(final_output="sym")
        if name == "SymbolicSimplifyAgent":
            return SimpleNamespace(final_output="simp")
        if name == "StemChoiceAgent":
            return SimpleNamespace(final_output='{"twin_stem": "Q", "choices": [1], "rationale": "r"}')
        if name == "FormatterAgent":
            if always_fail or call_counts[name] == 1:
                return SimpleNamespace(final_output="not json")
            return SimpleNamespace(
                final_output=(
                    '{"twin_stem": "Q", "choices": [1], '
                    '"answer_index": 0, "answer_value": 1, "rationale": "r"}'
                )
            )
        if name == "QAAgent":
            return SimpleNamespace(final_output="pass")
        raise AssertionError("unexpected agent")

    monkeypatch.setattr(pipeline.AgentsRunner, "run_sync", mock_run_sync)

    out = pipeline.generate_twin("p", "s")
    if always_fail:
        assert out.error is not None and out.error.startswith("FormatterAgent failed")
        assert call_counts.get("FormatterAgent") == pipeline._JSON_MAX_RETRIES
    else:
        assert out.error is None
        assert out.twin_stem == "Q"
        assert call_counts.get("FormatterAgent") == 2

