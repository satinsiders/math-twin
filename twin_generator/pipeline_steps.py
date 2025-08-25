"""Individual step functions for the twin generator pipeline."""
from __future__ import annotations

import json
from typing import Any, cast

from . import constants as C
from .agents import (
    ConceptAgent,
    FormatterAgent,
    OperationsAgent,
    ParserAgent,
    SampleAgent,
    StemChoiceAgent,
    TemplateAgent,
    SymbolicSolveAgent,
    SymbolicSimplifyAgent,
)
from .pipeline_helpers import (
    _TOOLS,
    _TEMPLATE_TOOLS,
    _TEMPLATE_MAX_RETRIES,
    invoke_agent,
)
from .tools.calc import _calc_answer
from .tools.graph import _render_graph
from .tools.html_table import _make_html_table
from .utils import _normalize_graph_points
from .pipeline_state import PipelineState


def _step_parse(state: PipelineState) -> PipelineState:
    out, err = invoke_agent(
        ParserAgent,
        state.problem_text + "\n" + state.solution,
        tools=_TOOLS,
        max_retries=1,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    state.parsed = cast(dict[str, Any], out)
    return state


def _step_concept(state: PipelineState) -> PipelineState:
    out, err = invoke_agent(
        ConceptAgent,
        str(state.parsed),
        tools=_TOOLS,
        expect_json=False,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    state.concept = cast(str, out)
    return state


def _step_template(state: PipelineState) -> PipelineState:
    payload = json.dumps({"parsed": state.parsed, "concept": state.concept})
    out, err = invoke_agent(
        TemplateAgent,
        payload,
        tools=_TEMPLATE_TOOLS,
        max_retries=_TEMPLATE_MAX_RETRIES,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    state.template = cast(dict[str, Any], out)
    return state


def _step_sample(state: PipelineState) -> PipelineState:
    out, err = invoke_agent(
        SampleAgent,
        json.dumps({"template": state.template}),
        tools=_TOOLS,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    if not isinstance(out, dict):
        state.error = "SampleAgent produced non-dict params"
        return state
    state.params = out
    return state


def _step_symbolic(state: PipelineState) -> PipelineState:
    payload = json.dumps({"template": state.template, "params": state.params})
    sol, err = invoke_agent(
        SymbolicSolveAgent,
        payload,
        tools=_TOOLS,
        expect_json=False,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.symbolic_error = err.replace("SymbolicSolveAgent", "Symbolic agents")
        return state
    state.symbolic_solution = cast(str, sol)
    simp, err = invoke_agent(
        SymbolicSimplifyAgent,
        state.symbolic_solution,
        tools=_TOOLS,
        expect_json=False,
    )
    if err:
        state.symbolic_error = err.replace("SymbolicSimplifyAgent", "Symbolic agents")
        return state
    state.symbolic_simplified = cast(str, simp)
    return state


def _step_operations(state: PipelineState) -> PipelineState:
    ops = state.template.get("operations") if isinstance(state.template, dict) else []
    if not ops:
        state.skip_qa = True
        return state

    # Only include the template, params, and intermediates explicitly referenced by
    # the operations rather than the entire pipeline state.  This keeps the payload
    # concise and avoids leaking unrelated data to the agent.
    extra_fields: set[str] = set()
    for op in ops:
        if not isinstance(op, dict):
            continue
        for key, value in op.items():
            if key in {"expr", "output", "outputs"}:
                continue
            if isinstance(value, str):
                if hasattr(state, value):
                    extra_fields.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and hasattr(state, item):
                        extra_fields.add(item)

    data: dict[str, Any] = {"template": state.template, "params": state.params}
    for field in extra_fields:
        val = getattr(state, field)
        if val is not None:
            data[field] = val

    payload = json.dumps({"data": data, "operations": ops})
    out, err = invoke_agent(
        OperationsAgent, payload, tools=_TOOLS, qa_feedback=state.qa_feedback
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    if not isinstance(out, dict):
        state.error = "OperationsAgent produced non-dict output"
        return state
    params_out = out.get("params")
    if isinstance(params_out, dict):
        state.params = params_out
    expected_outputs: set[str] = set()
    for op in ops:
        if isinstance(op, dict):
            out_key = op.get("output")
            if isinstance(out_key, str):
                expected_outputs.add(out_key)
            else:
                out_vals = op.get("outputs")
                if isinstance(out_vals, list):
                    expected_outputs.update(
                        str(o) for o in out_vals if isinstance(o, str)
                    )

    for key, value in out.items():
        if key == "params" or key not in expected_outputs:
            continue
        if hasattr(state, key):
            setattr(state, key, value)
        else:
            state.extras[key] = value
    return state


def _select_graph_spec(
    visual: dict[str, Any], user_spec: Any, force: bool
) -> Any:  # noqa: ANN401 - generic return
    """Choose the graph spec given visual config, user override, and force flag."""
    vtype = visual.get("type")
    if force:
        return user_spec or visual.get("data") or C.DEFAULT_GRAPH_SPEC
    if vtype == "graph":
        return visual.get("data") or user_spec or C.DEFAULT_GRAPH_SPEC
    return None


def _render_table(visual: dict[str, Any]) -> str | None:
    """Render a table visual to HTML if applicable."""
    if visual.get("type") == "table":
        return _make_html_table(json.dumps(visual.get("data", {})))
    return None


def _step_visual(state: PipelineState) -> PipelineState:
    visual = state.template.get("visual") if isinstance(state.template, dict) else None
    if not isinstance(visual, dict):
        visual = {"type": "none"}

    spec = _select_graph_spec(visual, state.graph_spec, bool(state.force_graph))
    if spec is not None:
        if isinstance(spec, dict):
            if state.force_graph and not spec.get("points"):
                spec["points"] = C.DEFAULT_GRAPH_SPEC.get("points", [])
            _normalize_graph_points(cast(dict[str, Any], spec))
        try:
            state.graph_path = _render_graph(json.dumps(spec))
        except Exception as exc:
            state.error = f"Invalid graph spec: {exc}"
        return state

    table_html = _render_table(visual)
    if table_html is not None:
        state.table_html = table_html
    return state


def _step_answer(state: PipelineState) -> PipelineState:
    expr = state.template.get("answer_expression", "0") if isinstance(state.template, dict) else "0"
    state.answer = _calc_answer(expr, json.dumps(state.params))
    return state


def _step_stem_choice(state: PipelineState) -> PipelineState:
    payload: dict[str, Any] = {
        "template": state.template,
        "params": state.params,
    }
    if state.graph_path:
        payload["graph_path"] = state.graph_path
    if state.table_html:
        payload["table_html"] = state.table_html

    out, err = invoke_agent(
        StemChoiceAgent,
        json.dumps(payload),
        tools=_TOOLS,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    state.stem_data = out
    return state


def _step_format(state: PipelineState) -> PipelineState:
    answer_value = str(state.answer)

    payload: dict[str, Any] = {
        "twin_stem": state.stem_data.get("twin_stem") if state.stem_data else None,
        "choices": state.stem_data.get("choices") if state.stem_data else None,
        "answer_value": answer_value,
        "rationale": state.stem_data.get("rationale") if state.stem_data else None,
    }
    if state.graph_path:
        payload["graph_path"] = state.graph_path
    if state.table_html:
        payload["table_html"] = state.table_html

    out, err = invoke_agent(
        FormatterAgent,
        json.dumps(payload),
        tools=_TOOLS,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    out_dict = cast(dict[str, Any], out)

    # Pass-through assets in case the formatter dropped them
    if state.graph_path:
        out_dict.setdefault("graph_path", state.graph_path)
    if state.table_html:
        out_dict.setdefault("table_html", state.table_html)

    state.twin_stem = out_dict.get("twin_stem")
    state.choices = out_dict.get("choices")
    state.answer_index = out_dict.get("answer_index")
    state.answer_value = out_dict.get("answer_value")
    state.rationale = out_dict.get("rationale")
    state.errors = out_dict.get("errors")
    state.graph_path = out_dict.get("graph_path", state.graph_path)
    state.table_html = out_dict.get("table_html", state.table_html)
    return state
