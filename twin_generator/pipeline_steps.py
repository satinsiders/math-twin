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
from .tools import _calc_answer, _make_html_table, _render_graph, _sanitize_params
from .utils import coerce_answers, validate_output
from .pipeline_state import PipelineState


def _step_parse(state: PipelineState) -> PipelineState:
    out, err = invoke_agent(
        ParserAgent,
        state.problem_text + "\n" + state.solution,
        tools=_TOOLS,
        max_retries=1,
    )
    if err:
        state.error = err
        return state
    state.parsed = cast(dict[str, Any], out)
    return state


def _step_concept(state: PipelineState) -> PipelineState:
    out, err = invoke_agent(
        ConceptAgent, str(state.parsed), tools=_TOOLS, expect_json=False
    )
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
    )
    if err:
        state.error = err
        return state
    state.template = cast(dict[str, Any], out)
    return state


def _step_sample(state: PipelineState) -> PipelineState:
    out, err = invoke_agent(
        SampleAgent, json.dumps({"template": state.template}), tools=_TOOLS
    )
    if err:
        state.error = err
        return state
    if not isinstance(out, dict):
        state.error = "SampleAgent produced non-dict params"
        return state
    sanitized, _ = _sanitize_params(out)
    state.params = {k: out[k] for k in sanitized}
    return state


def _step_symbolic(state: PipelineState) -> PipelineState:
    payload = json.dumps({"template": state.template, "params": state.params})
    sol, err = invoke_agent(
        SymbolicSolveAgent, payload, tools=_TOOLS, expect_json=False
    )
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
    out, err = invoke_agent(OperationsAgent, payload, tools=_TOOLS)
    if err:
        state.error = err
        return state
    if isinstance(out, dict):
        params_out = out.get("params")
        if isinstance(params_out, dict):
            sanitized, _ = _sanitize_params(params_out)
            state.params = {k: params_out[k] for k in sanitized}
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


def _step_visual(state: PipelineState) -> PipelineState:
    visual = state.template.get("visual") if isinstance(state.template, dict) else None
    if not isinstance(visual, dict):
        visual = {"type": "none"}
    force = bool(state.force_graph)
    user_spec = state.graph_spec

    vtype = visual.get("type")
    if vtype == "graph":
        spec = visual.get("data", {}) or C.DEFAULT_GRAPH_SPEC
        state.graph_path = _render_graph(json.dumps(spec))
        return state

    if force:
        gspec = user_spec or C.DEFAULT_GRAPH_SPEC
        state.graph_path = _render_graph(json.dumps(gspec))
        return state

    if vtype == "table":
        state.table_html = _make_html_table(json.dumps(visual.get("data", {})))
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

    out, err = invoke_agent(StemChoiceAgent, json.dumps(payload), tools=_TOOLS)
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

    out, err = invoke_agent(FormatterAgent, json.dumps(payload), tools=_TOOLS)
    if err:
        state.error = err
        return state
    out_dict = cast(dict[str, Any], out)

    # Pass-through assets in case the formatter dropped them
    if state.graph_path:
        out_dict.setdefault("graph_path", state.graph_path)
    if state.table_html:
        out_dict.setdefault("table_html", state.table_html)

    out_dict = coerce_answers(out_dict)
    out_dict = validate_output(out_dict)

    state.twin_stem = out_dict.get("twin_stem")
    state.choices = out_dict.get("choices")
    state.answer_index = out_dict.get("answer_index")
    state.answer_value = out_dict.get("answer_value")
    state.rationale = out_dict.get("rationale")
    state.errors = out_dict.get("errors")
    state.graph_path = out_dict.get("graph_path", state.graph_path)
    state.table_html = out_dict.get("table_html", state.table_html)
    return state
