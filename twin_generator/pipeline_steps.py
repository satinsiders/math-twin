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


def _step_parse(data: dict[str, Any]) -> dict[str, Any]:
    out, err = invoke_agent(
        ParserAgent,
        data["problem_text"] + "\n" + data["solution"],
        tools=_TOOLS,
        max_retries=1,
    )
    if err:
        data["error"] = err
        return data
    data["parsed"] = cast(dict[str, Any], out)
    return data


def _step_concept(data: dict[str, Any]) -> dict[str, Any]:
    out, err = invoke_agent(
        ConceptAgent, str(data["parsed"]), tools=_TOOLS, expect_json=False
    )
    if err:
        data["error"] = err
        return data
    data["concept"] = cast(str, out)
    return data


def _step_template(data: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps({"parsed": data["parsed"], "concept": data["concept"]})
    out, err = invoke_agent(
        TemplateAgent,
        payload,
        tools=_TEMPLATE_TOOLS,
        max_retries=_TEMPLATE_MAX_RETRIES,
    )
    if err:
        data["error"] = err
        return data
    data["template"] = cast(dict[str, Any], out)
    return data


def _step_sample(data: dict[str, Any]) -> dict[str, Any]:
    out, err = invoke_agent(
        SampleAgent, json.dumps({"template": data["template"]}), tools=_TOOLS
    )
    if err:
        data["error"] = err
        return data
    if not isinstance(out, dict):
        data["error"] = "SampleAgent produced non-dict params"
        return data
    _, skipped = _sanitize_params(out)
    if skipped:
        bad = ", ".join(f"{k}={out[k]!r}" for k in skipped)
        data["error"] = f"SampleAgent produced non-numeric params: {bad}"
        return data
    data["params"] = out
    return data


def _step_symbolic(data: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps({"template": data["template"], "params": data["params"]})
    sol, err = invoke_agent(
        SymbolicSolveAgent, payload, tools=_TOOLS, expect_json=False
    )
    if err:
        data["symbolic_error"] = err.replace("SymbolicSolveAgent", "Symbolic agents")
        return data
    data["symbolic_solution"] = cast(str, sol)
    simp, err = invoke_agent(
        SymbolicSimplifyAgent, data["symbolic_solution"], tools=_TOOLS, expect_json=False
    )
    if err:
        data["symbolic_error"] = err.replace("SymbolicSimplifyAgent", "Symbolic agents")
        return data
    data["symbolic_simplified"] = cast(str, simp)
    return data


def _step_operations(data: dict[str, Any]) -> dict[str, Any]:
    ops = data.get("template", {}).get("operations") or []
    if not ops:
        data["skip_qa"] = True
        return data

    payload = json.dumps({"data": data, "operations": ops})
    out, err = invoke_agent(OperationsAgent, payload, tools=_TOOLS)
    if err:
        data["error"] = err
        return data
    if isinstance(out, dict):
        params_out = out.get("params")
        if isinstance(params_out, dict):
            _, skipped = _sanitize_params(params_out)
            if skipped:
                bad = ", ".join(f"{k}={params_out[k]!r}" for k in skipped)
                data["error"] = f"OperationsAgent produced non-numeric params: {bad}"
                return data
        data.update(out)
    return data


def _step_visual(data: dict[str, Any]) -> dict[str, Any]:
    visual = data.get("template", {}).get("visual")
    if not isinstance(visual, dict):
        visual = {"type": "none"}
    force = bool(data.get("force_graph"))
    user_spec = data.get("graph_spec")

    vtype = visual.get("type")
    if vtype == "graph":
        spec = visual.get("data", {}) or C.DEFAULT_GRAPH_SPEC
        data["graph_path"] = _render_graph(json.dumps(spec))
        return data

    if force:
        gspec = user_spec or C.DEFAULT_GRAPH_SPEC
        data["graph_path"] = _render_graph(json.dumps(gspec))
        return data

    if vtype == "table":
        data["table_html"] = _make_html_table(json.dumps(visual.get("data", {})))
    return data


def _step_answer(data: dict[str, Any]) -> dict[str, Any]:
    expr = data["template"].get("answer_expression", "0")
    data["answer"] = _calc_answer(expr, json.dumps(data["params"]))
    return data


def _step_stem_choice(data: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "template": data["template"],
        "params": data["params"],
    }
    if "graph_path" in data:
        payload["graph_path"] = data["graph_path"]
    if "table_html" in data:
        payload["table_html"] = data["table_html"]

    out, err = invoke_agent(StemChoiceAgent, json.dumps(payload), tools=_TOOLS)
    if err:
        data["error"] = err
        return data
    data["stem_data"] = out
    return data


def _step_format(data: dict[str, Any]) -> dict[str, Any]:
    answer_value = str(data["answer"])

    payload: dict[str, Any] = {
        "twin_stem": data["stem_data"].get("twin_stem"),
        "choices": data["stem_data"].get("choices"),
        "answer_value": answer_value,
        "rationale": data["stem_data"].get("rationale"),
    }
    if "graph_path" in data:
        payload["graph_path"] = data["graph_path"]
    if "table_html" in data:
        payload["table_html"] = data["table_html"]

    out, err = invoke_agent(FormatterAgent, json.dumps(payload), tools=_TOOLS)
    if err:
        data["error"] = err
        return data
    out = cast(dict[str, Any], out)

    # Pass-through assets in case the formatter dropped them
    if "graph_path" in data:
        out["graph_path"] = data["graph_path"]
    if "table_html" in data:
        out["table_html"] = data["table_html"]

    out = coerce_answers(out)
    out = validate_output(out)
    return out
