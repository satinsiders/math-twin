"""Linear pipeline orchestration of agents, tools, and helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from agents.run import Runner as AgentsRunner  # type: ignore

from . import constants as C
from .constants import GraphSpec
from .agents import (
    ConceptAgent,
    FormatterAgent,
    ParserAgent,
    SampleAgent,
    StemChoiceAgent,
    TemplateAgent,
)
from .tools import (
    _calc_answer,      # internal helper functions (NOT the FunctionTool wrappers!)
    _make_html_table,
    _render_graph,
)
from .utils import (
    coerce_answers,
    get_final_output,
    safe_json,
    validate_output,
)

__all__ = ["generate_twin"]


# ---------------------------------------------------------------------------
# Simple sequential graph executor – keeps dependencies minimal.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Graph:
    steps: list[Callable[[dict[str, Any]], dict[str, Any]]]


class _Runner:
    """Minimal re‑implementation of a sequential task executor."""

    def __init__(self, graph: _Graph, *, verbose: bool = False) -> None:
        self.graph = graph
        self.verbose = verbose

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401 – generic return
        data = dict(inputs)
        for step in self.graph.steps:
            if self.verbose:
                name = step.__name__.replace("_step_", "").lstrip("_")
                print(f"[twin-generator] {name}…")
            data = step(data)
        return {"output": data}


# ---------------------------------------------------------------------------
# Pipeline steps – each mutates and returns `data` dict.
# ---------------------------------------------------------------------------


def _step_parse(data: dict[str, Any]) -> dict[str, Any]:
    res = AgentsRunner.run_sync(ParserAgent, input=data["problem_text"] + "\n" + data["solution"])
    data["parsed"] = get_final_output(res)
    return data


def _step_concept(data: dict[str, Any]) -> dict[str, Any]:
    res = AgentsRunner.run_sync(ConceptAgent, input=str(data["parsed"]))
    data["concept"] = get_final_output(res)
    return data


def _step_template(data: dict[str, Any]) -> dict[str, Any]:
    res = AgentsRunner.run_sync(TemplateAgent, input=json.dumps({"parsed": data["parsed"], "concept": data["concept"]}))
    data["template"] = safe_json(get_final_output(res))
    return data


def _step_sample(data: dict[str, Any]) -> dict[str, Any]:
    res = AgentsRunner.run_sync(SampleAgent, input=json.dumps({"template": data["template"]}))
    data["params"] = safe_json(get_final_output(res))
    return data


def _step_visual(data: dict[str, Any]) -> dict[str, Any]:
    visual = data.get("template", {}).get("visual") or {"type": "none"}
    force = bool(data.get("force_graph"))
    user_spec = data.get("graph_spec")

    if visual.get("type") == "graph":
        spec = visual.get("data", {}) or C.DEFAULT_GRAPH_SPEC
        data["graph_path"] = _render_graph(json.dumps(spec))
        return data

    if force:
        gspec = user_spec or C.DEFAULT_GRAPH_SPEC
        data["graph_path"] = _render_graph(json.dumps(gspec))
        return data

    if visual.get("type") == "table":
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

    res = AgentsRunner.run_sync(StemChoiceAgent, input=json.dumps(payload))
    data["stem_data"] = safe_json(get_final_output(res))
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

    res = AgentsRunner.run_sync(FormatterAgent, input=json.dumps(payload))
    out = safe_json(get_final_output(res))

    # Pass‑through assets in case the formatter dropped them
    if "graph_path" in data:
        out["graph_path"] = data["graph_path"]
    if "table_html" in data:
        out["table_html"] = data["table_html"]

    out = coerce_answers(out)
    out = validate_output(out)
    return out


_PIPELINE = _Graph(
    steps=[
        _step_parse,
        _step_concept,
        _step_template,
        _step_sample,
        _step_visual,
        _step_answer,
        _step_stem_choice,
        _step_format,
    ]
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_twin(
    problem_text: str,
    solution_text: str,
    *,
    force_graph: bool = False,
    graph_spec: GraphSpec | None = None,
    verbose: bool = False,
) -> dict[str, Any]:  # noqa: ANN401 – generic return
    """Generate a twin SAT‑style math question given a source problem/solution."""
    runner = _Runner(_PIPELINE, verbose=verbose)
    result = runner.run(
        {
            "problem_text": problem_text,
            "solution": solution_text,
            "force_graph": force_graph,
            "graph_spec": graph_spec,
        }
    )
    return cast(dict[str, Any], result["output"])
