"""Linear pipeline orchestration of agents, tools, and helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, cast

from agents.run import Runner as AgentsRunner  # type: ignore

from . import constants as C
from .constants import GraphSpec
from .agents import (
    ConceptAgent,
    FormatterAgent,
    ParserAgent,
    QAAgent,
    SampleAgent,
    StemChoiceAgent,
    TemplateAgent,
    SymbolicSolveAgent,
    SymbolicSimplifyAgent,
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
    """Minimal re‑implementation of a sequential task executor with QA checks.

    The runner executes each pipeline step sequentially and performs a QA check
    after every step.  If the QA step fails the corresponding pipeline step is
    retried until QA passes.  Optionally a maximum number of QA retries can be
    supplied to prevent infinite loops.
    """

    def __init__(
        self,
        graph: _Graph,
        *,
        verbose: bool = False,
        qa_max_retries: int | None = None,
    ) -> None:
        self.graph = graph
        self.verbose = verbose
        self.qa_max_retries = qa_max_retries

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401 – generic return
        data = dict(inputs)
        steps = list(self.graph.steps)
        idx = 0
        while idx < len(steps):
            step = steps[idx]
            name = step.__name__.replace("_step_", "").lstrip("_")
            attempts = 0
            while True:
                before = dict(data)
                if self.verbose:
                    print(
                        f"[twin-generator] step {idx + 1}/{len(steps)}: {name} "
                        f"attempt {attempts + 1}"
                    )
                data = step(data)
                skip_qa = bool(data.pop("skip_qa", False))
                if "error" in data:
                    break
                next_steps = data.pop("next_steps", None)
                if skip_qa:
                    if next_steps:
                        steps[idx + 1 : idx + 1] = next_steps
                    break
                try:
                    qa_in = json.dumps({"step": name, "data": data})
                    qa_res = AgentsRunner.run_sync(QAAgent, input=qa_in)
                    qa_out = get_final_output(qa_res).strip().lower()
                except Exception as exc:  # pragma: no cover - defensive
                    data["error"] = f"QAAgent failed: {exc}"
                    break
                if self.verbose:
                    print(
                        f"[twin-generator] step {idx + 1}/{len(steps)}: {name} "
                        f"QA round {attempts + 1}: {qa_out}"
                    )
                if qa_out == "pass":
                    if next_steps:
                        steps[idx + 1 : idx + 1] = next_steps
                    break
                attempts += 1
                if (
                    self.qa_max_retries is not None
                    and attempts >= self.qa_max_retries
                ):
                    data["error"] = f"QA failed for {name}: {qa_out}"
                    break
                data = before
            if "error" in data:
                break
            idx += 1
        return {"output": data}


# ---------------------------------------------------------------------------
# Pipeline steps – each mutates and returns `data` dict.
# ---------------------------------------------------------------------------


def _step_parse(data: dict[str, Any]) -> dict[str, Any]:
    try:
        res = AgentsRunner.run_sync(ParserAgent, input=data["problem_text"] + "\n" + data["solution"])
        data["parsed"] = get_final_output(res)
    except Exception as exc:  # pragma: no cover - defensive
        data["error"] = f"ParserAgent failed: {exc}"
    return data


def _step_concept(data: dict[str, Any]) -> dict[str, Any]:
    try:
        res = AgentsRunner.run_sync(ConceptAgent, input=str(data["parsed"]))
        data["concept"] = get_final_output(res)
    except Exception as exc:  # pragma: no cover - defensive
        data["error"] = f"ConceptAgent failed: {exc}"
    return data


def _step_template(data: dict[str, Any]) -> dict[str, Any]:
    try:
        res = AgentsRunner.run_sync(
            TemplateAgent,
            input=json.dumps({"parsed": data["parsed"], "concept": data["concept"]}),
        )
        data["template"] = safe_json(get_final_output(res))
    except Exception as exc:  # pragma: no cover - defensive
        data["error"] = f"TemplateAgent failed: {exc}"
    return data


def _step_sample(data: dict[str, Any]) -> dict[str, Any]:
    try:
        res = AgentsRunner.run_sync(
            SampleAgent,
            input=json.dumps({"template": data["template"]}),
        )
        data["params"] = safe_json(get_final_output(res))
    except Exception as exc:  # pragma: no cover - defensive
        data["error"] = f"SampleAgent failed: {exc}"
    return data


def _step_symbolic(data: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.dumps({"template": data["template"], "params": data["params"]})
        res = AgentsRunner.run_sync(SymbolicSolveAgent, input=payload)
        data["symbolic_solution"] = get_final_output(res)
        res2 = AgentsRunner.run_sync(SymbolicSimplifyAgent, input=data["symbolic_solution"])
        data["symbolic_simplified"] = get_final_output(res2)
    except Exception as exc:  # pragma: no cover - defensive
        data["symbolic_error"] = f"Symbolic agents failed: {exc}"
    return data


def _step_operations(data: dict[str, Any]) -> dict[str, Any]:
    ops = data.get("template", {}).get("operations") or []
    steps: list[Callable[[dict[str, Any]], dict[str, Any]]] = []
    for idx, op in enumerate(ops):
        kind = op.get("kind")
        condition = op.get("condition")
        if kind == "sympy":
            expr = op.get("expr", "0")
            out_key = op.get("output", f"sym_{idx}")

            def _op_step(
                d: dict[str, Any],
                *,
                _expr: str = expr,
                _out: str = out_key,
                _cond: str | None = condition,
            ) -> dict[str, Any]:
                if _cond and not d.get(cast(str, _cond)):
                    return d
                params = {}
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        params[k] = v
                    elif isinstance(v, str):
                        try:
                            float(v)
                            params[k] = float(v)
                        except Exception:
                            pass
                d[_out] = _calc_answer(_expr, json.dumps(params))
                return d

            _op_step.__name__ = f"_step_op_{idx}"
            steps.append(_op_step)
        elif kind == "agent":
            agent_name = op.get("agent")
            input_key = op.get("input_key")
            out_key = op.get("output", f"agent_{idx}")

            def _agent_step(
                d: dict[str, Any],
                *,
                _agent_name: str = cast(str, agent_name),
                _input_key: str = cast(str, input_key),
                _out: str = out_key,
                _cond: str | None = condition,
            ) -> dict[str, Any]:
                if _cond and not d.get(cast(str, _cond)):
                    return d
                agent = globals().get(_agent_name)
                if agent is None:
                    d["error"] = f"Unknown agent {_agent_name}"
                    return d
                res = AgentsRunner.run_sync(agent, input=d.get(_input_key))
                d[_out] = get_final_output(res)
                return d

            _agent_step.__name__ = f"_step_op_{idx}"
            steps.append(_agent_step)

    if steps:
        data["next_steps"] = steps
    else:
        # No operations to perform; skip QA for this no-op step
        data["skip_qa"] = True
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

    try:
        res = AgentsRunner.run_sync(StemChoiceAgent, input=json.dumps(payload))
        data["stem_data"] = safe_json(get_final_output(res))
    except Exception as exc:  # pragma: no cover - defensive
        data["error"] = f"StemChoiceAgent failed: {exc}"
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

    try:
        res = AgentsRunner.run_sync(FormatterAgent, input=json.dumps(payload))
        out = safe_json(get_final_output(res))
    except Exception as exc:  # pragma: no cover - defensive
        data["error"] = f"FormatterAgent failed: {exc}"
        return data

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
        _step_symbolic,
        _step_operations,
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
