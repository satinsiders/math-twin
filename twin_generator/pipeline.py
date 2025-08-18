"""Linear pipeline orchestration of agents, tools, and helpers."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, cast

from agents.run import Runner as AgentsRunner  # type: ignore

from . import constants as C
from .constants import GraphSpec
from .agents import (
    ConceptAgent,
    FormatterAgent,
    OperationsAgent,
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
    _sanitize_params,
    calc_answer_tool,
    make_html_table_tool,
    render_graph_tool,
)
from .utils import (
    coerce_answers,
    get_final_output,
    safe_json,
    validate_output,
)

__all__ = ["generate_twin"]


# Default tools available to most agents.  TemplateAgent is intentionally given
# a more restrictive list that excludes ``render_graph_tool`` and ``make_html_table_tool``.
_TOOLS = [calc_answer_tool, render_graph_tool, make_html_table_tool]
_TEMPLATE_TOOLS = [calc_answer_tool]
_TEMPLATE_MAX_RETRIES = 3
_JSON_MAX_RETRIES = 3

# Steps whose outputs are expected to be JSON‑serializable before running QA.
# The registry can be extended in tests or by downstream code if new steps are
# introduced that require JSON validation.
_JSON_STEPS = {
    "parse",
    "concept",
    "template",
    "sample",
    "symbolic",
    "operations",
    "visual",
    "answer",
    "stem_choice",
    "format",
}


def invoke_agent(
    agent: Any,
    payload: str,
    *,
    tools: list[Any] | None = None,
    expect_json: bool = True,
    max_retries: int = _JSON_MAX_RETRIES,
) -> tuple[Any | None, str | None]:
    """Run an agent and parse its output.

    Parameters
    ----------
    agent:
        Agent instance to execute via ``AgentsRunner.run_sync``.
    payload:
        Textual input supplied to the agent.
    tools:
        Optional list of tools passed through to the agent runner.
    expect_json:
        When ``True`` the agent's final output is parsed as JSON using
        :func:`safe_json`.
    max_retries:
        Number of attempts to retry the agent when JSON parsing fails.

    Returns
    -------
    tuple
        ``(result, error)`` where ``result`` is the parsed JSON object or raw
        string returned by the agent and ``error`` is an error message when the
        invocation fails or ``None`` on success.

    Error Semantics
    ---------------
    * Exceptions raised by ``AgentsRunner.run_sync`` are captured and surfaced
      as ``"<AgentName> failed: <exc>"``.
    * If ``expect_json`` is ``True`` and the output cannot be parsed after
      ``max_retries`` attempts, the returned error follows the same format.
    """

    agent_name = getattr(agent, "name", getattr(agent, "__name__", str(agent)))
    attempts = 0
    while True:
        try:
            res = AgentsRunner.run_sync(agent, input=payload, tools=tools)
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"{agent_name} failed: {exc}"

        out = get_final_output(res)
        if not expect_json:
            return out, None

        try:
            return safe_json(out), None
        except ValueError as exc:
            attempts += 1
            if attempts >= max_retries:
                return None, f"{agent_name} failed: {exc}"


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
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    def _execute_step(
        self, step: Callable[[dict[str, Any]], dict[str, Any]], data: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        bool,
        list[Callable[[dict[str, Any]], dict[str, Any]]] | None,
        dict[str, Any],
    ]:
        before = dict(data)
        result = step(data)
        skip_qa = bool(result.pop("skip_qa", False))
        next_steps = result.pop("next_steps", None)
        return result, skip_qa, next_steps, before

    def _qa_check(
        self,
        name: str,
        data: dict[str, Any],
        idx: int,
        attempts: int,
        total_steps: int,
        json_required: bool,
    ) -> tuple[bool, str]:
        try:
            qa_in = json.dumps({"step": name, "data": data})
        except (TypeError, ValueError) as exc:
            if not json_required:
                raise RuntimeError(f"QAAgent failed: {exc}")
            qa_out = f"non-serializable data: {exc}"
            self.logger.debug(
                "[twin-generator] step %d/%d: %s QA round %d: %s",
                idx + 1,
                total_steps,
                name,
                attempts + 1,
                qa_out,
            )
            return False, qa_out
        try:
            qa_res = AgentsRunner.run_sync(QAAgent, input=qa_in, tools=_TOOLS)
            qa_out = get_final_output(qa_res).strip().lower()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"QAAgent failed: {exc}")
        self.logger.debug(
            "[twin-generator] step %d/%d: %s QA round %d: %s",
            idx + 1,
            total_steps,
            name,
            attempts + 1,
            qa_out,
        )
        return qa_out == "pass", qa_out

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = dict(inputs)
        steps = list(self.graph.steps)
        idx = 0
        while idx < len(steps):
            step = steps[idx]
            name = step.__name__.replace("_step_", "").lstrip("_")
            attempts = 0
            while True:
                self.logger.debug(
                    "[twin-generator] step %d/%d: %s attempt %d",
                    idx + 1,
                    len(steps),
                    name,
                    attempts + 1,
                )
                data, skip_qa, next_steps, before = self._execute_step(step, data)
                if "error" in data:
                    return {"output": data}
                if skip_qa:
                    if next_steps:
                        steps[idx + 1 : idx + 1] = next_steps
                    break
                json_required = name in _JSON_STEPS
                try:
                    passed, qa_out = self._qa_check(
                        name, data, idx, attempts, len(steps), json_required
                    )
                except RuntimeError as exc:
                    return {"output": {"error": str(exc)}}
                if passed:
                    if next_steps:
                        steps[idx + 1 : idx + 1] = next_steps
                    break
                attempts += 1
                if (
                    self.qa_max_retries is not None
                    and attempts >= self.qa_max_retries
                ):
                    return {
                        "output": {
                            "error": f"QA failed for {name}: {qa_out}",
                        }
                    }
                data = before
            idx += 1
        return {"output": data}


# ---------------------------------------------------------------------------
# Pipeline steps – each mutates and returns `data` dict.
# ---------------------------------------------------------------------------


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
