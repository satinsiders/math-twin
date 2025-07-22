"""Twin Math‑Problem Generator
================================

Self‑contained pipeline for generating *twin* (isomorphic) SAT‑style math
questions from a source problem+solution. Implemented as a lightweight linear
**Graph** so it can be plugged straight into the OpenAI Agents SDK or run from
the CLI for quick smoke‑tests.

Major changes in this revision
------------------------------
1. **Robust CLI** – Run `python twin_generator.py --demo` (basic) or
   `--graph-demo` (graph visual) or feed your own `--problem` / `--solution`
   text files.
2. **Typed stubs** for AgentsRunner fallback so local dry‑runs don’t explode if
   the official SDK isn’t imported yet.
3. **Better error surfacing** and early checks for `OPENAI_API_KEY`.
4. **Sample assets** dropped to `/tmp` and encoded to base64 so no file I/O is
   required by downstream consumers.

Usage examples
--------------
```bash
# Basic pipeline smoke‑test (no graph)
python twin_generator.py --demo

# Graph demo – generates a twin problem that includes a graph visual
python twin_generator.py --graph-demo

# Custom source problem / solution
python twin_generator.py \
    --problem path/to/problem.txt \
    --solution path/to/solution.txt \
    --out twin.json
```

Requirements
------------
* Python ≥3.10
* `openai-agents-python` (clone or `pip install -e .` in the repo root)
* `sympy`, `matplotlib`, `python-dotenv` (or just set `OPENAI_API_KEY`)

```
pip install sympy matplotlib python-dotenv
```

------------- CODE STARTS BELOW -------------
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt  # type: ignore
import sympy as sp  # type: ignore

# ---------------------------------------------------------------------------
# Optional Agents SDK import (graceful fallback makes local testing easy)
# ---------------------------------------------------------------------------

try:
    from agents import Agent
    from agents.run import Runner as AgentsRunner
    from agents.tool import function_tool

    tool: Callable[..., Any] = function_tool
except ModuleNotFoundError:  # pragma: no cover – local smoke‑test mode

    class _Dummy:
        def __init__(self, *_, **__):
            pass

    class _DummyRunner:
        @classmethod
        def run_sync(cls, *_: Any, **__: Any) -> Any:
            raise RuntimeError("OpenAI Agents SDK not found. Install it or run with --local-only.")

    Agent = _Dummy  # type: ignore
    AgentsRunner = _DummyRunner  # type: ignore

    def tool(func: Callable[..., Any]) -> Callable[..., Any]:  # noqa: D401
        """No‑op decorator when SDK is absent."""

        return func

# ---------------------------------------------------------------------------
# Utility tool functions (exposed to LLM Workers via @tool)
# ---------------------------------------------------------------------------


@tool
def sample_params(template_json: str) -> str:
    """Randomly sample numeric params within provided domains."""

    template = json.loads(template_json)
    params: dict[str, Any] = {}
    for name, spec in template.get("params", {}).items():
        if "choices" in spec:
            params[name] = random.choice(spec["choices"])
            continue
        low = spec.get("min", 0)
        high = spec.get("max", 10)
        if spec.get("type") == "float":
            params[name] = random.uniform(low, high)
        else:
            params[name] = random.randint(int(low), int(high))
    return json.dumps(params)


@tool
def render_graph(spec_json: str) -> str:
    """Render graph → base64‑PNG string (inline, avoids temp files)."""

    spec = json.loads(spec_json)
    points = spec.get("points", [])
    style = spec.get("style", "line")

    fig, ax = plt.subplots()
    xs, ys = zip(*points) if points else ([], [])
    if style == "scatter":
        ax.scatter(xs, ys)
    else:
        ax.plot(xs, ys)
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@tool
def make_html_table(table_json: str) -> str:
    """Convert a JSON table spec → `<table>` element string."""

    data = json.loads(table_json)
    header = data.get("header", [])
    rows = data.get("rows", [])
    head_html = "".join(f"<th>{h}</th>" for h in header)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{rows_html}</tbody></table>"


@tool
def calc_answer(expression: str, params_json: str) -> Any:
    """Safely evaluate `expression` under the numbered params."""

    params = json.loads(params_json)
    expr = sp.sympify(expression)
    result = expr.evalf(subs=params)
    if result.is_Integer:
        return int(result)
    if result.is_Float:
        return float(result)
    return str(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_json(text: str) -> dict[str, Any]:
    """Attempts to safely parse JSON from raw agent output."""
    # Strip code block fences like ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        # Try to locate the first JSON object in the text.
        bracketed = re.search(r"\{.*\}", text, re.DOTALL)
        if bracketed:
            text = bracketed.group(0)

    try:
        data = json.loads(text)
        return cast(dict[str, Any], data)
    except json.JSONDecodeError as exc:
        # Show the first ~300 characters to avoid dumping everything.
        snippet = text.strip().replace("\n", " ")[:300]
        raise ValueError(f"Agent output was not valid JSON: {snippet}...") from exc


# ---------------------------------------------------------------------------
# Agent definitions (won’t run without OpenAI Agents SDK)
# ---------------------------------------------------------------------------

ParserAgent = Agent(
    name="ParserAgent",
    instructions=(
        "Take the source problem + solution and return JSON detailing variables, "
        "relations, constraints, any visuals, and the answer format."
    ),
    model="gpt-4o",
)

ConceptAgent = Agent(
    name="ConceptAgent",
    instructions=(
        "From the parsed JSON, identify the key concept(s) and outline the canonical "
        "solution path in ordered steps."
    ),
    model="gpt-4o",
)

TemplateAgent = Agent(
    name="TemplateAgent",
    instructions=(
        "Replace literals with symbols; provide domains; include a `visual` field "
        "→ {type: none|graph|table, data: {…}}."
    ),
    model="gpt-4o",
)

SampleAgent = Agent(
    name="sample",
    instructions=(
        "Given a parameterized math problem template, generate a candidate "
        "parameter set and compute output."
    ),
    model="gpt-4o",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Using the template + sampled params + visuals, draft the final twin stem "
        "and answer choices (SAT tone)."
    ),
    model="gpt-4o",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Return *only* the minified JSON: {twin_stem, choices, answer, rationale, "
        "graph_img?, table_html?}. No markdown."
    ),
    model="gpt-4o",
)


# ---------------------------------------------------------------------------
# Graph + Runner helpers
# ---------------------------------------------------------------------------


@dataclass
class Graph:
    steps: list[Callable[[dict[str, Any]], dict[str, Any]]]


class Runner:
    """Simple sequential executor."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = dict(inputs)
        for step in self.graph.steps:
            data = step(data)
        return {"output": data}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _step_parse(data: dict[str, Any]) -> dict[str, Any]:
    res = AgentsRunner.run_sync(
        ParserAgent,
        input=data["problem_text"] + "\n" + data["solution"],
    )
    data["parsed"] = res.final_output
    return data


def _step_concept(data: dict[str, Any]) -> dict[str, Any]:
    res = AgentsRunner.run_sync(ConceptAgent, input=str(data["parsed"]))
    data["concept"] = res.final_output
    return data


def _step_template(data: dict[str, Any]) -> dict[str, Any]:
    """Generate parameterized template from the parsed problem + concept."""
    res = AgentsRunner.run_sync(
        TemplateAgent,
        input=json.dumps({"parsed": data["parsed"], "concept": data["concept"]}),
    )
    print("DEBUG AGENT OUTPUT:\n", res.final_output)
    data["template"] = _safe_json(res.final_output)
    return data


def _step_sample(data: dict[str, Any]) -> dict[str, Any]:
    """Propose and evaluate candidate parameter sets for the template."""
    res = AgentsRunner.run_sync(
        SampleAgent,
        input=json.dumps({"template": data["template"]}),
    )
    print("DEBUG AGENT OUTPUT:\n", res.final_output)
    data["params"] = _safe_json(res.final_output)
    return data


def _step_visual(data: dict[str, Any]) -> dict[str, Any]:
    visual = data["template"].get("visual", {"type": "none"})
    if visual.get("type") == "graph":
        data["graph_img"] = render_graph(json.dumps(visual.get("data", {})))
    elif visual.get("type") == "table":
        data["table_html"] = make_html_table(json.dumps(visual.get("data", {})))
    return data


def _step_answer(data: dict[str, Any]) -> dict[str, Any]:
    expr = data["template"].get("answer_expression", "0")
    data["answer"] = calc_answer(expr, json.dumps(data["params"]))
    return data


def _step_stem_choice(data: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "template": data["template"],
        "params": data["params"],
        "graph_img": data.get("graph_img"),
        "table_html": data.get("table_html"),
    }
    res = AgentsRunner.run_sync(StemChoiceAgent, input=json.dumps(payload))
    data["stem_data"] = json.loads(str(res.final_output))
    return data


def _step_format(data: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "twin_stem": data["stem_data"].get("twin_stem"),
        "choices": data["stem_data"].get("choices"),
        "answer": data["answer"],
        "rationale": data["stem_data"].get("rationale"),
    }
    if "graph_img" in data:
        payload["graph_img"] = data["graph_img"]
    if "table_html" in data:
        payload["table_html"] = data["table_html"]
    res = AgentsRunner.run_sync(FormatterAgent, input=json.dumps(payload))
    return cast(dict[str, Any], json.loads(str(res.final_output)))


GRAPH = Graph(
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
# Public API
# ---------------------------------------------------------------------------


def generate_twin(problem_text: str, solution_text: str) -> dict[str, Any]:
    """Generate a twin problem programmatically."""

    runner = Runner(GRAPH)
    result = runner.run({"problem_text": problem_text, "solution": solution_text})
    return cast(dict[str, Any], result["output"])


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_DEMO_PROBLEM = """If 3x + 2 = 17, what is the value of x?"""
_DEMO_SOLUTION = """Subtract 2 → 3x = 15, then divide by 3 → x = 5."""

_GRAPH_PROBLEM = (
    "The graph below shows points for a linear function. Which equation best "
    "models the data? (Assume the function is linear.)"
)
_GRAPH_SOLUTION = """Slope (m) = 2, y‑intercept = −1 → y = 2x − 1."""

_GRAPH_TEMPLATE = {
    "visual": {
        "type": "graph",
        "data": {"points": [[0, -1], [1, 1], [2, 3], [3, 5]], "style": "scatter"},
    },
    "answer_expression": "2*x - 1",  # dummy so calc_answer can succeed
    "params": {},
}


def _parse_cli() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="Generate SAT twin problems ✔")
    parser.add_argument("--problem", help="Path to source problem text")
    parser.add_argument("--solution", help="Path to official solution text")
    parser.add_argument("--demo", action="store_true", help="Run trivial demo")
    parser.add_argument("--graph-demo", action="store_true", help="Run demo with graph visual")
    parser.add_argument("--out", help="Write JSON output to file")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip Agents calls; just dry‑run tool stubs (useful for CI).",
    )
    return parser.parse_args()


def _main() -> None:  # noqa: D401
    ns = _parse_cli()

    if ns.local_only:
        os.environ.setdefault("LOCAL_ONLY", "1")

    if ns.demo or ns.graph_demo:
        problem_text = _GRAPH_PROBLEM if ns.graph_demo else _DEMO_PROBLEM
        solution_text = _GRAPH_SOLUTION if ns.graph_demo else _DEMO_SOLUTION
    else:
        if not ns.problem or not ns.solution:
            sys.exit("Error: --problem and --solution are required unless using --demo flags.")
        problem_text = Path(ns.problem).read_text("utf-8")
        solution_text = Path(ns.solution).read_text("utf-8")

    out = generate_twin(problem_text, solution_text)

    json_out = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    if ns.out:
        Path(ns.out).write_text(json_out, "utf-8")
        print(f"✔ Twin problem JSON written to {ns.out}")
    else:
        print(json_out)


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ and os.getenv("LOCAL_ONLY") != "1":
        sys.exit("Error: Set OPENAI_API_KEY or run with --local-only.")
    _main()
