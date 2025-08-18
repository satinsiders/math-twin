"""Shared helpers and constants for the twin generator pipeline."""
from __future__ import annotations

from typing import Any

from agents.run import Runner as AgentsRunner  # type: ignore

from .tools import calc_answer_tool, make_html_table_tool, render_graph_tool
from .utils import get_final_output, safe_json

__all__ = [
    "AgentsRunner",
    "invoke_agent",
    "_TOOLS",
    "_TEMPLATE_TOOLS",
    "_TEMPLATE_MAX_RETRIES",
    "_JSON_MAX_RETRIES",
    "_JSON_STEPS",
]

# Default tools available to most agents. TemplateAgent is intentionally given a
# more restrictive list that excludes ``render_graph_tool`` and
# ``make_html_table_tool``.
_TOOLS = [calc_answer_tool, render_graph_tool, make_html_table_tool]
_TEMPLATE_TOOLS = [calc_answer_tool]
_TEMPLATE_MAX_RETRIES = 3
_JSON_MAX_RETRIES = 3

# Steps whose outputs are expected to be JSON-serializable before running QA.
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
    """Run an agent and parse its output."""
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
