"""Runner interface for executing agents synchronously."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence, cast

import json

from twin_generator import tools as tg_tools


class Runner:
    """Minimal runner that executes an :class:`Agent` via the OpenAI *Responses* API.

    The real project this repository was extracted from relies on the
    ``openai`` package to execute small helper agents.  The original
    implementation is intentionally tiny – the goal is simply to take an
    ``Agent`` instance and a piece of ``input`` and return the model's textual
    output.  The tests for this kata monkeypatch :func:`run_sync`, so the runner
    needs to be small and dependency free at import time while still working
    when used in practice.

    Only the modern Responses API is supported.  Older Chat Completions
    interfaces have been removed to ensure the project targets the Agents SDK
    and Responses API exclusively.  The method always returns an object with a
    ``final_output`` attribute containing the model's response.
    """

    @staticmethod
    def run_sync(
        agent: Any,
        input: Any,
        *,
        tools: Sequence[Any] | None = None,
    ) -> Any:  # pragma: no cover - exercised via mocks
        """Execute ``agent`` with ``input`` and return a namespace containing
        the model's response in ``final_output``.

        Parameters
        ----------
        agent:
            Object with ``instructions`` and optional ``model`` attributes.
        input:
            Data passed to the agent.  It is converted to ``str`` and used as the
            user prompt.
        tools:
            Optional collection of OpenAI tool definitions the model may call.
        """

        try:  # Import lazily so the dependency is optional for testing.
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("openai package is required to run agents") from exc

        if not hasattr(openai, "OpenAI"):
            raise RuntimeError(
                "openai.OpenAI client with Responses API support is required"
            )

        client: Any = openai.OpenAI()
        if not hasattr(client, "responses"):
            raise RuntimeError(
                "openai client does not support Responses API; upgrade your package"
            )

        model = getattr(agent, "model", None) or "gpt-4o-mini"
        system_msg = getattr(agent, "instructions", "")
        user_msg = str(input)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "input": cast(Any, messages),
        }
        tool_map: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = list(tools)
            for t in tools:
                name = t.get("name") if isinstance(t, dict) else None
                if isinstance(name, str):
                    fn = getattr(tg_tools, name, None)
                    if callable(fn):
                        tool_map[name] = fn

        resp: Any = client.responses.create(**kwargs)

        while getattr(resp, "required_action", None):
            action = resp.required_action
            if getattr(action, "type", None) != "submit_tool_outputs":
                break
            submit = action.submit_tool_outputs
            calls = getattr(submit, "tool_calls", [])
            outputs: list[dict[str, str]] = []
            for call in calls:
                func = getattr(call, "function", None)
                if func is None:
                    continue
                name = getattr(func, "name", "")
                args_json = getattr(func, "arguments", "{}")
                args = json.loads(args_json)
                fn = tool_map.get(name)
                if fn is None:
                    raise RuntimeError(f"unknown tool requested: {name}")
                result = fn(**args)
                outputs.append({
                    "tool_call_id": getattr(call, "id", ""),
                    "output": str(result),
                })
            resp = client.responses.submit_tool_outputs(
                response_id=getattr(resp, "id"),
                tool_outputs=outputs,
            )

        final_output = getattr(resp, "output_text", str(resp))

        return SimpleNamespace(final_output=final_output)
