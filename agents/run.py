"""Runner interface for executing agents synchronously."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Sequence, cast


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
        if tools:
            kwargs["tools"] = list(tools)
        resp: Any = client.responses.create(**kwargs)

        # Build mapping of tool names to callable functions from ``twin_generator.tools``
        tool_funcs: dict[str, Any] = {}
        try:  # Import lazily to avoid heavy deps when not needed
            from twin_generator import tools as _tg_tools

            for attr in dir(_tg_tools):
                if attr.endswith("_tool"):
                    spec = getattr(_tg_tools, attr)
                    if isinstance(spec, dict):
                        name = spec.get("name")
                    else:  # pragma: no cover - defensive
                        name = getattr(spec, "name", None)
                    func = getattr(_tg_tools, str(name), None)
                    if func:
                        tool_funcs[attr] = func
                        tool_funcs[str(name)] = func
        except Exception:  # pragma: no cover - defensive
            pass

        # Loop while the API requires tool outputs or is still processing
        while getattr(resp, "status", None) != "completed":
            required = getattr(resp, "required_action", None)
            if required:
                submit = getattr(required, "submit_tool_outputs", None)
                tool_calls = [] if submit is None else getattr(submit, "tool_calls", [])
                outputs: list[dict[str, str]] = []
                for call in tool_calls:
                    func_info = getattr(call, "function", None)
                    name = getattr(func_info, "name", None) if func_info else None
                    args_json = (
                        getattr(func_info, "arguments", "{}") if func_info else "{}"
                    )
                    args = json.loads(args_json)
                    func = tool_funcs.get(str(name))
                    if not func:
                        raise RuntimeError(f"tool {name!r} not implemented")
                    result = func(**args)
                    call_id = getattr(call, "id", None)
                    outputs.append(
                        {"tool_call_id": str(call_id), "output": str(result)}
                    )
                resp = client.responses.submit_tool_outputs(
                    response_id=getattr(resp, "id"),
                    tool_outputs=outputs,
                )
            else:
                resp = client.responses.get(getattr(resp, "id"))

        final_output = getattr(resp, "output_text", str(resp))

        return SimpleNamespace(final_output=final_output)
