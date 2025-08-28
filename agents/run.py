"""Runner interface for executing agents synchronously."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast, Sequence, ClassVar


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

    _SANITIZED_CACHE: ClassVar[dict[str, dict[str, Any]]] = {}

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
        user_msg_raw = str(input)

        # Detect image-aware flow for vision agents (e.g., GraphVisionAgent)
        is_vision = bool(getattr(agent, "requires_vision", False)) or (
            getattr(agent, "name", "").lower() == "graphvisionagent"
        )

        messages: list[dict[str, Any]]
        if is_vision:
            graph_url: str | None = None
            extra_text: str = ""
            try:
                payload = json.loads(user_msg_raw)
            except Exception:
                payload = {"prompt": user_msg_raw}
            graph_url = payload.get("graph_url") if isinstance(payload, dict) else None
            # Preserve any useful context to help the model interpret the image
            for key in ("problem", "solution", "parsed"):
                if isinstance(payload, dict) and key in payload and payload[key] is not None:
                    extra_text += f"\n{key}: {payload[key]}"
            # Fallback to the entire raw message if no structured fields
            if not extra_text.strip():
                extra_text = user_msg_raw
            user_content: list[dict[str, Any]] = []
            user_content.append({"type": "input_text", "text": extra_text.strip()})
            if graph_url:
                user_content.append({"type": "input_image", "image_url": str(graph_url)})
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg_raw},
            ]

        kwargs: dict[str, Any] = {
            "model": model,
            "input": cast(Any, messages),
        }

        # Optional: enable higher reasoning effort for accuracy-critical agents
        effort = getattr(agent, "reasoning_effort", None)
        if effort:
            try:
                kwargs["reasoning"] = {"effort": str(effort)}
            except Exception:
                pass

        sanitized_tools, tool_map = Runner._sanitize_tools(tools)
        if sanitized_tools:
            kwargs["tools"] = sanitized_tools

        resp: Any = client.responses.create(**kwargs)

        resp = Runner._execute_tool_calls(client, resp, tool_map)

        # Extract a reliable text output from the Responses object.
        final_output = Runner._extract_output_text(resp)

        return SimpleNamespace(final_output=final_output)

    @staticmethod
    def _extract_output_text(resp: Any) -> str:
        """Best-effort extraction of textual output from a Responses object.

        Handles multiple SDK shapes:
        - ``resp.output_text`` when available
        - ``resp.output`` as a list of messages with nested content/text/value
        - Falls back to ``str(resp)`` if no text could be found
        """

        def _is_nonempty_str(val: Any) -> bool:
            return isinstance(val, str) and val.strip() != ""

        # 1) Prefer consolidated property if present and non-empty
        consolidated = getattr(resp, "output_text", None)
        if _is_nonempty_str(consolidated):
            return consolidated

        # 2) Walk the contemporary Responses shape: resp.output -> [message]
        # Each message often has .content -> [parts], each part may have .text.value
        try:
            output = getattr(resp, "output", None)
        except Exception:
            output = None

        texts: list[str] = []

        def _collect(obj: Any, depth: int = 0) -> None:
            # Guard against overly deep or cyclic structures
            if depth > 5 or obj is None:
                return
            # Direct strings
            if isinstance(obj, str):
                if obj.strip():
                    texts.append(obj)
                return
            # Lists/tuples of items
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    _collect(it, depth + 1)
                return
            # Dictionaries: prefer likely text-bearing keys
            if isinstance(obj, dict):
                for key in ("text", "value", "content"):
                    if key in obj:
                        _collect(obj[key], depth + 1)
                return
            # Generic objects: probe common attributes seen in SDKs
            for attr in ("text", "value", "content"):
                try:
                    val = getattr(obj, attr)
                except Exception:
                    val = None
                if val is not None:
                    _collect(val, depth + 1)

        if output is not None:
            _collect(output)

        # If we managed to gather any text fragments, join them
        if texts:
            joined = "\n".join(t for t in texts if t.strip())
            if joined.strip():
                return joined

        # 3) Avoid dumping the entire response object; return empty string
        # so callers can decide how to handle missing text.
        return ""

    @classmethod
    def _sanitize_tools(
        cls, tools: Sequence[Any] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tool_map: dict[str, Any] = {}
        sanitized: list[dict[str, Any]] = []
        if not tools:
            return sanitized, tool_map
        for t in tools:
            name = t.get("name")
            if name:
                tool_map[name] = t
                cached = cls._SANITIZED_CACHE.get(name)
                if cached is None:
                    sanitized_tool = {
                        k: v for k, v in t.items() if not k.startswith("_")
                    }
                    cls._SANITIZED_CACHE[name] = sanitized_tool.copy()
                    sanitized.append(sanitized_tool.copy())
                else:
                    sanitized.append(cached.copy())
            else:
                sanitized.append({k: v for k, v in t.items() if not k.startswith("_")})
        return sanitized, tool_map

    @staticmethod
    def _execute_tool_calls(
        client: Any, resp: Any, tool_map: dict[str, Any]
    ) -> Any:
        max_iterations = 10
        iterations = 0
        while getattr(resp, "status", None) == "requires_action":
            if iterations >= max_iterations:
                status = getattr(resp, "status", None)
                raise RuntimeError(
                    "tool calls did not resolve after "
                    f"{max_iterations} iterations; status: {status}"
                )
            action = getattr(resp, "required_action")
            submit = getattr(action, "submit_tool_outputs")
            calls = getattr(submit, "tool_calls", [])
            outputs = []
            for call in calls:
                name = cast(str, getattr(call.function, "name"))
                tool = tool_map.get(name)
                if not tool:
                    raise RuntimeError(f"tool {name} not provided")
                func = tool.get("_func")
                if not callable(func):
                    raise RuntimeError(f"tool {name} is missing callable")
                try:
                    args = json.loads(getattr(call.function, "arguments", "{}"))
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"tool {name} provided invalid JSON arguments: {e}"
                    ) from e
                result = func(**args)
                outputs.append({"tool_call_id": call.id, "output": json.dumps(result, default=str)})
            resp = client.responses.submit_tool_outputs(
                response_id=resp.id, tool_outputs=outputs
            )
            iterations += 1
        return resp
