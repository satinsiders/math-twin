"""Runner interface for executing agents synchronously."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast, Sequence


class Runner:
    """Minimal runner that executes an :class:`Agent` via the OpenAI API.

    The real project this repository was extracted from relies on the
    `openai` package to execute small helper agents.  The original implementation
    is intentionally tiny – the goal is simply to take an ``Agent`` instance and
    a piece of ``input`` and return the model's textual output.  The tests for
    this kata monkeypatch :func:`run_sync`, so the runner needs to be small and
    dependency free at import time while still working when used in practice.

    The method below attempts to import ``openai`` lazily so importing the
    package does not require the dependency.  When invoked it uses whichever API
    surface is available (``responses.create`` for newer versions or
    ``chat.completions.create``/``ChatCompletion.create`` for older ones) and
    always returns an object with a ``final_output`` attribute.
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

        model = getattr(agent, "model", None) or "gpt-4o-mini"
        system_msg = getattr(agent, "instructions", "")
        user_msg = str(input)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        final_output: str

        # The OpenAI python client went through multiple API styles.  We try the
        # modern one first, falling back to older interfaces.
        if hasattr(openai, "OpenAI"):
            client: Any = openai.OpenAI()
            if hasattr(client, "responses"):
                kwargs: dict[str, Any] = {
                    "model": model,
                    "input": cast(Any, messages),
                }
                if tools:
                    kwargs["tools"] = list(tools)
                resp: Any = client.responses.create(**kwargs)
                final_output = getattr(resp, "output_text", str(resp))
            else:  # pragma: no cover - depends on library version
                kwargs = {
                    "model": model,
                    "messages": cast(Any, messages),
                }
                if tools:
                    kwargs["tools"] = list(tools)
                resp = cast(Any, client.chat.completions.create(**kwargs))
                final_output = resp.choices[0].message["content"]
        else:  # pragma: no cover - legacy client
            chat_cls = getattr(openai, "ChatCompletion")  # type: ignore[attr-defined]
            kwargs = {"model": model, "messages": cast(Any, messages)}
            if tools:
                kwargs["functions"] = [t["function"] for t in tools]
            resp = cast(Any, chat_cls.create(**kwargs))
            final_output = resp.choices[0].message["content"]

        return SimpleNamespace(final_output=final_output)
