"""Runner interface for executing agents synchronously."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


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
    def run_sync(agent: Any, input: Any) -> Any:  # pragma: no cover - exercised via mocks
        """Execute ``agent`` with ``input`` and return a namespace containing
        the model's response in ``final_output``.

        Parameters
        ----------
        agent:
            Object with ``instructions`` and optional ``model`` attributes.
        input:
            Data passed to the agent.  It is converted to ``str`` and used as the
            user prompt.
        """

        try:  # Import lazily so the dependency is optional for testing.
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("openai package is required to run agents") from exc

        model = getattr(agent, "model", None) or "gpt-4o-mini"
        system_msg = getattr(agent, "instructions", "")
        user_msg = str(input)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        final_output: str

        # The OpenAI python client went through multiple API styles.  We try the
        # modern one first, falling back to older interfaces.
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI()
            if hasattr(client, "responses"):
                resp = client.responses.create(model=model, input=messages)
                final_output = getattr(resp, "output_text", str(resp))
            else:  # pragma: no cover - depends on library version
                resp = client.chat.completions.create(model=model, messages=messages)
                final_output = resp.choices[0].message["content"]
        else:  # pragma: no cover - legacy client
            resp = openai.ChatCompletion.create(model=model, messages=messages)
            final_output = resp.choices[0].message["content"]

        return SimpleNamespace(final_output=final_output)
