"""Utilities for defining simple function-based tools."""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar, get_type_hints


F = TypeVar("F", bound=Callable[..., Any])


def _annotation_to_type(ann: Any) -> str:
    """Map a Python type annotation to a JSON schema primitive."""
    if ann in (int, float):
        return "number"
    if ann is bool:
        return "boolean"
    if ann in (dict, list):
        return "object"
    return "string"


def function_tool(func: F) -> dict[str, Any]:
    """Return an OpenAI tool definition for ``func``.

    The returned dictionary follows the "tool" schema understood by the
    OpenAI APIs and can be supplied directly when invoking a model so it may
    call the wrapped function via tool-calling.
    """

    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, str)
        properties[name] = {"type": _annotation_to_type(ann)}
        if param.default is inspect._empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    # The OpenAI Responses/Agents APIs expect a top-level ``name`` field for
    # each tool in addition to the ``type`` and ``parameters`` schema.
    name = func.__name__
    description = func.__doc__ or ""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": schema,
        "_func": func,
    }
