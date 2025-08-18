"""Utilities for defining simple function-based tools."""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar, get_args, get_origin, get_type_hints, Union


F = TypeVar("F", bound=Callable[..., Any])


_INTROSPECTION_CACHE: dict[Callable[..., Any], tuple[inspect.Signature, dict[str, Any]]] = {}


def _annotation_to_type(ann: Any) -> str:
    """Map a Python type annotation to a JSON schema primitive.

    Supported annotations include primitives (``int``, ``float``, ``bool``,
    ``str``), container types such as ``list[T]`` and ``dict[K, V]``, and
    ``Optional``/``Union`` where all members resolve to the same JSON schema
    type. Mixed unions default to ``"string"``.
    """

    origin = get_origin(ann)
    if origin is Union:
        args = [a for a in get_args(ann) if a is not type(None)]  # Optional
        if not args:
            return "string"
        types = {_annotation_to_type(a) for a in args}
        if len(types) == 1:
            return types.pop()
        return "string"

    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin is dict:
        return "object"
    if origin is not None:
        ann = origin

    if ann in (int, float):
        return "number"
    if ann is bool:
        return "boolean"
    if ann is dict:
        return "object"
    if ann in (list, tuple, set, frozenset):
        return "array"
    return "string"


def function_tool(func: F) -> dict[str, Any]:
    """Return an OpenAI tool definition for ``func``.

    The returned dictionary follows the "tool" schema understood by the
    OpenAI APIs and can be supplied directly when invoking a model so it may
    call the wrapped function via tool-calling.
    """

    if func in _INTROSPECTION_CACHE:
        sig, hints = _INTROSPECTION_CACHE[func]
    else:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        _INTROSPECTION_CACHE[func] = (sig, hints)

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
