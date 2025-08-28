"""Utilities for defining simple function-based tools."""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar, get_args, get_origin, get_type_hints, Union


F = TypeVar("F", bound=Callable[..., Any])


_INTROSPECTION_CACHE: dict[Callable[..., Any], tuple[inspect.Signature, dict[str, Any]]] = {}


def _annotation_to_schema(ann: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON schema object understood by the OpenAI tools API.

    Produces minimally complete schemas, including ``items`` for arrays and
    ``additionalProperties`` for dicts where possible. Falls back to ``string``
    when the type cannot be determined reliably.
    """

    origin = get_origin(ann)

    # Optional/Union handling: prefer a single underlying type, else string
    if origin is Union:
        args = [a for a in get_args(ann) if a is not type(None)]
        if not args:
            return {"type": "string"}
        # Try to unify identical schemas
        schemas = [_annotation_to_schema(a) for a in args]
        # If all schemas share the same top-level type, keep it; else string
        types = {s.get("type", "string") for s in schemas}
        if len(types) == 1:
            # If array, ensure items is present
            t = types.pop()
            if t == "array":
                # choose first non-string items if available
                for s in schemas:
                    if s.get("items"):
                        return {"type": "array", "items": s["items"]}
                return {"type": "array", "items": {"type": "string"}}
            if t == "object":
                # For objects, we can't merge properties reliably; keep generic
                return {"type": "object"}
            return {"type": t}
        return {"type": "string"}

    # Containers with element types
    if origin in (list, tuple, set, frozenset):
        args = get_args(ann)
        item_ann = args[0] if args else Any
        item_schema = _annotation_to_schema(item_ann)
        # Ensure a valid item type
        if "type" not in item_schema:
            item_schema = {"type": "string"}
        return {"type": "array", "items": {"type": item_schema.get("type", "string")}}

    if origin is dict:
        args = get_args(ann)
        val_ann = args[1] if len(args) == 2 else Any
        val_schema = _annotation_to_schema(val_ann)
        return {"type": "object", "additionalProperties": {"type": val_schema.get("type", "string")}}

    # Collapse typing aliases to concrete classes
    if origin is not None:
        ann = origin

    # Primitives
    if ann in (int, float):
        return {"type": "number"}
    if ann is bool:
        return {"type": "boolean"}
    if ann in (str,):
        return {"type": "string"}
    if ann in (dict,):
        return {"type": "object"}
    if ann in (list, tuple, set, frozenset):
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string"}


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
        try:
            hints = get_type_hints(func)
        except Exception:
            # Fallback for older Python versions or unsupported annotations (e.g., PEP 604 unions on 3.9)
            hints = getattr(func, "__annotations__", {}) or {}
        _INTROSPECTION_CACHE[func] = (sig, hints)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, str)
        properties[name] = _annotation_to_schema(ann)
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
