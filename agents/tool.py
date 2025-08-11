"""Utilities for defining simple function-based tools."""

from typing import Any, Callable, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def function_tool(func: F) -> F:
    """Return the provided function as a tool with preserved type."""
    return func
