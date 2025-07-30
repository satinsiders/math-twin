"""Package‑wide constants and demo assets."""

from typing import Any

_DEMO_PROBLEM = """If 3x + 2 = 17, what is the value of x?"""
_DEMO_SOLUTION = """Subtract 2 → 3x = 15, then divide by 3 → x = 5."""

DEFAULT_GRAPH_SPEC: dict[str, Any] = {
    "points": [[0, -1], [1, 1], [2, 3], [3, 5]],  # slope 2, intercept −1
    "style": "line",
    "title": "y = 2x − 1",
}

_GRAPH_PROBLEM = (
    "The graph below shows points for a linear function. Which equation best "
    "models the data? (Assume the function is linear.)"
)
_GRAPH_SOLUTION = """Slope (m) = 2, y‑intercept = −1 → y = 2x − 1."""

__all__ = [
    "_DEMO_PROBLEM",
    "_DEMO_SOLUTION",
    "DEFAULT_GRAPH_SPEC",
    "_GRAPH_PROBLEM",
    "_GRAPH_SOLUTION",
]
