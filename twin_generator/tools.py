"""FunctionTool wrappers exposed to the OpenAI Agents SDK."""
from __future__ import annotations

import html as _html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from agents.tool import function_tool

__all__ = [
    "make_html_table_tool",
    "render_graph_tool",
    "calc_answer_tool",
]


# ---------------------------------------------------------------------------
# HTML table helper
# ---------------------------------------------------------------------------

def _make_html_table(table_json: str) -> str:
    """Convert a JSON table spec → `<table>` element string (values escaped)."""
    data = json.loads(table_json)
    header = data.get("header", [])
    rows = data.get("rows", [])

    head_html = "".join(f"<th>{_html.escape(str(h))}</th>" for h in header)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{rows_html}</tbody></table>"


make_html_table_tool = function_tool(_make_html_table)


# ---------------------------------------------------------------------------
# Graph renderer helper
# ---------------------------------------------------------------------------

def _render_graph(spec_json: str) -> str:
    """Render a graph to a **PNG file** and return the file path (string)."""
    import matplotlib  # type: ignore
    _prefer_tk = os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND") == "TkAgg"
    try:
        if _prefer_tk:
            matplotlib.use("TkAgg")
        else:
            matplotlib.use("Agg")
    except Exception:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    spec = json.loads(spec_json)
    points: list[list[float]] = spec.get("points", [])
    style: str = spec.get("style", "line")
    title: str | None = spec.get("title")

    fig, ax = plt.subplots(figsize=(6, 6))

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    if points:
        xs, ys = zip(*points)
        xs = tuple(float(x) for x in xs)
        ys = tuple(float(y) for y in ys)
    else:
        xs = ()
        ys = ()

    if style == "scatter":
        ax.scatter(xs, ys)
    else:
        ax.plot(xs, ys)

    if title:
        ax.set_title(str(title))

    ax.grid(True)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_aspect("equal", adjustable="box")

    # Make the visible range adapt to the plotted data
    if points:
        ax.relim()
        ax.autoscale_view()
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # --- move the axes spines to x=0, y=0 and hide the others ---
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # ticks now belong on the visible spines
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")

    # Write PNG to a temp file and return path
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    png_path = Path(path)
    fig.savefig(png_path, format="png")
    plt.close(fig)
    return str(png_path)


render_graph_tool = function_tool(_render_graph)


# ---------------------------------------------------------------------------
# Exact math evaluator helper
# ---------------------------------------------------------------------------


def _calc_answer(expression: str, params_json: str) -> Any:  # noqa: ANN401 – generic return
    """Evaluate `expression` under the provided `params`.

    * Prefers exact symbolic simplification
    * Falls back to numeric evaluation
    * Coerces near‑integers to ``int`` when appropriate
    """
    import sympy as sp
    params = json.loads(params_json)
    expr = sp.sympify(expression)
    exact = expr.subs(params)

    try:
        exact_simpl = sp.simplify(exact)
    except Exception:
        exact_simpl = exact

    # Symbolic → numeric when free symbols remain
    if getattr(exact_simpl, "free_symbols", set()):
        result = sp.N(exact_simpl)
    else:
        result = exact_simpl

    # Convert to Python primitives
    try:
        if result.is_integer:  # type: ignore[attr-defined]
            return int(result)  # type: ignore[misc]
    except AttributeError:
        pass

    try:
        f = float(result)
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return f
    except Exception:
        return str(result)


calc_answer_tool = function_tool(_calc_answer)
