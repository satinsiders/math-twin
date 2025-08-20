"""FunctionTool wrappers exposed to the OpenAI Agents SDK."""
from __future__ import annotations

import html as _html
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

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


def _select_matplotlib_backend() -> None:
    """Choose a usable Matplotlib backend once at import time."""
    import matplotlib  # type: ignore

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "tkagg"}:  # already configured
        return

    env_backend = os.environ.get("MPLBACKEND", "").lower()
    prefer_tk = bool(os.environ.get("DISPLAY")) or env_backend == "tkagg"
    if prefer_tk:
        try:
            matplotlib.use("TkAgg")
            return
        except Exception as exc:  # pragma: no cover - depends on system backend
            warnings.warn(
                f"Preferred GUI backend 'TkAgg' unavailable; falling back to 'Agg': {exc}",
                RuntimeWarning,
            )
    matplotlib.use("Agg")


_select_matplotlib_backend()


def _render_graph(spec_json: str) -> str:
    """Render a graph to a **PNG file** and return the file path (string)."""
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


def _sanitize_params(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return params convertible to numeric SymPy expressions and skipped keys.

    Any key whose value cannot be converted to a SymPy expression or results in
    an expression with free symbols is skipped.  The cleaned dictionary uses the
    SymPy representations of the values while ``skipped`` contains keys that were
    discarded.
    """
    import sympy as sp

    cleaned: dict[str, Any] = {}
    skipped: list[str] = []
    for key, val in raw.items():
        try:
            sym_val = sp.sympify(val)
        except sp.SympifyError:
            skipped.append(key)
            continue
        if not getattr(sym_val, "is_number", False):
            skipped.append(key)
            continue
        cleaned[key] = sym_val
    return cleaned, skipped


def _run_with_timeout(func: Callable[..., Any], seconds: int, *args: Any, **kwargs: Any) -> Any:
    """Execute ``func`` with a time limit, raising ``TimeoutError`` on expiry."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=seconds)
        except FutureTimeoutError as exc:
            raise TimeoutError("timed out") from exc


def _calc_answer(expression: str, params_json: str) -> Any:  # noqa: ANN401 – generic return
    """Evaluate `expression` under the provided `params`.

    * Prefers exact symbolic simplification
    * Falls back to numeric evaluation
    * Coerces near‑integers to ``int`` when appropriate
    """
    import sympy as sp
    import warnings

    params = json.loads(params_json)
    sanitized, skipped = _sanitize_params(params)
    if skipped:
        warnings.warn(f"Skipped non-numeric params: {', '.join(skipped)}")

    def _make_safe(func: Any, fallback: Any) -> Any:
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return _run_with_timeout(func, 2, *args, **kwargs)
            except Exception:
                return fallback(*args, **kwargs)

        return _wrapper

    local_ops = {
        "diff": _make_safe(sp.diff, sp.Derivative),
        "integrate": _make_safe(sp.integrate, sp.Integral),
        "limit": _make_safe(sp.limit, sp.Limit),
        "summation": _make_safe(sp.summation, sp.Sum),
        "Derivative": _make_safe(lambda *a, **k: sp.Derivative(*a, **k).doit(), sp.Derivative),
        "Integral": _make_safe(lambda *a, **k: sp.Integral(*a, **k).doit(), sp.Integral),
        "Limit": _make_safe(lambda *a, **k: sp.Limit(*a, **k).doit(), sp.Limit),
        "Sum": _make_safe(lambda *a, **k: sp.Sum(*a, **k).doit(), sp.Sum),
    }

    try:
        expr = sp.sympify(expression, locals=local_ops)
    except Exception:
        expr = sp.sympify(expression)

    def _eval_advanced(e: sp.Expr, depth: int = 0) -> sp.Expr:
        if depth > 5:
            return e
        targets = list(e.atoms(sp.Derivative, sp.Integral, sp.Limit, sp.Sum))
        if not targets:
            return e
        for t in targets:
            try:
                res = _run_with_timeout(t.doit, 2)
            except Exception:
                res = t
            e = e.xreplace({t: res})
        return _eval_advanced(e, depth + 1)

    expr = _eval_advanced(expr)
    expr = expr.subs(sanitized)
    expr = _eval_advanced(expr)

    try:
        exact_simpl = _run_with_timeout(sp.simplify, 5, expr)
    except Exception:
        exact_simpl = expr

    if getattr(exact_simpl, "free_symbols", set()):
        try:
            result = _run_with_timeout(sp.N, 2, exact_simpl)
        except Exception:
            result = exact_simpl
    else:
        result = exact_simpl

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
