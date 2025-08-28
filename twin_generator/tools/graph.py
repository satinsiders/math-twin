"""Graph rendering helpers."""
from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from agents.tool import function_tool

__all__ = ["render_graph_tool", "_render_graph"]


def _render_graph(spec_json: str) -> str:
    """Render a graph to a **PNG file** and return the file path (string)."""
    # Lazy import so the package works without matplotlib unless a graph is requested
    try:
        import matplotlib  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "matplotlib is required to render graphs. Install it or run without visuals."
        ) from exc

    # Select a usable backend on demand
    try:
        backend = matplotlib.get_backend().lower()
    except Exception:
        backend = ""
    if backend not in {"agg", "tkagg"}:
        env_backend = os.environ.get("MPLBACKEND", "").lower()
        prefer_tk = bool(os.environ.get("DISPLAY")) or env_backend == "tkagg"
        if prefer_tk:
            try:
                matplotlib.use("TkAgg")
            except Exception as exc:  # pragma: no cover - depends on system backend
                warnings.warn(
                    f"Preferred GUI backend 'TkAgg' unavailable; falling back to 'Agg': {exc}",
                    RuntimeWarning,
                )
                matplotlib.use("Agg")
        else:
            matplotlib.use("Agg")

    spec = json.loads(spec_json)
    raw_points: list[Any] = spec.get("points", [])
    style: str = spec.get("style", "line")
    title: str | None = spec.get("title")

    # Validate and normalize points
    points: list[tuple[float, float]] = []
    for idx, pt in enumerate(raw_points):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise ValueError(
                f"Point {idx} invalid: expected [x, y] format, got {pt!r}"
            )
        x, y = pt[:2]
        try:
            x_f = float(x)
            y_f = float(y)
        except (TypeError, ValueError):
            raise ValueError(
                f"Point {idx} invalid: expected [x, y] format, got {pt!r}"
            )
        points.append((x_f, y_f))

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
render_graph_tool["name"] = "render_graph_tool"
