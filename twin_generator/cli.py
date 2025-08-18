"""Command‑line interface wrapper around :pyfunc:`twin_generator.generate_twin`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import constants as C
from .pipeline import generate_twin

__all__ = ["main"]


def _preview_graph(path: str) -> None:
    """Display the generated graph PNG in a Matplotlib window (best‑effort)."""
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        import matplotlib.pyplot as plt  # imported lazily to avoid GUI deps
    except ImportError as exc:
        print(
            f"⚠️ Could not preview graph image; missing dependency: {exc}",
            file=sys.stderr,
        )
        return

    try:
        img = Image.open(path).convert("RGBA")
        img.show()
        arr = np.array(img)
        fig, ax = plt.subplots()
        ax.imshow(arr)
        ax.axis("off")
        plt.show()
    except Exception as exc:
        print(f"⚠️ Could not preview graph image: {exc}", file=sys.stderr)


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: D401 – imperative mood
    parser = argparse.ArgumentParser(description="Generate SAT twin problems ✔")
    parser.add_argument("--problem", help="Path to source problem text")
    parser.add_argument("--solution", help="Path to official solution text")
    parser.add_argument("--demo", action="store_true", help="Run trivial demo")
    parser.add_argument("--graph-demo", action="store_true", help="Run demo with graph visual")
    parser.add_argument("--out", help="Write JSON output to file")
    parser.add_argument("--preview", action="store_true", help="Preview graph PNG if generated")
    parser.add_argument("--verbose", action="store_true", help="Print progress steps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: D401 – imperative mood
    ns = _parse_cli(argv)

    if ns.demo or ns.graph_demo:
        problem_text: str = C._GRAPH_PROBLEM if ns.graph_demo else C._DEMO_PROBLEM
        solution_text: str = C._GRAPH_SOLUTION if ns.graph_demo else C._DEMO_SOLUTION
    else:
        if not ns.problem or not ns.solution:
            sys.exit("Error: --problem and --solution are required unless using --demo flags.")
        problem_text = Path(ns.problem).read_text("utf-8").strip()
        solution_text = Path(ns.solution).read_text("utf-8").strip()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Error: Set OPENAI_API_KEY before running.")

    out: dict[str, Any] = generate_twin(
        problem_text,
        solution_text,
        force_graph=bool(ns.graph_demo),
        graph_spec=C.DEFAULT_GRAPH_SPEC if ns.graph_demo else None,
        verbose=bool(ns.verbose),
    )

    auto_preview = ns.preview or ns.graph_demo
    if auto_preview and "graph_path" in out:
        _preview_graph(str(out["graph_path"]))

    json_out = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    if ns.out:
        Path(ns.out).write_text(json_out, "utf-8")
        print(f"✔ Twin problem JSON written to {ns.out}")
    else:
        print(json_out)


if __name__ == "__main__":  # pragma: no cover
    main()
