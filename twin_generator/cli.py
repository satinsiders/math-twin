"""Command‑line interface wrapper around :pyfunc:`twin_generator.generate_twin`."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dataclasses import asdict

from . import constants as C
from .pipeline import PipelineState, generate_twin

# --- Flexible prompt parsing helpers -----------------------------------------------------------
import re
from typing import Any, Tuple


def _extract_first_url(text: str) -> str | None:
    url_re = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
    m = url_re.search(text)
    return m.group(0) if m else None


def _split_problem_solution(text: str) -> Tuple[str, str]:
    """Heuristically split a free-form prompt into problem/solution parts.

    Recognizes case-insensitive markers like "Solution:" or "Answer:". If no
    marker is present, returns the entire text as the problem and an empty
    solution.
    """
    # Common markers that users might include ad-hoc
    markers = [r"solution\s*:", r"answer\s*:"]
    pattern = re.compile(r"|".join(markers), re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return text, ""
    idx = m.start()
    problem = text[:idx].strip()
    solution = text[idx:].split(":", 1)[-1].strip()
    return problem, solution


def _coerce_prompt_to_inputs(prompt: str) -> tuple[str, str, dict[str, Any]]:
    """Interpret a single prompt string into pipeline inputs.

    Returns (problem_text, solution_text, extra_kwargs) where extra_kwargs may
    include keys like graph_url, force_graph, graph_spec.
    """
    extras: dict[str, Any] = {}

    # 1) JSON object with explicit fields
    if prompt.lstrip().startswith("{"):
        try:
            obj = json.loads(prompt)
            if isinstance(obj, dict):
                problem = str(obj.get("problem", "") or "")
                solution = str(obj.get("solution", "") or "")
                if obj.get("graph_url"):
                    extras["graph_url"] = str(obj["graph_url"])  # type: ignore[index]
                if obj.get("force_graph"):
                    extras["force_graph"] = bool(obj["force_graph"])  # type: ignore[index]
                if obj.get("graph_spec"):
                    extras["graph_spec"] = obj["graph_spec"]  # type: ignore[index]
                return problem, solution, extras
        except Exception:
            # fall through to other strategies
            pass

    # 2) If prompt looks like a file path, try to load it (safely)
    def _looks_like_path(s: str) -> bool:
        # Avoid treating multi-line or extremely long strings as paths
        if "\n" in s or "\r" in s:
            return False
        # Heuristic: cap at a reasonable filename length to dodge OS errors
        return len(s) <= 240

    if _looks_like_path(prompt):
        try:
            p = Path(prompt)
            # Guard exists()/is_file() behind try/except to catch OSError from long bogus names
            if p.exists() and p.is_file():
                try:
                    text = p.read_text("utf-8")
                    if p.suffix.lower() == ".json":
                        return _coerce_prompt_to_inputs(text)
                    # Split heuristically if the file contains both problem and solution
                    problem, solution = _split_problem_solution(text)
                    # Opportunistically extract a URL (e.g., to a graph image)
                    url = _extract_first_url(text)
                    if url:
                        extras["graph_url"] = url
                    return problem, solution, extras
                except Exception:
                    # Treat the value as free text if reading failed
                    pass
        except (OSError, ValueError):
            # Not a valid path on this system; fall back to free text
            pass

    # 3) Free text: split on Solution/Answer markers and mine for an image URL
    problem, solution = _split_problem_solution(prompt)
    url = _extract_first_url(prompt)
    if url:
        extras["graph_url"] = url
    return problem, solution, extras

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
    parser.add_argument(
        "--problem",
        help="Path to source problem text (use '-' to read from stdin)",
    )
    parser.add_argument(
        "--solution",
        help="Path to official solution text (use '-' to read from stdin)",
    )
    parser.add_argument("--demo", action="store_true", help="Run trivial demo")
    parser.add_argument("--graph-demo", action="store_true", help="Run demo with graph visual")
    parser.add_argument("--graph-url", help="Optional URL to an existing graph image to analyze/use")
    parser.add_argument("--out", help="Write JSON output to file")
    parser.add_argument(
        "--twin-only",
        action="store_true",
        help=(
            "Print only the final twin problem (stem + choices) to stdout. "
            "Use --out to persist the full JSON if needed."
        ),
    )
    parser.add_argument("--preview", action="store_true", help="Preview graph PNG if generated")
    parser.add_argument(
        "--log-level",
        choices=["WARNING", "INFO", "DEBUG"],
        default="WARNING",
        help="Logging level for twin_generator",
    )
    # Flexible single-argument prompt mode (captures quoted free text)
    parser.add_argument(
        "prompt",
        nargs="?",
        help=(
            "Single quoted prompt that may include the problem, optional solution, and an optional image URL. "
            "If it points to a file, the file will be read; JSON with keys {problem, solution, graph_url} is also accepted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: D401 – imperative mood
    ns = _parse_cli(argv)

    logging.basicConfig(level=logging.WARNING)
    level = getattr(logging, ns.log_level)
    pkg_logger = logging.getLogger("twin_generator")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    handler.setLevel(level)
    pkg_logger.addHandler(handler)
    pkg_logger.propagate = False
    pkg_logger.setLevel(level)

    # Early API key check for faster feedback when actually running agents
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Error: Set OPENAI_API_KEY before running.")

    # Disallow mixing demo/file flags with flexible prompt to avoid confusion
    if ns.prompt is not None and (ns.problem or ns.solution or ns.demo or ns.graph_demo):
        sys.exit("Error: 'prompt' cannot be combined with --problem/--solution/--demo/--graph-demo.")

    # Resolve inputs from one of three modes: prompt | demo | explicit files
    extra_kwargs: dict[str, Any] = {}
    if ns.prompt is not None:
        # Treat empty prompt as an implicit demo for convenience
        if not ns.prompt.strip():
            problem_text = C._DEMO_PROBLEM
            solution_text = C._DEMO_SOLUTION
        else:
            problem_text, solution_text, extra_kwargs = _coerce_prompt_to_inputs(ns.prompt)
        # Allow explicit --graph-url to override mined URL
        if ns.graph_url:
            extra_kwargs["graph_url"] = ns.graph_url
    elif ns.demo or ns.graph_demo:
        problem_text = C._GRAPH_PROBLEM if ns.graph_demo else C._DEMO_PROBLEM
        solution_text = C._GRAPH_SOLUTION if ns.graph_demo else C._DEMO_SOLUTION
        if ns.graph_demo:
            extra_kwargs["force_graph"] = True
            extra_kwargs["graph_spec"] = C.DEFAULT_GRAPH_SPEC
        if ns.graph_url:
            extra_kwargs["graph_url"] = ns.graph_url
    else:
        # Explicit file mode
        if not ns.problem or not ns.solution:
            sys.exit("Error: --problem and --solution are required unless using prompt/demo modes.")
        if ns.problem == "-" and ns.solution == "-":
            sys.exit(
                "Error: cannot read both --problem and --solution from stdin; provide one via a file path."
            )

        def _read_arg(path: str, label: str) -> str:
            try:
                if path == "-":
                    data = sys.stdin.read()
                else:
                    data = Path(path).read_text("utf-8")
            except Exception as exc:
                sys.exit(f"Error reading {label}: {exc}")
            if not data.strip():
                sys.exit(f"Error: {label} is empty.")
            return data  # preserve original formatting (no strip)

        problem_text = _read_arg(ns.problem, "--problem")
        solution_text = _read_arg(ns.solution, "--solution")
        if ns.graph_url:
            extra_kwargs["graph_url"] = ns.graph_url

    out: PipelineState = generate_twin(
        problem_text,
        solution_text,
        force_graph=bool(extra_kwargs.get("force_graph", False)),
        graph_spec=extra_kwargs.get("graph_spec"),
        graph_url=extra_kwargs.get("graph_url"),
        verbose=ns.log_level in {"INFO", "DEBUG"},
    )

    auto_preview = ns.preview or ns.graph_demo
    if auto_preview and out.graph_path:
        _preview_graph(str(out.graph_path))

    json_out = json.dumps(asdict(out), ensure_ascii=False, separators=(",", ":"))
    if ns.out:
        Path(ns.out).write_text(json_out, "utf-8")
        print(f"✔ Twin problem JSON written to {ns.out}")

    # Final stdout behavior
    if ns.twin_only:
        # Emit only the final twin problem (stem + choices) plus answer and rationale
        minimal = {
            "twin_stem": out.twin_stem,
            "choices": out.choices,
            "answer": out.answer_value,
            "rationale": out.rationale,
        }
        print(json.dumps(minimal, ensure_ascii=False, separators=(",", ":")))
    elif not ns.out:
        # Default behavior: print the full JSON to stdout when not writing to file
        print(json_out)


if __name__ == "__main__":  # pragma: no cover
    main()
