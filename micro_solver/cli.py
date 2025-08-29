from __future__ import annotations

import argparse
from typing import Optional

from .state import MicroState
from .orchestrator import MicroGraph, MicroRunner
from .steps import DEFAULT_MICRO_STEPS, build_steps


def solve(problem_text: str, *, verbose: bool = False) -> MicroState:
    steps = build_steps(max_iters=None)
    graph = MicroGraph(steps=steps)
    runner = MicroRunner(graph, verbose=verbose)
    state = MicroState(problem_text=problem_text)
    return runner.run(state)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Micro‑solver CLI (experimental)")
    parser.add_argument("text", help="Problem text to solve")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Configure logging for micro-solver when verbose
    if args.verbose:
        import logging
        logger = logging.getLogger("micro_solver")
        logger.setLevel(logging.INFO)
        # Ensure at least one handler is present
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

        logging.getLogger("micro_solver.orchestrator").setLevel(logging.INFO)
        logging.getLogger("micro_solver.steps").setLevel(logging.INFO)

    out = solve(args.text, verbose=args.verbose)
    if out.error:
        print(f"error: {out.error}")
        return 1

    # Prefer a concrete final answer. If absent, surface an informative fallback
    if out.final_answer is not None:
        print(out.final_answer)
        return 0

    # Fallbacks when no final answer
    if out.candidate_answers:
        last = out.candidate_answers[-1]
        if args.verbose:
            print(f"candidate-only (unverified): {last}")
        else:
            # In non-verbose mode, show the candidate directly so it isn't just 'None'
            print(last)
        return 0

    # No candidate either: print a concise explanation when available
    if out.final_explanation:
        print(out.final_explanation)
    else:
        print("No final answer; no candidate extracted. Use --verbose for details.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
