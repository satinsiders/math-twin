from __future__ import annotations

"""Progress‑driven operator scheduler for the micro‑solver rebuild."""

import math
import random
import re
from copy import deepcopy
from typing import Sequence

from .state import MicroState
from .operators import Operator, DEFAULT_OPERATORS
from .steps_meta import _micro_monitor_dof
from .certificate import build_certificate
from .sym_utils import parse_relation_sides, evaluate_with_env, evaluate_numeric


def _total_residual_l2(state: MicroState) -> float:
    vals: list[float] = []
    for rel in state.C["symbolic"]:
        op, lhs, rhs = parse_relation_sides(rel)
        if op != "=":
            continue
        env = state.V["symbolic"].get("env", {})
        ok_l, val_l = evaluate_with_env(lhs, env)
        if not ok_l:
            ok_l, val_l = evaluate_numeric(lhs)
        ok_r, val_r = evaluate_with_env(rhs, env)
        if not ok_r:
            ok_r, val_r = evaluate_numeric(rhs)
        if ok_l and ok_r:
            try:
                vals.append(abs(float(val_l) - float(val_r)))
            except Exception:
                continue
    return float(math.sqrt(sum(v * v for v in vals)))


def _count_satisfied_ineq(state: MicroState) -> int:
    count = 0
    env = state.V["symbolic"].get("env", {})
    for rel in state.C["symbolic"]:
        op, lhs, rhs = parse_relation_sides(rel)
        if op not in ("<", "<=", ">", ">="):
            continue
        ok_l, val_l = evaluate_with_env(lhs, env)
        if not ok_l:
            ok_l, val_l = evaluate_numeric(lhs)
        ok_r, val_r = evaluate_with_env(rhs, env)
        if not ok_r:
            ok_r, val_r = evaluate_numeric(rhs)
        if not (ok_l and ok_r):
            continue
        try:
            if op == "<" and val_l < val_r:
                count += 1
            elif op == "<=" and val_l <= val_r:
                count += 1
            elif op == ">" and val_l > val_r:
                count += 1
            elif op == ">=" and val_l >= val_r:
                count += 1
        except Exception:
            continue
    return count


def _bounds_volume(bounds: dict[str, tuple[float | None, float | None]] | None) -> float:
    if not bounds:
        return 0.0
    vol = 1.0
    any_bound = False
    for low, high in bounds.values():
        if low is None or high is None:
            continue
        any_bound = True
        try:
            span = float(high) - float(low)
        except Exception:
            span = 0.0
        if span < 0:
            span = 0.0
        vol *= span
    return float(vol if any_bound else 0.0)


def decompose_goal(state: MicroState) -> MicroState:
    """Split multi-part goals and seed ``plan_steps`` with subgoals.

    The micro-solver sometimes receives problems with composite goals such as
    "solve for x and y".  This helper detects such phrasing and decomposes the
    goal into individual objectives.  ``state.goal`` is converted into a list of
    subgoal strings and ``state.plan_steps`` is initialised with a planning step
    per subgoal so downstream agents can reason about them independently.

    The split is intentionally lightweight – it looks for conjunctions like
    ``and``/","/``;`` and preserves any leading verb phrase (e.g. ``solve for``).
    If no split is detected the state is returned unchanged.
    """

    goal = state.goal
    if not goal or isinstance(goal, list):
        return state

    parts: list[str] = []

    # Handle patterns like "solve for x and y" preserving the prefix
    match = re.match(r"(.+?for\s+)(.+)", goal)
    if match and re.search(r"\band\b|[,;]", match.group(2)):
        prefix = match.group(1)
        rest = match.group(2)
        tokens = [t.strip() for t in re.split(r"[,;]|\band\b", rest) if t.strip()]
        if len(tokens) > 1:
            parts = [prefix + t for t in tokens]

    if not parts:
        tokens = [t.strip() for t in re.split(r"[,;]|\band\b", goal) if t.strip()]
        if len(tokens) > 1:
            parts = tokens

    if len(parts) <= 1:
        return state

    state.goal = parts
    state.plan_steps = [{"action": "subgoal", "goal": g} for g in parts]
    return state


def update_metrics(state: MicroState) -> MicroState:
    """Refresh solver metrics like degrees of freedom and progress score."""

    prev_metrics = dict(getattr(state, "M", {}))
    state = _micro_monitor_dof(state)
    redundant_idx = list(state.M.get("redundant_constraints_idx", []))
    if redundant_idx:
        removed = [r for i, r in enumerate(state.C["symbolic"]) if i in redundant_idx]
        state.C["symbolic"] = [r for i, r in enumerate(state.C["symbolic"]) if i not in redundant_idx]
        state = _micro_monitor_dof(state)
        state.M["redundant_constraints_idx"] = redundant_idx
        state.M["redundant_constraints"] = removed
    metrics = dict(getattr(state, "M", {}))

    prev_dof = prev_metrics.get("degrees_of_freedom")
    dof = metrics.get("degrees_of_freedom")
    if dof is not None:
        if dof < 0 or (prev_dof is not None and prev_dof > 0 and dof > 0):
            metrics["needs_replan"] = True

    prev_res = metrics.get("residual_l2")
    res = _total_residual_l2(state)
    metrics["residual_l2"] = res
    metrics["residual_l2_change"] = (
        float(prev_res - res) if prev_res is not None else 0.0
    )

    ineq = _count_satisfied_ineq(state)
    metrics["ineq_satisfied"] = float(ineq)

    prev_vol = metrics.get("bounds_volume")
    vol = _bounds_volume(state.V["symbolic"].get("derived", {}).get("bounds"))
    metrics["bounds_volume"] = vol
    metrics["bounds_volume_reduction"] = (
        float(prev_vol - vol) if prev_vol is not None else 0.0
    )

    prev_sample = metrics.get("sample_size")
    sample = state.V["symbolic"].get("derived", {}).get("sample")
    sample_size = float(len(sample)) if isinstance(sample, dict) else 0.0
    metrics["sample_size"] = sample_size
    metrics["sample_size_reduction"] = (
        float(prev_sample - sample_size) if prev_sample is not None else 0.0
    )

    state.M = metrics

    metrics["progress_score"] = float(
        -abs(state.M.get("degrees_of_freedom", 0))
        + metrics.get("residual_l2_change", 0.0)
        + metrics.get("ineq_satisfied", 0.0)
        + metrics.get("bounds_volume_reduction", 0.0)
        + metrics.get("sample_size_reduction", 0.0)
    )
    try:
        state.log.append(
            "metrics: "
            f"dof={metrics.get('degrees_of_freedom')} "
            f"res={metrics.get('residual_l2')} "
            f"score={metrics.get('progress_score')}"
        )
    except Exception:
        pass
    return state


def goal_satisfied(state: MicroState) -> bool:
    return state.A["symbolic"].get("final") is not None


def select_operator(state: MicroState, operators: Sequence[Operator]) -> Operator | None:
    """Pick the applicable operator with the highest score."""

    best_op: Operator | None = None
    best_score = float("-inf")
    best_delta = float("-inf")
    for op in operators:
        try:
            if not op.applicable(state):
                continue
            score_fn = getattr(op, "score", None)
            score = float(score_fn(state)) if callable(score_fn) else 0.0
            if score > best_score:
                best_score = score
                state_copy = deepcopy(state)
                _, delta = op.apply(state_copy)
                best_delta = float(delta)
                best_op = op
            elif score == best_score:
                state_copy = deepcopy(state)
                _, delta = op.apply(state_copy)
                delta = float(delta)
                if best_delta <= 0 and delta > 0:
                    best_delta = delta
                    best_op = op
        except Exception:
            continue
    return best_op


def replan(state: MicroState) -> MicroState:
    """Extended replan heuristic switching representations and branches."""
    # Decompose multi-part goals into individual subgoals
    if isinstance(state.goal, str) and re.search(r"\band\b|[,;]", state.goal):
        state = decompose_goal(state)

    # Representation swap
    if len(state.representations) > 1:
        try:
            idx = state.representations.index(state.representation)
        except ValueError:
            idx = -1
        curr_rep = state.representation
        next_rep = state.representations[(idx + 1) % len(state.representations)]
        if next_rep != curr_rep:
            # Swap active view containers so symbolic bucket always refers to current view
            for bucket in ("R", "C", "V", "A"):
                data = getattr(state, bucket)
                data[curr_rep], data[next_rep] = data[next_rep], data[curr_rep]
            state.representation = next_rep

    # Reseed numeric solver initial conditions
    state.numeric_seed = random.random()

    # Rotate or branch case splits when available
    if state.case_splits:
        state.active_case = (state.active_case + 1) % len(state.case_splits)
        state.C["symbolic"] = list(state.case_splits[state.active_case])
    else:
        # Fallback: rotate relations to explore different forms
        state.C["symbolic"] = list(reversed(state.C["symbolic"]))

    state.M["needs_replan"] = False
    try:
        state.log.append(
            f"replan: rep={state.representation} case={state.active_case}"
        )
    except Exception:
        pass
    return state


def solve(state: MicroState, operators: Sequence[Operator], *, max_iters: int = 10) -> MicroState:
    """Iteratively apply operators chosen by progress signals."""

    for _ in range(max_iters):
        state = update_metrics(state)
        if goal_satisfied(state):
            break
        if state.M.get("needs_replan") or state.M.get("stalls", 0) > 3:
            state = replan(state)
            state.M["stalls"] = 0
            continue
        op = select_operator(state, operators)
        if op is None:
            break
        before = state.M.get("progress_score", 0.0)
        state, _delta = op.apply(state)
        state = update_metrics(state)
        if state.M.get("progress_score", 0.0) <= before:
            state.M["stalls"] = state.M.get("stalls", 0) + 1
        else:
            state.M["stalls"] = 0
    try:
        state.A["symbolic"]["certificate"] = build_certificate(state)
    except Exception:
        pass
    return state


def solve_with_defaults(state: MicroState, *, max_iters: int = 10) -> MicroState:
    """Solve ``state`` using the built-in default operator pool."""

    return solve(state, DEFAULT_OPERATORS, max_iters=max_iters)
