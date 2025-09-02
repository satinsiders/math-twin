from __future__ import annotations

"""Progress‑driven operator scheduler for the micro‑solver rebuild."""

import math
import random
from typing import Sequence

from .state import MicroState
from .operators import Operator, DEFAULT_OPERATORS
from .steps_meta import _micro_monitor_dof
from .certificate import build_certificate
from .sym_utils import parse_relation_sides, evaluate_with_env, evaluate_numeric


def _total_residual_l2(state: MicroState) -> float:
    vals: list[float] = []
    for rel in state.relations:
        op, lhs, rhs = parse_relation_sides(rel)
        if op != "=":
            continue
        ok_l, val_l = evaluate_with_env(lhs, state.env)
        if not ok_l:
            ok_l, val_l = evaluate_numeric(lhs)
        ok_r, val_r = evaluate_with_env(rhs, state.env)
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
    for rel in state.relations:
        op, lhs, rhs = parse_relation_sides(rel)
        if op not in ("<", "<=", ">", ">="):
            continue
        ok_l, val_l = evaluate_with_env(lhs, state.env)
        if not ok_l:
            ok_l, val_l = evaluate_numeric(lhs)
        ok_r, val_r = evaluate_with_env(rhs, state.env)
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


def update_metrics(state: MicroState) -> MicroState:
    """Refresh solver metrics like degrees of freedom and progress score."""

    state = _micro_monitor_dof(state)
    metrics = dict(getattr(state, "M", {}))

    prev_res = metrics.get("residual_l2")
    res = _total_residual_l2(state)
    metrics["residual_l2"] = res
    metrics["residual_l2_change"] = (
        float(prev_res - res) if prev_res is not None else 0.0
    )

    ineq = _count_satisfied_ineq(state)
    metrics["ineq_satisfied"] = float(ineq)

    prev_vol = metrics.get("bounds_volume")
    vol = _bounds_volume(state.derived.get("bounds"))
    metrics["bounds_volume"] = vol
    metrics["bounds_volume_reduction"] = (
        float(prev_vol - vol) if prev_vol is not None else 0.0
    )

    state.M = metrics

    state.progress_score = float(
        -abs(state.degrees_of_freedom)
        + metrics.get("residual_l2_change", 0.0)
        + metrics.get("ineq_satisfied", 0.0)
        + metrics.get("bounds_volume_reduction", 0.0)
    )
    return state


def goal_satisfied(state: MicroState) -> bool:
    return state.final_answer is not None


def select_operator(state: MicroState, operators: Sequence[Operator]) -> Operator | None:
    """Pick the applicable operator with the highest score."""

    best_op: Operator | None = None
    best_score = float("-inf")
    for op in operators:
        try:
            if not op.applicable(state):
                continue
            score_fn = getattr(op, "score", None)
            score = float(score_fn(state)) if callable(score_fn) else 0.0
            if score > best_score:
                best_score = score
                best_op = op
        except Exception:
            continue
    return best_op


def replan(state: MicroState) -> MicroState:
    """Extended replan heuristic switching representations and branches."""

    # Representation swap
    if len(state.representations) > 1:
        try:
            idx = state.representations.index(state.representation)
        except ValueError:
            idx = -1
        state.representation = state.representations[(idx + 1) % len(state.representations)]

    # Reseed numeric solver initial conditions
    state.numeric_seed = random.random()

    # Rotate or branch case splits when available
    if state.case_splits:
        state.active_case = (state.active_case + 1) % len(state.case_splits)
        state.relations = list(state.case_splits[state.active_case])
    else:
        # Fallback: rotate relations to explore different forms
        state.relations = list(reversed(state.relations))

    state.needs_replan = False
    return state


def solve(state: MicroState, operators: Sequence[Operator], *, max_iters: int = 10) -> MicroState:
    """Iteratively apply operators chosen by progress signals."""

    for _ in range(max_iters):
        state = update_metrics(state)
        if goal_satisfied(state):
            break
        if state.needs_replan or state.stalls > 3:
            state = replan(state)
            state.stalls = 0
            continue
        op = select_operator(state, operators)
        if op is None:
            break
        before = state.progress_score
        state, _delta = op.apply(state)
        state = update_metrics(state)
        if state.progress_score <= before:
            state.stalls += 1
        else:
            state.stalls = 0
    try:
        state.certificate = build_certificate(state)
    except Exception:
        pass
    return state


def solve_with_defaults(state: MicroState, *, max_iters: int = 10) -> MicroState:
    """Solve ``state`` using the built-in default operator pool."""

    return solve(state, DEFAULT_OPERATORS, max_iters=max_iters)
