from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Set, Tuple

from .state import MicroState
from .sym_utils import evaluate_numeric, evaluate_with_env, parse_relation_sides

logger = logging.getLogger("micro_solver.steps")


def local_qa(state: MicroState, prev_relations: list[str], prev_idx: int) -> tuple[bool, str]:
    """Basic QA check to ensure rewrite changed state."""
    try:
        changed = (state.C["symbolic"] != prev_relations) or (state.current_step_idx != prev_idx)
        if not state.C["symbolic"]:
            return False, "empty-relations-after-rewrite"
        if not changed:
            return False, "no-change-after-rewrite"
        return True, "pass"
    except Exception as exc:  # pragma: no cover
        return False, f"qa-error:{exc}"


def maybe_eval_target(state: MicroState) -> bool:
    """Evaluate the target expression and record candidate answers."""
    try:
        target_expr = None
        if isinstance(state.R["symbolic"]["canonical_repr"], dict):
            target_expr = state.R["symbolic"]["canonical_repr"].get("target")
        if isinstance(target_expr, str) and target_expr.strip():
            ok, val = evaluate_with_env(target_expr, state.V["symbolic"]["env"] or {})
            if ok:
                state.A["symbolic"]["candidates"].append(val)
                return True
    except Exception:
        pass
    return False


def promote_env_from_relations(state: MicroState) -> None:
    """Populate environment variables from existing relations."""
    try:
        for r in state.C["symbolic"] or []:
            op, lhs, rhs = parse_relation_sides(r)
            if op != "=":
                continue
            name = (lhs or "").strip()
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
                continue
            ok, val = evaluate_with_env(rhs, state.V["symbolic"]["env"] or {})
            if ok:
                try:
                    state.V["symbolic"]["env"][name] = val
                except Exception:
                    pass
    except Exception:
        pass


def stable_unique(items: list[str]) -> list[str]:
    """Return a list with stable ordering and unique elements."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def state_digest(state: MicroState) -> str:
    """Compute a digest of the state's relations and environment."""
    try:
        rel_key = "|".join(sorted(map(str, state.C["symbolic"] or [])))
        env_items = ",".join(
            f"{k}={state.V['symbolic']['env'].get(k)}" for k in sorted(state.V['symbolic']['env'].keys())
        )
        h = hashlib.md5()
        h.update(rel_key.encode("utf-8", errors="ignore"))
        h.update(env_items.encode("utf-8", errors="ignore"))
        return h.hexdigest()
    except Exception:
        return ""


def target_symbols(state: MicroState) -> set[str]:
    """Extract symbolic variables from the target expression."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        transformations = (*standard_transformations, implicit_multiplication_application)
        tgt = (state.R["symbolic"]["canonical_repr"] or {}).get("target") if isinstance(state.R["symbolic"]["canonical_repr"], dict) else None
        if isinstance(tgt, str) and tgt.strip():
            expr = parse_expr(tgt, transformations=transformations)
            return {str(s) for s in getattr(expr, "free_symbols", set())}
    except Exception:
        pass
    return set()


def numeric_solvable_count(relations: list[str], symbols: set[str]) -> int:
    """Count how many symbols admit a numeric solution from relations."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        transformations = (*standard_transformations, implicit_multiplication_application)
        eqs: list[sp.Eq] = []
        for r in relations:
            try:
                op, lhs, rhs = parse_relation_sides(r)
                if op != "=":
                    continue
                eL = parse_expr(lhs, transformations=transformations)
                eR = parse_expr(rhs, transformations=transformations)
                eqs.append(sp.Eq(eL, eR))
            except Exception:
                continue
        count = 0
        for name in symbols:
            try:
                sym = sp.Symbol(name)
                sol = sp.solve(eqs, sym, dict=True)
                if sol and sol[0].get(sym) is not None:
                    val = sol[0][sym]
                    if not getattr(val, "free_symbols", set()):
                        count += 1
            except Exception:
                continue
        return count
    except Exception:
        return 0


def progress_metrics(state: MicroState) -> tuple[float, int, int, int, int, int, int]:
    """Compute heuristic progress metrics for the current state."""
    score = 0.0
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        transformations = (*standard_transformations, implicit_multiplication_application)
    except Exception:
        sp = None  # type: ignore
        transformations = None  # type: ignore
    try:
        score += 2.0 * float(len(state.V["symbolic"]["env"] or {}))
    except Exception:
        pass
    num_evaluable = 0
    free_syms: set[str] = set()
    eq_count = 0
    all_syms: set[str] = set()
    for r in state.C["symbolic"] or []:
        try:
            op, lhs, rhs = parse_relation_sides(r)
            if op == "":
                ok, _ = evaluate_with_env(lhs, state.V["symbolic"]["env"] or {})
                if not ok:
                    ok, _ = evaluate_numeric(lhs)
                if ok:
                    num_evaluable += 1
            else:
                okL, _L = evaluate_with_env(lhs, state.V["symbolic"]["env"] or {})
                okR, _R = evaluate_with_env(rhs, state.V["symbolic"]["env"] or {})
                if not okL:
                    okL, _L = evaluate_numeric(lhs)
                if not okR:
                    okR, _R = evaluate_numeric(rhs)
                if okL and okR:
                    num_evaluable += 1
            if op == "=":
                eq_count += 1
            if sp and transformations:
                try:
                    eL = parse_expr(lhs, transformations=transformations)
                    eR = parse_expr(rhs or "0", transformations=transformations)
                    symset = getattr(eL, "free_symbols", set()) | getattr(eR, "free_symbols", set())
                    for s in symset:
                        free_syms.add(str(s))
                        all_syms.add(str(s))
                except Exception:
                    pass
        except Exception:
            pass
    score += 1.0 * float(num_evaluable)
    try:
        score -= 0.5 * float(len(free_syms))
    except Exception:
        pass
    tgt_syms = target_symbols(state)
    bound = 0
    try:
        for t in tgt_syms:
            if t in (state.V["symbolic"]["env"] or {}):
                bound += 1
    except Exception:
        pass
    try:
        target_expr = (state.R["symbolic"]["canonical_repr"] or {}).get("target") if isinstance(state.R["symbolic"]["canonical_repr"], dict) else None
        if isinstance(target_expr, str) and target_expr.strip():
            ok, _ = evaluate_with_env(target_expr, state.V["symbolic"]["env"] or {})
            if ok:
                score += 5.0
    except Exception:
        pass
    score += 0.3 * float(eq_count)
    num_solvable = 0
    try:
        if tgt_syms:
            num_solvable += numeric_solvable_count(state.C["symbolic"], tgt_syms)
        others = set(all_syms) - set(tgt_syms)
        if others:
            num_solvable += max(0, numeric_solvable_count(state.C["symbolic"], others))
    except Exception:
        pass
    score += 3.0 * float(bound)
    unbound = max(0, len(tgt_syms) - bound)
    score -= 0.7 * float(unbound)
    score += 1.0 * float(num_solvable)
    return score, eq_count, num_evaluable, len(free_syms), bound, unbound, num_solvable
