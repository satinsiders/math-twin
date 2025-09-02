from __future__ import annotations

"""Utilities for analysing constraint independence and repairing rank issues.

This module provides small helpers that operate on relation strings used by the
micro‑solver. The functions avoid heavy symbolic manipulation where possible and
fall back to conservative heuristics when SymPy is unavailable.

Example workflow::

    jac = numeric_jacobian(relations, variables)
    redundant = mark_redundant_constraints(relations, variables)
    repaired, info = attempt_rank_repair(relations, variables)

``numeric_jacobian`` computes the Jacobian matrix for equality relations with
respect to the provided variables. ``mark_redundant_constraints`` uses that
Jacobian to detect linearly dependent rows. ``attempt_rank_repair`` drops those
rows and tries simple substitutions to keep the system well determined.
"""

from typing import Sequence, Tuple, Dict, Any, Optional


def _collect_symbols(relations: Sequence[str]) -> list[str]:
    """Return sorted list of symbols appearing in equality relations."""
    try:
        from .sym_utils import parse_relation_sides, _parse_expr
    except Exception:  # SymPy unavailable or import failure
        return []

    syms: set[str] = set()
    for r in relations:
        try:
            op, lhs, rhs = parse_relation_sides(r)
            if op != "=":
                continue
            expr = _parse_expr(lhs) - _parse_expr(rhs)
            syms.update(str(s) for s in getattr(expr, "free_symbols", set()))
        except Exception:
            continue
    return sorted(syms)


def numeric_jacobian(
    relations: Sequence[str],
    variables: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, Any]] = None,
):
    """Return numeric Jacobian matrix for equality relations.

    The Jacobian is evaluated at ``env`` when provided, otherwise at zeros. When
    SymPy is unavailable, ``None`` is returned.
    """
    try:
        import sympy as sp
        from .sym_utils import parse_relation_sides, _parse_expr
    except Exception:
        return None

    vars_list = list(variables) if variables else _collect_symbols(relations)
    exprs = []
    for r in relations:
        op, lhs, rhs = parse_relation_sides(r)
        if op != "=":
            continue
        try:
            exprs.append(_parse_expr(lhs) - _parse_expr(rhs))
        except Exception:
            continue
    if not exprs or not vars_list:
        return sp.Matrix([])

    syms = [sp.Symbol(v) for v in vars_list]
    J = sp.Matrix(exprs).jacobian(syms)
    subs: Dict[Any, Any] = {}
    if env:
        for v in vars_list:
            val = env.get(v)
            if isinstance(val, (int, float)):
                subs[sp.Symbol(v)] = float(val)
    if subs:
        J = J.subs(subs)
    return J.applyfunc(lambda e: e.evalf())


def mark_redundant_constraints(
    relations: Sequence[str],
    variables: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, Any]] = None,
) -> list[int]:
    """Return indices of relations that appear linearly dependent.

    The detection is based on the row rank of the Jacobian. When rank
    computation fails, an empty list is returned.
    """
    try:
        import sympy as sp
    except Exception:
        return []

    J = numeric_jacobian(relations, variables, env)
    if J is None or J.rows == 0:
        return []
    try:
        _, pivots = J.T.rref()  # pivot columns of transpose = independent rows
        independent = set(pivots)
        return [i for i in range(J.rows) if i not in independent]
    except Exception:
        return []


def attempt_rank_repair(
    relations: Sequence[str],
    variables: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, Any]] = None,
) -> Tuple[list[str], Dict[str, Any]]:
    """Remove redundant constraints and attempt simple substitutions.

    Returns ``(new_relations, info)`` where ``info`` records removed relations
    and any substitutions performed.
    """
    vars_list = list(variables) if variables else _collect_symbols(relations)
    redundant = mark_redundant_constraints(relations, vars_list, env)
    if not redundant:
        return list(relations), {"removed": [], "substitutions": {}}

    new_rel = [r for i, r in enumerate(relations) if i not in redundant]
    subs: Dict[str, str] = {}

    try:
        import sympy as sp
        from .sym_utils import parse_relation_sides, _parse_expr, rewrite_relations
    except Exception:
        return new_rel, {"removed": [relations[i] for i in redundant], "substitutions": {}}

    for idx in redundant:
        r = relations[idx]
        op, lhs, rhs = parse_relation_sides(r)
        if op != "=":
            continue
        eq = _parse_expr(lhs) - _parse_expr(rhs)
        for v in vars_list:
            sym = sp.Symbol(v)
            try:
                sol = sp.solve(eq, sym)
                if sol:
                    subs[v] = sp.sstr(sol[0])
                    break
            except Exception:
                continue
    if subs:
        try:
            new_rel = rewrite_relations(
                new_rel,
                {"action": "substitute", "args": {"replacements": subs}},
            )
        except Exception:
            pass
    return new_rel, {"removed": [relations[i] for i in redundant], "substitutions": subs}
