from __future__ import annotations

"""SymPy-backed helpers for deterministic micro-steps.

These utilities are intentionally small and defensive. They parse expressions
with implicit multiplication enabled and perform bounded simplification.
"""

from typing import Any, Optional, Tuple, Iterable
import re


def simplify_expr(expr_str: str) -> str:
    """Return a simplified equivalent expression string using SymPy.

    Falls back to the original string on any parsing/simplification error.
    """
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        transformations = (*standard_transformations, implicit_multiplication_application)
        expr = parse_expr(str(expr_str), transformations=transformations)
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
        s = sp.sstr(expr)
        return s
    except Exception:
        return str(expr_str)


def verify_candidate(relations: list[str], candidate: str, *, varname: Optional[str] = None) -> bool:
    """Best-effort verification that a candidate expression satisfies relations.

    Heuristics:
    - If ``varname`` is provided (e.g., 'x'), check equations of the form ``varname = expression``
      or transform equations involving ``varname`` by substitution and check equivalence.
    - Only equality relations are checked; inequalities are ignored (return True if no checks fail).

    Returns True if no equality check fails (vacuously true if nothing could be checked).
    """
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        x = sp.Symbol(str(varname or "x"))
        cand = parse_expr(str(candidate), transformations=(*standard_transformations, implicit_multiplication_application))

        def _parse_side(side: str) -> Any:
            return parse_expr(side, transformations=(*standard_transformations, implicit_multiplication_application))

        def _parse_relation(rel: str) -> Tuple[str, Any, Any]:  # op, lhs, rhs
            # Support =, <=, >=, <, >, !=
            m = re.search(r"(<=|>=|!=|=|<|>)", rel)
            if not m:
                return "=", _parse_side(rel), sp.Integer(0)
            op = m.group(1)
            lhs = rel[: m.start(1)]
            rhs = rel[m.end(1) :]
            return op, _parse_side(lhs), _parse_side(rhs)

        ok = True
        checked = 0
        for r in relations:
            try:
                op, lhs_e, rhs_e = _parse_relation(r)
            except Exception:
                continue
            # Substitute candidate for the variable symbol wherever it appears
            try:
                lhs_sub = lhs_e.subs({x: cand})
                rhs_sub = rhs_e.subs({x: cand})
            except Exception:
                continue
            try:
                if op == "=":
                    diff = sp.simplify(lhs_sub - rhs_sub)
                    if getattr(diff, "is_zero", None) is False or (diff != 0):
                        ok = False
                elif op == "<=":
                    rel = sp.Le(lhs_sub, rhs_sub)
                    val = bool(rel)
                    if not val:
                        ok = False
                elif op == ">=":
                    rel = sp.Ge(lhs_sub, rhs_sub)
                    val = bool(rel)
                    if not val:
                        ok = False
                elif op == "<":
                    rel = sp.Lt(lhs_sub, rhs_sub)
                    val = bool(rel)
                    if not val:
                        ok = False
                elif op == ">":
                    rel = sp.Gt(lhs_sub, rhs_sub)
                    val = bool(rel)
                    if not val:
                        ok = False
                elif op == "!=":
                    rel = sp.Ne(lhs_sub, rhs_sub)
                    val = bool(rel)
                    if not val:
                        ok = False
                checked += 1
            except Exception:
                # If we cannot evaluate reliably, skip
                continue

        # Only accept when we actually checked at least one equality
        return ok if checked > 0 else False
    except Exception:
        # If SymPy is unavailable or parsing fails, be non-blocking
        return False


def evaluate_numeric(expr_str: str) -> tuple[bool, Any]:  # noqa: ANN401 - generic
    """Attempt to evaluate an expression to a Python number.

    Returns (ok, value). ok is True only when there are no free symbols and the
    expression can be converted to int/float. Integers are preferred when within
    tight tolerance of an exact integer.
    """
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        expr = parse_expr(str(expr_str), transformations=(*standard_transformations, implicit_multiplication_application))
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
        if getattr(expr, "free_symbols", set()):
            return False, None
        try:
            val = float(expr)
            # Prefer exact integers when close
            if abs(val - round(val)) < 1e-9:
                return True, int(round(val))
            return True, val
        except Exception:
            return False, None
    except Exception:
        return False, None


def _clean_for_sympy(s: str) -> str:
    """Best-effort cleanup of natural-language tails and Unicode quirks.

    - Strip LaTeX dollar signs
    - Normalize unicode minus to ASCII '-'
    - If ' is ' appears and no comparison operator is present, drop the trailing
      natural-language clause starting at ' is '
    """
    s2 = str(s).strip()
    try:
        s2 = s2.replace("$", "")
        s2 = s2.replace("\u2212", "-")
        if not re.search(r"(<=|>=|!=|=|<|>)", s2):
            s2 = re.split(r"\bis\b", s2, 1)[0].strip()
    except Exception:
        pass
    return s2


def _parse_expr(s: str):  # internal helper
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    cleaned = _clean_for_sympy(s)
    return parse_expr(str(cleaned), transformations=(*standard_transformations, implicit_multiplication_application))


def parse_relation_sides(rel: str) -> Tuple[str, str, str]:
    """Return (op, lhs_str, rhs_str) for a relation string.

    Recognizes =, <=, >=, <, >, !=.
    If no operator is found, returns op='' (empty), lhs=rel, rhs=''.
    This allows callers to distinguish genuine equalities from bare expressions.
    """
    m = re.search(r"(<=|>=|!=|=|<|>)", rel)
    if not m:
        return "", rel, ""
    op = m.group(1)
    lhs = rel[: m.start(1)]
    rhs = rel[m.end(1) :]
    return op, lhs.strip(), rhs.strip()


def rewrite_relations(relations: list[str], step: dict) -> list[str]:
    """Apply a small set of deterministic algebraic rewrites to relations.

    Supported actions (case-insensitive substring matching):
    - add / subtract both sides: args {value|term}
    - multiply / divide both sides: args {by}
    - substitute: args {replacements:{str->str}}
    - assign: args {target:str, value:str} → appends or replaces a relation

    Returns new relations list on success, or the original list on failure.
    """
    try:
        import sympy as sp
    except Exception:
        return relations
    try:
        action = str(step.get("action", "")).lower()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
    except Exception:
        return relations

    def _lr(expr: str) -> Tuple[Any, Any]:
        if re.search(r"(<=|>=|!=|=|<|>)", expr):
            opm = re.search(r"(<=|>=|!=|=|<|>)", expr)
            lhs = expr[: opm.start(1)]
            rhs = expr[opm.end(1) :]
            return _parse_expr(lhs), _parse_expr(rhs)
        if "=" in expr:
            lhs, rhs = expr.split("=", 1)
            return _parse_expr(lhs), _parse_expr(rhs)
        # Fallback: treat as equality to 0 after cleanup
        return _parse_expr(expr), sp.Integer(0)

    new_rels: list[str] = []

    if any(k in action for k in ("add", "subtract", "sub", "+", "-")):
        term = args.get("value") or args.get("term")
        if term is None:
            return relations
        try:
            t = _parse_expr(term)
        except Exception:
            return relations
        sign = -1 if ("subtract" in action or "-" in action or " sub" in action) else 1
        for r in relations:
            try:
                L, R = _lr(r)
                L2 = sp.simplify(L + sign * t)
                R2 = sp.simplify(R + sign * t)
                new_rels.append(f"{sp.sstr(L2)} = {sp.sstr(R2)}")
            except Exception:
                new_rels.append(r)
        return new_rels

    if any(k in action for k in ("multiply", "divide", "*", "/")):
        by = args.get("by")
        if by is None:
            return relations
        try:
            b = _parse_expr(by)
        except Exception:
            return relations
        if "divide" in action or "/" in action:
            opL = lambda x: sp.simplify(x / b)  # noqa: E731
        else:
            opL = lambda x: sp.simplify(x * b)  # noqa: E731
        for r in relations:
            try:
                L, R = _lr(r)
                new_rels.append(f"{sp.sstr(opL(L))} = {sp.sstr(opL(R))}")
            except Exception:
                new_rels.append(r)
        return new_rels

    if "substitute" in action or "subs" in action or "replace" in action:
        repl = args.get("replacements") if isinstance(args.get("replacements"), dict) else {}
        rep_map: dict[Any, Any] = {}
        for k, v in repl.items():
            try:
                rep_map[_parse_expr(k)] = _parse_expr(v)
            except Exception:
                continue
        for r in relations:
            try:
                L, R = _lr(r)
                L2 = L.xreplace(rep_map)
                R2 = R.xreplace(rep_map)
                new_rels.append(f"{sp.sstr(L2)} = {sp.sstr(R2)}")
            except Exception:
                new_rels.append(r)
        return new_rels

    if "expand" in action:
        for r in relations:
            try:
                L, R = _lr(r)
                new_rels.append(f"{sp.sstr(sp.expand(L))} = {sp.sstr(sp.expand(R))}")
            except Exception:
                new_rels.append(r)
        return new_rels

    if "factor" in action:
        for r in relations:
            try:
                L, R = _lr(r)
                new_rels.append(f"{sp.sstr(sp.factor(L))} = {sp.sstr(sp.factor(R))}")
            except Exception:
                new_rels.append(r)
        return new_rels

    if "simplify" in action:
        for r in relations:
            try:
                L, R = _lr(r)
                new_rels.append(f"{sp.sstr(sp.simplify(L))} = {sp.sstr(sp.simplify(R))}")
            except Exception:
                new_rels.append(r)
        return new_rels

    if "assign" in action or ("target" in args and "value" in args):
        tgt = str(args.get("target", "")).strip()
        val = str(args.get("value", "")).strip()
        if tgt and val:
            try:
                L = _parse_expr(tgt.split("=", 1)[0] if "=" in tgt else tgt)
                R = _parse_expr(val)
                new_rels = relations + [f"{sp.sstr(L)} = {sp.sstr(R)}"]
                return new_rels
            except Exception:
                return relations
    return relations


def solve_for(relations: list[str], target: Optional[str]) -> list[str]:
    """Attempt to solve equality relations for ``target`` symbol.

    Returns a list of solution expressions (as strings). On failure, returns [].
    """
    if not target:
        return []
    try:
        import sympy as sp
        sym = sp.Symbol(str(target))
        eqs: list[sp.Eq] = []
        for r in relations:
            op, lhs, rhs = parse_relation_sides(r)
            if op != "=":
                continue
            try:
                L = _parse_expr(lhs)
                R = _parse_expr(rhs)
                eqs.append(sp.Eq(L, R))
            except Exception:
                continue
        if not eqs:
            return []
        try:
            sol = sp.solve(eqs, sym, dict=True)
        except Exception:
            # Fallback to solveset for single equation
            try:
                if len(eqs) == 1:
                    S = sp.solveset(eqs[0].lhs - eqs[0].rhs, sym, domain=sp.S.Complexes)
                    if hasattr(S, "args") and S.args:
                        return [sp.sstr(a) for a in S.args]
                    if S is sp.S.EmptySet:
                        return []
                    return [str(S)]
            except Exception:
                return []
            return []
        results: list[str] = []
        for mapping in sol:
            val = mapping.get(sym)
            if val is not None:
                results.append(sp.sstr(val))
        return results
    except Exception:
        return []


def solve_any(relations: list[str]) -> list[str]:
    """Attempt to solve the system for any symbol appearing in the relations.

    Returns a list of solution expressions (as strings) that are fully determined
    (i.e., have no free symbols). On failure, returns [].
    """
    try:
        import sympy as sp
    except Exception:
        return []
    try:
        # Build equations and collect symbols
        eqs: list[sp.Eq] = []
        symbols: set[Any] = set()
        for r in relations:
            op, lhs, rhs = parse_relation_sides(r)
            if op != "=":
                continue
            try:
                L = _parse_expr(lhs)
                R = _parse_expr(rhs)
            except Exception:
                continue
            eqs.append(sp.Eq(L, R))
            try:
                symbols |= set(L.free_symbols) | set(R.free_symbols)
            except Exception:
                pass
        if not eqs or not symbols:
            return []
        # Try solving for all symbols jointly
        try:
            sol = sp.solve(eqs, list(symbols), dict=True)
        except Exception:
            sol = []
        results: list[str] = []
        for mapping in sol or []:
            for sym, val in mapping.items():
                try:
                    if not getattr(val, "free_symbols", set()):
                        results.append(sp.sstr(val))
                except Exception:
                    continue
        # If none were fully determined, try single-equation solves as a last resort
        if not results and len(eqs) == 1:
            try:
                sym_list = list(symbols)
                single = sp.solve(eqs[0], sym_list, dict=True)
                for m in single:
                    for _sym, val in m.items():
                        try:
                            if not getattr(val, "free_symbols", set()):
                                results.append(sp.sstr(val))
                        except Exception:
                            continue
            except Exception:
                pass
        return results
    except Exception:
        return []
