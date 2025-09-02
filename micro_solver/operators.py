from __future__ import annotations

"""Operator interface and basic implementations for the micro‑solver rebuild.

Operators perform small state transitions and return a progress signal.  They
are intentionally lightweight so they can be scheduled dynamically based on
observed progress rather than a fixed strategy tree.
"""

from dataclasses import dataclass
from typing import Tuple, Any

from .state import MicroState
from .sym_utils import (
    rewrite_relations,
    simplify_expr,
    verify_candidate,
    solve_for,
    solve_any,
    parse_relation_sides,
    evaluate_numeric,
    evaluate_with_env,
)


def _apply_env(relations: list[str], env: dict[str, Any]) -> list[str]:
    """Return relations with known environment bindings substituted."""
    if not env:
        return relations
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        trans = (*standard_transformations, implicit_multiplication_application)
        rep = {sp.Symbol(k): parse_expr(str(v), transformations=trans) for k, v in env.items()}
        new_rels: list[str] = []
        for r in relations:
            try:
                op, lhs, rhs = parse_relation_sides(r)
                if op != "=":
                    new_rels.append(r)
                    continue
                L = parse_expr(lhs, transformations=trans).xreplace(rep)
                R = parse_expr(rhs, transformations=trans).xreplace(rep)
                new_rels.append(f"{sp.sstr(L)} = {sp.sstr(R)}")
            except Exception:
                new_rels.append(r)
        return new_rels
    except Exception:
        return relations


class Operator:
    """Protocol for reasoning or calculation operators."""

    name: str

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - interface
        return True

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, state: MicroState) -> float:  # pragma: no cover - default
        """Return a heuristic estimate of progress without mutating ``state``."""
        return 0.0


@dataclass
class SimplifyOperator(Operator):
    """Canonicalize all relations using :func:`simplify_expr`."""

    name: str = "simplify"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.C["symbolic"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        before = sum(len(r) for r in state.C["symbolic"])
        state.C["symbolic"] = [simplify_expr(r) for r in state.C["symbolic"]]
        after = sum(len(r) for r in state.C["symbolic"])
        delta = float(before - after)
        return state, delta

    def score(self, state: MicroState) -> float:
        before = sum(len(r) for r in state.C["symbolic"])
        after = sum(len(simplify_expr(r)) for r in state.C["symbolic"])
        return float(before - after)


@dataclass
class SubstituteOperator(Operator):
    """Perform deterministic substitutions on all relations."""

    replacements: dict[str, str]
    name: str = "substitute"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(self.replacements) and bool(state.C["symbolic"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        step = {"action": "substitute", "args": {"replacements": self.replacements}}
        new_rel = rewrite_relations(state.C["symbolic"], step)
        delta = float(len(state.C["symbolic"]) - len(new_rel))
        state.C["symbolic"] = new_rel
        return state, delta

    def score(self, state: MicroState) -> float:
        step = {"action": "substitute", "args": {"replacements": self.replacements}}
        new_rel = rewrite_relations(state.C["symbolic"], step)
        return float(len(state.C["symbolic"]) - len(new_rel))


@dataclass
class FeasibleSampleOperator(Operator):
    """Toy numeric sampler that records a random point for free variables."""

    name: str = "feasible_sample"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.V["symbolic"]["variables"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        import random

        sample: dict[str, float] = {}
        for v in state.V["symbolic"]["variables"]:
            low, high = state.domain.get(v, (None, None))
            tags = state.qual.get(v, set())
            # Apply qualitative sign hints
            if "positive" in tags:
                low = max(low or 0.0, 0.0)
            if "nonnegative" in tags:
                low = max(low or 0.0, 0.0)
            if "negative" in tags:
                high = min(high or 0.0, 0.0)
            if "nonpositive" in tags:
                high = min(high or 0.0, 0.0)
            # Default bounds when unspecified
            if low is None:
                low = -1.0
            if high is None:
                high = 1.0
            if low >= high:
                high = low + 1.0
            sample[v] = random.uniform(low, high)
        state.V["symbolic"]["derived"]["sample"] = sample
        return state, 0.0

    def score(self, state: MicroState) -> float:
        return float(len(state.V["symbolic"].get("variables", [])))


@dataclass
class SolveOperator(Operator):
    """Solve relations for a target symbol when system is determined."""

    name: str = "solve"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return (
            state.M.get("degrees_of_freedom", 0) == 0
            and bool(state.C["symbolic"])
            and not state.A["symbolic"]["candidates"]
        )

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        # Pick the first variable that is not yet bound in the environment.
        # When all variables are bound already, fall back to the first variable
        # so that its value can still be surfaced as a candidate answer.
        target = next((v for v in state.V["symbolic"]["variables"] if v not in state.V["symbolic"]["env"]), None)
        if target is None and state.V["symbolic"]["variables"]:
            target = state.V["symbolic"]["variables"][0]

        # Substitute known bindings into the relations before solving
        rels = _apply_env(state.C["symbolic"], state.V["symbolic"]["env"])

        sols: list[Any]
        if target in state.V["symbolic"]["env"]:
            sols = [str(state.V["symbolic"]["env"][target])]
        else:
            sols = solve_for(rels, target) if target else []
            if not sols:
                sols = solve_any(rels)
        if sols:
            state.A["symbolic"]["candidates"].extend(sols)
            return state, 1.0
        return state, 0.0

    def score(self, state: MicroState) -> float:
        target = next((v for v in state.V["symbolic"]["variables"] if v not in state.V["symbolic"]["env"]), None)
        if target is None and state.V["symbolic"]["variables"]:
            target = state.V["symbolic"]["variables"][0]
        rels = _apply_env(state.C["symbolic"], state.V["symbolic"].get("env", {}))
        sols: list[Any]
        if target in state.V["symbolic"].get("env", {}):
            sols = [str(state.V["symbolic"]["env"].get(target))]
        else:
            sols = solve_for(rels, target) if target else []
            if not sols:
                sols = solve_any(rels)
        return 1.0 if sols else 0.0


@dataclass
class VerifyOperator(Operator):
    """Verify the latest candidate against original relations."""

    name: str = "verify"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return (
            state.M.get("degrees_of_freedom", 0) == 0
            and bool(state.A["symbolic"]["candidates"])
            and state.A["symbolic"]["final"] is None
        )

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            candidate = str(state.A["symbolic"]["candidates"][-1])
        except Exception:
            return state, 0.0

        # Choose the variable corresponding to the candidate: first unbound symbol
        var = next((v for v in state.V["symbolic"]["variables"] if v not in state.V["symbolic"]["env"]), None)

        # Substitute known bindings into the relations before verification
        rels = _apply_env(state.C["symbolic"], state.V["symbolic"]["env"])

        if verify_candidate(rels, candidate, varname=var):
            state.A["symbolic"]["final"] = candidate
            return state, 1.0
        return state, 0.0

    def score(self, state: MicroState) -> float:
        try:
            candidate = str(state.A["symbolic"]["candidates"][-1])
        except Exception:
            return 0.0
        var = next((v for v in state.V["symbolic"]["variables"] if v not in state.V["symbolic"]["env"]), None)
        rels = _apply_env(state.C["symbolic"], state.V["symbolic"]["env"])
        return 1.0 if verify_candidate(rels, candidate, varname=var) else 0.0


@dataclass
class EliminateOperator(Operator):
    """Eliminate one variable by solving and substituting.

    Progress signal: number of occurrences of the eliminated symbol removed
    from the relations."""

    name: str = "eliminate"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return len(state.V["symbolic"]["variables"]) > 1 and bool(state.C["symbolic"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        target = state.V["symbolic"]["variables"][-1]
        before = sum(r.count(target) for r in state.C["symbolic"])
        new_rel = rewrite_relations(
            state.C["symbolic"],
            {"action": "eliminate_symbol", "args": {"symbol": target}},
        )
        after = sum(r.count(target) for r in new_rel)
        delta = float(before - after)
        if delta > 0:
            state.C["symbolic"] = new_rel
            state.V["symbolic"]["variables"] = [v for v in state.V["symbolic"]["variables"] if v != target]
        return state, delta

    def score(self, state: MicroState) -> float:
        target = state.V["symbolic"]["variables"][-1]
        before = sum(r.count(target) for r in state.C["symbolic"])
        new_rel = rewrite_relations(
            state.C["symbolic"],
            {"action": "eliminate_symbol", "args": {"symbol": target}},
        )
        after = sum(r.count(target) for r in new_rel)
        return float(before - after)


@dataclass
class TransformOperator(Operator):
    """Apply a deterministic algebraic rewrite (expand, factor, …).

    Progress signal: change in total relation string length (positive when the
    rewritten form is shorter)."""

    action: str = "expand"
    name: str = "transform"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.C["symbolic"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        before = sum(len(r) for r in state.C["symbolic"])
        new_rel = rewrite_relations(state.C["symbolic"], {"action": self.action})
        after = sum(len(r) for r in new_rel)
        state.C["symbolic"] = new_rel
        return state, float(before - after)

    def score(self, state: MicroState) -> float:
        before = sum(len(r) for r in state.C["symbolic"])
        new_rel = rewrite_relations(state.C["symbolic"], {"action": self.action})
        after = sum(len(r) for r in new_rel)
        return float(before - after)


@dataclass
class DiffOperator(Operator):
    """Differentiate a derived expression.

    Expects ``state.derived['expression']`` as a SymPy parsable string and an
    optional ``state.derived['variable']`` (defaults to ``x``).  The derivative
    is stored in ``state.derived['derivative']``.

    Progress signal: change in string length between the original expression
    and the derivative (positive when the derivative is shorter).
    """

    name: str = "diff"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        deriv = state.derived
        return isinstance(deriv, dict) and "expression" in deriv

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        deriv = state.derived
        expr = deriv.get("expression") if isinstance(deriv, dict) else None
        if expr is None:
            return state, 0.0
        try:
            import sympy as sp

            var = deriv.get("variable", "x") if isinstance(deriv, dict) else "x"
            sym = sp.Symbol(str(var))
            expr_sym = sp.sympify(str(expr))
            deriv = sp.diff(expr_sym, sym)
            result = sp.sstr(deriv)
            if isinstance(state.derived, dict):
                state.derived["derivative"] = result
            delta = float(len(str(expr)) - len(result))
            return state, delta
        except Exception:
            return state, 0.0

    def score(self, state: MicroState) -> float:
        deriv = state.derived
        expr = deriv.get("expression") if isinstance(deriv, dict) else None
        if expr is None:
            return 0.0
        try:
            import sympy as sp

            var = deriv.get("variable", "x") if isinstance(deriv, dict) else "x"
            sym = sp.Symbol(str(var))
            expr_sym = sp.sympify(str(expr))
            res = sp.diff(expr_sym, sym)
            return float(len(str(expr)) - len(sp.sstr(res)))
        except Exception:
            return 0.0


@dataclass
class IntegrateOperator(Operator):
    """Integrate a derived expression symbolically.

    Expects ``state.derived['expression']`` as a SymPy parsable string and an
    optional ``state.derived['variable']`` (defaults to ``x``).  The antiderivative
    is stored in ``state.derived['integral']``.

    Progress signal: change in string length between the original expression
    and the integral (positive when the integral is shorter).
    """

    name: str = "integrate"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        deriv = state.derived
        return isinstance(deriv, dict) and "expression" in deriv

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        deriv = state.derived
        expr = deriv.get("expression") if isinstance(deriv, dict) else None
        if expr is None:
            return state, 0.0
        try:
            import sympy as sp

            var = deriv.get("variable", "x") if isinstance(deriv, dict) else "x"
            sym = sp.Symbol(str(var))
            expr_sym = sp.sympify(str(expr))
            integ = sp.integrate(expr_sym, sym)
            result = sp.sstr(integ)
            if isinstance(state.derived, dict):
                state.derived["integral"] = result
            delta = float(len(str(expr)) - len(result))
            return state, delta
        except Exception:
            return state, 0.0

    def score(self, state: MicroState) -> float:
        deriv = state.derived
        expr = deriv.get("expression") if isinstance(deriv, dict) else None
        if expr is None:
            return 0.0
        try:
            import sympy as sp

            var = deriv.get("variable", "x") if isinstance(deriv, dict) else "x"
            sym = sp.Symbol(str(var))
            expr_sym = sp.sympify(str(expr))
            res = sp.integrate(expr_sym, sym)
            return float(len(str(expr)) - len(sp.sstr(res)))
        except Exception:
            return 0.0


@dataclass
class CaseSplitOperator(Operator):
    """Split simple squared equalities into linear cases.

    Progress signal: number of case relations generated."""

    name: str = "case_split"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.C["symbolic"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
            trans = (*standard_transformations, implicit_multiplication_application)
            cases: list[str] = []
            for r in state.C["symbolic"]:
                op, lhs, rhs = parse_relation_sides(r)
                if op != "=":
                    continue
                L = parse_expr(lhs, transformations=trans)
                R = parse_expr(rhs, transformations=trans)
                if L.is_Pow and L.exp == 2 and len(L.free_symbols) == 1 and R.is_number:
                    sym = list(L.free_symbols)[0]
                    root = sp.sqrt(R)
                    cases.append(f"{sp.sstr(sym)} = {sp.sstr(root)}")
                    cases.append(f"{sp.sstr(sym)} = {sp.sstr(-root)}")
                    break
            if cases:
                state.V["symbolic"]["derived"]["cases"] = cases
                return state, float(len(cases))
        except Exception:
            pass
        return state, 0.0

    def score(self, state: MicroState) -> float:
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
            trans = (*standard_transformations, implicit_multiplication_application)
            for r in state.C["symbolic"]:
                op, lhs, rhs = parse_relation_sides(r)
                if op != "=":
                    continue
                L = parse_expr(lhs, transformations=trans)
                R = parse_expr(rhs, transformations=trans)
                if L.is_Pow and L.exp == 2 and len(L.free_symbols) == 1 and R.is_number:
                    sym = list(L.free_symbols)[0]
                    root = sp.sqrt(R)
                    return float(2)
        except Exception:
            pass
        return 0.0


@dataclass
class BoundInferOperator(Operator):
    """Infer numeric bounds from inequality relations.

    Progress signal: number of bound endpoints added or tightened."""

    name: str = "bound_infer"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.C["symbolic"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            import sympy as sp
            bounds = dict(state.domain)
            changes = 0
            for r in state.C["symbolic"]:
                op, lhs, rhs = parse_relation_sides(r)
                if op not in ("<", "<=", ">", ">="):
                    continue
                try:
                    sym = sp.Symbol(lhs.strip())
                    val = float(sp.sympify(rhs))
                except Exception:
                    continue
                key = str(sym)
                low, high = bounds.get(key, (None, None))
                if op in (">", ">="):
                    if low is None or val > low:
                        low = val
                        changes += 1
                else:  # < or <=
                    if high is None or val < high:
                        high = val
                        changes += 1
                bounds[key] = (low, high)
            if changes:
                state.domain = bounds
                state.V["symbolic"]["derived"]["bounds"] = bounds
            return state, float(changes)
        except Exception:
            return state, 0.0

    def score(self, state: MicroState) -> float:
        try:
            import sympy as sp
            bounds = dict(state.domain)
            changes = 0
            for r in state.C["symbolic"]:
                op, lhs, rhs = parse_relation_sides(r)
                if op not in ("<", "<=", ">", ">="):
                    continue
                try:
                    sym = sp.Symbol(lhs.strip())
                    val = float(sp.sympify(rhs))
                except Exception:
                    continue
                key = str(sym)
                low, high = bounds.get(key, (None, None))
                if op in (">", ">="):
                    if low is None or val > low:
                        low = val
                        changes += 1
                else:
                    if high is None or val < high:
                        high = val
                        changes += 1
                bounds[key] = (low, high)
            return float(changes)
        except Exception:
            return 0.0


@dataclass
class DomainPruneOperator(Operator):
    """Remove sampled values that violate known bounds or qualitative tags.

    Progress signal: number of sample entries removed."""

    name: str = "domain_prune"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        sample = state.V["symbolic"]["derived"].get("sample")
        return isinstance(sample, dict) and bool(sample)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        sample = dict(state.V["symbolic"]["derived"].get("sample", {}))
        removed = 0
        for k, v in list(sample.items()):
            low, high = state.domain.get(k, (None, None))
            tags = state.qual.get(k, set())
            if (low is not None and v < low) or (high is not None and v > high):
                sample.pop(k)
                removed += 1
                continue
            if "positive" in tags and v <= 0:
                sample.pop(k)
                removed += 1
                continue
            if "nonnegative" in tags and v < 0:
                sample.pop(k)
                removed += 1
                continue
            if "negative" in tags and v >= 0:
                sample.pop(k)
                removed += 1
                continue
            if "nonpositive" in tags and v > 0:
                sample.pop(k)
                removed += 1
                continue
        if removed:
            state.V["symbolic"]["derived"]["sample"] = sample
        return state, float(removed)

    def score(self, state: MicroState) -> float:
        sample = dict(state.V["symbolic"].get("derived", {}).get("sample", {}))
        removed = 0
        for k, v in list(sample.items()):
            low, high = state.domain.get(k, (None, None))
            tags = state.qual.get(k, set())
            if (low is not None and v < low) or (high is not None and v > high):
                removed += 1
                continue
            if "positive" in tags and v <= 0:
                removed += 1
                continue
            if "nonnegative" in tags and v < 0:
                removed += 1
                continue
            if "negative" in tags and v >= 0:
                removed += 1
                continue
            if "nonpositive" in tags and v > 0:
                removed += 1
                continue
        return float(removed)


@dataclass
class NumericSolveOperator(Operator):
    """Evaluate explicit assignments numerically.

    Progress signal: number of candidate answers appended (0 or 1)."""

    name: str = "numeric_solve"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.C["symbolic"]) and not state.A["symbolic"]["candidates"]

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        for r in state.C["symbolic"]:
            op, lhs, rhs = parse_relation_sides(r)
            if op != "=":
                continue
            ok, val = evaluate_with_env(rhs, state.V["symbolic"]["env"])
            if not ok:
                ok, val = evaluate_numeric(rhs)
            if ok:
                state.A["symbolic"]["candidates"].append(str(val))
                return state, 1.0
        return state, 0.0

    def score(self, state: MicroState) -> float:
        for r in state.C["symbolic"]:
            op, lhs, rhs = parse_relation_sides(r)
            if op != "=":
                continue
            ok, val = evaluate_with_env(rhs, state.V["symbolic"].get("env", {}))
            if not ok:
                ok, val = evaluate_numeric(rhs)
            if ok:
                return 1.0
        return 0.0


@dataclass
class GridRefineOperator(Operator):
    """Refine the numeric sample grid by rounding values.

    Progress signal: number of sample entries updated."""

    name: str = "grid_refine"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        sample = state.V["symbolic"]["derived"].get("sample")
        return isinstance(sample, dict) and bool(sample)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        sample = dict(state.V["symbolic"]["derived"].get("sample", {}))
        changes = 0
        for k, v in sample.items():
            try:
                rv = round(float(v), 3)
                if rv != v:
                    sample[k] = rv
                    changes += 1
            except Exception:
                continue
        if changes:
            state.V["symbolic"]["derived"]["sample"] = sample
        return state, float(changes)

    def score(self, state: MicroState) -> float:
        sample = dict(state.V["symbolic"].get("derived", {}).get("sample", {}))
        changes = 0
        for v in sample.values():
            try:
                rv = round(float(v), 3)
                if rv != v:
                    changes += 1
            except Exception:
                continue
        return float(changes)


@dataclass
class QuadratureOperator(Operator):
    """Compute a definite integral stored in ``derived``.

    Expects ``state.V["symbolic"]["derived"]['integrand']`` (expression in ``x``) and
    ``state.V["symbolic"]["derived"]['interval']`` as ``(a, b)``.
    Progress signal: 1.0 when integral value is produced."""

    name: str = "quadrature"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return "integrand" in state.V["symbolic"]["derived"] and "interval" in state.V["symbolic"]["derived"]

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            import sympy as sp
            x = sp.Symbol("x")
            f_expr = sp.sympify(str(state.V["symbolic"]["derived"].get("integrand")))
            a, b = state.V["symbolic"]["derived"].get("interval")
            val = float(sp.integrate(f_expr, (x, a, b)))
            state.V["symbolic"]["derived"]["integral"] = val
            return state, 1.0
        except Exception:
            return state, 0.0

    def score(self, state: MicroState) -> float:
        try:
            import sympy as sp
            x = sp.Symbol("x")
            f_expr = sp.sympify(str(state.V["symbolic"].get("derived", {}).get("integrand")))
            a, b = state.V["symbolic"].get("derived", {}).get("interval", (None, None))
            if a is None or b is None:
                return 0.0
            sp.integrate(f_expr, (x, a, b))
            return 1.0
        except Exception:
            return 0.0


@dataclass
class RationalizeOperator(Operator):
    """Convert numeric candidates to rational form.

    Progress signal: number of candidate answers changed."""

    name: str = "rationalize"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return any("." in str(a) for a in state.A["symbolic"]["candidates"])

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            import sympy as sp
            new_answers = []
            changes = 0
            for a in state.A["symbolic"]["candidates"]:
                try:
                    r = sp.Rational(str(a)).limit_denominator()
                    if str(r) != str(a):
                        changes += 1
                    new_answers.append(str(r))
                except Exception:
                    new_answers.append(str(a))
            if changes:
                state.A["symbolic"]["candidates"] = new_answers
            return state, float(changes)
        except Exception:
            return state, 0.0

    def score(self, state: MicroState) -> float:
        try:
            import sympy as sp
            changes = 0
            for a in state.A["symbolic"]["candidates"]:
                try:
                    r = sp.Rational(str(a)).limit_denominator()
                    if str(r) != str(a):
                        changes += 1
                except Exception:
                    continue
            return float(changes)
        except Exception:
            return 0.0


# Default operator pool used by the high-level scheduler entrypoint.
#
# The set is intentionally small; it demonstrates the operator protocol with a
# mix of symbolic and validation steps while keeping the scheduling loop
# lightweight.  Additional operators can be appended by callers as needed.
DEFAULT_OPERATORS: list[Operator] = [
    SolveOperator(),
    VerifyOperator(),
    SimplifyOperator(),
    EliminateOperator(),
    TransformOperator(),
    CaseSplitOperator(),
    BoundInferOperator(),
    DomainPruneOperator(),
    NumericSolveOperator(),
    GridRefineOperator(),
    QuadratureOperator(),
    RationalizeOperator(),
    DiffOperator(),
    IntegrateOperator(),
]
