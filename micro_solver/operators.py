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


@dataclass
class SimplifyOperator(Operator):
    """Canonicalize all relations using :func:`simplify_expr`."""

    name: str = "simplify"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.relations)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        before = sum(len(r) for r in state.relations)
        state.relations = [simplify_expr(r) for r in state.relations]
        after = sum(len(r) for r in state.relations)
        delta = float(before - after)
        return state, delta


@dataclass
class SubstituteOperator(Operator):
    """Perform deterministic substitutions on all relations."""

    replacements: dict[str, str]
    name: str = "substitute"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(self.replacements) and bool(state.relations)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        step = {"action": "substitute", "args": {"replacements": self.replacements}}
        new_rel = rewrite_relations(state.relations, step)
        delta = float(len(state.relations) - len(new_rel))
        state.relations = new_rel
        return state, delta


@dataclass
class FeasibleSampleOperator(Operator):
    """Toy numeric sampler that records a random point for free variables."""

    name: str = "feasible_sample"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.variables)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        import random

        sample = {v: random.random() for v in state.variables}
        state.derived["sample"] = sample
        return state, 0.0


@dataclass
class SolveOperator(Operator):
    """Solve relations for a target symbol when system is determined."""

    name: str = "solve"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return (
            state.degrees_of_freedom == 0
            and bool(state.relations)
            and not state.candidate_answers
        )

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        # Pick the first variable that is not yet bound in the environment.
        # When all variables are bound already, fall back to the first variable
        # so that its value can still be surfaced as a candidate answer.
        target = next((v for v in state.variables if v not in state.env), None)
        if target is None and state.variables:
            target = state.variables[0]

        # Substitute known bindings into the relations before solving
        rels = _apply_env(state.relations, state.env)

        sols: list[Any]
        if target in state.env:
            sols = [str(state.env[target])]
        else:
            sols = solve_for(rels, target) if target else []
            if not sols:
                sols = solve_any(rels)
        if sols:
            state.candidate_answers.extend(sols)
            return state, 1.0
        return state, 0.0


@dataclass
class VerifyOperator(Operator):
    """Verify the latest candidate against original relations."""

    name: str = "verify"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return (
            state.degrees_of_freedom == 0
            and bool(state.candidate_answers)
            and state.final_answer is None
        )

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            candidate = str(state.candidate_answers[-1])
        except Exception:
            return state, 0.0

        # Choose the variable corresponding to the candidate: first unbound symbol
        var = next((v for v in state.variables if v not in state.env), None)

        # Substitute known bindings into the relations before verification
        rels = _apply_env(state.relations, state.env)

        if verify_candidate(rels, candidate, varname=var):
            state.final_answer = candidate
            return state, 1.0
        return state, 0.0


@dataclass
class EliminateOperator(Operator):
    """Eliminate one variable by solving and substituting.

    Progress signal: number of occurrences of the eliminated symbol removed
    from the relations."""

    name: str = "eliminate"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return len(state.variables) > 1 and bool(state.relations)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        target = state.variables[-1]
        before = sum(r.count(target) for r in state.relations)
        new_rel = rewrite_relations(
            state.relations,
            {"action": "eliminate_symbol", "args": {"symbol": target}},
        )
        after = sum(r.count(target) for r in new_rel)
        delta = float(before - after)
        if delta > 0:
            state.relations = new_rel
            state.variables = [v for v in state.variables if v != target]
        return state, delta


@dataclass
class TransformOperator(Operator):
    """Apply a deterministic algebraic rewrite (expand, factor, …).

    Progress signal: change in total relation string length (positive when the
    rewritten form is shorter)."""

    action: str = "expand"
    name: str = "transform"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.relations)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        before = sum(len(r) for r in state.relations)
        new_rel = rewrite_relations(state.relations, {"action": self.action})
        after = sum(len(r) for r in new_rel)
        state.relations = new_rel
        return state, float(before - after)


@dataclass
class CaseSplitOperator(Operator):
    """Split simple squared equalities into linear cases.

    Progress signal: number of case relations generated."""

    name: str = "case_split"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.relations)

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
            for r in state.relations:
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
                state.derived["cases"] = cases
                return state, float(len(cases))
        except Exception:
            pass
        return state, 0.0


@dataclass
class BoundInferOperator(Operator):
    """Infer numeric bounds from inequality relations.

    Progress signal: number of bound endpoints added or tightened."""

    name: str = "bound_infer"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.relations)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            import sympy as sp
            bounds = dict(state.derived.get("bounds", {}))
            changes = 0
            for r in state.relations:
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
                state.derived["bounds"] = bounds
            return state, float(changes)
        except Exception:
            return state, 0.0


@dataclass
class NumericSolveOperator(Operator):
    """Evaluate explicit assignments numerically.

    Progress signal: number of candidate answers appended (0 or 1)."""

    name: str = "numeric_solve"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return bool(state.relations) and not state.candidate_answers

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        for r in state.relations:
            op, lhs, rhs = parse_relation_sides(r)
            if op != "=":
                continue
            ok, val = evaluate_with_env(rhs, state.env)
            if not ok:
                ok, val = evaluate_numeric(rhs)
            if ok:
                state.candidate_answers.append(str(val))
                return state, 1.0
        return state, 0.0


@dataclass
class GridRefineOperator(Operator):
    """Refine the numeric sample grid by rounding values.

    Progress signal: number of sample entries updated."""

    name: str = "grid_refine"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        sample = state.derived.get("sample")
        return isinstance(sample, dict) and bool(sample)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        sample = dict(state.derived.get("sample", {}))
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
            state.derived["sample"] = sample
        return state, float(changes)


@dataclass
class QuadratureOperator(Operator):
    """Compute a definite integral stored in ``derived``.

    Expects ``state.derived['integrand']`` (expression in ``x``) and
    ``state.derived['interval']`` as ``(a, b)``.
    Progress signal: 1.0 when integral value is produced."""

    name: str = "quadrature"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return "integrand" in state.derived and "interval" in state.derived

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            import sympy as sp
            x = sp.Symbol("x")
            f_expr = sp.sympify(str(state.derived.get("integrand")))
            a, b = state.derived.get("interval")
            val = float(sp.integrate(f_expr, (x, a, b)))
            state.derived["integral"] = val
            return state, 1.0
        except Exception:
            return state, 0.0


@dataclass
class RationalizeOperator(Operator):
    """Convert numeric candidates to rational form.

    Progress signal: number of candidate answers changed."""

    name: str = "rationalize"

    def applicable(self, state: MicroState) -> bool:  # pragma: no cover - trivial
        return any("." in str(a) for a in state.candidate_answers)

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        try:
            import sympy as sp
            new_answers = []
            changes = 0
            for a in state.candidate_answers:
                try:
                    r = sp.Rational(str(a)).limit_denominator()
                    if str(r) != str(a):
                        changes += 1
                    new_answers.append(str(r))
                except Exception:
                    new_answers.append(str(a))
            if changes:
                state.candidate_answers = new_answers
            return state, float(changes)
        except Exception:
            return state, 0.0


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
    NumericSolveOperator(),
    GridRefineOperator(),
    QuadratureOperator(),
    RationalizeOperator(),
]
