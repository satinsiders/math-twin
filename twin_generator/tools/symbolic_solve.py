"""Symbolic equation solver tool."""
from __future__ import annotations

import json
from typing import Any, Iterable

from agents.tool import function_tool

__all__ = ["symbolic_solve_tool", "_symbolic_solve"]


def _parse_equation(expr: str) -> Any:
    import sympy as sp

    if "=" in expr:
        lhs, rhs = expr.split("=", 1)
        return sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
    return sp.Eq(sp.sympify(expr), 0)


def _to_symbols(sym: Any) -> Iterable[Any]:
    import sympy as sp

    if isinstance(sym, (list, tuple)):
        return sp.symbols(list(sym))
    return (sp.Symbol(sym),)


def _symbolic_solve(eq_json: str) -> str:
    """Return exact solution(s) to a symbolic equation.

    Parameters
    ----------
    eq_json:
        JSON string with two fields:
        - ``equation``: a SymPy-readable expression. An ``=`` sign may be
          provided; otherwise the expression is assumed to equal 0.
        - ``variable``: single symbol name or list of names to solve for.

    Returns
    -------
    str
        JSON string encoding a list of solution dictionaries with simplified
        expressions as strings.
    """
    import sympy as sp

    payload = json.loads(eq_json)
    eq_field = payload["equation"]
    var_field = payload["variable"]

    equations = [_parse_equation(eq_field)] if isinstance(eq_field, str) else [
        _parse_equation(e) for e in eq_field
    ]
    symbols = _to_symbols(var_field)

    solutions = sp.solve(equations, symbols, dict=True)
    simplified = [
        {str(k): str(sp.simplify(v)) for k, v in sol.items()} for sol in solutions
    ]
    return json.dumps(simplified)

symbolic_solve_tool = function_tool(_symbolic_solve)
symbolic_solve_tool["name"] = "symbolic_solve_tool"
