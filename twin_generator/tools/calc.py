"""Exact math evaluator helpers."""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

from agents.tool import function_tool

__all__ = ["calc_answer_tool", "_calc_answer"]


def _sanitize_params(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return params convertible to numeric SymPy expressions and skipped keys."""
    import sympy as sp

    cleaned: dict[str, Any] = {}
    skipped: list[str] = []
    for key, val in raw.items():
        try:
            sym_val = sp.sympify(val)
        except sp.SympifyError:
            skipped.append(key)
            continue
        if not getattr(sym_val, "is_number", False):
            skipped.append(key)
            continue
        cleaned[key] = sym_val
    return cleaned, skipped


def _run_with_timeout(func: Callable[..., Any], seconds: int, *args: Any, **kwargs: Any) -> Any:
    """Execute ``func`` with a time limit, raising ``TimeoutError`` on expiry."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=seconds)
        except FutureTimeoutError as exc:
            raise TimeoutError("timed out") from exc


def _calc_answer(expression: str, params_json: str) -> Any:  # noqa: ANN401 – generic return
    """Evaluate `expression` under the provided `params`.

    * Prefers exact symbolic simplification
    * Falls back to numeric evaluation
    * Coerces near‑integers to ``int`` when appropriate
    """
    import sympy as sp
    from sympy.core.relational import Relational

    params = json.loads(params_json)
    sanitized, skipped = _sanitize_params(params)
    if skipped:
        warnings.warn(f"Skipped non-numeric params: {', '.join(skipped)}")

    error_msg = (
        f"Could not evaluate expression '{expression}'. "
        "Remove the equation sign, use '*' for multiplication, and provide values for all variables."
    )

    def _make_safe(func: Any, fallback: Any) -> Any:
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return _run_with_timeout(func, 2, *args, **kwargs)
            except Exception:
                return fallback(*args, **kwargs)

        return _wrapper

    local_ops = {
        "diff": _make_safe(sp.diff, sp.Derivative),
        "integrate": _make_safe(sp.integrate, sp.Integral),
        "limit": _make_safe(sp.limit, sp.Limit),
        "summation": _make_safe(sp.summation, sp.Sum),
        "Derivative": _make_safe(lambda *a, **k: sp.Derivative(*a, **k).doit(), sp.Derivative),
        "Integral": _make_safe(lambda *a, **k: sp.Integral(*a, **k).doit(), sp.Integral),
        "Limit": _make_safe(lambda *a, **k: sp.Limit(*a, **k).doit(), sp.Limit),
        "Sum": _make_safe(lambda *a, **k: sp.Sum(*a, **k).doit(), sp.Sum),
    }

    try:
        expr = sp.sympify(expression, locals=local_ops)
    except Exception:
        expr = sp.sympify(expression)

    if isinstance(expr, Relational):
        raise ValueError(error_msg)

    def _eval_advanced(e: sp.Expr, depth: int = 0) -> sp.Expr:
        if depth > 5:
            return e
        targets = list(e.atoms(sp.Derivative, sp.Integral, sp.Limit, sp.Sum))
        if not targets:
            return e
        for t in targets:
            try:
                res = _run_with_timeout(t.doit, 2)
            except Exception:
                res = t
            e = e.xreplace({t: res})
        return _eval_advanced(e, depth + 1)

    expr = _eval_advanced(expr)
    expr = expr.subs(sanitized)
    expr = _eval_advanced(expr)

    try:
        exact_simpl = _run_with_timeout(sp.simplify, 5, expr)
    except Exception:
        exact_simpl = expr

    if getattr(exact_simpl, "free_symbols", set()):
        try:
            result = _run_with_timeout(sp.N, 2, exact_simpl)
        except Exception:
            result = exact_simpl
    else:
        result = exact_simpl

    if getattr(result, "free_symbols", set()):
        raise ValueError(error_msg)

    try:
        if result.is_integer:  # type: ignore[attr-defined]
            return int(result)  # type: ignore[misc]
    except AttributeError:
        pass

    try:
        f = float(result)
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return f
    except Exception:
        raise ValueError(error_msg)


calc_answer_tool = function_tool(_calc_answer)
