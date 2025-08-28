"""Tools for analyzing and synthesizing function graphs.

These tools help agents compute points from analytic expressions and fit
functions to observed points extracted from a graph image.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional, List

from agents.tool import function_tool

__all__ = [
    "sample_function_points_tool",
    "fit_function_tool",
]


def _parse_params(params_json: Optional[str]) -> dict[str, Any]:
    if not params_json:
        return {}
    try:
        data = json.loads(params_json)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _eval_expr_at_x(expr: str, x: float, params: dict[str, Any]) -> float:
    import sympy as sp  # type: ignore

    x_sym = sp.Symbol("x")
    try:
        f = sp.sympify(expr)
    except Exception as exc:
        raise ValueError(f"Invalid expression: {exc}")
    subs = {sp.Symbol(k): sp.sympify(v) for k, v in params.items()}
    y = sp.N(f.subs(subs).subs({x_sym: x}))
    return float(y)


def _sort_points(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = list(points)
    pts.sort(key=lambda p: (p[0], p[1]))
    return pts


def _polyfit(points: list[tuple[float, float]], degree: int) -> tuple[list[float], float]:
    import numpy as np  # type: ignore

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    coeffs = np.polyfit(xs, ys, degree)
    p = np.poly1d(coeffs)
    ss_res = float(np.sum((ys - p(xs)) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    return coeffs.tolist(), r2


def _linreg(X: list[float], Y: list[float]) -> tuple[float, float, float]:
    """Return slope, intercept, r2 for Y ~ a*X + b."""
    import numpy as np  # type: ignore

    x = np.array(X, dtype=float)
    y = np.array(Y, dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return 0.0, float(np.mean(y) if len(y) else 0.0), 0.0
    y_pred = a * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    return float(a), float(b), r2


def _equation_from_coeffs(coeffs: list[float]) -> tuple[str, dict[str, float]]:
    # coeffs: highest power first, e.g., [a, b, c] for ax^2 + bx + c
    deg = len(coeffs) - 1
    params: dict[str, float] = {}
    terms: list[str] = []
    for i, c in enumerate(coeffs):
        power = deg - i
        name = chr(ord("a") + i)
        params[name] = float(c)
        if power == 0:
            terms.append(f"{name}")
        elif power == 1:
            terms.append(f"{name}*x")
        else:
            terms.append(f"{name}*x**{power}")
    eq = " + ".join(terms)
    return eq, params


def _fit_best_poly(points: list[tuple[float, float]], max_degree: int) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for deg in range(1, max(1, max_degree) + 1):
        try:
            coeffs, r2 = _polyfit(points, deg)
        except Exception:
            continue
        eq, params = _equation_from_coeffs(coeffs)
        cand = {"type": "polynomial", "degree": deg, "equation": eq, "parameters": params, "r2": r2}
        if best is None or cand["r2"] > best.get("r2", -1.0):
            best = cand
    return best or {"type": "unknown", "equation": "", "parameters": {}, "r2": 0.0}


def _fit_exponential(points: list[tuple[float, float]]) -> dict[str, Any]:
    # y = A * exp(B x); requires y>0 for log transform
    X: list[float] = []
    LY: list[float] = []
    for x, y in points:
        if y is None:
            continue
        try:
            if y > 0:
                X.append(float(x))
                import math

                LY.append(math.log(float(y)))
        except Exception:
            continue
    if len(X) < 2:
        return {"type": "exponential", "equation": "A*exp(B*x)", "parameters": {}, "r2": 0.0}
    a, b, r2 = _linreg(X, LY)
    import math

    A = math.exp(b)
    B = a
    return {
        "type": "exponential",
        "equation": "A*exp(B*x)",
        "parameters": {"A": float(A), "B": float(B)},
        "r2": float(r2),
    }


def _fit_logarithmic(points: list[tuple[float, float]]) -> dict[str, Any]:
    # y = A * ln(x) + B; requires x>0
    LX: list[float] = []
    Y: list[float] = []
    for x, y in points:
        try:
            if x > 0:
                import math

                LX.append(math.log(float(x)))
                Y.append(float(y))
        except Exception:
            continue
    if len(LX) < 2:
        return {"type": "log", "equation": "A*log(x) + B", "parameters": {}, "r2": 0.0}
    A, B, r2 = _linreg(LX, Y)
    return {
        "type": "log",
        "equation": "A*log(x) + B",
        "parameters": {"A": float(A), "B": float(B)},
        "r2": float(r2),
    }


def _fit_power(points: list[tuple[float, float]]) -> dict[str, Any]:
    # y = A * x^k; requires x>0,y>0; log transform: ln y = ln A + k ln x
    LX: list[float] = []
    LY: list[float] = []
    for x, y in points:
        try:
            if x > 0 and y > 0:
                import math

                LX.append(math.log(float(x)))
                LY.append(math.log(float(y)))
        except Exception:
            continue
    if len(LX) < 2:
        return {"type": "power", "equation": "A*x**k", "parameters": {}, "r2": 0.0}
    k, lnA, r2 = _linreg(LX, LY)
    import math

    A = math.exp(lnA)
    return {
        "type": "power",
        "equation": "A*x**k",
        "parameters": {"A": float(A), "k": float(k)},
        "r2": float(r2),
    }


def _fit_trig(points: list[tuple[float, float]], max_freqs: int = 50) -> dict[str, Any]:
    # y ~= a*sin(w x) + b*cos(w x) + d; grid-search w
    import numpy as np  # type: ignore
    import math

    if len(points) < 4:
        return {"type": "trig", "equation": "A*sin(B*x + C) + D", "parameters": {}, "r2": 0.0}
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    lo, hi = float(np.min(xs)), float(np.max(xs))
    span = max(1e-6, hi - lo)
    # Candidate angular frequencies w over a loose range
    ws = np.linspace(2 * math.pi / (span * 6), 2 * math.pi / (span * 0.5), max_freqs)
    best: dict[str, Any] | None = None
    for w in ws:
        S = np.sin(w * xs)
        C = np.cos(w * xs)
        A = np.column_stack([S, C, np.ones_like(xs)])
        try:
            a, b, d = np.linalg.lstsq(A, ys, rcond=None)[0]
        except Exception:
            continue
        y_pred = a * S + b * C + d
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
        Aamp = float(math.hypot(a, b))
        phase = float(math.atan2(b, a))
        cand = {
            "type": "trig",
            "equation": "A*sin(B*x + C) + D",
            "parameters": {"A": float(Aamp), "B": float(w), "C": float(phase), "D": float(d)},
            "r2": float(r2),
        }
        if best is None or cand["r2"] > best.get("r2", -1.0):
            best = cand
    return best or {"type": "trig", "equation": "A*sin(B*x + C) + D", "parameters": {}, "r2": 0.0}


def _parse_points(points_json: str) -> list[tuple[float, float]]:
    data = json.loads(points_json)
    pts: list[tuple[float, float]] = []
    if isinstance(data, dict) and "points" in data:
        data = data["points"]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    pts.append((float(item[0]), float(item[1])))
                except Exception:
                    continue
    return _sort_points(pts)


def _bound_from_points(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return -10.0, 10.0
    xs = [p[0] for p in points]
    return min(xs), max(xs)


def _sample_points(expr: str, params: dict[str, Any], x_values: Optional[List[float]], n: int, x_min: Optional[float], x_max: Optional[float]) -> list[list[float]]:
    if x_values is None:
        if x_min is None or x_max is None:
            x_min, x_max = -10.0, 10.0
        x_values = _linspace(float(x_min), float(x_max), max(2, n))
    pts: list[list[float]] = []
    for x in x_values:
        y = _eval_expr_at_x(expr, float(x), params)
        if not (y != y):  # filter NaN
            pts.append([float(x), float(y)])
    return pts


def _fit_family(points: list[tuple[float, float]], family: Optional[str], max_degree: int) -> dict[str, Any]:
    fam = (family or "").strip().lower()
    if fam in ("", "poly", "polynomial", "auto"):
        return _fit_best_poly(points, max_degree)
    if fam in ("linear", "line"):
        return _fit_best_poly(points, 1)
    if fam in ("quadratic", "quad"):
        return _fit_best_poly(points, 2)
    if fam in ("exponential", "exp"):
        return _fit_exponential(points)
    if fam in ("logarithmic", "log"):
        return _fit_logarithmic(points)
    if fam in ("power", "powerlaw", "power-law"):
        return _fit_power(points)
    if fam in ("trig", "trigonometric", "sin", "sine"):
        return _fit_trig(points)
    return _fit_best_poly(points, max_degree)


def _fit_best_model(points: list[tuple[float, float]], max_degree: int = 3) -> dict[str, Any]:
    candidates = [
        _fit_best_poly(points, max_degree),
        _fit_exponential(points),
        _fit_logarithmic(points),
        _fit_power(points),
        _fit_trig(points),
    ]
    best = max(candidates, key=lambda d: float(d.get("r2", 0.0)))
    return best


def _infer_bounds(points: list[tuple[float, float]]) -> tuple[float, float]:
    lo, hi = _bound_from_points(points)
    if lo == hi:
        lo -= 5.0
        hi += 5.0
    return lo, hi


def _sample_function_points(
    expr: str,
    params_json: Optional[str] = None,
    x_values: Optional[List[float]] = None,
    n: int = 50,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> dict[str, Any]:
    params = _parse_params(params_json)
    pts = _sample_points(expr, params, x_values, n, x_min, x_max)
    return {"points": pts}


def _fit_function(points_json: str, family: Optional[str] = None, max_degree: int = 3, families: Optional[List[str]] = None) -> dict[str, Any]:
    data = json.loads(points_json)
    # Multi-series: return fits per series
    if isinstance(data, dict) and isinstance(data.get("series"), list):
        out_series: list[dict[str, Any]] = []
        for s in data.get("series", []):
            if not isinstance(s, dict):
                continue
            label = s.get("label")
            pts = s.get("points", [])
            points = _sort_points([(float(p[0]), float(p[1])) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2])
            if not points:
                out_series.append({"label": label, "fit": {"type": "unknown", "equation": "", "parameters": {}, "r2": 0.0}})
                continue
            if families:
                cand_best: dict[str, Any] | None = None
                for fam in families:
                    cand = _fit_family(points, fam, max_degree)
                    if cand_best is None or float(cand.get("r2", 0.0)) > float(cand_best.get("r2", 0.0)):
                        cand_best = cand
                fit = cand_best or _fit_best_model(points, max_degree)
            elif (family or "").strip().lower() in ("auto", ""):
                fit = _fit_best_model(points, max_degree)
            else:
                fit = _fit_family(points, family, max_degree)
            lo, hi = _infer_bounds(points)
            try:
                sampled = _sample_points(fit.get("equation", ""), fit.get("parameters", {}), None, 50, lo, hi)
            except Exception:
                sampled = []
            out_series.append({"label": label, "fit": fit, "suggested_points": sampled})
        return {"series": out_series}

    # Single series
    points = _parse_points(points_json)
    if not points:
        return {"type": "unknown", "equation": "", "parameters": {}, "r2": 0.0}
    if families:
        fit = None
        for fam in families:
            cand = _fit_family(points, fam, max_degree)
            if fit is None or float(cand.get("r2", 0.0)) > float(fit.get("r2", 0.0)):
                fit = cand
        fit = fit or _fit_best_model(points, max_degree)
    elif (family or "").strip().lower() in ("auto", ""):
        fit = _fit_best_model(points, max_degree)
    else:
        fit = _fit_family(points, family, max_degree)
    lo, hi = _infer_bounds(points)
    try:
        sampled = _sample_points(fit.get("equation", ""), fit.get("parameters", {}), None, 50, lo, hi)
    except Exception:
        sampled = []
    fit["suggested_points"] = sampled
    return fit


sample_function_points_tool = function_tool(_sample_function_points)
sample_function_points_tool["name"] = "sample_function_points_tool"

fit_function_tool = function_tool(_fit_function)
fit_function_tool["name"] = "fit_function_tool"
