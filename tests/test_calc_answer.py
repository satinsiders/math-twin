from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from twin_generator.tools import _calc_answer  # noqa: E402
import twin_generator.tools as tools  # noqa: E402


def test_calc_answer_diff() -> None:
    assert _calc_answer("diff(x**2, x)", '{"x": 3}') == 6


def test_calc_answer_integral() -> None:
    assert _calc_answer("integrate(x, (x, 0, 2))", '{}') == 2


def test_calc_answer_limit() -> None:
    assert _calc_answer("limit(sin(x)/x, x, 0)", '{}') == 1


def test_calc_answer_summation() -> None:
    assert _calc_answer("summation(x, (x, 1, 3))", '{}') == 6


def test_calc_answer_nested() -> None:
    expr = "integrate(diff(x**3, x), (x, 0, 2))"
    assert _calc_answer(expr, '{}') == 8


def test_calc_answer_skips_bad_params() -> None:
    with pytest.warns(UserWarning):
        assert _calc_answer("x + y", '{"x": 1, "y": "oops"}') == "y + 1.0"


def test_calc_answer_timeout_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _timeout(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise TimeoutError

    monkeypatch.setattr(tools, "_run_with_timeout", _timeout)
    assert tools._calc_answer("diff(x**2, x)", '{}') == "Derivative(x**2, x)"
