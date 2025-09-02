import pytest

from micro_solver.state import MicroState
from micro_solver.operators import DiffOperator, IntegrateOperator, DEFAULT_OPERATORS


def test_diff_operator_computes_derivative_and_progress() -> None:
    state = MicroState()
    state.derived = {}
    op = DiffOperator()
    assert not op.applicable(state)
    state.derived["expression"] = "x**2"
    assert op.applicable(state)
    state, delta = op.apply(state)
    assert state.derived["derivative"] == "2*x"
    assert delta == float(len("x**2") - len("2*x"))


def test_integrate_operator_computes_integral_and_progress() -> None:
    state = MicroState()
    state.derived = {}
    state.derived["expression"] = "2*x"
    op = IntegrateOperator()
    assert op.applicable(state)
    state, delta = op.apply(state)
    assert state.derived["integral"] == "x**2"
    assert delta == float(len("2*x") - len("x**2"))


def test_default_operators_include_new_ones() -> None:
    assert any(isinstance(o, DiffOperator) for o in DEFAULT_OPERATORS)
    assert any(isinstance(o, IntegrateOperator) for o in DEFAULT_OPERATORS)
