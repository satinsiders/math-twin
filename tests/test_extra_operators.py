import pytest

from micro_solver.state import MicroState
from micro_solver.operators import (
    EliminateOperator,
    TransformOperator,
    CaseSplitOperator,
    BoundInferOperator,
    NumericSolveOperator,
    GridRefineOperator,
    QuadratureOperator,
    RationalizeOperator,
    DomainPruneOperator,
    FeasibleSampleOperator,
)


def test_eliminate_operator_removes_symbol() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x", "y"]
    state.C["symbolic"] = ["x + y = 3", "y = 1"]
    state, delta = EliminateOperator().apply(state)
    assert all("y" not in r for r in state.C["symbolic"])
    assert "y" not in state.V["symbolic"]["variables"]
    assert delta > 0


def test_transform_operator_factor() -> None:
    state = MicroState()
    state.C["symbolic"] = ["x**2 + 2*x + 1 = 0"]
    state, delta = TransformOperator(action="factor").apply(state)
    assert state.C["symbolic"] == ["(x + 1)**2 = 0"]
    assert delta > 0


def test_case_split_operator_generates_cases() -> None:
    state = MicroState()
    state.C["symbolic"] = ["x**2 = 1"]
    state, delta = CaseSplitOperator().apply(state)
    assert state.V["symbolic"]["derived"].get("cases") == ["x = 1", "x = -1"]
    assert delta == 2


def test_bound_infer_operator_collects_bounds() -> None:
    state = MicroState()
    state.C["symbolic"] = ["x >= 0", "x < 5"]
    state, delta = BoundInferOperator().apply(state)
    assert state.domain["x"] == (0.0, 5.0)
    assert delta == 2


def test_numeric_solve_operator_evaluates_expression() -> None:
    state = MicroState()
    state.C["symbolic"] = ["x = 2 + 3"]
    state, delta = NumericSolveOperator().apply(state)
    assert state.A["symbolic"]["candidates"] == ["5"]
    assert delta == 1.0


def test_grid_refine_operator_rounds_sample() -> None:
    state = MicroState()
    state.V["symbolic"]["derived"]["sample"] = {"x": 0.3333333}
    state, delta = GridRefineOperator().apply(state)
    assert state.V["symbolic"]["derived"]["sample"]["x"] == 0.333
    assert delta == 1.0


def test_quadrature_operator_computes_integral() -> None:
    state = MicroState()
    state.V["symbolic"]["derived"]["integrand"] = "x"
    state.V["symbolic"]["derived"]["interval"] = (0, 1)
    state, delta = QuadratureOperator().apply(state)
    assert state.V["symbolic"]["derived"]["integral"] == pytest.approx(0.5)
    assert delta == 1.0


def test_rationalize_operator_converts_candidates() -> None:
    state = MicroState()
    state.A["symbolic"]["candidates"] = ["0.5", "2"]
    state, delta = RationalizeOperator().apply(state)
    assert state.A["symbolic"]["candidates"] == ["1/2", "2"]
    assert delta == 1.0


def test_feasible_sample_operator_respects_bounds() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x >= 1", "x <= 2"]
    state, _ = BoundInferOperator().apply(state)
    import random

    random.seed(0)
    state, _ = FeasibleSampleOperator().apply(state)
    sample = state.V["symbolic"]["derived"]["sample"]["x"]
    assert 1.0 <= sample <= 2.0


def test_domain_prune_operator_removes_invalid_samples() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x", "y"]
    state.domain = {"x": (0.0, 1.0)}
    state.qual = {"y": {"nonnegative"}}
    state.V["symbolic"]["derived"]["sample"] = {"x": 2.0, "y": -1.0}
    state, delta = DomainPruneOperator().apply(state)
    assert "x" not in state.V["symbolic"]["derived"]["sample"]
    assert "y" not in state.V["symbolic"]["derived"]["sample"]
    assert delta == 2.0
