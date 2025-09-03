from __future__ import annotations

import pytest

from micro_solver.scheduler import update_metrics, select_operator
from micro_solver.state import MicroState
from micro_solver.operators import Operator


class BaselineOp(Operator):
    name = "base"

    def applicable(self, state: MicroState) -> bool:
        return True

    def apply(self, state: MicroState):
        return state, 0.0


class MetricOp(Operator):
    name = "metric"

    def applicable(self, state: MicroState) -> bool:
        return True

    def apply(self, state: MicroState):
        return state, 0.0

    def score(self, state: MicroState) -> float:
        return state.M.get("ineq_satisfied", 0.0)


class ZeroDeltaOp(Operator):
    name = "zero"

    def applicable(self, state: MicroState) -> bool:
        return True

    def apply(self, state: MicroState):
        return state, 0.0


class PositiveDeltaOp(Operator):
    name = "positive"

    def applicable(self, state: MicroState) -> bool:
        return True

    def apply(self, state: MicroState):
        return state, 1.0


def test_update_metrics_tracks_progress() -> None:
    state = MicroState()
    state.C["symbolic"] = ["x = 3", "x >= 0", "x <= 10"]
    state.V["symbolic"]["variables"] = ["x"]
    state.V["symbolic"]["env"] = {"x": 5}
    state.V["symbolic"]["derived"] = {"bounds": {"x": (0.0, 10.0)}}
    state = update_metrics(state)
    assert state.M["residual_l2"] == pytest.approx(2.0)
    assert state.M["residual_l2_change"] == pytest.approx(0.0)
    assert state.M["ineq_satisfied"] == pytest.approx(2.0)
    assert state.M["bounds_volume"] == pytest.approx(10.0)
    p1 = state.M["progress_score"]

    state.V["symbolic"]["env"]["x"] = 3
    state.V["symbolic"]["derived"]["bounds"]["x"] = (0.0, 8.0)
    state = update_metrics(state)
    assert state.M["residual_l2"] == pytest.approx(0.0)
    assert state.M["residual_l2_change"] == pytest.approx(2.0)
    assert state.M["bounds_volume_reduction"] == pytest.approx(2.0)
    assert state.M["progress_score"] > p1

    assert state.M["sample_size"] == pytest.approx(0.0)

    state.V["symbolic"]["derived"]["sample"] = {"x": 1.0, "y": 2.0}
    state = update_metrics(state)
    assert state.M["sample_size"] == pytest.approx(2.0)
    assert state.M["sample_size_reduction"] == pytest.approx(-2.0)
    p2 = state.M["progress_score"]

    state.V["symbolic"]["derived"]["sample"].pop("y")
    state = update_metrics(state)
    assert state.M["sample_size"] == pytest.approx(1.0)
    assert state.M["sample_size_reduction"] == pytest.approx(1.0)
    assert state.M["progress_score"] > p2


def test_select_operator_uses_metric_scores() -> None:
    state = MicroState()
    state.M["ineq_satisfied"] = 5.0
    ops = [BaselineOp(), MetricOp()]
    chosen = select_operator(state, ops)
    assert isinstance(chosen, MetricOp)

    state.M["ineq_satisfied"] = 0.0
    chosen = select_operator(state, ops)
    assert isinstance(chosen, BaselineOp)


def test_select_operator_breaks_ties_with_delta() -> None:
    state = MicroState()
    ops = [ZeroDeltaOp(), PositiveDeltaOp()]
    chosen = select_operator(state, ops)
    assert isinstance(chosen, PositiveDeltaOp)


def test_update_metrics_drops_redundant_relations() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x", "y"]
    state.C["symbolic"] = ["x + y = 2", "2x + 2y = 4", "x - y = 0"]
    state = update_metrics(state)
    assert "2x + 2y = 4" not in state.C["symbolic"]
    assert state.M["redundant_constraints_idx"] == [1]
    assert state.M["redundant_constraints"] == ["2x + 2y = 4"]
