import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from micro_solver.sym_utils import parse_relation_sides
from micro_solver.operators import BoundInferOperator
from micro_solver.state import MicroState
from micro_solver.steps_meta import _micro_monitor_dof


def test_parse_relation_sides_unicode() -> None:
    assert parse_relation_sides("x ≤ 5") == ("<=", "x", "5")
    assert parse_relation_sides("y ≥ 3") == (">=", "y", "3")


def test_bound_infer_operator_unicode() -> None:
    state = MicroState()
    state.C["symbolic"] = ["x ≥ 0", "x ≤ 5"]
    state, delta = BoundInferOperator().apply(state)
    assert state.domain["x"] == (0.0, 5.0)
    assert delta == 2


def test_ineq_count_unicode() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x ≥ 0", "x ≤ 5", "x = 3"]
    state = _micro_monitor_dof(state)
    assert state.M["ineq_count"] == 2
