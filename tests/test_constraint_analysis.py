from micro_solver.constraint_analysis import (
    mark_redundant_constraints,
    attempt_rank_repair,
    build_independence_graph,
)
from micro_solver.steps_meta import _micro_monitor_dof
from micro_solver.state import MicroState


def test_mark_redundant_constraints_detects_dependency() -> None:
    rels = ["x + y = 2", "2x + 2y = 4", "x - y = 0"]
    redundant = mark_redundant_constraints(rels, ["x", "y"])
    assert redundant == [1]


def test_attempt_rank_repair_removes_redundant() -> None:
    rels = ["x + y = 2", "2x + 2y = 4", "x - y = 0"]
    repaired, info = attempt_rank_repair(rels, ["x", "y"])
    assert len(repaired) == 2
    assert info["removed"] == ["2x + 2y = 4"]


def test_inequality_does_not_shift_indices() -> None:
    rels = ["x >= 0", "x + y = 2", "2x + 2y = 4"]
    redundant = mark_redundant_constraints(rels, ["x", "y"])
    assert redundant == [2]
    repaired, info = attempt_rank_repair(rels, ["x", "y"])
    assert repaired == ["x >= 0", "x + y = 2"]
    assert info["removed"] == ["2x + 2y = 4"]


def test_build_independence_graph_detects_redundant() -> None:
    rels = ["x + y = 2", "2x + 2y = 4", "x - y = 0"]
    graph = build_independence_graph(rels, ["x", "y"])
    assert graph["redundant"] == [1]


def test_monitor_dof_records_redundant() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x", "y"]
    state.C["symbolic"] = ["x + y = 2", "2x + 2y = 4", "x - y = 0"]
    state = _micro_monitor_dof(state)
    assert state.M["redundant_constraints"] == ["2x + 2y = 4"]
