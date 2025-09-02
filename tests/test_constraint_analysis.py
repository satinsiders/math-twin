from micro_solver.constraint_analysis import (
    mark_redundant_constraints,
    attempt_rank_repair,
)


def test_mark_redundant_constraints_detects_dependency() -> None:
    rels = ["x + y = 2", "2x + 2y = 4", "x - y = 0"]
    redundant = mark_redundant_constraints(rels, ["x", "y"])
    assert redundant == [1]


def test_attempt_rank_repair_removes_redundant() -> None:
    rels = ["x + y = 2", "2x + 2y = 4", "x - y = 0"]
    repaired, info = attempt_rank_repair(rels, ["x", "y"])
    assert len(repaired) == 2
    assert info["removed"] == ["2x + 2y = 4"]
