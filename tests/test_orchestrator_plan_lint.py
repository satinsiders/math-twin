import pathlib
import sys
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from micro_solver.orchestrator import MicroGraph, MicroRunner
from micro_solver.state import MicroState


def test_invalid_plan_steps_abort():
    state = MicroState(plan_steps=[{"action": "compute", "args": {"result": 5}}])
    runner = MicroRunner(MicroGraph([]))
    with pytest.raises(RuntimeError) as exc:
        runner.run(state, lint_plan=True)
    assert "arg-forbidden:result" in str(exc.value)


def test_lint_bypass_allows_invalid_plan():
    state = MicroState(plan_steps=[{"action": "compute", "args": {"result": 5}}])
    runner = MicroRunner(MicroGraph([]))
    runner.run(state, lint_plan=False)

