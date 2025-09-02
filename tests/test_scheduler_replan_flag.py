from typing import Tuple

from micro_solver.scheduler import solve
from micro_solver.state import MicroState
from micro_solver.operators import Operator


class DummyOperator(Operator):
    name = "dummy"

    def applicable(self, state: MicroState) -> bool:
        return True

    def apply(self, state: MicroState) -> Tuple[MicroState, float]:
        state.A["symbolic"]["final"] = 42
        return state, 1.0


def test_scheduler_applies_operator_without_spurious_replan() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]  # ensures non-zero degrees of freedom
    result = solve(state, [DummyOperator()], max_iters=2)
    assert result.A["symbolic"].get("final") == 42
