from micro_solver.scheduler import solve_with_defaults
from micro_solver.state import MicroState
import pytest


def test_certificate_records_best_candidate() -> None:
    state = MicroState()
    state.V["symbolic"]["variables"] = ["x"]
    state.C["symbolic"] = ["x + 2 = 5"]
    result = solve_with_defaults(state, max_iters=4)
    cert = result.A["symbolic"].get("certificate")
    assert cert is not None
    assert cert.verified is True
    assert str(cert.value) == "3"
    # Residual for solved relation should be close to zero
    assert cert.residuals.get("x + 2 = 5", 1.0) == pytest.approx(0.0)
