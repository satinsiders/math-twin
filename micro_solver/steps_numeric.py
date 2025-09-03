from __future__ import annotations

"""Populate numeric representation from existing symbolic data."""

from copy import deepcopy

from .state import MicroState


def _micro_numeric(state: MicroState) -> MicroState:
    """Initialize numeric view (R/C/V/A) by copying the symbolic view."""
    try:
        state.R["numeric"] = deepcopy(state.R.get("symbolic", {}))
        state.C["numeric"] = list(state.C.get("symbolic", []))
        state.V["numeric"] = deepcopy(state.V.get("symbolic", {}))
        state.A["numeric"] = deepcopy(state.A.get("symbolic", {}))
    except Exception as exc:  # pragma: no cover - defensive
        state.error = f"numeric-populate-failed:{exc}"
    return state
