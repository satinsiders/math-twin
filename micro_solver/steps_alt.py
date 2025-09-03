from __future__ import annotations

"""Populate alternative representation from existing symbolic data."""

from copy import deepcopy

from .state import MicroState


def _micro_alt(state: MicroState) -> MicroState:
    """Initialize alt view (R/C/V/A) by copying the symbolic view."""
    try:
        state.R["alt"] = deepcopy(state.R.get("symbolic", {}))
        state.C["alt"] = list(state.C.get("symbolic", []))
        state.V["alt"] = deepcopy(state.V.get("symbolic", {}))
        state.A["alt"] = deepcopy(state.A.get("symbolic", {}))
    except Exception as exc:  # pragma: no cover - defensive
        state.error = f"alt-populate-failed:{exc}"
    return state
