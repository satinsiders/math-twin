from __future__ import annotations

"""Shared candidate verification helpers for scheduler and operators."""

from typing import Any

from .state import MicroState
from .sym_utils import verify_candidate, evaluate_with_env, evaluate_numeric


def verify_candidate_state(state: MicroState, *, via: str) -> bool:
    """Verify ``state.A['symbolic']['candidate']`` and promote to final if valid.

    Returns ``True`` if verification succeeded. Records ``verification_context``
    in ``state.M`` and, when successful, sets ``final`` and
    ``final_confidence='verified'``. Verification may use a custom callable
    ``state.E['verifier']`` or compare against ``state.E['canonical_target']``;
    otherwise it falls back to relation substitution via ``verify_candidate``.
    """

    candidate = state.A["symbolic"].get("candidate")
    if candidate is None or state.A["symbolic"].get("final") is not None:
        return False

    env = state.V["symbolic"].get("env", {})
    context: dict[str, Any] = {
        "via": via,
        "dof_at_verify": state.M.get("degrees_of_freedom"),
        "evidence": None,
    }

    ok = False
    verifier = state.E.get("verifier") if isinstance(state.E, dict) else None
    if callable(verifier):
        try:
            ok = bool(verifier(candidate))
            context["evidence"] = ok
        except Exception as exc:  # pragma: no cover - defensive
            context["evidence"] = f"verifier_error:{exc}"
            ok = False
    else:
        target = state.E.get("canonical_target") if isinstance(state.E, dict) else None
        if target is not None:
            ok_cand, val_cand = evaluate_with_env(str(candidate), env)
            if not ok_cand:
                ok_cand, val_cand = evaluate_numeric(str(candidate))
            ok_tar, val_tar = evaluate_with_env(str(target), env)
            if not ok_tar:
                ok_tar, val_tar = evaluate_numeric(str(target))
            if ok_cand and ok_tar:
                try:
                    ok = float(val_cand) == float(val_tar)
                except Exception:
                    ok = val_cand == val_tar
                context["evidence"] = {"candidate": val_cand, "target": val_tar}
        else:
            ok = verify_candidate(state.C["symbolic"], str(candidate))
            context["evidence"] = ok

    state.M["verification_context"] = context
    if ok:
        state.A["symbolic"]["final"] = candidate
        state.A["symbolic"]["final_confidence"] = "verified"
    return ok
