from __future__ import annotations

"""Utility functions for building solver certificates.

The certificate captures the best candidate found and how well it satisfies the
original constraints.  This is a lightweight, anytime artefact that downstream
consumers can inspect for transparency.
"""

from typing import Any, Dict, Optional

from .candidate import Candidate
from .sym_utils import parse_relation_sides, _parse_expr  # type: ignore


def _compute_residuals(relations: list[str], candidate: Any, *, varname: Optional[str] = None) -> Dict[str, float]:
    """Return residuals for equality relations after substituting ``candidate``.

    Non-equality relations are ignored. Residuals are absolute numeric
    differences between left and right hand sides.
    """

    residuals: Dict[str, float] = {}
    try:
        import sympy as sp

        sym = sp.Symbol(str(varname or "x"))
        cand = _parse_expr(str(candidate))
    except Exception:
        return residuals
    for rel in relations:
        op, lhs, rhs = parse_relation_sides(rel)
        if op != "=":
            continue
        try:
            L = _parse_expr(lhs).subs({sym: cand})
            R = _parse_expr(rhs).subs({sym: cand})
            residual = abs(float(sp.N(L - R)))
            residuals[rel] = residual
        except Exception:
            continue
    return residuals


def build_certificate(state: Any) -> Candidate:
    """Create a :class:`Candidate` summary for ``state``.

    Uses ``state.final_answer`` when available, otherwise falls back to the last
    entry in ``state.candidate_answers``.  The candidate always includes the
    residuals against the original ``state.relations`` and a ``verified`` flag
    indicating whether the candidate passed verification.
    """

    cand_val = getattr(state, "final_answer", None)
    verified = cand_val is not None
    if cand_val is None:
        try:
            cand_val = state.candidate_answers[-1]
        except Exception:
            cand_val = None
    residuals: Dict[str, float] = {}
    if cand_val is not None:
        var = None
        try:
            from .steps_candidate import _infer_target_var  # type: ignore

            var = _infer_target_var(state)
        except Exception:
            var = None
        residuals = _compute_residuals(list(getattr(state, "relations", [])), cand_val, varname=var)
    return Candidate(value=cand_val, residuals=residuals, verified=verified)
