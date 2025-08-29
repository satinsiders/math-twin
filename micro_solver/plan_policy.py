from __future__ import annotations

"""Plan policy linter for micro-solver decomposition steps.

Use ``lint_plan(plan_steps)`` to validate that a plan is strictly planning-only
and does not compute results. Returns a dict with ``ok`` and a list of
``issues`` strings.
"""

from typing import Any, Dict, List
import re


# Always-disallowed keys because they imply precomputed outputs embedded in the plan
DISALLOWED_ARG_KEYS = {
    "result",
    "results",
}


def _num_like(x: Any) -> bool:
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", x.strip()):
        return True
    return False


def lint_plan(steps: Any) -> Dict[str, Any]:  # noqa: ANN401 - generic
    """Return {ok: bool, issues: [str]} for the given plan steps.

    Rules:
    - Each step must be a dict with string ``action`` and dict ``args`` (may include ``id`` string).
    - Args must not contain keys that indicate computed results (DISALLOWED_ARG_KEYS).
    - Args must not contain purely numeric lists (e.g., [1, 11, 37]).
    - Symbolic references are allowed (strings, identifiers, expressions). Numeric strings are allowed as single values, but not as a list of numbers.
    """
    issues: List[str] = []
    if not isinstance(steps, list) or not steps:
        return {"ok": False, "issues": ["plan-steps-missing-or-empty"]}

    for i, st in enumerate(steps):
        if not isinstance(st, dict):
            issues.append(f"step-{i+1}:not-a-dict")
            continue
        action = st.get("action")
        if not isinstance(action, str) or not action.strip():
            issues.append(f"step-{i+1}:missing-action")
        args = st.get("args") if isinstance(st.get("args"), dict) else None
        if not isinstance(args, dict):
            issues.append(f"step-{i+1}:missing-args-object")
            continue

        # Disallowed keys (explicitly forbidden regardless of value)
        for k in DISALLOWED_ARG_KEYS:
            if k in args:
                issues.append(f"step-{i+1}:arg-forbidden:{k}")

        # Numeric-only lists are forbidden in any arg
        for k, v in args.items():
            if isinstance(v, list) and v and all(_num_like(it) for it in v):
                issues.append(f"step-{i+1}:arg-numeric-list:{k}")

    return {"ok": len(issues) == 0, "issues": issues}
