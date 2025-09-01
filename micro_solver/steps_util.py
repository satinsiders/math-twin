from __future__ import annotations

import json
from typing import Any, Optional, Tuple, cast

from agents.run import Runner as AgentsRunner  # type: ignore


def _as_json(s: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads(s))
    except Exception as exc:
        raise ValueError(f"invalid-json:{exc}")


def _invoke(
    agent: Any,
    payload: Any,
    *,
    expect_json: bool = True,
    tools: Optional[list] = None,
    qa_feedback: Optional[str] = None,
) -> Tuple[Any, Optional[str]]:  # noqa: ANN401 - generic
    try:
        if isinstance(payload, str):
            raw = payload if not qa_feedback else f"{payload}\n\n[qa_feedback]: {qa_feedback}"
        else:
            try:
                data = dict(payload)
            except Exception:
                data = {"input": payload}
            if qa_feedback and "qa_feedback" not in data:
                data["qa_feedback"] = qa_feedback
            import json as _json
            raw = _json.dumps(data)
        res = AgentsRunner.run_sync(agent, input=raw, tools=tools)
        out = cast(str, getattr(res, "final_output", ""))
        out = out.strip()
        if expect_json:
            return _as_json(out), None
        return out, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)

