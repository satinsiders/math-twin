"""Internal helpers for JSON handling, answer coercion, validation, etc."""
from __future__ import annotations

import json
import os
import re
from typing import Any, cast

# ---------------------------------------------------------------------------
# Generic agent output handling
# ---------------------------------------------------------------------------

def get_final_output(res: Any) -> str:  # noqa: ANN401 – generic return
    """Extract the best‑guess textual payload from an Agents SDK response."""
    out = getattr(res, "final_output", None) or getattr(res, "output", None) or getattr(res, "content", None) or res
    return str(out)


# ---------------------------------------------------------------------------
# JSON safety wrapper
# ---------------------------------------------------------------------------

def safe_json(text: str) -> dict[str, Any]:
    """Best‑effort JSON repair loader with fenced‑code and bracket fallbacks."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        bracketed = re.search(r"\{[\s\S]*\}", text)
        if bracketed:
            text = bracketed.group(0)

    try:
        return cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError as exc:
        repaired = re.sub(r"\\([^\"\\/bfnrtu])", r"\\\\\1", text)
        try:
            return cast(dict[str, Any], json.loads(repaired))
        except Exception:
            snippet = text.strip().replace("\n", " ")[:300]
            raise ValueError(f"Agent output was not valid JSON: {snippet}...") from exc


# ---------------------------------------------------------------------------
# Answer coercion & validation
# ---------------------------------------------------------------------------

def coerce_answers(out: dict[str, Any]) -> dict[str, Any]:
    choices = out.get("choices") or []
    ans_idx = out.get("answer_index")
    ans_val = out.get("answer_value")
    ans_amb = out.get("answer")  # legacy support

    def _find_index(val: Any) -> int | None:  # noqa: ANN401 – generic param
        sval = str(val).strip()
        for i, c in enumerate(choices):
            if str(c).strip() == sval:
                return i
        try:
            f = float(sval)
            for i, c in enumerate(choices):
                try:
                    if abs(float(str(c)) - f) < 1e-9:
                        return i
                except Exception:
                    pass
        except Exception:
            pass
        return None

    if isinstance(ans_idx, int) and 0 <= ans_idx < len(choices):
        ans_val = choices[ans_idx]
    else:
        if ans_val is not None:
            idx = _find_index(ans_val)
            if idx is not None:
                ans_idx = idx
                ans_val = choices[idx]
        if ans_idx is None and ans_amb is not None:
            if isinstance(ans_amb, int) and 0 <= ans_amb < len(choices):
                ans_idx = ans_amb
                ans_val = choices[ans_idx]
            else:
                idx = _find_index(ans_amb)
                if idx is not None:
                    ans_idx = idx
                    ans_val = choices[idx]

    out["answer_index"] = ans_idx
    out["answer_value"] = ans_val
    out["answer"] = ans_val  # backward compatibility
    return out


def validate_output(block: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    q = block.get("twin_stem") or block.get("question")
    choices = block.get("choices") or []
    ans_val = block.get("answer_value", block.get("answer"))
    ans_idx = block.get("answer_index")

    if not isinstance(q, str) or not q.strip():
        errors.append("Empty question stem.")
    if not isinstance(choices, list) or not choices:
        errors.append("Choices missing or empty.")

    if not isinstance(ans_idx, int) or not (0 <= ans_idx < len(choices)):
        errors.append("answer_index invalid or missing.")
    elif ans_val != choices[ans_idx]:
        errors.append("answer_value mismatch.")

    if "graph_path" in block and not os.path.isfile(str(block["graph_path"])):
        errors.append("graph_path does not exist.")

    block["errors"] = errors
    return block
