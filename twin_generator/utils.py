"""Internal helpers for JSON handling, answer coercion, validation, etc."""
from __future__ import annotations

import json
import os
import re
from types import ModuleType
from typing import Any, Callable, cast

try:  # pragma: no cover - optional dependency
    import json5 as _json5  # type: ignore
except Exception:  # pragma: no cover - fall back to stdlib
    _json5 = None
json5: ModuleType | None = _json5

# ---------------------------------------------------------------------------
# Generic agent output handling
# ---------------------------------------------------------------------------


def get_final_output(res: Any) -> str:  # noqa: ANN401 – generic return
    """Extract the best‑guess textual payload from an Agents SDK response."""
    for attr in ("final_output", "output", "content"):
        if hasattr(res, attr):
            val = getattr(res, attr)
            if val is not None:
                return str(val)
    return str(res)


# ---------------------------------------------------------------------------
# JSON safety helpers
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """Return the most likely JSON substring from *text*."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    bracketed = re.search(r"\{[\s\S]*\}", text)
    if bracketed:
        return bracketed.group(0)
    return text


def _repair_json(text: str) -> str:
    """Attempt light‑weight JSON repairs and return the adjusted string."""
    repaired = text
    # Replace only single quotes that act as string delimiters, leaving apostrophes
    # inside strings intact. This pattern mirrors typical JSON token boundaries and
    # avoids overzealous replacements.
    repaired = re.sub(
        r"(?<![\\w])'([^'\\]*(?:\\.[^'\\]*)*)'",
        r'"\1"',
        repaired,
    )
    # Strip both line (`//`) and block (`/* */`) comments
    repaired = re.sub(r"//.*?(?=\n|$)", "", repaired)
    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    if open_braces > close_braces:
        repaired += "}" * (open_braces - close_braces)
    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    if open_brackets > close_brackets:
        repaired += "]" * (open_brackets - close_brackets)
    repaired = re.sub(r",\s*(?=[}\]])", "", repaired)
    repaired = re.sub(r"\\([^\"\\/bfnrtu])", r"\\\\\1", repaired)
    return repaired


def _parsers() -> list[Callable[[str], Any]]:
    parsers: list[Callable[[str], Any]] = []
    if json5 is not None:
        parsers.append(json5.loads)
    parsers.append(json.loads)
    return parsers


def safe_json(text: str) -> dict[str, Any]:
    """Best‑effort JSON loader with tolerant parsing and repair attempts."""
    text = text.strip()
    if not text:
        raise ValueError("Agent output was empty")

    original_snippet = text.replace("\n", " ")[:300]
    text = _extract_json_block(text)

    for parser in _parsers():
        try:
            return cast(dict[str, Any], parser(text))
        except Exception:
            pass

    repaired = _repair_json(text)
    for parser in _parsers():
        try:
            return cast(dict[str, Any], parser(repaired))
        except Exception:
            pass

    repaired_snippet = repaired.strip().replace("\n", " ")[:300]
    raise ValueError(
        "Agent output was not valid JSON even after repair. "
        f"Original snippet: {original_snippet}... Repaired snippet: {repaired_snippet}..."
    )


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
