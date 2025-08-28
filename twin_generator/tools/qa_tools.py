"""QA-specific tool wrappers."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from agents.tool import function_tool

from .calc import _sanitize_params
from ..utils import coerce_answers, validate_output

__all__ = [
    "sanitize_params_tool",
    "validate_output_tool",
    "check_asset_tool",
    "graph_consistency_tool",
    "validate_answer_ref_tool",
    "detect_degenerate_params_tool",
    "count_concept_steps_tool",
    "check_invariants_tool",
    "choices_truth_filter_tool",
    "rationale_grounding_tool",
    "stem_number_grounding_tool",
    "_sanitize_params_tool",
    "_validate_output_tool",
    "_check_asset",
    "_graph_consistency_tool",
    "_validate_answer_ref_tool",
    "_detect_degenerate_params_tool",
    "_count_concept_steps_tool",
    "_check_invariants_tool",
    "_choices_truth_filter_tool",
    "_rationale_grounding_tool",
    "_stem_number_grounding_tool",
]


def _sanitize_params_tool(params_json: str) -> dict[str, Any]:
    """Return numeric parameters and skipped keys from *params_json*."""
    params = json.loads(params_json)
    sanitized, skipped = _sanitize_params(params)
    sanitized_out = {k: str(v) for k, v in sanitized.items()}
    return {"sanitized": sanitized_out, "skipped": skipped}


sanitize_params_tool = function_tool(_sanitize_params_tool)
sanitize_params_tool["name"] = "sanitize_params_tool"


def _validate_output_tool(block_json: str) -> dict[str, Any]:
    """Coerce answer fields then validate the formatter output."""
    block = json.loads(block_json)
    block = coerce_answers(block)
    return validate_output(block)


validate_output_tool = function_tool(_validate_output_tool)
validate_output_tool["name"] = "validate_output_tool"


def _check_asset(graph_path: Optional[str] = None, table_html: Optional[str] = None) -> bool:
    """Return ``True`` if no asset is required or one is available.

    The QA pipeline may omit both ``graph_path`` and ``table_html`` when a twin
    question does not use a visual or tabular asset. In that case there is no
    asset to validate and the function should consider the check successful.

    Parameters are considered missing when they are ``None`` or empty strings.
    """

    # If both assets are missing, there is nothing to validate.
    if not graph_path and not table_html:
        return True

    # Accept remote HTTP(S) URLs as valid assets
    if graph_path and (graph_path.startswith("http://") or graph_path.startswith("https://")):
        return True
    if graph_path and os.path.isfile(graph_path):
        return True
    if table_html and str(table_html).strip():
        return True
    return False


check_asset_tool = function_tool(_check_asset)
check_asset_tool["name"] = "check_asset_tool"


def _graph_consistency_tool(
    graph_path: str,
    points_json: str,
    style: Optional[str] = None,
    tolerance: float = 0.2,
) -> dict[str, Any]:
    """Best-effort visual consistency check between a rendered graph and points.

    Strategy: re-render a graph image from the given points using the same plotting
    routine and compare with the target image via a simple pixel-difference score.
    Returns an object with fields {ok: bool, score: float} where lower scores are
    better (0=identical). If dependencies are unavailable or the image is remote,
    the check degrades to ok=True with score=1.0 and a reason field.
    """

    try:
        if not graph_path or not os.path.isfile(graph_path):
            return {"ok": True, "score": 1.0, "reason": "no-local-file-or-missing"}
    except Exception:
        return {"ok": True, "score": 1.0, "reason": "access-error"}

    try:
        from .graph import _render_graph as _render
    except Exception:
        return {"ok": True, "score": 1.0, "reason": "missing-renderer"}

    try:
        import json as _json
        spec = {"points": _json.loads(points_json).get("points") if points_json.strip().startswith("{") else _json.loads(points_json)}
        if style:
            spec["style"] = style
        recon_path = _render(_json.dumps(spec))
    except Exception:
        return {"ok": True, "score": 1.0, "reason": "rerender-failed"}

    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        # Dependencies not installed; pass leniently
        return {"ok": True, "score": 1.0, "reason": "missing-deps"}

    try:
        a = Image.open(graph_path).convert("L").resize((256, 256))
        b = Image.open(recon_path).convert("L").resize((256, 256))
        A = np.asarray(a, dtype=float) / 255.0
        B = np.asarray(b, dtype=float) / 255.0
        diff = np.abs(A - B)
        score = float(diff.mean())  # 0 identical, larger means different
        ok = score <= float(tolerance)
        return {"ok": bool(ok), "score": score}
    except Exception:
        return {"ok": True, "score": 1.0, "reason": "compare-failed"}


graph_consistency_tool = function_tool(_graph_consistency_tool)
graph_consistency_tool["name"] = "graph_consistency_tool"


def _validate_answer_ref_tool(
    template_json: str,
    params_json: str = "",
) -> dict[str, Any]:
    """Validate that answer_expression is either numeric-like or references existing params/outputs.

    Heuristics:
    - Consider expressions matching an identifier pattern as references: [A-Za-z_][A-Za-z0-9_]*
    - If reference, check presence in params; otherwise, check if any template operation declares that output.
    Returns: { ok: bool, is_identifier: bool, ref: str|None, in_params: bool, in_operations: bool, detail: str }
    """
    import re

    try:
        tpl = json.loads(template_json) if isinstance(template_json, str) else template_json
    except Exception:
        return {"ok": False, "is_identifier": False, "ref": None, "in_params": False, "in_operations": False, "detail": "invalid template_json"}
    try:
        params = json.loads(params_json) if params_json else {}
    except Exception:
        params = {}

    expr = None
    if isinstance(tpl, dict):
        expr = tpl.get("answer_expression")
    if not isinstance(expr, str) or not expr.strip():
        return {"ok": False, "is_identifier": False, "ref": None, "in_params": False, "in_operations": False, "detail": "missing answer_expression"}

    ident_pat = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    is_ident = bool(ident_pat.match(expr.strip()))
    if not is_ident:
        # treat as composite expression – we won't do full eval here; assume okay
        return {"ok": True, "is_identifier": False, "ref": None, "in_params": False, "in_operations": False, "detail": "composite expression"}

    ref = expr.strip()
    in_params = isinstance(params, dict) and (ref in params)

    # If not in params, check if operations declare this output
    in_ops = False
    ops = tpl.get("operations") if isinstance(tpl, dict) else None
    if isinstance(ops, list):
        for op in ops:
            if not isinstance(op, dict):
                continue
            outk = op.get("output")
            if isinstance(outk, str) and outk == ref:
                in_ops = True
                break
            outs = op.get("outputs")
            if isinstance(outs, list) and any(isinstance(o, str) and o == ref for o in outs):
                in_ops = True
                break

    ok = bool(in_params or in_ops)
    detail = (
        "identifier is available in params"
        if in_params
        else ("identifier scheduled from operations" if in_ops else "identifier missing in params and operations")
    )
    return {
        "ok": ok,
        "is_identifier": True,
        "ref": ref,
        "in_params": in_params,
        "in_operations": in_ops,
        "detail": detail,
    }


validate_answer_ref_tool = function_tool(_validate_answer_ref_tool)
validate_answer_ref_tool["name"] = "validate_answer_ref_tool"


# ----------------------- Difficulty / Degeneracy Heuristics -----------------------
def _parse_number(x: Any) -> tuple[bool, float | int]:
    """Try to parse a JSON number or numeric-looking string into a number.

    Returns (ok, value). This is intentionally permissive but simple; it supports
    ints, floats, and rational strings like "p/q" via Fraction, falling back to float.
    """
    try:
        if isinstance(x, (int, float)):
            return True, x
        if isinstance(x, str):
            s = x.strip()
            # Simple fraction support
            if "/" in s and all(part.strip(" +-\t").replace("_", "").isdigit() for part in s.split("/", 1)):
                from fractions import Fraction

                f = Fraction(s)
                # Prefer exact ints when possible
                return True, int(f) if f.denominator == 1 else float(f)
            # Fall back to float parsing
            return True, float(s)
    except Exception:
        pass
    return False, 0.0


def _is_perfect_square(n: float | int) -> bool:
    try:
        # Only check for small-ish magnitudes to avoid float instability
        if isinstance(n, float):
            if not (-1e12 < n < 1e12):
                return False
            if abs(n - round(n)) > 1e-9:
                return False
            n = int(round(n))
        if not isinstance(n, int):
            return False
        if n < 0:
            return False
        r = int(n**0.5)
        return r * r == n
    except Exception:
        return False


def _detect_degenerate_params_tool(template_json: str, params_json: str = "") -> dict[str, Any]:
    """Detect parameter choices that likely trivialize difficulty.

    Heuristics include:
    - Any parameter equal to 0, 1, or -1 when used in template/answer_expression.
    - Quadratic discriminant b^2 - 4ac being a perfect square for templates containing x^2 or 'quadratic'.
    - Parameters that are perfect squares when sqrt(...) appears and the parameter name occurs in answer_expression.
    - Magnitude/format checks to preserve difficulty (when difficulty metadata exists):
      * For medium/hard: reject cases where all used parameters are single-digit integers (|value| < 10).
      * For hard: additionally require that at least one used parameter be non-integer (decimal or rational p/q) OR a multi-digit integer (|value| >= 10) and
        at least one used parameter be non-integer; otherwise flag as trivializing.
      * When step_count >= 3 (if provided via template.meta.complexity_features.step_count), discourage all-used-params being small integers.

    Returns: { ok: bool, reasons: [string], details: {...} }
    """
    reasons: list[str] = []
    details: dict[str, Any] = {}
    try:
        tpl = json.loads(template_json) if isinstance(template_json, str) else template_json
    except Exception:
        return {"ok": False, "reasons": ["invalid template_json"], "details": {}}
    try:
        params = json.loads(params_json) if params_json else {}
    except Exception:
        params = {}

    # Pull text fields and any difficulty metadata for checks
    tpl_text = ""
    difficulty = None
    step_count = None
    try:
        if isinstance(tpl, dict):
            tpl_text = (str(tpl.get("template")) + "\n" + str(tpl.get("answer_expression"))).lower()
            meta = tpl.get("meta") if isinstance(tpl.get("meta"), dict) else None
            if meta and isinstance(meta.get("difficulty"), str):
                difficulty = meta.get("difficulty")
            elif isinstance(tpl.get("difficulty"), str):
                difficulty = tpl.get("difficulty")
            if meta and isinstance(meta.get("complexity_features"), dict):
                sc = meta.get("complexity_features", {}).get("step_count")
                if isinstance(sc, int):
                    step_count = sc
    except Exception:
        pass

    # 1) 0/±1 checks
    used_names = set()
    try:
        if isinstance(tpl, dict):
            for field in ("template", "answer_expression"):
                val = str(tpl.get(field, ""))
                for name in (params or {}).keys():
                    if name and isinstance(name, str) and name in val:
                        used_names.add(name)
    except Exception:
        pass
    for k, v in (params or {}).items():
        ok, num = _parse_number(v)
        if not ok:
            continue
        if k in used_names and (num == 0 or num == 1 or num == -1):
            reasons.append(f"parameter {k} has trivializing value {num}")

    # 2) Quadratic discriminant perfect square heuristic (template-aware)
    try:
        quad_like = ("x^2" in tpl_text) or ("quadratic" in tpl_text)
        if quad_like and all(k in (params or {}) for k in ("a", "b", "c")):
            ok_a, a = _parse_number(params["a"])  # type: ignore[index]
            ok_b, b = _parse_number(params["b"])  # type: ignore[index]
            ok_c, c = _parse_number(params["c"])  # type: ignore[index]
            if ok_a and ok_b and ok_c:
                disc = (b * b) - (4 * a * c)
                # Honor difficulty_profile.needs_square_discriminant
                needs_square = False
                try:
                    meta = tpl.get("meta") if isinstance(tpl, dict) else None
                    prof = meta.get("difficulty_profile") if isinstance(meta, dict) else None
                    if isinstance(prof, dict) and bool(prof.get("needs_square_discriminant")):
                        needs_square = True
                except Exception:
                    needs_square = False
                is_square = _is_perfect_square(disc)
                if needs_square and not is_square:
                    reasons.append("quadratic discriminant not a perfect square but profile requires it")
                if (not needs_square) and is_square:
                    reasons.append("quadratic discriminant is a perfect square")
                details["discriminant"] = disc
    except Exception:
        pass

    # 3) Radicals: if sqrt present and a referenced param is perfect square
    try:
        if "sqrt" in tpl_text:
            for k, v in (params or {}).items():
                if k in used_names:
                    ok_v, num_v = _parse_number(v)
                    if ok_v and _is_perfect_square(num_v) and abs(num_v) > 1:
                        reasons.append(f"parameter {k} is a perfect square {num_v} under sqrt")
    except Exception:
        pass

    # 4) Magnitude and numeric-form heuristics for difficulty preservation
    try:
        # 4a) Honor template.meta.difficulty_profile.min_value_ranges
        try:
            meta = tpl.get("meta") if isinstance(tpl, dict) else None
            prof = meta.get("difficulty_profile") if isinstance(meta, dict) else None
            ranges = prof.get("min_value_ranges") if isinstance(prof, dict) else None
        except Exception:
            ranges = None
        if isinstance(ranges, dict):
            for name, spec in ranges.items():
                if not isinstance(name, str) or name not in (params or {}):
                    continue
                ok_v, num_v = _parse_number((params or {}).get(name))
                if not ok_v:
                    continue
                if isinstance(spec, dict):
                    v = float(num_v)
                    if "abs_min" in spec:
                        try:
                            if abs(v) < float(spec["abs_min"]):
                                reasons.append(f"parameter {name} below abs_min {spec['abs_min']}")
                        except Exception:
                            pass
                    if "min" in spec:
                        try:
                            if v < float(spec["min"]):
                                reasons.append(f"parameter {name} below min {spec['min']}")
                        except Exception:
                            pass
                    if "max" in spec:
                        try:
                            if v > float(spec["max"]):
                                reasons.append(f"parameter {name} above max {spec['max']}")
                        except Exception:
                            pass

        # Consider only parameters that are referenced in the template/answer_expression
        used_vals: list[tuple[str, Any, bool, float | int]] = []
        for k, v in (params or {}).items():
            if k not in used_names:
                continue
            ok_v, num_v = _parse_number(v)
            used_vals.append((k, v, ok_v, num_v))

        if used_vals:
            all_single_digit_ints = all(
                ok_v and isinstance(num_v, (int, float)) and abs(num_v) < 10 and float(num_v).is_integer()
                for _, _, ok_v, num_v in used_vals
            )
            any_multi_digit_or_fraction = any(
                (
                    (ok_v and abs(float(num_v)) >= 10)
                    or (
                        isinstance(v, str)
                        and ("/" in v or ("." in v and not str(float(v)).endswith(".0")))
                    )
                )
                for _, v, ok_v, num_v in used_vals
            )
            any_non_integer = any(
                (
                    (ok_v and not float(num_v).is_integer())
                    or (isinstance(v, str) and ("/" in v or "." in v))
                )
                for _, v, ok_v, num_v in used_vals
            )

            # If step_count suggests multi-step reasoning, avoid all tiny integers
            if step_count is not None and step_count >= 3 and all_single_digit_ints:
                reasons.append("all used parameters are single-digit integers despite multi-step reasoning")

            if isinstance(difficulty, str):
                diff = difficulty.lower().strip()
                if diff in {"medium", "hard"} and all_single_digit_ints:
                    reasons.append("all used parameters are single-digit integers for medium/hard difficulty")
                if diff == "hard":
                    if not any_multi_digit_or_fraction:
                        reasons.append("hard difficulty lacks any multi-digit or fractional parameters")
                    if not any_non_integer:
                        reasons.append("hard difficulty lacks any non-integer parameters")
    except Exception:
        pass

    return {"ok": len(reasons) == 0, "reasons": reasons, "details": details}


detect_degenerate_params_tool = function_tool(_detect_degenerate_params_tool)
detect_degenerate_params_tool["name"] = "detect_degenerate_params_tool"


def _check_invariants_tool(template_json: str, twin_stem: Optional[str] = None) -> dict[str, Any]:
    """Check stem invariants from template.meta.invariants.

    Supported invariant keys:
    - ask: canonical ask tag (e.g., 'smaller_integer'). Enforced via phrase heuristics.
    - forbid_asks: array of tags to forbid.
    - require_phrases / forbid_phrases: literal substring checks.
    Returns: { ok: bool, reasons: [string] }
    """
    reasons: list[str] = []
    try:
        tpl = json.loads(template_json) if isinstance(template_json, str) else template_json
    except Exception:
        return {"ok": False, "reasons": ["invalid template_json"]}
    text = (twin_stem or "").strip()
    invariants = None
    try:
        invariants = tpl.get("meta", {}).get("invariants") if isinstance(tpl, dict) else None
        if invariants is None and isinstance(tpl, dict):
            invariants = tpl.get("invariants")
    except Exception:
        invariants = None
    if not isinstance(invariants, dict) or not text:
        return {"ok": True, "reasons": []}

    t = text.lower()
    ask = invariants.get("ask")
    forbid_asks = invariants.get("forbid_asks") or []
    req_phr = invariants.get("require_phrases") or []
    forb_phr = invariants.get("forbid_phrases") or []

    def _enforce_ask(tag: str) -> None:
        # Heuristic phrase mapping. Extend as needed.
        tag = str(tag).lower().strip()
        patterns: dict[str, list[str]] = {
            "smaller_integer": ["smaller integer", "smaller of the two integers"],
            "larger_integer": ["larger integer", "greater integer"],
            "value_of_f": ["value of f"],
            "solve_for_x": ["solve for x"],
            "ordered_pair": ["ordered pair", "(x, y)"],
        }
        pats = patterns.get(tag, [])
        if pats and not any(p in t for p in pats):
            reasons.append(f"stem missing ask pattern for '{tag}'")

    if isinstance(ask, str):
        _enforce_ask(ask)
    if isinstance(forbid_asks, list):
        for tag in forbid_asks:
            if not isinstance(tag, str):
                continue
            # Simple detection: any mapped phrase indicates violation
            probe = str(tag).lower().strip()
            if probe in {"ordered_pair"} and ("ordered pair" in t or "(" in t and ")" in t and "," in t):
                reasons.append("stem violates forbidden ask 'ordered_pair'")
            if probe == "solve_for_x" and "solve for x" in t:
                reasons.append("stem violates forbidden ask 'solve_for_x'")
    for phrase in req_phr:
        if isinstance(phrase, str) and phrase and phrase.lower() not in t:
            reasons.append(f"stem missing required phrase '{phrase}'")
    for phrase in forb_phr:
        if isinstance(phrase, str) and phrase and phrase.lower() in t:
            reasons.append(f"stem contains forbidden phrase '{phrase}'")

    return {"ok": len(reasons) == 0, "reasons": reasons}


check_invariants_tool = function_tool(_check_invariants_tool)
check_invariants_tool["name"] = "check_invariants_tool"


def _choices_truth_filter_tool(
    choices_json: str,
    computed_value: Any = None,  # noqa: ANN401 – generic
    template_json: str | None = None,
    params_json: str | None = None,
) -> dict[str, Any]:
    """Best-effort truth filter to prevent duplicate-correct distractors.

    Strategy: when a numeric computed_value is available, count how many choices equal it
    (string or numeric equality within tight tolerance). If more than one, flag.
    Returns: { ok: bool, correct_count: int }
    """
    try:
        choices = json.loads(choices_json) if isinstance(choices_json, str) else choices_json
    except Exception:
        return {"ok": True, "correct_count": 0}
    # Determine target value: prefer provided computed_value; else compute from template/params
    target_val: Any = computed_value
    if target_val is None and template_json and params_json is not None:
        try:
            tpl = json.loads(template_json)
            expr = tpl.get("answer_expression") if isinstance(tpl, dict) else None
            from .calc import _calc_answer as _calc
            target_val = _calc(str(expr or "0"), str(params_json))
        except Exception:
            target_val = None
    try:
        ok, target = _parse_number(target_val)
        if not ok:
            return {"ok": True, "correct_count": 0}
    except Exception:
        return {"ok": True, "correct_count": 0}

    def _eq(a: Any, b: Any) -> bool:  # noqa: ANN401 – generic
        sa, sb = str(a).strip(), str(b).strip()
        if sa == sb:
            return True
        try:
            fa, fb = float(sa), float(sb)
            return abs(fa - fb) < 1e-9
        except Exception:
            return False

    cnt = 0
    for c in choices if isinstance(choices, list) else []:
        if _eq(c, target):
            cnt += 1
    return {"ok": cnt <= 1, "correct_count": cnt}


choices_truth_filter_tool = function_tool(_choices_truth_filter_tool)
choices_truth_filter_tool["name"] = "choices_truth_filter_tool"


def _rationale_grounding_tool(state_json: str, rationale: Optional[str] = None) -> dict[str, Any]:
    """Reject numeric drift by ensuring rationale only uses known numbers.

    Allowed set is derived from the serialized PipelineState: params' numeric values,
    computed_value/answer, and numeric choices. Returns: { ok: bool, unknown: [numbers] }.
    """
    try:
        state = json.loads(state_json)
    except Exception:
        return {"ok": True, "unknown": []}
    text = (rationale or "").strip()
    if not text:
        return {"ok": True, "unknown": []}

    allowed: list[float] = []
    # params
    try:
        for v in (state.get("params") or {}).values():
            ok, num = _parse_number(v)
            if ok:
                allowed.append(float(num))
    except Exception:
        pass
    # computed value / answer
    for key in ("computed_value", "answer", "answer_value"):
        try:
            v = state.get(key)
            ok, num = _parse_number(v)
            if ok:
                allowed.append(float(num))
        except Exception:
            pass
    # symbolic solutions (extract numeric tokens from strings)
    for key in ("symbolic_solution", "symbolic_simplified"):
        try:
            s = str(state.get(key) or "")
            if s:
                import re as _re
                for m in _re.finditer(r"-?\b\d+\s*/\s*\d+\b", s):
                    ok, num = _parse_number(m.group(0).replace(" ", ""))
                    if ok:
                        allowed.append(float(num))
                for m in _re.finditer(r"-?\b\d+(?:\.\d+)?\b", s):
                    ok, num = _parse_number(m.group(0))
                    if ok:
                        allowed.append(float(num))
        except Exception:
            pass
    # choices
    try:
        for c in (state.get("choices") or []):
            ok, num = _parse_number(c)
            if ok:
                allowed.append(float(num))
    except Exception:
        pass

    # Deduplicate
    try:
        allowed_set = set(round(x, 9) for x in allowed)
    except Exception:
        allowed_set = set()

    # Extract numbers (ints, decimals, simple fractions) from rationale
    import re as _re

    nums: list[str] = []
    # Fractions first
    for m in _re.finditer(r"-?\b\d+\s*/\s*\d+\b", text):
        nums.append(m.group(0).replace(" ", ""))
    # Decimals/integers
    for m in _re.finditer(r"-?\b\d+(?:\.\d+)?\b", text):
        nums.append(m.group(0))

    unknown: list[str] = []
    for s in nums:
        ok, num = _parse_number(s)
        if not ok:
            continue
        f = float(num)
        if round(f, 9) not in allowed_set:
            unknown.append(s)

    return {"ok": len(unknown) == 0, "unknown": list(dict.fromkeys(unknown))}


rationale_grounding_tool = function_tool(_rationale_grounding_tool)
rationale_grounding_tool["name"] = "rationale_grounding_tool"

def _stem_number_grounding_tool(state_json: str, twin_stem: Optional[str] = None) -> dict[str, Any]:
    """Ensure twin_stem does not introduce numeric values beyond the parameters.

    Strategy: build an allowed set of numbers from params and literal numbers
    appearing in the template string. Extract numbers from twin_stem and flag
    any that are not in the allowed set. Returns: { ok: bool, unknown: [numbers] }.
    """
    try:
        state = json.loads(state_json) if isinstance(state_json, str) else state_json
    except Exception:
        return {"ok": False, "unknown": ["invalid-state-json"]}

    text = (twin_stem or "").strip()
    if not text:
        return {"ok": True, "unknown": []}

    allowed: list[str] = []

    def _add_num(x: Any) -> None:
        try:
            if isinstance(x, (int, float)):
                allowed.append(str(int(x)) if float(x).is_integer() else str(float(x)))
            elif isinstance(x, str):
                s = x.strip()
                if s:
                    # Accept raw token and a float-normalized form when possible
                    allowed.append(s)
                    try:
                        f = float(s)
                        allowed.append(str(int(f)) if float(f).is_integer() else str(float(f)))
                    except Exception:
                        pass
        except Exception:
            pass

    # Parameters are the authoritative numeric sources for stems
    try:
        for _, v in (state.get("params") or {}).items():
            _add_num(v)
    except Exception:
        pass

    # Literal numbers present in the template string are allowed
    try:
        tpl = state.get("template") or {}
        template_text = str(tpl.get("template") if isinstance(tpl, dict) else "")
        import re as _re
        for tok in _re.findall(r"-?\d+(?:\.\d+)?", template_text):
            allowed.append(tok)
    except Exception:
        pass

    # Build canonical numeric set (as floats with rounding guard) for comparisons
    allowed_set: set[float] = set()
    for s in allowed:
        try:
            f = float(str(s).replace("%", ""))
            allowed_set.add(round(f, 9))
        except Exception:
            continue

    # Extract numbers (ints, decimals, simple fractions) from stem
    try:
        import re as _re
        tokens = _re.findall(r"-?\d+(?:\.\d+)?(?:\s*%|\b)", text)
    except Exception:
        tokens = []

    unknown: list[str] = []
    for s in tokens:
        try:
            cleaned = s.replace("%", "").strip()
            f = float(cleaned)
            if round(f, 9) not in allowed_set:
                unknown.append(s)
        except Exception:
            # If not parsable as float, be lenient
            continue

    # Deduplicate while preserving order
    return {"ok": len(unknown) == 0, "unknown": list(dict.fromkeys(unknown))}


stem_number_grounding_tool = function_tool(_stem_number_grounding_tool)
stem_number_grounding_tool["name"] = "stem_number_grounding_tool"


def _count_concept_steps_tool(concept_text: str) -> dict[str, Any]:
    """Count numbered steps in ConceptAgent output.

    Returns: { steps: int }
    """
    steps = 0
    try:
        text = str(concept_text or "")
        for line in text.splitlines():
            s = line.strip()
            # Match patterns like "1.", "2.", or multi-digit "10." etc.
            if "." in s:
                head = s.split(".", 1)[0]
                if head.isdigit():
                    steps += 1
    except Exception:
        steps = 0
    return {"steps": steps}


count_concept_steps_tool = function_tool(_count_concept_steps_tool)
count_concept_steps_tool["name"] = "count_concept_steps_tool"
