from __future__ import annotations

from typing import Any

from .state import MicroState
from . import agents as A
from .steps_util import _invoke


def _micro_normalize(state: MicroState) -> MicroState:
    try:
        s = (state.problem_text or "").replace("\u2212", "-")
        state.R["symbolic"]["normalized_text"] = s.strip()
        state.skip_qa = True
    except Exception as exc:
        state.error = f"normalize-failed:{exc}"
    return state


def _micro_tokenize(state: MicroState) -> MicroState:
    out, err = _invoke(
        A.TokenizerAgent,
        state.R["symbolic"].get("normalized_text") or state.problem_text,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"TokenizerAgent:{err}"
        return state
    try:
        state.R["symbolic"]["sentences"] = list(map(str, out.get("sentences", [])))
        tps = out.get("tokens_per_sentence")
        tok = out.get("tokens")
        sentences = state.R["symbolic"].get("sentences", [])
        tokens_per_sentence: list[list[str]] = []
        if isinstance(tps, list) and all(isinstance(row, list) for row in tps):
            tokens_per_sentence = [[str(x) for x in row] for row in tps]
        elif isinstance(tok, list) and tok and all(isinstance(row, list) for row in tok):
            tokens_per_sentence = [[str(x) for x in row] for row in tok]
        elif isinstance(tok, list) and tok and len(sentences) == 1:
            tokens_per_sentence = [[str(x) for x in tok]]
        if not tokens_per_sentence or len(tokens_per_sentence) != len(sentences):
            tokens_per_sentence = [[str(x) for x in s.split()] for s in sentences]
        flat_tokens: list[str] = [tok for row in tokens_per_sentence for tok in row]
        state.R["symbolic"]["tokens_per_sentence"] = tokens_per_sentence
        state.R["symbolic"]["tokens"] = flat_tokens
    except Exception as exc:
        state.error = f"tokenize-parse:{exc}"
    return state


def _micro_entities(state: MicroState) -> MicroState:
    # Provide raw text to the agent for context alongside tokens/sentences
    payload: dict[str, Any] = {
        "sentences": state.R["symbolic"].get("sentences", []),
        "tokens": state.R["symbolic"].get("tokens", []),
        "text": state.problem_text,
    }
    out, err = _invoke(A.EntityExtractorAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"EntityExtractorAgent:{err}"
        return state
    vs = state.V["symbolic"]
    vs["variables"] = [str(x) for x in out.get("variables", [])]
    vs["constants"] = [str(x) for x in out.get("constants", [])]
    vs["identifiers"] = [str(x) for x in out.get("identifiers", [])]
    vs["points"] = [str(x) for x in out.get("points", [])]
    vs["functions"] = [str(x) for x in out.get("functions", [])]
    vs["parameters"] = [str(x) for x in out.get("parameters", [])]
    q = out.get("quantities") or []
    # Coerce quantity values to numeric when possible; keep unit and sentence_idx
    try:
        norm_q = []
        for d in q:
            if not isinstance(d, dict):
                continue
            val = d.get("value")
            num_val = None
            if isinstance(val, (int, float)):
                num_val = val
            else:
                try:
                    s = str(val)
                    if s and s.replace(".", "", 1).lstrip("-+").isdigit():
                        num_val = float(s)
                        if abs(num_val - round(num_val)) < 1e-9:
                            num_val = int(round(num_val))
                except Exception:
                    num_val = None
            entry = {"value": (num_val if num_val is not None else str(val)), "sentence_idx": int(d.get("sentence_idx", 0))}
            if d.get("unit") is not None:
                entry["unit"] = str(d.get("unit"))
            norm_q.append(entry)
        state.V["symbolic"]["quantities"] = norm_q
    except Exception:
        state.V["symbolic"]["quantities"] = []
    # Deterministic augmentation: ensure numeric literals appear in constants/quantities
    try:
        import re as _re
        numbers: set[str] = set()
        for tok in state.R["symbolic"].get("tokens", []) or []:
            for m in _re.finditer(r"-?\d+(?:\.\d+)?", str(tok)):
                numbers.add(m.group(0))
        norm_txt = state.R["symbolic"].get("normalized_text")
        if norm_txt:
            for m in _re.finditer(r"-?\d+(?:\.\d+)?", str(norm_txt)):
                numbers.add(m.group(0))
        if numbers:
            existing = set(map(str, vs.get("constants", [])))
            for num in sorted(numbers, key=lambda s: (len(s), s)):
                if num not in existing:
                    vs.setdefault("constants", []).append(num)
            present_vals = {str(d.get("value")) for d in (vs.get("quantities", []) or []) if isinstance(d, dict)}
            for num in numbers:
                if num not in present_vals:
                    vs.setdefault("quantities", []).append({"value": num, "sentence_idx": 0})
    except Exception:
        pass
    return state


def _micro_relations(state: MicroState) -> MicroState:
    payload = {
        "sentences": state.R["symbolic"].get("sentences", []),
        "tokens": state.R["symbolic"].get("tokens", []),
        "text": state.problem_text,
    }
    out, err = _invoke(A.RelationExtractorAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"RelationExtractorAgent:{err}"
        return state
    rels = [str(x) for x in out.get("relations", [])]
    state.C["symbolic"] = rels
    return state


def _micro_goal(state: MicroState) -> MicroState:
    # Pass full problem text to improve goal inference when sentences are sparse/empty
    payload = {"sentences": state.R["symbolic"].get("sentences", []), "text": state.problem_text}
    out, err = _invoke(
        A.GoalInterpreterAgent,
        payload,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"GoalInterpreterAgent:{err}"
        return state
    state.goal = str(out.get("goal")) if out.get("goal") is not None else None
    return state


def _micro_classify(state: MicroState) -> MicroState:
    out, err = _invoke(
        A.TypeClassifierAgent,
        {"relations": state.C["symbolic"], "goal": state.goal},
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"TypeClassifierAgent:{err}"
        return state
    state.problem_type = str(out.get("problem_type")) if out.get("problem_type") is not None else None
    return state


def _micro_repr(state: MicroState) -> MicroState:
    payload = {
        "variables": state.V["symbolic"].get("variables", []),
        "constants": state.V["symbolic"].get("constants", []),
        "quantities": state.V["symbolic"].get("quantities", []),
        "relations": state.C["symbolic"],
        "goal": state.goal,
        "problem_type": state.problem_type,
    }
    out, err = _invoke(A.RepresentationAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"RepresentationAgent:{err}"
        return state
    if isinstance(out, dict):
        state.R["symbolic"]["canonical_repr"] = out
    return state
