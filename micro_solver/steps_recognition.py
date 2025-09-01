from __future__ import annotations

from typing import Any

from .state import MicroState
from . import agents as A
from .steps_util import _invoke


def _micro_normalize(state: MicroState) -> MicroState:
    try:
        s = (state.problem_text or "").replace("\u2212", "-")
        state.normalized_text = s.strip()
        state.skip_qa = True
    except Exception as exc:
        state.error = f"normalize-failed:{exc}"
    return state


def _micro_tokenize(state: MicroState) -> MicroState:
    out, err = _invoke(
        A.TokenizerAgent,
        state.normalized_text or state.problem_text,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"TokenizerAgent:{err}"
        return state
    try:
        state.sentences = list(map(str, out.get("sentences", [])))
        tps = out.get("tokens_per_sentence")
        tok = out.get("tokens")

        tokens_per_sentence: list[list[str]] = []
        if isinstance(tps, list) and all(isinstance(row, list) for row in tps):
            tokens_per_sentence = [[str(x) for x in row] for row in tps]
        elif isinstance(tok, list) and tok and all(isinstance(row, list) for row in tok):
            # Some models may put nested lists under 'tokens'
            tokens_per_sentence = [[str(x) for x in row] for row in tok]
        elif isinstance(tok, list) and tok and len(state.sentences) == 1:
            # Flat token list with a single sentence
            tokens_per_sentence = [[str(x) for x in tok]]

        # If tokens_per_sentence missing or mismatched, fall back to simple whitespace split
        if not tokens_per_sentence or len(tokens_per_sentence) != len(state.sentences):
            tokens_per_sentence = [[str(x) for x in s.split()] for s in state.sentences]

        flat_tokens: list[str] = [tok for row in tokens_per_sentence for tok in row]
        state.tokens_per_sentence = tokens_per_sentence
        state.tokens = flat_tokens
    except Exception as exc:
        state.error = f"tokenize-parse:{exc}"
    return state


def _micro_entities(state: MicroState) -> MicroState:
    # Provide raw text to the agent for context alongside tokens/sentences
    payload: dict[str, Any] = {"sentences": state.sentences, "tokens": state.tokens, "text": state.problem_text}
    out, err = _invoke(A.EntityExtractorAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"EntityExtractorAgent:{err}"
        return state
    state.variables = [str(x) for x in out.get("variables", [])]
    state.constants = [str(x) for x in out.get("constants", [])]
    state.identifiers = [str(x) for x in out.get("identifiers", [])]
    state.points = [str(x) for x in out.get("points", [])]
    state.functions = [str(x) for x in out.get("functions", [])]
    state.parameters = [str(x) for x in out.get("parameters", [])]
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
        state.quantities = norm_q
    except Exception:
        state.quantities = []
    # Deterministic augmentation: ensure numeric literals appear in constants/quantities
    try:
        import re as _re
        numbers: set[str] = set()
        for tok in state.tokens or []:
            for m in _re.finditer(r"-?\d+(?:\.\d+)?", str(tok)):
                numbers.add(m.group(0))
        if state.normalized_text:
            for m in _re.finditer(r"-?\d+(?:\.\d+)?", str(state.normalized_text)):
                numbers.add(m.group(0))
        if numbers:
            existing = set(map(str, state.constants or []))
            for num in sorted(numbers, key=lambda s: (len(s), s)):
                if num not in existing:
                    state.constants.append(num)
            present_vals = {str(d.get("value")) for d in (state.quantities or []) if isinstance(d, dict)}
            for num in numbers:
                if num not in present_vals:
                    state.quantities.append({"value": num, "sentence_idx": 0})
    except Exception:
        pass
    return state


def _micro_relations(state: MicroState) -> MicroState:
    payload = {"sentences": state.sentences, "tokens": state.tokens, "text": state.problem_text}
    out, err = _invoke(A.RelationExtractorAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"RelationExtractorAgent:{err}"
        return state
    state.relations = [str(x) for x in out.get("relations", [])]
    state.equations = list(state.relations)
    return state


def _micro_goal(state: MicroState) -> MicroState:
    # Pass full problem text to improve goal inference when sentences are sparse/empty
    payload = {"sentences": state.sentences, "text": state.problem_text}
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
        {"relations": state.relations, "goal": state.goal},
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
        "variables": state.variables,
        "constants": state.constants,
        "quantities": state.quantities,
        "relations": state.relations,
        "goal": state.goal,
        "problem_type": state.problem_type,
    }
    out, err = _invoke(A.RepresentationAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"RepresentationAgent:{err}"
        return state
    if isinstance(out, dict):
        state.canonical_repr = out
    return state
