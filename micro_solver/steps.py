from __future__ import annotations

import json
from typing import Any, Optional, Tuple, cast
import logging

from agents.run import Runner as AgentsRunner  # type: ignore

from .state import MicroState
from . import agents as A
from .sym_utils import simplify_expr, verify_candidate, evaluate_numeric, rewrite_relations, solve_for, solve_any, parse_relation_sides


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
            raw = json.dumps(data)
        res = AgentsRunner.run_sync(agent, input=raw, tools=tools)
        out = cast(str, getattr(res, "final_output", ""))
        out = out.strip()
        if expect_json:
            return _as_json(out), None
        return out, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


# ----------------------------- Recognition -----------------------------
def _micro_normalize(state: MicroState) -> MicroState:
    # Light local normalization to reduce work for the Tokenizer agent
    try:
        s = (state.problem_text or "").replace("\u2212", "-")  # minus sign
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
        state.tokens = list(map(str, out.get("tokens", [])))
    except Exception as exc:
        state.error = f"tokenize-parse:{exc}"
    return state


def _micro_entities(state: MicroState) -> MicroState:
    payload = {"sentences": state.sentences, "tokens": state.tokens}
    out, err = _invoke(A.EntityExtractorAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"EntityExtractorAgent:{err}"
        return state
    state.variables = [str(x) for x in out.get("variables", [])]
    state.constants = [str(x) for x in out.get("constants", [])]
    q = out.get("quantities") or []
    try:
        state.quantities = [
            {"value": str(d.get("value")), **({"unit": str(d.get("unit"))} if d.get("unit") is not None else {}), "sentence_idx": int(d.get("sentence_idx", 0))}
            for d in q if isinstance(d, dict)
        ]
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
        # Insert into constants if missing
        if numbers:
            existing = set(map(str, state.constants or []))
            for num in sorted(numbers, key=lambda s: (len(s), s)):
                if num not in existing:
                    state.constants.append(num)
            # Also add bare numeric quantities when not present
            present_vals = {str(d.get("value")) for d in (state.quantities or []) if isinstance(d, dict)}
            for num in numbers:
                if num not in present_vals:
                    state.quantities.append({"value": num, "sentence_idx": 0})
    except Exception:
        pass
    return state


def _micro_relations(state: MicroState) -> MicroState:
    payload = {"sentences": state.sentences, "tokens": state.tokens}
    out, err = _invoke(A.RelationExtractorAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"RelationExtractorAgent:{err}"
        return state
    state.relations = [str(x) for x in out.get("relations", [])]
    state.equations = list(state.relations)
    return state


def _micro_goal(state: MicroState) -> MicroState:
    out, err = _invoke(
        A.GoalInterpreterAgent,
        {"sentences": state.sentences},
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


# ----------------------------- Reasoning ------------------------------
def _micro_schema(state: MicroState) -> MicroState:
    payload = {
        "type": state.problem_type,
        "relations": state.relations,
        "target": state.goal,
    }
    out, err = _invoke(A.SchemaRetrieverAgent, payload, qa_feedback=state.qa_feedback)
    state.qa_feedback = None
    if err:
        state.error = f"SchemaRetrieverAgent:{err}"
        return state
    state.schemas = [str(x) for x in out.get("schemas", [])]
    return state


def _micro_strategies(state: MicroState) -> MicroState:
    out, err = _invoke(
        A.StrategyEnumeratorAgent,
        {"schemas": state.schemas, "relations": state.relations, "target": state.goal},
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"StrategyEnumeratorAgent:{err}"
        return state
    state.strategies = [str(x) for x in out.get("strategies", [])]
    return state


def _micro_choose_strategy(state: MicroState) -> MicroState:
    # Probe each strategy with a precondition check and pick the first ok
    for s in state.strategies or []:
        out, err = _invoke(
            A.PreconditionCheckerAgent,
            {"strategy": s, "relations": state.relations},
            qa_feedback=state.qa_feedback,
        )
        if err:
            continue
        if bool(out.get("ok", False)):
            state.chosen_strategy = s
            break
    if not state.chosen_strategy:
        # Fallback: pick the first if available
        if state.strategies:
            state.chosen_strategy = state.strategies[0]
        else:
            state.error = "no-strategy"
    return state


def _micro_decompose(state: MicroState) -> MicroState:
    out, err = _invoke(
        A.StepDecomposerAgent,
        {"strategy": state.chosen_strategy, "relations": state.relations, "target": state.goal},
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"StepDecomposerAgent:{err}"
        return state
    steps = [d for d in (out.get("plan_steps") or []) if isinstance(d, dict)]
    state.plan_steps = steps
    state.current_step_idx = 0
    # Plan summary logs
    try:
        n = len(steps)
        if n:
            logger.info("[micro-solver] plan: %d step(s)", n)
            pref_keys = (
                "var",
                "variable",
                "by",
                "to",
                "term",
                "factor",
                "lhs",
                "rhs",
                "expr",
                "value",
                "side",
            )

            def _short(v: Any) -> str:  # noqa: ANN401 - generic
                try:
                    if isinstance(v, (int, float)):
                        return str(v)
                    s = str(v)
                    return s if len(s) <= 24 else s[:21] + "…"
                except Exception:
                    return "?"

            for i, s in enumerate(steps, 1):
                action = s.get("action")
                sid = s.get("id")
                args = s.get("args") if isinstance(s.get("args"), dict) else {}
                arg_items: list[tuple[str, str]] = []
                try:
                    # Prefer well-known keys, then fill with first few others
                    seen: set[str] = set()
                    for k in pref_keys:
                        if k in args and len(arg_items) < 3:
                            arg_items.append((k, _short(args[k])))
                            seen.add(k)
                    if len(arg_items) < 3:
                        for k in args:
                            if k in seen:
                                continue
                            arg_items.append((k, _short(args[k])))
                            if len(arg_items) >= 3:
                                break
                except Exception:
                    arg_items = []

                arg_str = ""
                if arg_items:
                    arg_str = " [" + ", ".join(f"{k}={v}" for k, v in arg_items) + "]"
                id_str = f" ({sid})" if sid else ""
                logger.info("[micro-solver] plan %d/%d: %s%s%s", i, n, str(action), id_str, arg_str)
    except Exception:
        pass
    return state


def _micro_next_action(state: MicroState) -> MicroState:
    out, err = _invoke(
        A.NextActionAgent,
        {"plan_steps": state.plan_steps, "current_idx": state.current_step_idx},
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"NextActionAgent:{err}"
        return state
    # Store the next step transiently in derived
    state.derived["next_step"] = out.get("next_step")
    return state


# ----------------------------- Calculation ---------------------------
def _micro_rewrite(state: MicroState) -> MicroState:
    step = state.derived.get("next_step") if isinstance(state.derived, dict) else None
    if not isinstance(step, dict):
        state.error = "missing-next-step"
        return state
    out, err = _invoke(
        A.RewriteAgent,
        {"relations": state.relations, "step": step},
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = f"RewriteAgent:{err}"
        return state
    new_rels = [str(x) for x in out.get("new_relations", [])]
    if new_rels:
        state.relations = new_rels
        state.equations = list(new_rels)
        state.intermediate.append({"op": step.get("action"), "out": new_rels})
    # advance pointer
    state.current_step_idx = min(state.current_step_idx + 1, max(0, len(state.plan_steps) - 1))
    return state


def _micro_extract_candidate(state: MicroState) -> MicroState:
    """Language-agnostic extraction of a candidate result from relations.

    Strategy (no alias or specific LHS names):
    1) From equalities (scanned from the end): pick the first side (RHS preferred) that
       is fully numeric-evaluable. If found, use that expression.
    2) If none, pick the last numeric-evaluable standalone expression.
    3) If still none, pick the RHS of the last equality; if no equality exists, pick
       the last relation as-is.
    """
    expr: Optional[str] = None

    # 1) Prefer numeric-evaluable sides of equalities, scanning from last to first
    # Only collect genuine equalities (avoid treating bare expressions as '= 0')
    import re as _re
    eqs: list[tuple[str, str]] = []
    for r in state.relations:
        op, lhs, rhs = parse_relation_sides(r)
        if op == "=" and _re.search(r"=", r):
            eqs.append((lhs, rhs))

    for lhs, rhs in reversed(eqs):
        ok_r, _val_r = evaluate_numeric(rhs)
        if ok_r:
            expr = rhs.strip()
            break
        ok_l, _val_l = evaluate_numeric(lhs)
        if ok_l:
            expr = lhs.strip()
            break

    # 2) Standalone numeric-evaluable expressions (no '=')
    if expr is None:
        for r in reversed(state.relations):
            op, lhs, rhs = parse_relation_sides(r)
            # Skip relations that actually contain any comparator
            if _re.search(r"(<=|>=|!=|=|<|>)", r):
                continue
            ok, _val = evaluate_numeric(r)
            if ok:
                expr = r.strip()
                break

    # 3) Fallback to structural heuristics: last equality RHS, else last relation
    if expr is None:
        if eqs:
            expr = eqs[-1][1].strip()
        elif state.relations:
            expr = state.relations[-1].strip()

    if expr is not None:
        state.candidate_answers.append(expr)
    else:
        state.skip_qa = True
    return state


def _micro_verify(state: MicroState) -> MicroState:
    for cand in state.candidate_answers:
        out, err = _invoke(
            A.VerifyAgent,
            {"relations": state.relations, "candidate": cand, "goal": state.goal, "problem_type": state.problem_type},
            qa_feedback=state.qa_feedback,
        )
        if err:
            continue
        if bool(out.get("ok", False)):
            state.final_answer = cand
            break
    if state.final_answer is None and state.candidate_answers:
        # fallback: take the last candidate without QA approval
        state.final_answer = state.candidate_answers[-1]
    return state


logger = logging.getLogger("micro_solver.steps")


def _micro_execute_plan(state: MicroState, *, max_iters: Optional[int] = None) -> MicroState:
    """Iteratively execute next_action → rewrite up to ``max_iters`` or plan completion.

    Runs a local micro-QA after each rewrite to maintain atomic correctness,
    mirroring the orchestrator's QA approach but scoped to this loop.
    """
    # local QA helper (deterministic, avoids pedantic LLM checks for atomic rewrites)
    def _local_qa(prev_relations: list[str], prev_idx: int) -> tuple[bool, str]:
        try:
            changed = (state.relations != prev_relations) or (state.current_step_idx != prev_idx)
            if not state.relations:
                return False, "empty-relations-after-rewrite"
            if not changed:
                return False, "no-change-after-rewrite"
            return True, "pass"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"qa-error:{exc}"

    iters = 0
    n = len(state.plan_steps or [])
    # If no plan, do nothing
    if n == 0:
        state.skip_qa = True
        return state
    # Track progress to avoid infinite loops if next_action stagnates
    no_progress = 0
    last_idx = state.current_step_idx
    last_relations = list(state.relations)
    # Execute until plan is exhausted (no hard iteration budget by default)
    while state.current_step_idx < n and (max_iters is None or iters < max_iters):
        # Next action
        out, err = _invoke(
            A.NextActionAgent,
            {"plan_steps": state.plan_steps, "current_idx": state.current_step_idx},
            qa_feedback=state.qa_feedback,
        )
        if err:
            state.error = f"NextActionAgent:{err}"
            return state
        step = out.get("next_step")
        if not isinstance(step, dict):
            break
        logger.info("[micro-solver] iterate %d: next action %s", iters + 1, str(step.get("action")))
        # Rewrite (deterministic first; fallback to agent)
        prev_relations = list(state.relations)
        prev_idx = state.current_step_idx
        # Try deterministic rewrite for common actions
        new_rels = rewrite_relations(state.relations, step)
        if new_rels == state.relations:
            # Not handled deterministically; fall back to agent
            r_out, r_err = _invoke(A.RewriteAgent, {"relations": state.relations, "step": step})
            if r_err:
                state.error = f"RewriteAgent:{r_err}"
                return state
            new_rels = [str(x) for x in (r_out.get("new_relations") or [])]
        if new_rels:
            state.relations = new_rels
            state.equations = list(new_rels)
            state.intermediate.append({"op": step.get("action"), "out": new_rels})
        # Advance to the next plan step; allow reaching n to end the loop cleanly
        state.current_step_idx = state.current_step_idx + 1

        ok, reason = _local_qa(prev_relations, prev_idx)
        logger.info("[micro-solver] iterate %d: rewrite QA: %s", iters + 1, reason or ("pass" if ok else ""))
        if not ok:
            state.error = f"QA failed in iterate: {reason}"
            return state
        iters += 1
        # Progress detection
        progressed = (state.current_step_idx != last_idx) or (state.relations != last_relations)
        if progressed:
            no_progress = 0
            last_idx = state.current_step_idx
            last_relations = list(state.relations)
        else:
            no_progress += 1
            if no_progress >= 3:
                logger.info("[micro-solver] execute_plan: no further progress; exiting loop")
                break

    logger.info("[micro-solver] execute_plan: completed (%d iterations)", iters)
    return state


def _micro_simplify_candidate_sympy(state: MicroState) -> MicroState:
    """Simplify the last candidate with SymPy for deterministic form."""
    if not state.candidate_answers:
        state.skip_qa = True
        return state
    try:
        last = str(state.candidate_answers[-1])
        simp = simplify_expr(last)
        state.candidate_answers[-1] = simp
        ok, val = evaluate_numeric(simp)
        if ok:
            state.final_answer = val
    except Exception:
        pass
    return state


def _micro_verify_sympy(state: MicroState) -> MicroState:
    """Verify the candidate using SymPy substitution when possible; fallback to LLM verify."""
    if not state.candidate_answers:
        state.skip_qa = True
        return state
    # Determine target variable
    var = _infer_target_var(state)
    for cand in list(state.candidate_answers):
        s = str(cand)
        if verify_candidate(state.relations, s, varname=var):
            ok, val = evaluate_numeric(s)
            state.final_answer = (val if ok else s)
            return state
    # Fallback to LLM verify
    return _micro_verify(state)


# Convenience top‑level graph for a simple end‑to‑end solve pass
DEFAULT_MICRO_STEPS = [
    _micro_normalize,
    _micro_tokenize,
    _micro_entities,
    _micro_relations,
    _micro_goal,
    _micro_classify,
    _micro_repr,
    _micro_schema,
    _micro_strategies,
    _micro_choose_strategy,
    _micro_decompose,
    _micro_execute_plan,
    # Try a direct SymPy solve for the inferred target before heuristic extraction
    # to catch cases where the plan did not isolate the target explicitly.
    # (Non-blocking: if nothing is found, continue.)
    # This is a deterministic step and will set a candidate if successful.
    #
    # Implemented as an inline wrapper to access current state cleanly.
]


def _micro_solve_sympy(state: MicroState) -> MicroState:
    """Attempt a deterministic solve.

    - If a target can be inferred, solve for that target.
    - Otherwise, try solving for any symbol(s) and capture any fully determined value.
    """
    target = _infer_target_var(state)
    sols: list[str] = []
    if target:
        sols = solve_for(state.relations, target)
    if not sols:
        sols = solve_any(state.relations)
    if sols:
        state.candidate_answers.append(str(sols[-1]))
    else:
        state.skip_qa = True
    return state


DEFAULT_MICRO_STEPS = [
    _micro_normalize,
    _micro_tokenize,
    _micro_entities,
    _micro_relations,
    _micro_goal,
    _micro_classify,
    _micro_repr,
    _micro_schema,
    _micro_strategies,
    _micro_choose_strategy,
    _micro_decompose,
    _micro_execute_plan,
    _micro_solve_sympy,
    _micro_extract_candidate,
    _micro_simplify_candidate_sympy,
    _micro_verify_sympy,
]


def build_steps(*, max_iters: Optional[int] = None) -> list:
    """Return the default micro-steps with a configurable execute-plan budget.

    The execute-plan step is wrapped to carry the desired max iteration count
    while preserving a stable step name for QA messages.
    """
    def _exec(state: MicroState) -> MicroState:
        return _micro_execute_plan(state, max_iters=max_iters)

    # Preserve recognizable name for QA/logging
    _exec.__name__ = _micro_execute_plan.__name__

    return [
        _micro_normalize,
        _micro_tokenize,
        _micro_entities,
        _micro_relations,
        _micro_goal,
        _micro_classify,
        _micro_repr,
        _micro_schema,
        _micro_strategies,
        _micro_choose_strategy,
        _micro_decompose,
        _exec,
        _micro_solve_sympy,
        _micro_extract_candidate,
        _micro_simplify_candidate_sympy,
        _micro_verify_sympy,
    ]
# ----------------------------- Target inference -----------------------
def _infer_target_var(state: MicroState) -> Optional[str]:
    """Infer the target variable/token using minimal assumptions.

    This function now only handles the clear 'solve for X' phrasing or explicit
    canonical targets. It avoids alias lists so the pipeline remains
    language-agnostic. When it cannot infer a target, later steps rely on
    structure/numeric evaluation rather than names.
    """
    # 1) From goal like "solve for x"
    try:
        if state.goal and "solve for" in state.goal.lower():
            part = state.goal.lower().split("solve for", 1)[1].strip()
            if part:
                return part.split()[0].strip(" ,.:;\n\t")
    except Exception:
        pass
    # 2) From canonical representation target
    try:
        if isinstance(state.canonical_repr, dict):
            tgt = state.canonical_repr.get("target")
            if isinstance(tgt, str) and tgt.strip():
                if "=" in tgt:
                    lhs = tgt.split("=", 1)[0]
                    return lhs.strip()
                # fallback: first token (language-agnostic)
                return tgt.strip().split()[0]
    except Exception:
        pass
    # 3) From plan step args (e.g., args.target = "…")
    try:
        for s in state.plan_steps or []:
            if not isinstance(s, dict):
                continue
            args = s.get("args") if isinstance(s.get("args"), dict) else None
            if not isinstance(args, dict):
                continue
            tgt = args.get("target")
            if isinstance(tgt, str) and tgt.strip():
                if "=" in tgt:
                    lhs = tgt.split("=", 1)[0]
                    return lhs.strip()
                return tgt.strip().split()[0]
    except Exception:
        pass
    return None
