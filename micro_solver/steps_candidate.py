from __future__ import annotations

from typing import Any, Optional, cast

from .state import MicroState
from . import agents as A
from .steps_util import _invoke
from .sym_utils import simplify_expr, verify_candidate, evaluate_numeric, evaluate_with_env, rewrite_relations, solve_for, solve_any, parse_relation_sides


def _micro_extract_candidate(state: MicroState) -> MicroState:
    expr: Optional[str] = None
    try:
        target_expr = None
        if isinstance(state.canonical_repr, dict):
            target_expr = state.canonical_repr.get("target")
        if isinstance(target_expr, str) and target_expr.strip():
            ok_t, val_t = evaluate_with_env(target_expr, state.env or {})
            if ok_t:
                # Record as candidate only; verification will finalize if justified
                state.candidate_answers.append(val_t)
                state.skip_qa = True
                return state
    except Exception:
        pass

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

    if expr is None:
        for r in reversed(state.relations):
            op, lhs, rhs = parse_relation_sides(r)
            if _re.search(r"(<=|>=|!=|=|<|>)", r):
                continue
            ok, _val = evaluate_numeric(r)
            if ok:
                expr = r.strip()
                break

    if expr is None:
        if eqs:
            expr = eqs[-1][1].strip()
        elif state.relations:
            expr = state.relations[-1].strip()

    if expr is not None:
        ok_num, val_num = evaluate_numeric(expr)
        if not ok_num:
            out, err = _invoke(
                A.CandidateSynthesizerAgent,
                {"relations": state.relations, "goal": state.goal, "problem_type": state.problem_type, "plan_steps": state.plan_steps},
                qa_feedback=state.qa_feedback,
            )
            state.qa_feedback = None
            if not err and isinstance(out, dict):
                cand = str(out.get("candidate", "")).strip()
                ok2, val2 = evaluate_numeric(cand)
                if ok2:
                    state.candidate_answers.append(val2)
                elif cand:
                    state.candidate_answers.append(cand)
            # Always skip QA for extraction; rely on verify step
            state.skip_qa = True
            return state
        # Numeric: avoid trivial 0 unless explicitly justified; otherwise try synthesis
        if isinstance(val_num, (int, float)) and float(val_num) == 0.0:
            out, err = _invoke(
                A.CandidateSynthesizerAgent,
                {"relations": state.relations, "goal": state.goal, "problem_type": state.problem_type, "plan_steps": state.plan_steps},
                qa_feedback=state.qa_feedback,
            )
            state.qa_feedback = None
            if not err and isinstance(out, dict):
                cand = str(out.get("candidate", "")).strip()
                ok2, val2 = evaluate_numeric(cand)
                if ok2 and float(val2) != 0.0:
                    state.candidate_answers.append(val2)
                    state.skip_qa = True
                    return state
                if cand and cand != "0":
                    state.candidate_answers.append(cand)
                    state.skip_qa = True
                    return state
            # As a last resort, do not emit a trivial 0 candidate
            state.skip_qa = True
            return state
        # Numeric nonzero: store candidate only
        state.candidate_answers.append(val_num)
        state.skip_qa = True
        return state

    state.skip_qa = True
    return state


def _micro_simplify_candidate_sympy(state: MicroState) -> MicroState:
    if not state.candidate_answers:
        state.skip_qa = True
        return state
    try:
        last_raw = state.candidate_answers[-1]
        last = str(last_raw)
        simp = simplify_expr(last)
        ok, val = evaluate_numeric(simp)
        if ok:
            state.candidate_answers[-1] = val
        else:
            state.candidate_answers[-1] = simp
    except Exception:
        pass
    return state


def _infer_target_var(state: MicroState) -> Optional[str]:
    try:
        if state.goal and "solve for" in state.goal.lower():
            part = state.goal.lower().split("solve for", 1)[1].strip()
            if part:
                return part.split()[0].strip(" ,.:;\n\t")
    except Exception:
        pass
    try:
        if isinstance(state.canonical_repr, dict):
            tgt = state.canonical_repr.get("target")
            if isinstance(tgt, str) and tgt.strip():
                if "=" in tgt:
                    lhs = tgt.split("=", 1)[0]
                    return lhs.strip()
                return tgt.strip().split()[0]
    except Exception:
        pass
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
    # Do not set final_answer on fallback; leave decision to verification success only
    return state


def _micro_verify_sympy(state: MicroState) -> MicroState:
    if not state.candidate_answers:
        state.skip_qa = True
        return state
    var = _infer_target_var(state)
    for cand in list(state.candidate_answers):
        s = str(cand)
        if verify_candidate(state.relations, s, varname=var):
            ok, val = evaluate_numeric(s)
            state.final_answer = (val if ok else s)
            return state
    return _micro_verify(state)


def _micro_solve_sympy(state: MicroState) -> MicroState:
    # Only attempt solving when the system appears determined
    if state.eq_count == 0 or state.degrees_of_freedom != 0:
        state.skip_qa = True
        return state
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
