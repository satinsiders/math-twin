from __future__ import annotations

import logging
from typing import Any, Optional

from .state import MicroState
from . import agents as A
from .steps_util import _invoke
from .sym_utils import evaluate_numeric, evaluate_with_env, rewrite_relations, parse_relation_sides


logger = logging.getLogger("micro_solver.steps")


def _micro_execute_plan(state: MicroState, *, max_iters: Optional[int] = None) -> MicroState:
    def _local_qa(prev_relations: list[str], prev_idx: int) -> tuple[bool, str]:
        try:
            changed = (state.relations != prev_relations) or (state.current_step_idx != prev_idx)
            if not state.relations:
                return False, "empty-relations-after-rewrite"
            if not changed:
                return False, "no-change-after-rewrite"
            return True, "pass"
        except Exception as exc:  # pragma: no cover
            return False, f"qa-error:{exc}"

    def _maybe_eval_target() -> bool:
        try:
            target_expr = None
            if isinstance(state.canonical_repr, dict):
                target_expr = state.canonical_repr.get("target")
            if isinstance(target_expr, str) and target_expr.strip():
                ok, val = evaluate_with_env(target_expr, state.env or {})
                if ok:
                    # Record as a candidate only; defer finalization to verify step
                    state.candidate_answers.append(val)
                    return True
        except Exception:
            pass
        return False

    import re as _re

    def _promote_env_from_relations() -> None:
        try:
            for r in state.relations or []:
                op, lhs, rhs = parse_relation_sides(r)
                if op != "=":
                    continue
                name = (lhs or "").strip()
                if not _re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
                    continue
                ok, val = evaluate_with_env(rhs, state.env or {})
                if ok:
                    try:
                        state.env[name] = val
                    except Exception:
                        pass
        except Exception:
            pass

    def _stable_unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in items:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _state_digest() -> str:
        try:
            import hashlib
            rel_key = "|".join(sorted(map(str, state.relations or [])))
            env_items = ",".join(f"{k}={state.env.get(k)}" for k in sorted(state.env.keys()))
            h = hashlib.md5()
            h.update(rel_key.encode("utf-8", errors="ignore"))
            h.update(env_items.encode("utf-8", errors="ignore"))
            return h.hexdigest()
        except Exception:
            return ""

    def _target_symbols() -> set[str]:
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
            transformations = (*standard_transformations, implicit_multiplication_application)
            tgt = (state.canonical_repr or {}).get("target") if isinstance(state.canonical_repr, dict) else None
            if isinstance(tgt, str) and tgt.strip():
                expr = parse_expr(tgt, transformations=transformations)
                return {str(s) for s in getattr(expr, "free_symbols", set())}
        except Exception:
            pass
        return set()

    def _numeric_solvable_count(relations: list[str], symbols: set[str]) -> int:
        # Count how many given symbols admit a numeric solution from the current relations
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
            transformations = (*standard_transformations, implicit_multiplication_application)
            eqs: list[sp.Eq] = []
            for r in relations:
                try:
                    op, lhs, rhs = parse_relation_sides(r)
                    if op != "=":
                        continue
                    eL = parse_expr(lhs, transformations=transformations)
                    eR = parse_expr(rhs, transformations=transformations)
                    eqs.append(sp.Eq(eL, eR))
                except Exception:
                    continue
            count = 0
            for name in symbols:
                try:
                    sym = sp.Symbol(name)
                    sol = sp.solve(eqs, sym, dict=True)
                    if sol and sol[0].get(sym) is not None:
                        val = sol[0][sym]
                        if not getattr(val, "free_symbols", set()):
                            count += 1
                except Exception:
                    continue
            return count
        except Exception:
            return 0

    def _progress_metrics() -> tuple[float, int, int, int, int, int, int]:
        # Returns (score, eq_count, num_evaluable, free_sym_count, target_bound, target_unbound, num_solvable)
        score = 0.0
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
            transformations = (*standard_transformations, implicit_multiplication_application)
        except Exception:
            sp = None  # type: ignore
            transformations = None  # type: ignore
        try:
            score += 2.0 * float(len(state.env or {}))
        except Exception:
            pass
        num_evaluable = 0
        free_syms: set[str] = set()
        eq_count = 0
        # Collect all symbols from relations for solvability check
        all_syms: set[str] = set()
        for r in state.relations or []:
            try:
                op, lhs, rhs = parse_relation_sides(r)
                if op == "":
                    ok, _ = evaluate_with_env(lhs, state.env or {})
                    if not ok:
                        ok, _ = evaluate_numeric(lhs)
                    if ok:
                        num_evaluable += 1
                else:
                    okL, _L = evaluate_with_env(lhs, state.env or {})
                    okR, _R = evaluate_with_env(rhs, state.env or {})
                    if not okL:
                        okL, _L = evaluate_numeric(lhs)
                    if not okR:
                        okR, _R = evaluate_numeric(rhs)
                    if okL and okR:
                        num_evaluable += 1
                if op == "=":
                    eq_count += 1
                if sp and transformations:
                    try:
                        eL = parse_expr(lhs, transformations=transformations)
                        eR = parse_expr(rhs or "0", transformations=transformations)
                        symset = getattr(eL, "free_symbols", set()) | getattr(eR, "free_symbols", set())
                        for s in symset:
                            free_syms.add(str(s))
                            all_syms.add(str(s))
                    except Exception:
                        pass
            except Exception:
                pass
        score += 1.0 * float(num_evaluable)
        try:
            score -= 0.5 * float(len(free_syms))
        except Exception:
            pass
        # Target-aware bonuses/penalties
        tgt_syms = _target_symbols()
        bound = 0
        try:
            for t in tgt_syms:
                if t in (state.env or {}):
                    bound += 1
        except Exception:
            pass
        try:
            target_expr = (state.canonical_repr or {}).get("target") if isinstance(state.canonical_repr, dict) else None
            if isinstance(target_expr, str) and target_expr.strip():
                ok, _ = evaluate_with_env(target_expr, state.env or {})
                if ok:
                    score += 5.0
        except Exception:
            pass
        # Bonus for more equalities (more constraints)
        score += 0.3 * float(eq_count)
        # Numeric-solvable symbols (target-aware priority): try target symbols first, then all symbols
        num_solvable = 0
        try:
            if tgt_syms:
                num_solvable += _numeric_solvable_count(state.relations, tgt_syms)
            # Also include general solvability for others (light weight)
            others = set(all_syms) - set(tgt_syms)
            if others:
                num_solvable += max(0, _numeric_solvable_count(state.relations, others))
        except Exception:
            pass
        # Target binding/penalty
        score += 3.0 * float(bound)
        unbound = max(0, len(tgt_syms) - bound)
        score -= 0.7 * float(unbound)
        # Reward newly-solvable symbols
        score += 1.0 * float(num_solvable)
        return score, eq_count, num_evaluable, len(free_syms), bound, unbound, num_solvable

    iters = 0
    n = len(state.plan_steps or [])
    no_progress = 0
    last_idx = state.current_step_idx
    last_relations = list(state.relations)
    if n > 0:
        logger.info("[micro-solver] execute_plan: plan_steps completed (%d iterations)", iters)
    else:
        logger.info("[micro-solver] execute_plan: no static plan; entering atomic")

    # Atomic loop
    tries = 0
    budget = max_iters if max_iters is not None else 50
    atomic_history: list[dict[str, Any]] = []
    attempted: set[str] = set()
    last_env = dict(state.env)
    while tries < budget:
        if _maybe_eval_target():
            # Update progress in derived and log
            try:
                ps, eqc, nev, frees = _progress_metrics()
                state.derived["progress_score"] = ps
                state.derived["progress_delta"] = None
                state.derived["atomic_iters"] = tries
                state.derived["eq_count"] = eqc
                state.derived["num_evaluable"] = nev
                state.derived["free_symbols"] = frees
            except Exception:
                pass
            return state

        base_hist = [{"action": (st or {}).get("action")} for st in (state.plan_steps or [])]
        hist = base_hist + [{"action": h.get("action"), "ok": h.get("ok"), "reason": h.get("reason")} for h in atomic_history]
        ap_out, ap_err = _invoke(
            A.AtomicPlannerAgent,
            {
                "relations": state.relations,
                "goal": state.goal,
                "canonical_target": (state.canonical_repr or {}).get("target") if isinstance(state.canonical_repr, dict) else None,
                "env": state.env,
                "history": hist,
            },
            qa_feedback=state.qa_feedback,
        )
        state.qa_feedback = None
        steps_prop: list[dict[str, Any]] = []
        if not ap_err and isinstance(ap_out, dict):
            if isinstance(ap_out.get("steps"), list):
                steps_prop = [st for st in ap_out.get("steps") if isinstance(st, dict)]
            elif isinstance(ap_out.get("step"), dict):
                steps_prop = [ap_out.get("step")]  # type: ignore
        if ap_err or not steps_prop:
            state.qa_feedback = "no-atomic-step"
            logger.info("[micro-solver] iterate atomic %d: planner produced no step", tries + 1)
            no_progress += 1
            if no_progress >= 5:
                logger.info("[micro-solver] execute_plan: no further progress; exiting loop")
                break
            continue

        # Evaluate candidates by simulated progress
        base_score, base_eqc, base_nev, base_free, base_bound, base_unbound, base_solv = _progress_metrics()
        best_step = None
        best_score = -1e9
        for cand in steps_prop[:3]:
            try:
                act_c = str(cand.get("action", "")).lower()
                args_c = cand.get("args") if isinstance(cand.get("args"), dict) else {}
            except Exception:
                continue
            key = f"{act_c}|{str(args_c)}|{_state_digest()}"
            if key in attempted:
                continue
            sim_rels = rewrite_relations(state.relations, cand)
            # Compute score for simulated relations
            if sim_rels and sim_rels != state.relations:
                saved = state.relations
                try:
                    state.relations = sim_rels
                    sim_score, _, _, _, _, _, _ = _progress_metrics()
                finally:
                    state.relations = saved
            else:
                sim_score = base_score
            if sim_score > best_score:
                best_score = sim_score
                best_step = cand
        step2 = best_step or steps_prop[0]
        try:
            act = str(step2.get("action", "")).lower()
            args = step2.get("args") if isinstance(step2.get("args"), dict) else {}
        except Exception:
            act, args = "", {}
        attempted.add(f"{act}|{str(args)}|{_state_digest()}")
        logger.info("[micro-solver] iterate atomic %d: action %s", tries + 1, str(step2.get("action")))

        prev_rel = list(state.relations)
        prev_env = dict(state.env)
        prev_score = base_score

        new_rels2 = rewrite_relations(state.relations, step2)
        # Guard: disallow unjustified assigns; require numeric RHS and structural justification
        assign_invalid = False
        if "assign" in act:
            try:
                rhs_val = args.get("value")
                tgt_sym = str(args.get("target", "")).strip()
                ok_rhs = False
                num_val = None
                if isinstance(rhs_val, str):
                    ok_rhs, num_val = evaluate_with_env(rhs_val, state.env or {})
                    if not ok_rhs:
                        ok_rhs, num_val = evaluate_numeric(rhs_val)
                # Structural justification: target must appear as a side of some equality, or be uniquely solvable from relations
                justified = False
                if tgt_sym:
                    # Check for explicit side equality 'tgt = expr' or 'expr = tgt'
                    for r in state.relations or []:
                        try:
                            opx, lhsx, rhsx = parse_relation_sides(r)
                        except Exception:
                            continue
                        if opx == "=" and (lhsx.strip() == tgt_sym or rhsx.strip() == tgt_sym):
                            justified = True
                            break
                    if not justified:
                        # Try solve for target; require numeric result and (if available) equality with proposed value
                        try:
                            import sympy as sp
                            from sympy.parsing.sympy_parser import (
                                implicit_multiplication_application,
                                parse_expr,
                                standard_transformations,
                            )
                            transformations = (*standard_transformations, implicit_multiplication_application)
                            sym = sp.Symbol(tgt_sym)
                            eqs: list[sp.Eq] = []
                            for r in state.relations:
                                try:
                                    op0, l0, r0 = parse_relation_sides(r)
                                    if op0 != "=":
                                        continue
                                    eL = parse_expr(l0, transformations=transformations)
                                    eR = parse_expr(r0, transformations=transformations)
                                    eqs.append(sp.Eq(eL, eR))
                                except Exception:
                                    continue
                            if eqs:
                                sol = sp.solve(eqs, sym, dict=True)
                                if sol and sol[0].get(sym) is not None:
                                    val = sol[0][sym]
                                    if not getattr(val, "free_symbols", set()):
                                        justified = True
                                        # If proposed rhs is numeric too, require consistency
                                        if ok_rhs and num_val is not None:
                                            try:
                                                vfloat = float(val)
                                                ok_consist = abs(vfloat - float(num_val)) < 1e-9
                                                if not ok_consist:
                                                    justified = False
                                            except Exception:
                                                pass
                        except Exception:
                            pass
                if not (ok_rhs and justified):
                    assign_invalid = True
                    new_rels2 = list(state.relations)
            except Exception:
                assign_invalid = True
                new_rels2 = list(state.relations)

        # Deterministic env-based substitute across all relations
        blocked_llm_for_subst = False
        if (not new_rels2 or new_rels2 == state.relations) and ("substitute_env" in act or "substitute" in act or "subs" in act or "replace" in act):
            try:
                import sympy as sp
                from sympy.parsing.sympy_parser import (
                    implicit_multiplication_application,
                    parse_expr,
                    standard_transformations,
                )
                transformations = (*standard_transformations, implicit_multiplication_application)
                subs_map: dict[Any, Any] = {}
                for k, v in (state.env or {}).items():
                    try:
                        if isinstance(v, (int, float)):
                            subs_map[sp.Symbol(str(k))] = v
                    except Exception:
                        continue
                # Merge explicit replacements if provided (for 'substitute'); ignore for 'substitute_env'
                if "substitute_env" not in act:
                    rep = args.get("replacements") if isinstance(args.get("replacements"), dict) else {}
                    for rk, rv in rep.items():
                        try:
                            subs_map[sp.Symbol(str(rk))] = parse_expr(str(rv), transformations=transformations)
                        except Exception:
                            continue
                if subs_map:
                    temp: list[str] = []
                    for r in state.relations:
                        try:
                            op_idx = r.find("=")
                            if op_idx >= 0:
                                lhs = r[:op_idx]
                                rhs = r[op_idx+1:]
                                eL = parse_expr(lhs, transformations=transformations).subs(subs_map)
                                eR = parse_expr(rhs, transformations=transformations).subs(subs_map)
                                temp.append(f"{sp.sstr(sp.simplify(eL))} = {sp.sstr(sp.simplify(eR))}")
                            else:
                                e = parse_expr(r, transformations=transformations).subs(subs_map)
                                temp.append(sp.sstr(sp.simplify(e)))
                        except Exception:
                            temp.append(r)
                    new_rels2 = temp
                else:
                    # No substitutions applicable: block LLM rewrite for substitute to avoid relation collapse
                    blocked_llm_for_subst = True
            except Exception:
                pass

        # Deterministic bind_numeric using env if provided
        if (not new_rels2 or new_rels2 == state.relations) and ("bind_numeric" in act):
            try:
                import sympy as sp
                from sympy.parsing.sympy_parser import (
                    implicit_multiplication_application,
                    parse_expr,
                    standard_transformations,
                )
                transformations = (*standard_transformations, implicit_multiplication_application)
                tgt = str(args.get("target", "")).strip()
                expr = str(args.get("expr", "")).strip()
                if tgt and expr:
                    e = parse_expr(expr, transformations=transformations)
                    # substitute env first
                    subs_map: dict[Any, Any] = {}
                    for k, v in (state.env or {}).items():
                        if isinstance(v, (int, float)):
                            subs_map[sp.Symbol(str(k))] = v
                    if subs_map:
                        e = e.subs(subs_map)
                    e_simpl = sp.simplify(e)
                    if not getattr(e_simpl, "free_symbols", set()):
                        valf = float(e_simpl)
                        val_out = int(round(valf)) if abs(valf - round(valf)) < 1e-9 else valf
                        new_rels2 = list(state.relations) + [f"{tgt} = {val_out}"]
                        try:
                            state.env[tgt] = val_out
                        except Exception:
                            pass
            except Exception:
                pass

        if not new_rels2 or new_rels2 == state.relations:
            # Avoid LLM fallback for substitute when we had nothing to substitute
            if not (("substitute" in act or "substitute_env" in act) and blocked_llm_for_subst):
                r_out, r_err = _invoke(A.RewriteAgent, {"relations": state.relations, "step": step2})
                if not r_err and isinstance(r_out, dict):
                    try:
                        llm_rels = [str(x) for x in (r_out.get("new_relations") or [])]
                        # Safety: if LLM returns fewer relations than original for a preserving action, merge instead of replace
                        if ("substitute" in act or "substitute_env" in act or "simplify" in act or "normalize" in act) and len(llm_rels) < len(state.relations):
                            new_rels2 = list(state.relations) + llm_rels
                            from_se = True
                        else:
                            new_rels2 = llm_rels
                    except Exception:
                        new_rels2 = []

        from_se = False
        if not new_rels2 or new_rels2 == state.relations:
            se_out, se_err = _invoke(
                A.StepExecutorAgent,
                {"relations": state.relations, "step": step2},
                qa_feedback=state.qa_feedback,
            )
            state.qa_feedback = None
            if not se_err and isinstance(se_out, dict):
                try:
                    env_delta = se_out.get("env_delta") if isinstance(se_out.get("env_delta"), dict) else {}
                    if env_delta:
                        try:
                            state.env.update(env_delta)
                        except Exception:
                            pass
                    delta = None
                    if isinstance(se_out.get("new_relations_delta"), list):
                        delta = [str(x) for x in (se_out.get("new_relations_delta") or [])]
                    if delta:
                        new_rels2 = list(state.relations) + delta
                    else:
                        new_rels2 = [str(x) for x in (se_out.get("new_relations") or [])]
                    from_se = True
                except Exception:
                    new_rels2 = []

        if new_rels2:
            if from_se:
                state.relations = _stable_unique(list(state.relations) + list(new_rels2))
            else:
                state.relations = _stable_unique(new_rels2)
            _promote_env_from_relations()

        try:
            # Progress logging
            new_score, new_eqc, new_nev, new_free, new_bound, new_unbound, new_solv = _progress_metrics()
            delta = new_score - prev_score
            logger.info(
                "[micro-solver] iterate atomic %d: out relations=%d env_keys=%d score=%.2f Δ=%.2f",
                tries + 1,
                len(state.relations or []),
                len(state.env or {}),
                new_score,
                delta,
            )
            state.derived["progress_score"] = new_score
            state.derived["progress_delta"] = delta
            state.derived["eq_count"] = new_eqc
            state.derived["num_evaluable"] = new_nev
            state.derived["free_symbols"] = new_free
            state.derived["target_bound"] = new_bound
            state.derived["target_unbound"] = new_unbound
            state.derived["numeric_solvable"] = new_solv
        except Exception:
            pass

        # Local QA
        changed = (state.relations != prev_rel) or (state.env != prev_env)
        if not state.relations:
            ok2, reason2 = False, "empty-relations-after-atomic"
        elif not changed:
            ok2, reason2 = False, "no-change-after-atomic"
        else:
            # Accept if structure improved, target binding improved, or numeric-solvable increased
            improved_structure = (new_eqc > base_eqc) or (new_free < base_free)
            improved_target = (new_bound > base_bound) or (new_unbound < base_unbound)
            improved_solvable = (new_solv > base_solv)
            if "assign" in act and assign_invalid:
                ok2, reason2 = False, "assign-invalid"
            elif "assign" in act and (state.env == prev_env):
                ok2, reason2 = False, "assign-without-env-change"
            elif state.derived.get("progress_delta", 0) <= 0 and not _maybe_eval_target() and not (improved_structure or improved_target or improved_solvable):
                ok2, reason2 = False, "no-progress-score"
            else:
                ok2, reason2 = True, "pass"
        logger.info("[micro-solver] iterate atomic %d: QA: %s", tries + 1, reason2)
        if not ok2:
            state.qa_feedback = reason2
            logger.info("[micro-solver] iterate atomic %d: QA fail (%s); re-planning", tries + 1, reason2)
            try:
                atomic_history.append({"action": act, "ok": False, "reason": reason2})
            except Exception:
                pass
            no_progress += 1
            if no_progress >= 5:
                logger.info("[micro-solver] execute_plan: no further progress; exiting loop")
                break
            continue

        tries += 1
        try:
            atomic_history.append({"action": act, "ok": True, "reason": "pass"})
        except Exception:
            pass
        progressed = (state.relations != last_relations) or (state.env != last_env)
        if progressed:
            no_progress = 0
            last_relations = list(state.relations)
            last_env = dict(state.env)
        else:
            no_progress += 1
            if no_progress >= 3:
                # Attempt opportunistic binding of solvable single-symbol equations
                try:
                    import sympy as sp
                    from sympy.parsing.sympy_parser import (
                        implicit_multiplication_application,
                        parse_expr,
                        standard_transformations,
                    )
                    transformations = (*standard_transformations, implicit_multiplication_application)
                    bound_any = False
                    for r in list(state.relations):
                        try:
                            op, lhs, rhs = parse_relation_sides(r)
                            if op != "=":
                                continue
                            eL = parse_expr(lhs, transformations=transformations)
                            eR = parse_expr(rhs, transformations=transformations)
                            syms = list((getattr(eL, "free_symbols", set()) | getattr(eR, "free_symbols", set())))
                            if len(syms) == 1:
                                sym = list(syms)[0]
                                sol = sp.solve(sp.Eq(eL, eR), sym, dict=True)
                                if sol:
                                    val = sol[0].get(sym)
                                    if val is not None and not getattr(val, "free_symbols", set()):
                                        name = str(sym)
                                        state.relations = _stable_unique(list(state.relations) + [f"{name} = {sp.sstr(val)}"]) 
                                        try:
                                            vfloat = float(val)
                                            state.env[name] = int(round(vfloat)) if abs(vfloat - round(vfloat)) < 1e-9 else vfloat
                                        except Exception:
                                            pass
                                        bound_any = True
                        except Exception:
                            continue
                    if bound_any:
                        no_progress = 0
                        continue
                except Exception:
                    pass
                break

    # Final progress record
    try:
        ps, eqc, nev, frees = _progress_metrics()
        state.derived["progress_score"] = ps
        state.derived["atomic_iters"] = tries
        state.derived["eq_count"] = eqc
        state.derived["num_evaluable"] = nev
        state.derived["free_symbols"] = frees
    except Exception:
        pass
    return state
