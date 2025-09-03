from __future__ import annotations

"""Sequential micro‑orchestrator for the micro‑solver.

Each step is a callable ``(MicroState) -> MicroState``. After every step the
orchestrator runs a light micro‑QA using ``MicroQAAgent`` to enforce minimal
post‑conditions, then proceeds or retries based on policy (default: no retry,
surface concise error). This keeps per‑agent load tiny while maintaining a
clean trace.
"""

import copy
import json
from dataclasses import dataclass
import logging
from typing import Any, Callable

from agents.run import Runner as AgentsRunner  # type: ignore

from .state import MicroState
from .agents import MicroQAAgent
from .certificate import build_certificate
from . import scheduler
from .plan_policy import lint_plan as lint_plan_steps


@dataclass
class MicroGraph:
    steps: list[Callable[[MicroState], MicroState]]


class MicroRunner:
    def __init__(
        self, graph: MicroGraph, *, verbose: bool = False, qa_max_retries: int = 5
    ) -> None:
        self.graph = graph
        self.verbose = verbose
        self.qa_max_retries = qa_max_retries
        # Structured logger similar to twin_generator
        self.logger = logging.getLogger("micro_solver.orchestrator")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _qa(
        self, step_name: str, before: MicroState, after: MicroState, out_obj: Any
    ) -> tuple[bool, str]:  # noqa: ANN401 - generic
        # (Legacy static-plan prechecks removed; dynamic atomic planning is used.)
        try:
            payload = json.dumps({
                "step": step_name,
                "data": {
                    # Minimal view of state that's generally safe to serialize
                    "problem_text": after.problem_text,
                    "sentences": after.R["symbolic"].get("sentences"),
                    "tokens": after.R["symbolic"].get("tokens"),
                    "tokens_per_sentence": after.R["symbolic"].get("tokens_per_sentence"),
                    "variables": after.V["symbolic"].get("variables"),
                    "constants": after.V["symbolic"].get("constants"),
                    "quantities": after.V["symbolic"].get("quantities"),
                    "relations": after.C["symbolic"],
                    "goal": after.goal,
                    "problem_type": after.problem_type,
                    "canonical_repr": after.R["symbolic"].get("canonical_repr"),
                    "schemas": after.schemas,
                    "strategies": after.strategies,
                    "plan_steps": after.plan_steps,
                    "current_step_idx": after.current_step_idx,
                    "equations": after.C["symbolic"],
                    "env": after.V["symbolic"].get("env"),
                    "derived": after.V["symbolic"].get("derived"),
                    "intermediate": after.A["symbolic"].get("intermediate"),
                    "candidate_answers": after.A["symbolic"].get("candidates"),
                    "final_answer": after.A["symbolic"].get("final"),
                },
                "out": out_obj,
            })
        except Exception as exc:
            return False, f"micro-qa:serialization-failed:{exc}"

        try:
            resp = AgentsRunner.run_sync(MicroQAAgent, input=payload)
            out_text = str(getattr(resp, "final_output", "")).strip()
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"micro-qa:error:{exc}"

        lower = out_text.lower()
        if lower == "pass" or lower.startswith("pass") or lower == "":
            return True, "pass"
        return False, out_text or "micro-qa:unknown-failure"

    def run(self, inputs: MicroState, *, lint_plan: bool = True) -> MicroState:
        state = copy.deepcopy(inputs)
        if lint_plan and state.plan_steps:
            lint_res = lint_plan_steps(state.plan_steps)
            if not lint_res.get("ok", False):
                issues = ", ".join(lint_res.get("issues", []))
                err = f"plan-policy-violations:{issues}"
                self.logger.error(err)
                state.error = err
                raise RuntimeError(err)
        # Step-specific minimal outputs for QA

        def _build_step_out(
            step_name: str, before: MicroState, after: MicroState
        ) -> dict[str, Any]:  # noqa: ANN401 - generic
            try:
                if step_name == "tokenize":
                    return {
                        "sentences": after.R["symbolic"].get("sentences"),
                        "tokens": after.R["symbolic"].get("tokens"),
                        "tokens_per_sentence": after.R["symbolic"].get("tokens_per_sentence"),
                    }
                if step_name == "entities":
                    return {
                        "variables": after.V["symbolic"].get("variables"),
                        "constants": after.V["symbolic"].get("constants"),
                        "quantities": after.V["symbolic"].get("quantities"),
                    }
                if step_name == "relations":
                    return {"relations": after.C["symbolic"]}
                if step_name == "goal":
                    return {"goal": after.goal}
                if step_name == "classify":
                    return {"problem_type": after.problem_type}
                if step_name == "repr":
                    return {"canonical_repr": after.R["symbolic"].get("canonical_repr")}
                if step_name == "schema":
                    return {"schemas": after.schemas}
                if step_name == "strategies":
                    return {"strategies": after.strategies}
                if step_name == "decompose":
                    return {"plan_steps": after.plan_steps}
                if step_name == "execute_plan":
                    return {
                        "relations": after.C["symbolic"],
                        "progress_score": after.M.get("progress_score"),
                        "degrees_of_freedom": after.M.get("degrees_of_freedom"),
                        "final_answer": after.A["symbolic"].get("final"),
                    }
            except Exception:
                pass
            # Fallback: generic delta
            return {
                "relations": after.C["symbolic"],
                "plan_steps": after.plan_steps,
                "final_answer": after.A["symbolic"].get("final"),
            }

        # Quick human-readable summary per step (verbose logging)
        def _summarize(step_name: str, before: MicroState, after: MicroState) -> str:
            def _trunc(s: Any, n: int = 64) -> str:
                try:
                    t = str(s)
                except Exception:
                    t = "?"
                return t if len(t) <= n else t[: n - 1] + "…"

            try:
                if step_name == "normalize":
                    return f"normalized_len={len(after.R['symbolic'].get('normalized_text') or '')}"
                if step_name == "tokenize":
                    return (
                        f"sentences={len(after.R['symbolic'].get('sentences') or [])} "
                        f"tokens={len(after.R['symbolic'].get('tokens') or [])}"
                    )
                if step_name == "entities":
                    return (
                        f"vars={len(after.V['symbolic'].get('variables') or [])} "
                        f"consts={len(after.V['symbolic'].get('constants') or [])} "
                        f"qty={len(after.V['symbolic'].get('quantities') or [])}"
                    )
                if step_name == "relations":
                    head = _trunc(after.C["symbolic"][0]) if after.C["symbolic"] else ""
                    return f"count={len(after.C["symbolic"])} head='{head}'"
                if step_name == "goal":
                    return f"goal='{_trunc(after.goal)}'"
                if step_name == "classify":
                    return f"type='{_trunc(after.problem_type)}'"
                if step_name == "repr":
                    targ = None
                    try:
                        cr = after.R["symbolic"].get("canonical_repr")
                        if isinstance(cr, dict):
                            targ = cr.get("target")
                    except Exception:
                        targ = None
                    return f"target='{_trunc(targ)}'"
                if step_name == "schema":
                    names = after.schemas or []
                    return f"schemas={len(names)}: {_trunc(', '.join(map(str, names[:3])))}"
                if step_name == "strategies":
                    names = after.strategies or []
                    return f"strategies={len(names)}: {_trunc(', '.join(map(str, names[:3])))}"
                if step_name == "decompose":
                    steps = after.plan_steps or []
                    acts = []
                    for st in steps[:3]:
                        try:
                            acts.append(str(st.get('action')))
                        except Exception:
                            pass
                    return f"plan_steps={len(steps)}: {_trunc(', '.join(acts))}"
                if step_name == "execute_plan":
                    base = f"relations={len(after.C['symbolic'])}"
                    tail = ""
                    try:
                        tail += f" dof={after.M.get('degrees_of_freedom')}"
                    except Exception:
                        pass
                    try:
                        tail += f" score={after.M.get('progress_score')}"
                    except Exception:
                        pass
                    try:
                        if after.A["symbolic"].get("final") is not None:
                            tail += f" final='{_trunc(after.A['symbolic'].get('final'))}'"
                    except Exception:
                        pass
                    return base + tail
            except Exception:
                return ""
            return ""

        for step in self.graph.steps:
            name = step.__name__.replace("_micro_", "").lstrip("_")
            idx = self.graph.steps.index(step)
            total = len(self.graph.steps)
            attempts = 0
            while True:
                self.logger.info(
                    "[micro-solver] step %d/%d: %s attempt %d",
                    idx + 1,
                    total,
                    name,
                    attempts + 1,
                )
                before = copy.deepcopy(state)
                if name == "execute_plan":
                    state = scheduler.solve_with_defaults(state)
                else:
                    state = step(state)
                # Emit a quick, human-readable summary for visibility
                try:
                    summary = _summarize(name, before, state)
                    if summary:
                        self.logger.info(
                            "[micro-solver] step %d/%d: %s ▸ %s",
                            idx + 1,
                            total,
                            name,
                            summary,
                        )
                except Exception:
                    pass
                if state.error:
                    # Treat agent/step errors as retryable up to qa_max_retries
                    err_reason = str(state.error)
                    attempts += 1
                    if attempts >= self.qa_max_retries:
                        # Exhausted retries; surface the error and stop
                        return state
                    # Log and retry with feedback
                    try:
                        self.logger.info(
                            "[micro-solver] step %d/%d: %s error (attempt %d): %s",
                            idx + 1,
                            total,
                            name,
                            attempts,
                            err_reason,
                        )
                    except Exception:
                        pass
                    before.qa_feedback = f"error:{err_reason}"
                    state = before
                    continue
                if state.skip_qa:
                    state.skip_qa = False
                    break
                out_obj = _build_step_out(name, before, state)
                ok, reason = self._qa(name, before, state, out_obj)
                self.logger.info(
                    "[micro-solver] step %d/%d: %s QA (attempt %d): %s",
                    idx + 1,
                    total,
                    name,
                    attempts + 1,
                    reason or ("pass" if ok else ""),
                )
                if ok:
                    state.qa_feedback = None
                    break
                attempts += 1
                if attempts >= self.qa_max_retries:
                    state.error = f"QA failed for {name}: {reason}"
                    return state
                # Revert and attach feedback for retry
                before.qa_feedback = reason
                state = before
            # Early exit if final solution is available
            if state.A["symbolic"].get("final") is not None:
                break
        if state.A["symbolic"].get("final") is not None:
            self.logger.info("[micro-solver] final solution: %s", state.A["symbolic"].get("final"))
        else:
            # Provide a more informative summary instead of a bare "(none)"
            # 1) If we have candidates, surface the last one as an unverified fallback
            fallback_msg = None
            try:
                last_cand = (
                    state.A["symbolic"]["candidates"][-1]
                    if state.A["symbolic"]["candidates"]
                    else None
                )
            except Exception:
                last_cand = None
            if last_cand is not None:
                fallback_msg = f"candidate-only (unverified): {last_cand}"
            else:
                # 2) Otherwise, show a short hint from the final relations
                try:
                    head_rel = state.C["symbolic"][-1] if state.C["symbolic"] else None
                except Exception:
                    head_rel = None
                if head_rel:
                    fallback_msg = f"no candidate; last relation: {head_rel}"
                else:
                    fallback_msg = "no candidate; no relations"

            # Attach to state for downstream consumers
            try:
                state.A["symbolic"]["explanation"] = (
                    state.A["symbolic"].get("explanation")
                    or f"No final answer computed; {fallback_msg}."
                )
            except Exception:
                pass

            self.logger.info("[micro-solver] final solution: %s", fallback_msg)
        try:
            state.A["symbolic"]["certificate"] = build_certificate(state)
        except Exception:
            pass
        return state
