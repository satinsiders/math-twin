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
from .plan_policy import lint_plan


@dataclass
class MicroGraph:
    steps: list[Callable[[MicroState], MicroState]]


class MicroRunner:
    def __init__(self, graph: MicroGraph, *, verbose: bool = False, qa_max_retries: int = 5) -> None:
        self.graph = graph
        self.verbose = verbose
        self.qa_max_retries = qa_max_retries
        # Structured logger similar to twin_generator
        self.logger = logging.getLogger("micro_solver.orchestrator")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _qa(self, step_name: str, before: MicroState, after: MicroState, out_obj: Any) -> tuple[bool, str]:  # noqa: ANN401 - generic
        # Deterministic prechecks for specific steps
        try:
            if step_name == "decompose":
                res = lint_plan(after.plan_steps or [])
                if not res.get("ok", True):
                    issues = res.get("issues", [])
                    reason = issues[0] if issues else "plan-policy-violation"
                    return False, reason
        except Exception:
            # If precheck fails internally, fall back to QAAgent
            pass
        try:
            payload = json.dumps({
                "step": step_name,
                "data": {
                    # Minimal view of state that's generally safe to serialize
                    "problem_text": after.problem_text,
                    "sentences": after.sentences,
                    "tokens": after.tokens,
                    "variables": after.variables,
                    "constants": after.constants,
                    "relations": after.relations,
                    "goal": after.goal,
                    "problem_type": after.problem_type,
                    "canonical_repr": after.canonical_repr,
                    "schemas": after.schemas,
                    "strategies": after.strategies,
                    "plan_steps": after.plan_steps,
                    "current_step_idx": after.current_step_idx,
                    "equations": after.equations,
                    "env": after.env,
                    "derived": after.derived,
                    "intermediate": after.intermediate,
                    "candidate_answers": after.candidate_answers,
                    "final_answer": after.final_answer,
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

    def run(self, inputs: MicroState) -> MicroState:
        state = copy.deepcopy(inputs)
        # Step-specific minimal outputs for QA
        def _build_step_out(step_name: str, before: MicroState, after: MicroState) -> dict[str, Any]:  # noqa: ANN401 - generic
            try:
                if step_name == "tokenize":
                    return {"sentences": after.sentences, "tokens": after.tokens}
                if step_name == "entities":
                    return {"variables": after.variables, "constants": after.constants, "quantities": after.quantities}
                if step_name == "relations":
                    return {"relations": after.relations}
                if step_name == "goal":
                    return {"goal": after.goal}
                if step_name == "classify":
                    return {"problem_type": after.problem_type}
                if step_name == "repr":
                    return {"canonical_repr": after.canonical_repr}
                if step_name == "schema":
                    return {"schemas": after.schemas}
                if step_name == "strategies":
                    return {"strategies": after.strategies}
                if step_name == "choose_strategy":
                    return {"chosen_strategy": after.chosen_strategy}
                if step_name == "decompose":
                    return {"plan_steps": after.plan_steps}
                if step_name == "execute_plan":
                    return {"current_step_idx": after.current_step_idx, "relations": after.relations}
                if step_name == "extract_candidate":
                    last = after.candidate_answers[-1] if after.candidate_answers else None
                    return {"candidate": last}
                if step_name == "simplify_candidate_sympy":
                    last = after.candidate_answers[-1] if after.candidate_answers else None
                    return {"candidate_simplified": last}
                if step_name in {"verify_sympy", "verify"}:
                    return {"final_answer": after.final_answer}
            except Exception:
                pass
            # Fallback: generic delta
            return {
                "relations": after.relations,
                "plan_steps": after.plan_steps,
                "final_answer": after.final_answer,
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
                    return f"normalized_len={len(after.normalized_text or '')}"
                if step_name == "tokenize":
                    return f"sentences={len(after.sentences)} tokens={len(after.tokens)}"
                if step_name == "entities":
                    return (
                        f"vars={len(after.variables)} consts={len(after.constants)} qty={len(after.quantities)}"
                    )
                if step_name == "relations":
                    head = _trunc(after.relations[0]) if after.relations else ""
                    return f"count={len(after.relations)} head='{head}'"
                if step_name == "goal":
                    return f"goal='{_trunc(after.goal)}'"
                if step_name == "classify":
                    return f"type='{_trunc(after.problem_type)}'"
                if step_name == "repr":
                    targ = None
                    try:
                        if isinstance(after.canonical_repr, dict):
                            targ = after.canonical_repr.get("target")
                    except Exception:
                        targ = None
                    return f"target='{_trunc(targ)}'"
                if step_name == "schema":
                    names = after.schemas or []
                    return f"schemas={len(names)}: {_trunc(', '.join(map(str, names[:3])))}"
                if step_name == "strategies":
                    names = after.strategies or []
                    return f"strategies={len(names)}: {_trunc(', '.join(map(str, names[:3])))}"
                if step_name == "choose_strategy":
                    return f"chosen='{_trunc(after.chosen_strategy)}'"
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
                    return f"idx={after.current_step_idx}/{len(after.plan_steps or [])} relations={len(after.relations)}"
                if step_name == "extract_candidate":
                    cand = after.candidate_answers[-1] if after.candidate_answers else None
                    return f"candidate='{_trunc(cand)}'"
                if step_name == "simplify_candidate_sympy":
                    cand = after.candidate_answers[-1] if after.candidate_answers else None
                    return f"simplified='{_trunc(cand)}'"
                if step_name in {"verify_sympy", "verify"}:
                    return f"final='{_trunc(after.final_answer)}'"
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
            if state.final_answer is not None:
                break
        if state.final_answer is not None:
            self.logger.info("[micro-solver] final solution: %s", state.final_answer)
        else:
            # Provide a more informative summary instead of a bare "(none)"
            # 1) If we have candidates, surface the last one as an unverified fallback
            fallback_msg = None
            try:
                last_cand = state.candidate_answers[-1] if state.candidate_answers else None
            except Exception:
                last_cand = None
            if last_cand is not None:
                fallback_msg = f"candidate-only (unverified): {last_cand}"
            else:
                # 2) Otherwise, show a short hint from the final relations
                try:
                    head_rel = state.relations[-1] if state.relations else None
                except Exception:
                    head_rel = None
                if head_rel:
                    fallback_msg = f"no candidate; last relation: {head_rel}"
                else:
                    fallback_msg = "no candidate; no relations"

            # Attach to state for downstream consumers
            try:
                state.final_explanation = (
                    state.final_explanation
                    or f"No final answer computed; {fallback_msg}."
                )
            except Exception:
                pass

            self.logger.info("[micro-solver] final solution: %s", fallback_msg)
        return state
