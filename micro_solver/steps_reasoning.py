from __future__ import annotations

from .state import MicroState
from . import agents as A
from .steps_util import _invoke


def _micro_schema(state: MicroState) -> MicroState:
    targets = state.goal if isinstance(state.goal, list) else [state.goal]
    schemas: list[str] = []
    for t in targets:
        payload = {
            "type": state.problem_type,
            "relations": state.C["symbolic"],
            "target": t,
        }
        out, err = _invoke(A.SchemaRetrieverAgent, payload, qa_feedback=state.qa_feedback)
        state.qa_feedback = None
        if err:
            state.error = f"SchemaRetrieverAgent:{err}"
            return state
        schemas.extend(str(x) for x in out.get("schemas", []))
    state.schemas = schemas
    return state


def _micro_strategies(state: MicroState) -> MicroState:
    targets = state.goal if isinstance(state.goal, list) else [state.goal]
    strategies: list[str] = []
    for t in targets:
        out, err = _invoke(
            A.StrategyEnumeratorAgent,
            {"schemas": state.schemas, "relations": state.C["symbolic"], "target": t},
            qa_feedback=state.qa_feedback,
        )
        state.qa_feedback = None
        if err:
            state.error = f"StrategyEnumeratorAgent:{err}"
            return state
        strategies.extend(str(x) for x in out.get("strategies", []))
    state.strategies = strategies
    return state


def _micro_choose_strategy(state: MicroState) -> MicroState:
    for s in state.strategies or []:
        out, err = _invoke(
            A.PreconditionCheckerAgent,
            {"strategy": s, "relations": state.C["symbolic"]},
            qa_feedback=state.qa_feedback,
        )
        if err:
            continue
        if bool(out.get("ok", False)):
            state.chosen_strategy = s
            break
    if not state.chosen_strategy:
        if state.strategies:
            state.chosen_strategy = state.strategies[0]
        else:
            state.error = "no-strategy"
    return state

