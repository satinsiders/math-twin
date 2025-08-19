"""Sequential execution runner for the twin generator pipeline."""
from __future__ import annotations

import json
import logging
import copy
from dataclasses import asdict, dataclass
from typing import Callable

from .agents import QAAgent
from .pipeline_helpers import AgentsRunner, _TOOLS
from .utils import get_final_output
from .pipeline_state import PipelineState


@dataclass(slots=True)
class _Graph:
    steps: list[Callable[[PipelineState], PipelineState]]


class _Runner:
    """Minimal sequential task executor with QA checks."""

    def __init__(
        self,
        graph: _Graph,
        *,
        verbose: bool = False,
        qa_max_retries: int | None = None,
    ) -> None:
        self.graph = graph
        self.verbose = verbose
        self.qa_max_retries = qa_max_retries
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _execute_step(
        self,
        step: Callable[[PipelineState], PipelineState],
        data: PipelineState,
    ) -> tuple[
        PipelineState,
        bool,
        list[Callable[[PipelineState], PipelineState]] | None,
        PipelineState,
    ]:
        before = copy.deepcopy(data)
        result = step(data)
        skip_qa = bool(result.skip_qa)
        next_steps = result.next_steps
        result.skip_qa = False
        result.next_steps = None
        return result, skip_qa, next_steps, before

    def _qa_check(
        self,
        name: str,
        data: PipelineState,
        idx: int,
        attempts: int,
        total_steps: int,
        json_required: bool,
    ) -> tuple[bool, str]:
        try:
            qa_in = json.dumps({"step": name, "data": asdict(data)})
        except (TypeError, ValueError) as exc:
            if not json_required:
                raise RuntimeError(f"QAAgent failed: {exc}")
            qa_out = f"non-serializable data: {exc}"
            self.logger.info(
                "[twin-generator] step %d/%d: %s QA round %d: %s",
                idx + 1,
                total_steps,
                name,
                attempts + 1,
                qa_out,
            )
            return False, qa_out
        try:
            qa_res = AgentsRunner.run_sync(QAAgent, input=qa_in, tools=_TOOLS)
            qa_out = get_final_output(qa_res).strip().lower()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"QAAgent failed: {exc}")
        self.logger.info(
            "[twin-generator] step %d/%d: %s QA round %d: %s",
            idx + 1,
            total_steps,
            name,
            attempts + 1,
            qa_out,
        )
        return qa_out == "pass", qa_out

    def run(self, inputs: PipelineState) -> PipelineState:
        data = copy.deepcopy(inputs)
        steps = list(self.graph.steps)
        idx = 0
        while idx < len(steps):
            step = steps[idx]
            name = step.__name__.replace("_step_", "").lstrip("_")
            attempts = 0
            while True:
                self.logger.info(
                    "[twin-generator] step %d/%d: %s attempt %d",
                    idx + 1,
                    len(steps),
                    name,
                    attempts + 1,
                )
                data, skip_qa, next_steps, before = self._execute_step(step, data)
                if data.error is not None:
                    return data
                if skip_qa:
                    if next_steps:
                        steps[idx + 1 : idx + 1] = next_steps
                    break
                from . import pipeline as pipeline_module
                json_required = name in pipeline_module._JSON_STEPS
                try:
                    passed, qa_out = self._qa_check(
                        name, data, idx, attempts, len(steps), json_required
                    )
                except RuntimeError as exc:
                    data.error = str(exc)
                    return data
                if passed:
                    if next_steps:
                        steps[idx + 1 : idx + 1] = next_steps
                    break
                attempts += 1
                if (
                    self.qa_max_retries is not None
                    and attempts >= self.qa_max_retries
                ):
                    data.error = f"QA failed for {name}: {qa_out}"
                    return data
                data = before
            idx += 1
        return data
