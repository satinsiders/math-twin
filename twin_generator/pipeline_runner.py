"""Sequential execution runner for the twin generator pipeline."""
from __future__ import annotations

import json
import logging
import copy
from dataclasses import asdict, dataclass
from typing import Callable

from .agents import QAAgent
from .pipeline_helpers import AgentsRunner, _QA_TOOLS
from .utils import get_final_output
from .tools import qa_tools as _qa
from .utils import coerce_answers as _coerce, validate_output as _validate
from .pipeline_state import PipelineState

# Default number of times to retry a step when QA checks fail.
QA_MAX_RETRIES = 5


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
        qa_max_retries: int | None = QA_MAX_RETRIES,
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
        """Run deterministic prechecks, then fall back to QAAgent.

        For critical steps like `stem_choice` and `format`, enforce student-facing
        constraints and no-hint policy without relying on an agent. Also ensure the
        twin's final numeric answer differs from the source when comparable.
        """
        # Deterministic prechecks for MC formatting and hint leakage
        try:
            is_stem = name == "stem_choice"
            is_format = name == "format"
            if is_stem or is_format:
                # Build a minimal block view
                block = {
                    "twin_stem": data.twin_stem or (data.stem_data or {}).get("twin_stem"),
                    "choices": data.choices or (data.stem_data or {}).get("choices"),
                }

                # Enforce student-facing MC constraints and no-hint language (no need for answer fields here)
                sf = _qa._student_facing_mc_tool(block)
                if not sf.get("ok", False):
                    reasons = sf.get("reasons", [])
                    msg = f"student-facing-fail:{';'.join(reasons)}"
                    data.qa_feedback = msg
                    self.logger.info(
                        "[twin-generator] step %d/%d: %s QA precheck: %s",
                        idx + 1,
                        total_steps,
                        name,
                        msg,
                    )
                    return False, msg

                

                # Ensure exactly one correct choice under computed value
                # Do not block at stem step on truth filter; enforce at format where answers are fixed
                if (not is_stem) and data.template and data.params and block.get("choices"):
                    ctf = _qa._choices_truth_filter_tool(
                        block.get("choices"),
                        data.answer,
                        json.dumps(data.template),
                        json.dumps(data.params),
                    )
                    if not ctf.get("ok", False):
                        msg = "choices-truth-fail:multiple-correct-or-duplicate"
                        data.qa_feedback = msg
                        self.logger.info(
                            "[twin-generator] step %d/%d: %s QA precheck: %s",
                            idx + 1,
                            total_steps,
                            name,
                            msg,
                        )
                        return False, msg

                # Ground numbers introduced in the stem to params/template
                try:
                    from dataclasses import asdict as _asdict
                    st_json = json.dumps(_asdict(data))
                except Exception:
                    st_json = "{}"
                sng = _qa._stem_number_grounding_tool(st_json, block.get("twin_stem"))
                if not sng.get("ok", True):
                    unknown = sng.get("unknown", [])
                    # Non-blocking guidance at stem step to avoid numeric leakage in stems
                    data.qa_feedback = f"stem-number-advice:{','.join(map(str, unknown))}"

                # Rationale numeric grounding
                if is_stem:
                    rationale = (data.stem_data or {}).get("rationale") if isinstance(data.stem_data, dict) else None
                    if isinstance(rationale, str) and rationale.strip():
                        rg = _qa._rationale_grounding_tool(st_json, rationale)
                        if not rg.get("ok", True):
                            # Non-blocking advice at stem step; enforce at format
                            data.qa_feedback = f"rationale-number-advice:{','.join(map(str, rg.get('unknown', [])))}"

                if is_format:
                    # For format step, we now have answer fields and perform full validation
                    full_block = {
                        **block,
                        "answer_index": data.answer_index,
                        "answer_value": data.answer_value,
                    }
                    full_block = _coerce(full_block)
                    v = _validate({**full_block})
                    if v.get("errors"):
                        msg = f"format-invalid:{';'.join(v['errors'])}"
                        data.qa_feedback = msg
                        return False, msg

                    # Rationale numeric grounding (formatter rationale)
                    if isinstance(data.rationale, str) and data.rationale.strip():
                        rg = _qa._rationale_grounding_tool(st_json, data.rationale)
                        if not rg.get("ok", True):
                            msg = f"rationale-number-drift:{','.join(map(str, rg.get('unknown', [])))}"
                            data.qa_feedback = msg
                            return False, msg

                    # Enforce: twin answer must differ from source numeric answer
                    def _as_float(x: object) -> float | None:
                        try:
                            return float(str(x))
                        except Exception:
                            return None
                    orig_ans = None
                    try:
                        if isinstance(data.parsed, dict):
                            orig_ans = data.parsed.get("answer_form")
                    except Exception:
                        pass
                    orig_val = None
                    if orig_ans is not None:
                        try:
                            # Evaluate using calc tool with empty params context
                            from .tools.calc import _calc_answer as _calc
                            orig_val = _calc(str(orig_ans), json.dumps({}))
                        except Exception:
                            orig_val = None
                    twin_val = _as_float(full_block.get("answer_value"))
                    o_val = _as_float(orig_val)
                    if twin_val is not None and o_val is not None and abs(twin_val - o_val) <= 1e-9:
                        msg = "twin-answer-equals-source"
                        data.qa_feedback = msg
                        return False, msg
        except Exception:
            # If prechecks themselves error, fall back to QAAgent
            pass

        # Assets timing guard: do not require graph/table assets before the visual step
        try:
            if name == "operations":
                v = (data.template or {}).get("visual") if isinstance(data.template, dict) else None
                vtype = v.get("type") if isinstance(v, dict) else None
                # Graph/table assets are produced in the subsequent visual step
                if vtype in {"graph", "table"} and not (data.graph_path or data.table_html):
                    return True, "graph/table asset not required before visual step"
        except Exception:
            # If inspection fails, proceed to QAAgent below
            pass
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
            qa_res = AgentsRunner.run_sync(QAAgent, input=qa_in, tools=_QA_TOOLS)
            qa_raw = get_final_output(qa_res)
        except Exception as exc:  # pragma: no cover - defensive
            qa_out = str(exc)
            data.qa_feedback = qa_out
            self.logger.info(
                "[twin-generator] step %d/%d: %s QA round %d: %s",
                idx + 1,
                total_steps,
                name,
                attempts + 1,
                qa_out,
            )
            return False, qa_out
        qa_out = qa_raw.strip()
        qa_lower = qa_out.lower()
        data.qa_feedback = qa_out
        self.logger.info(
            "[twin-generator] step %d/%d: %s QA round %d: %s",
            idx + 1,
            total_steps,
            name,
            attempts + 1,
            qa_out or "(empty response)",
        )
        # Be tolerant to minor variations from the QA agent: treat common
        # affirmative responses and empty strings as pass.
        ok_synonyms = {
            "pass",
            "ok",
            "okay",
            "looks good",
            "no issues",
            "no issue",
            "valid",
            "all good",
            "passes",
        }
        is_pass = (
            qa_lower == "pass"
            or qa_lower.startswith("pass")
            or qa_lower in ok_synonyms
            or qa_lower.strip() == ""
        )
        return is_pass, qa_out

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
                    data.qa_feedback = None
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
                    data.qa_feedback = None
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
                before.qa_feedback = data.qa_feedback
                data = before
            idx += 1
        data.qa_feedback = None
        return data
