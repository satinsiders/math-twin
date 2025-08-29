"""Individual step functions for the twin generator pipeline."""
from __future__ import annotations

import json
from typing import Any, cast
import copy
import re

from . import constants as C
from .agents import (
    ConceptAgent,
    FormatterAgent,
    OperationsAgent,
    ParserAgent,
    SampleAgent,
    StemChoiceAgent,
    TemplateAgent,
    SymbolicSolveAgent,
    SymbolicSimplifyAgent,
    # New
    GraphVisionAgent,
)
from .pipeline_helpers import (
    _TOOLS,
    _TEMPLATE_TOOLS,
    _TEMPLATE_MAX_RETRIES,
    invoke_agent,
)
from .tools.calc import _calc_answer
from .tools.graph import _render_graph
from .tools.html_table import _make_html_table
from .utils import _normalize_graph_points
from .pipeline_state import PipelineState


def _step_parse(state: PipelineState) -> PipelineState:
    # Provide explicit section markers to help the parser separate inputs
    parser_input = f"Problem:\n{state.problem_text}\n\nSolution:\n{state.solution}"
    out, err = invoke_agent(
        ParserAgent,
        parser_input,
        tools=_TOOLS,
        max_retries=1,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    state.parsed = cast(dict[str, Any], out)
    return state


def _step_concept(state: PipelineState) -> PipelineState:
    out, err = invoke_agent(
        ConceptAgent,
        str(state.parsed),
        tools=_TOOLS,
        expect_json=False,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    state.concept = cast(str, out)
    return state


def _step_template(state: PipelineState) -> PipelineState:
    payload = json.dumps({
        "parsed": state.parsed,
        "concept": state.concept,
        # Provide any graph analysis so the template can bind params/ops to visuals
        "graph_analysis": state.graph_analysis,
    })
    out, err = invoke_agent(
        TemplateAgent,
        payload,
        tools=_TEMPLATE_TOOLS,
        max_retries=_TEMPLATE_MAX_RETRIES,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    state.template = cast(dict[str, Any], out)
    return state


def _step_sample(state: PipelineState) -> PipelineState:
    def _sanitize_template_for_sampling(tpl: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the template with non-sampleable clutter removed.

        - Drop domains for clearly non-numeric symbols (e.g., function types like "R -> R").
        - Drop domains for symbols that are declared as operation outputs (derived values).
        - Remove obviously contradictory textual constraints like "a + b = r1 + r2" when
          the template also asserts r1, r2 are roots of f and f(x) = a x^2 + b x + c.
        """
        cleaned = copy.deepcopy(tpl)
        try:
            domains = cleaned.get("domains") if isinstance(cleaned, dict) else None
            ops = cleaned.get("operations") if isinstance(cleaned, dict) else None
            # 1) Remove domains for non-numeric symbols (heuristic)
            if isinstance(domains, dict):
                to_delete: list[str] = []
                for k, v in domains.items():
                    try:
                        sv = str(v).lower()
                    except Exception:
                        sv = str(v)
                    # obvious non-numeric descriptors
                    if "->" in sv or "function" in sv:
                        to_delete.append(k)
                # common symbols that shouldn't be sampled
                for sym in ("f", "x"):
                    if sym in domains:
                        to_delete.append(sym)
                # 2) Remove domains for operation outputs (derived values)
                derived: set[str] = set()
                if isinstance(ops, list):
                    for op in ops:
                        if isinstance(op, dict):
                            outk = op.get("output")
                            if isinstance(outk, str):
                                derived.add(outk)
                            outs = op.get("outputs")
                            if isinstance(outs, list):
                                for name in outs:
                                    if isinstance(name, str):
                                        derived.add(name)
                for name in derived:
                    if name in domains:
                        to_delete.append(name)
                for k in set(to_delete):
                    try:
                        del domains[k]
                    except Exception:
                        pass
            # 3) Remove contradictory textual constraint "a + b = r1 + r2"
            if isinstance(cleaned, dict) and isinstance(cleaned.get("template"), str):
                t = cleaned["template"]
                # Only strip if the quadratic coefficient form appears and roots are mentioned
                if re.search(r"\ba\s*\*?\s*x\s*\^\s*2\s*\+\s*b\s*\*?\s*x\s*\+\s*c", t.replace(" ", ""), re.IGNORECASE) or (
                    "a x^2 + b x + c" in t or "ax^2 + bx + c" in t
                ):
                    if "roots of f" in t or "real roots of f" in t or "r1" in t and "r2" in t:
                        cleaned["template"] = re.sub(r"(?i)\b[aA]\s*\+\s*[bB]\s*=\s*r1\s*\+\s*r2\b", "", t)
        except Exception:
            pass
        return cleaned

    tpl_for_sampling = (
        _sanitize_template_for_sampling(state.template)
        if isinstance(state.template, dict)
        else state.template
    )
    # Include forbidden answer values derived from the source to proactively avoid
    # sampling params that yield the same final answer as the original.
    payload: dict[str, Any] = {"template": tpl_for_sampling}
    try:
        original_ans = None
        if isinstance(state.parsed, dict):
            original_ans = state.parsed.get("answer_form")
        if original_ans is not None:
            try:
                original_val = _calc_answer(str(original_ans), json.dumps({}))
            except Exception:
                original_val = None
            if original_val is not None:
                payload["avoid_same_answer"] = True
                payload["forbidden_answer_values"] = [original_val]
    except Exception:
        pass

    out, err = invoke_agent(
        SampleAgent,
        json.dumps(payload),
        tools=_TOOLS,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    # Be tolerant to agents that may signal "no params" via JSON null or a literal
    # "null" string. Treat these as an empty params object rather than an error.
    if out is None or (isinstance(out, str) and out.strip().lower() == "null"):
        state.params = {}
        return state
    if not isinstance(out, dict):
        state.error = "SampleAgent produced non-dict params"
        return state
    state.params = out
    return state


def _step_symbolic(state: PipelineState) -> PipelineState:
    payload = json.dumps({"template": state.template, "params": state.params})
    sol, err = invoke_agent(
        SymbolicSolveAgent,
        payload,
        tools=_TOOLS,
        expect_json=False,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.symbolic_error = err.replace("SymbolicSolveAgent", "Symbolic agents")
        return state
    state.symbolic_solution = cast(str, sol)
    simp, err = invoke_agent(
        SymbolicSimplifyAgent,
        state.symbolic_solution,
        tools=_TOOLS,
        expect_json=False,
    )
    if err:
        state.symbolic_error = err.replace("SymbolicSimplifyAgent", "Symbolic agents")
        return state
    state.symbolic_simplified = cast(str, simp)
    return state


def _step_operations(state: PipelineState) -> PipelineState:
    ops = state.template.get("operations") if isinstance(state.template, dict) else []
    if not ops:
        state.skip_qa = True
        return state

    # Only include the template, params, and intermediates explicitly referenced by
    # the operations rather than the entire pipeline state.  This keeps the payload
    # concise and avoids leaking unrelated data to the agent.
    extra_fields: set[str] = set()
    for op in ops:
        if not isinstance(op, dict):
            continue
        for key, value in op.items():
            if key in {"expr", "output", "outputs"}:
                continue
            if isinstance(value, str):
                if hasattr(state, value):
                    extra_fields.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and hasattr(state, item):
                        extra_fields.add(item)

    data: dict[str, Any] = {"template": state.template, "params": state.params}
    for field in extra_fields:
        val = getattr(state, field)
        if val is not None:
            data[field] = val

    payload = json.dumps({"data": data, "operations": ops})
    out, err = invoke_agent(
        OperationsAgent, payload, tools=_TOOLS, qa_feedback=state.qa_feedback
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    if not isinstance(out, dict):
        state.error = "OperationsAgent produced non-dict output"
        return state
    params_out = out.get("params")
    if isinstance(params_out, dict):
        state.params = params_out
    expected_outputs: set[str] = set()
    for op in ops:
        if isinstance(op, dict):
            out_key = op.get("output")
            if isinstance(out_key, str):
                expected_outputs.add(out_key)
            else:
                out_vals = op.get("outputs")
                if isinstance(out_vals, list):
                    expected_outputs.update(
                        str(o) for o in out_vals if isinstance(o, str)
                    )

    for key, value in out.items():
        if key == "params" or key not in expected_outputs:
            continue
        if hasattr(state, key):
            setattr(state, key, value)
        else:
            state.extras[key] = value
    return state


def _select_graph_spec(
    visual: dict[str, Any], user_spec: Any, force: bool
) -> Any:  # noqa: ANN401 - generic return
    """Choose the graph spec given visual config, user override, and force flag."""
    vtype = visual.get("type")
    if force:
        return user_spec or visual.get("data") or C.DEFAULT_GRAPH_SPEC
    if vtype == "graph":
        # Do not auto-inject a default spec unless force is set; only use
        # explicit data or a user override to avoid creating visuals when
        # the source problem had none.
        return visual.get("data") or user_spec
    return None


def _render_table(visual: dict[str, Any]) -> str | None:
    """Render a table visual to HTML if applicable."""
    if visual.get("type") == "table":
        return _make_html_table(json.dumps(visual.get("data", {})))
    return None


def _step_visual(state: PipelineState) -> PipelineState:
    visual = state.template.get("visual") if isinstance(state.template, dict) else None
    if not isinstance(visual, dict):
        visual = {"type": "none"}

    spec = _select_graph_spec(visual, state.graph_spec, bool(state.force_graph))
    # If template requested a graph but provided no data and no override, demote to 'none'
    if (
        spec is None
        and isinstance(state.template, dict)
        and isinstance(state.template.get("visual"), dict)
        and state.template["visual"].get("type") == "graph"
        and not state.graph_url
    ):
        try:
            state.template["visual"]["type"] = "none"
        except Exception:
            pass
    # If an external URL is provided and no spec is requested, prefer direct URL
    if spec is None and state.graph_url:
        # Use the external image as-is; QA will allow URLs
        state.graph_path = state.graph_url
        return state

    if spec is not None:
        # Allow spec values to reference state/extras keys (e.g., "points": "graph_points")
        spec = _resolve_refs(spec, state)
        if isinstance(spec, dict):
            if state.force_graph and not spec.get("points"):
                spec["points"] = C.DEFAULT_GRAPH_SPEC.get("points", [])
            _normalize_graph_points(cast(dict[str, Any], spec))
        try:
            state.graph_path = _render_graph(json.dumps(spec))
        except Exception as exc:
            state.error = f"Invalid graph spec: {exc}"
        return state

    table_html = _render_table(visual)
    if table_html is not None:
        state.table_html = table_html
    return state


def _step_answer(state: PipelineState) -> PipelineState:
    expr = (
        state.template.get("answer_expression", "0")
        if isinstance(state.template, dict)
        else "0"
    )
    try:
        state.answer = _calc_answer(expr, json.dumps(state.params))
    except Exception as exc:
        # Fallback: if expr is a key in params, use its value (string expression)
        fallback: Any = None
        if isinstance(expr, str) and isinstance(state.params, dict) and expr in state.params:
            fallback = state.params.get(expr)
        if fallback is not None:
            state.answer = fallback
        else:
            # Non-fatal: some question types (e.g., equation selection) don't yield a numeric answer.
            state.answer = None
            try:
                state.extras["computed_value_error"] = str(exc)
            except Exception:
                pass
    # Preserve the computed numeric answer separately for sanity checks/visuals
    try:
        if state.answer is not None:
            state.extras["computed_value"] = state.answer
    except Exception:
        # Be resilient if extras isn't a dict for any reason
        pass

    # Avoid producing a twin with the exact same numeric answer as the source.
    # If the original parsed answer is numeric and equals the computed twin answer,
    # resample parameters up to a small cap.
    try:
        original_ans = None
        if isinstance(state.parsed, dict):
            original_ans = state.parsed.get("answer_form")
        # Attempt to evaluate original answer numerically if it looks numeric or an expression
        original_val = None
        if original_ans is not None:
            try:
                # Use the same calc routine with empty params context
                original_val = _calc_answer(str(original_ans), json.dumps({}))
            except Exception:
                original_val = None
        def _as_float(x: Any) -> float | None:
            try:
                if isinstance(x, (int, float)):
                    return float(x)
                # Strings that are numeric-like
                return float(str(x))
            except Exception:
                return None
        twin_val = _as_float(state.answer)
        orig_val = _as_float(original_val)
        if twin_val is not None and orig_val is not None:
            # Compare within a small tolerance for floating cases
            if abs(twin_val - orig_val) <= 1e-9:
                attempts = int(state.extras.get("resample_avoid_same_answer_attempts", 0) or 0)
                if attempts < 3:
                    state.extras["resample_avoid_same_answer_attempts"] = attempts + 1
                    # Schedule a re-sample + recompute chain, skip QA on this pass
                    state.skip_qa = True
                    state.next_steps = [
                        _step_sample,
                        _step_symbolic,
                        _step_operations,
                        _step_visual,
                        _step_answer,
                    ]
                else:
                    # Give up after a few attempts; record for diagnostics
                    state.extras["note_same_answer_allowed"] = True
    except Exception:
        # Be fault-tolerant: if any of the above fails, proceed without resampling
        pass
    return state


def _step_stem_choice(state: PipelineState) -> PipelineState:
    payload: dict[str, Any] = {
        "template": state.template,
        "params": state.params,
    }
    if state.graph_path:
        payload["graph_path"] = state.graph_path
    if state.table_html:
        payload["table_html"] = state.table_html

    out, err = invoke_agent(
        StemChoiceAgent,
        json.dumps(payload),
        tools=_TOOLS,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    if not isinstance(out, dict):
        state.error = "StemChoiceAgent produced non-dict output"
        return state
    state.stem_data = out
    return state


def _step_format(state: PipelineState) -> PipelineState:
    payload: dict[str, Any] = {
        "twin_stem": state.stem_data.get("twin_stem") if state.stem_data else None,
        "choices": state.stem_data.get("choices") if state.stem_data else None,
        "rationale": state.stem_data.get("rationale") if state.stem_data else None,
    }
    # Provide the computed value as a non-graded hint only; the formatter
    # is responsible for selecting the correct choice and emitting
    # answer_index/answer_value that match the choices.
    if state.answer is not None:
        payload["computed_value"] = state.answer
    if state.graph_path:
        payload["graph_path"] = state.graph_path
    if state.table_html:
        payload["table_html"] = state.table_html

    out, err = invoke_agent(
        FormatterAgent,
        json.dumps(payload),
        tools=_TOOLS,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        state.error = err
        return state
    out_dict = cast(dict[str, Any], out)

    # Pass-through assets in case the formatter dropped them
    if state.graph_path:
        out_dict.setdefault("graph_path", state.graph_path)
    if state.table_html:
        out_dict.setdefault("table_html", state.table_html)

    state.twin_stem = out_dict.get("twin_stem")
    state.choices = out_dict.get("choices")
    state.answer_index = out_dict.get("answer_index")
    state.answer_value = out_dict.get("answer_value")
    state.rationale = out_dict.get("rationale")
    state.errors = out_dict.get("errors")
    state.graph_path = out_dict.get("graph_path", state.graph_path)
    state.table_html = out_dict.get("table_html", state.table_html)

    # Final safeguard: if the formatted twin's answer matches the source numeric answer,
    # schedule a resample chain to enforce a different outcome.
    try:
        original_ans = None
        if isinstance(state.parsed, dict):
            original_ans = state.parsed.get("answer_form")
        original_val = None
        if original_ans is not None:
            try:
                original_val = _calc_answer(str(original_ans), json.dumps({}))
            except Exception:
                original_val = None
        def _as_float(x: Any) -> float | None:
            try:
                return float(str(x))
            except Exception:
                return None
        twin_val = _as_float(state.answer_value)
        orig_val = _as_float(original_val)
        if twin_val is not None and orig_val is not None and abs(twin_val - orig_val) <= 1e-9:
            attempts = int(state.extras.get("resample_avoid_same_answer_attempts", 0) or 0)
            if attempts < 3:
                state.extras["resample_avoid_same_answer_attempts"] = attempts + 1
                state.skip_qa = True
                state.next_steps = [
                    _step_sample,
                    _step_symbolic,
                    _step_operations,
                    _step_visual,
                    _step_answer,
                    _step_stem_choice,
                    _step_format,
                ]
    except Exception:
        pass
    return state


def _resolve_refs(obj: Any, state: PipelineState) -> Any:  # noqa: ANN401 - generic
    """Recursively replace string tokens that refer to state/extras keys.

    Any string value equal to the name of a PipelineState attribute or an
    ``extras`` key is replaced by that value. Other values are returned
    unchanged. Lists and dicts are processed recursively.
    """
    if isinstance(obj, str):
        if hasattr(state, obj):
            return getattr(state, obj)
        if isinstance(state.extras, dict) and obj in state.extras:
            return state.extras[obj]
        return obj
    if isinstance(obj, list):
        return [_resolve_refs(x, state) for x in obj]
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, state) for k, v in obj.items()}
    return obj


def _step_graph_analyze(state: PipelineState) -> PipelineState:
    """Analyze an external graph image URL if provided.

    Stores a structured JSON in ``state.graph_analysis`` with best-effort
    extraction of points, axes, and inferred function details. If unavailable
    or if analysis fails, the step is a no-op.
    """
    if not state.graph_url:
        # No external image to analyze; skip QA for this no-op step
        state.skip_qa = True
        return state
    payload: dict[str, Any] = {
        "graph_url": state.graph_url,
        "problem": state.problem_text,
        "solution": state.solution,
    }
    if state.parsed is not None:
        payload["parsed"] = state.parsed
    out, err = invoke_agent(
        GraphVisionAgent,
        json.dumps(payload),
        tools=None,
        qa_feedback=state.qa_feedback,
    )
    state.qa_feedback = None
    if err:
        # Non-fatal; keep going without analysis
        state.extras["graph_analysis_error"] = err
        state.skip_qa = True
        return state
    if isinstance(out, dict):
        state.graph_analysis = out
        # Convenience: expose common fields for operations/templates
        try:
            series = out.get("series") or []
            if series and isinstance(series, list) and isinstance(series[0], dict):
                pts = series[0].get("points")
                if isinstance(pts, list):
                    state.extras.setdefault("observed_points", pts)
        except Exception:
            pass
    # Analysis is advisory; do not block pipeline on QA for this step
    state.skip_qa = True
    return state
