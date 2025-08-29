"""Tool functions for the twin generator."""

from .html_table import _make_html_table, make_html_table_tool
from .graph import _render_graph, render_graph_tool
from .calc import _calc_answer, calc_answer_tool
from .qa_tools import (
    _check_asset,
    _sanitize_params_tool,
    _validate_output_tool,
    _graph_consistency_tool,
    _validate_answer_ref_tool,
    _detect_degenerate_params_tool,
    _count_concept_steps_tool,
    _choices_truth_filter_tool,
    _rationale_grounding_tool,
    _student_facing_mc_tool,
    check_asset_tool,
    sanitize_params_tool,
    validate_output_tool,
    graph_consistency_tool,
    validate_answer_ref_tool,
    detect_degenerate_params_tool,
    count_concept_steps_tool,
    choices_truth_filter_tool,
    rationale_grounding_tool,
    student_facing_mc_tool,
)
from .symbolic_solve import _symbolic_solve, symbolic_solve_tool
from .graph_analysis import (
    sample_function_points_tool,
    fit_function_tool,
)

__all__ = [
    # Public tools
    "make_html_table_tool",
    "render_graph_tool",
    "calc_answer_tool",
    "sanitize_params_tool",
    "validate_output_tool",
    "check_asset_tool",
    "graph_consistency_tool",
    "validate_answer_ref_tool",
    "detect_degenerate_params_tool",
    "count_concept_steps_tool",
    "choices_truth_filter_tool",
    "rationale_grounding_tool",
    "student_facing_mc_tool",
    "symbolic_solve_tool",
    "sample_function_points_tool",
    "fit_function_tool",
    # Private helpers (for advanced use/tests)
    "_make_html_table",
    "_render_graph",
    "_calc_answer",
    "_sanitize_params_tool",
    "_validate_output_tool",
    "_check_asset",
    "_graph_consistency_tool",
    "_validate_answer_ref_tool",
    "_detect_degenerate_params_tool",
    "_count_concept_steps_tool",
    "_choices_truth_filter_tool",
    "_rationale_grounding_tool",
    "_student_facing_mc_tool",
    "_symbolic_solve",
]
