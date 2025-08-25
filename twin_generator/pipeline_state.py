from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .constants import GraphSpec


@dataclass
class PipelineState:
    """Typed container for data flowing through the pipeline."""

    # Inputs
    problem_text: str = ""
    solution: str = ""
    force_graph: bool = False
    graph_spec: GraphSpec | None = None

    # Intermediate results
    parsed: dict[str, Any] | None = None
    concept: str | None = None
    template: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    symbolic_solution: str | None = None
    symbolic_simplified: str | None = None
    symbolic_error: str | None = None
    graph_path: str | None = None
    table_html: str | None = None
    answer: Any | None = None
    stem_data: dict[str, Any] | None = None

    # Final formatted output fields
    twin_stem: str | None = None
    choices: list[Any] | None = None
    answer_index: int | None = None
    answer_value: Any | None = None
    rationale: str | None = None
    errors: list[str] | None = None

    # Error handling
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # QA feedback from failed checks
    qa_feedback: str | None = None

    # Runner metadata
    skip_qa: bool = False
    next_steps: list[Callable[["PipelineState"], "PipelineState"]] | None = None
