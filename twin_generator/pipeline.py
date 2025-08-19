"""Linear pipeline orchestration of agents, tools, and helpers."""
from __future__ import annotations



from agents.run import Runner as AgentsRunner  # type: ignore  # noqa: F401

from . import constants as C  # noqa: F401
from .constants import GraphSpec
from .agents import (  # noqa: F401
    ConceptAgent,
    FormatterAgent,
    OperationsAgent,
    ParserAgent,
    SampleAgent,
    StemChoiceAgent,
    TemplateAgent,
    SymbolicSolveAgent,
    SymbolicSimplifyAgent,
)
from .pipeline_runner import _Graph, _Runner
from .pipeline_state import PipelineState
from .pipeline_steps import (
    _step_answer,
    _step_concept,
    _step_format,
    _step_operations,
    _step_parse,
    _step_sample,
    _step_stem_choice,
    _step_symbolic,
    _step_template,
    _step_visual,
)
from .pipeline_helpers import (  # noqa: F401
    _TOOLS,
    _TEMPLATE_TOOLS,
    _TEMPLATE_MAX_RETRIES,
    _JSON_MAX_RETRIES,
    _JSON_STEPS,
    invoke_agent,
)

__all__ = ["generate_twin", "PipelineState"]

_PIPELINE = _Graph(
    steps=[
        _step_parse,
        _step_concept,
        _step_template,
        _step_sample,
        _step_symbolic,
        _step_operations,
        _step_visual,
        _step_answer,
        _step_stem_choice,
        _step_format,
    ]
)


def generate_twin(
    problem_text: str,
    solution_text: str,
    *,
    force_graph: bool = False,
    graph_spec: GraphSpec | None = None,
    verbose: bool = False,
) -> PipelineState:
    """Generate a twin SAT-style math question given a source problem/solution."""
    runner = _Runner(_PIPELINE, verbose=verbose)
    state = PipelineState(
        problem_text=problem_text,
        solution=solution_text,
        force_graph=force_graph,
        graph_spec=graph_spec,
    )
    result = runner.run(state)
    return result
