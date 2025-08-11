"""Agent definitions used by the pipeline."""
from agents import Agent  # type: ignore

ParserAgent = Agent(
    name="ParserAgent",
    instructions=(
        "Take the source problem + solution and return JSON detailing variables, "
        "relations, constraints, any visuals, and the answer format."
    ),
    model="gpt-4o",
)

ConceptAgent = Agent(
    name="ConceptAgent",
    instructions=(
        "From the parsed JSON, identify the key concept(s) and outline the canonical "
        "solution path in ordered steps."
    ),
    model="gpt-4o",
)

TemplateAgent = Agent(
    name="TemplateAgent",
    instructions=(
        "Replace literals with symbols; provide domains; include a `visual` field "
        "→ {type: none|graph|table, data: {…}}."
    ),
    model="gpt-4o",
)

SampleAgent = Agent(
    name="SampleAgent",
    instructions=(
        "Given a parameterized math problem template, generate a candidate "
        "parameter set and compute output. Return only the parameter mapping "
        "as valid JSON without any prose."
    ),
    model="gpt-4o",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Using the *parameter template* and the sampled params, draft a **new SAT-style** equation problem "
        "that tests the same concept but with surface variation. Return only JSON with keys: "
        "twin_stem, choices[], rationale."
    ),
    model="gpt-4o",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Return minified JSON with fields: twin_stem, choices[], answer_index, "
        "answer_value, rationale, graph_path?, table_html?. Validate internal consistency."
    ),
    model="gpt-4o",
)

QAAgent = Agent(
    name="QAAgent",
    instructions=(
        "Validate the previous step's output for correctness and internal consistency. "
        "Return 'pass' if the output is sound, otherwise return a brief reason.",
    ),
    model="gpt-4o",
)

__all__ = [
    "ParserAgent",
    "ConceptAgent",
    "TemplateAgent",
    "SampleAgent",
    "StemChoiceAgent",
    "FormatterAgent",
    "QAAgent",
]

