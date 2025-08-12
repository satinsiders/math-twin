"""Agent definitions used by the pipeline."""
from agents import Agent  # type: ignore

ParserAgent = Agent(
    name="ParserAgent",
    instructions=(
        "Take the source problem + solution and return JSON detailing variables, "
        "relations, constraints, any visuals, and the answer format, ensuring coverage through "
        "extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

ConceptAgent = Agent(
    name="ConceptAgent",
    instructions=(
        "From the parsed JSON, identify the key concept(s) and outline the canonical "
        "solution path in ordered steps, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

TemplateAgent = Agent(
    name="TemplateAgent",
    instructions=(
        "Replace literals with symbols; provide domains; include a `visual` field "
        "→ {type: none|graph|table, data: {…}}, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

SampleAgent = Agent(
    name="SampleAgent",
    instructions=(
        "Given a parameterized math problem template, generate a candidate "
        "parameter set and compute output. Return only the parameter mapping "
        "as valid JSON without any prose, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Using the *parameter template* and the sampled params, draft a **new SAT-style** equation problem "
        "that tests the same concept but with surface variation. Return only JSON with keys: "
        "twin_stem, choices[], rationale, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Return minified JSON with fields: twin_stem, choices[], answer_index, "
        "answer_value, rationale, graph_path?, table_html?. Validate internal consistency while "
        "ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

QAAgent = Agent(
    name="QAAgent",
    instructions=(
        "Validate the previous step's output for correctness and internal consistency, ensuring coverage through "
        "extremely advanced math operations. Return 'pass' if the output is sound, otherwise return a brief reason."
    ),
    model="gpt-5-mini",
)

SymbolicSolveAgent = Agent(
    name="SymbolicSolveAgent",
    instructions=(
        "Handle heavy symbolic equation solving tasks with precision, ensuring coverage through "
        "extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

SymbolicSimplifyAgent = Agent(
    name="SymbolicSimplifyAgent",
    instructions=(
        "Perform deep symbolic simplification and manipulation while ensuring coverage through "
        "extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

OperationsAgent = Agent(
    name="OperationsAgent",
    instructions=(
        "Given the current pipeline data and a list of operations, invoke any provided tools "
        "to compute intermediate results. Return only JSON with any newly derived fields."
    ),
    model="gpt-5-mini",
)

__all__ = [
    "ParserAgent",
    "ConceptAgent",
    "TemplateAgent",
    "SampleAgent",
    "StemChoiceAgent",
    "FormatterAgent",
    "QAAgent",
    "SymbolicSolveAgent",
    "SymbolicSimplifyAgent",
    "OperationsAgent",
]
