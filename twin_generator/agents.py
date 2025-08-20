"""Agent definitions used by the pipeline."""
from agents import Agent  # type: ignore

ParserAgent = Agent(
    name="ParserAgent",
    instructions=(
        "Take the source problem + solution and return only a single JSON object "
        "with double-quoted keys/values and no trailing text detailing variables, "
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

# TemplateAgent expected schema:
# {
#   "template": "problem statement with symbolic parameters",
#   "domains": {"symbol": "domain description", ...},
#   "answer_expression": "expression using the symbols",
#   "operations": [{"expr": "...", "output": "..."}],
#   "visual": {"type": "none|graph|table", "data": {...}}
# }
# Example:
# {
#   "template": "Solve for x: a*x + b = c",
#   "domains": {"a": "nonzero real", "b": "real", "c": "real"},
#   "answer_expression": "(c - b) / a",
#   "operations": [{"expr": "a*x + b", "output": "lhs"}],
#   "visual": {"type": "none", "data": {}}
# }
TemplateAgent = Agent(
    name="TemplateAgent",
    instructions=(
        "Replace literals with symbols; provide domains; include a `visual` field "
        "→ {type: none|graph|table, data: {…}}. Return only a single JSON object with "
        "double-quoted keys/values and no trailing text, ensuring coverage through "
        "extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

SampleAgent = Agent(
    name="SampleAgent",
    instructions=(
        "Given a parameterized math problem template, generate a candidate "
        "parameter set and compute output. Return only a single JSON object "
        "with double-quoted keys/values and no trailing text representing the "
        "parameter mapping. Only include numeric parameter values required by "
        "the template—no extra commentary, derived objects, or additional fields. "
        "Each parameter value must be a plain number or numeric expression "
        "compatible with SymPy, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Using the *parameter template* and the sampled params, draft a **new SAT-style** equation problem "
        "that tests the same concept but with surface variation. Return only a single JSON object "
        "with double-quoted keys/values and no trailing text containing keys: "
        "twin_stem, choices[], rationale, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Return only a single minified JSON object with double-quoted keys/values "
        "and no trailing text containing fields: twin_stem, choices[], answer_index, "
        "answer_value, rationale, graph_path?, table_html?. Validate internal consistency while "
        "ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-mini",
)

QAAgent = Agent(
    name="QAAgent",
    instructions=(
        "Validate the previous step's output for correctness, strict JSON formatting "
        "(double-quoted keys/values with no trailing text), and internal consistency, ensuring coverage through "
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
        "to compute intermediate results. Return only a single JSON object with double-quoted "
        "keys/values and no trailing text containing any newly derived fields."
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
