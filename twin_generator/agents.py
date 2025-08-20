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
    model="gpt-5-nano",
)

ConceptAgent = Agent(
    name="ConceptAgent",
    instructions=(
        "From the parsed JSON, identify the key concept(s) and outline the canonical "
        "solution path in ordered steps, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-nano",
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
    model="gpt-5-nano",
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
    model="gpt-5-nano",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Using the *parameter template* and the sampled params, draft a **new SAT-style** equation problem "
        "that tests the same concept but with surface variation. Return only a single JSON object "
        "with double-quoted keys/values and no trailing text containing keys: "
        "twin_stem, choices[], rationale, ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-nano",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Return only a single minified JSON object with double-quoted keys/values "
        "and no trailing text containing fields: twin_stem, choices[], answer_index, "
        "answer_value, rationale, graph_path?, table_html?. Validate internal consistency while "
        "ensuring coverage through extremely advanced math operations."
    ),
    model="gpt-5-nano",
)

QAAgent = Agent(
    name="QAAgent",
    instructions=(
        "Validate the previous step's output for correctness, strict JSON formatting "
        "(double-quoted keys/values with no trailing text), and internal consistency, ensuring coverage through "
        "extremely advanced math operations. Return 'pass' if the output is sound, otherwise return a brief reason."
    ),
    model="gpt-5-nano",
)

SymbolicSolveAgent = Agent(
    name="SymbolicSolveAgent",
    instructions=(
        "Input: JSON {template, params}. Substitute params into template to form the target relation/"
        "expression and solve exactly for the intended unknown(s). Output: ONE SymPy-parsable STRING, nothing else. "
        "Use exact arithmetic (no floats). Operators/functions: **, *, sqrt, Abs, log, pi, E, I, oo; logic And/Or; "
        "sets Interval, Union, FiniteSet, EmptySet, ConditionSet. Default domain: Reals; honor constraints and nonzero "
        "denominators. Verify candidates by substitution into the original relation(s) and drop extraneous roots. "
        "Trig: give GENERAL solutions with integer k in Integers. Inequalities/solution sets: return SymPy set syntax "
        "(e.g., Union(Interval.Lopen(0,2), Interval(3,oo))). Multiple solutions: FiniteSet or Union; systems: FiniteSet "
        "of tuples or Piecewise only if required. If no solution: EmptySet. If unresolved under constraints: "
        "ConditionSet(var, condition, S.Reals). Deterministic ordering. No commentary."
    ),
    model="gpt-5-nano",
)

SymbolicSimplifyAgent = Agent(
    name="SymbolicSimplifyAgent",
    instructions=(
        "Input: ONE SymPy expression STRING. Output: ONE SymPy expression STRING that is equivalent and simpler; "
        "if no provable improvement exists, return the INPUT EXACTLY (idempotent no-op). Use exact arithmetic; no floats. "
        "Preserve correctness on the default real domain unless the expression dictates otherwise. Do not cancel factors "
        "that may be zero; do not combine logs unless positivity of arguments is guaranteed. Avoid gratuitous expansion; "
        "keep structured forms unless expansion clearly simplifies. Respect principal branches for roots/abs; introduce "
        "Piecewise only when needed. Merge adjacent intervals/guards when present. Deterministic symbol/term ordering. "
        "No commentary."
    ),
    model="gpt-5-nano",
)

OperationsAgent = Agent(
    name="OperationsAgent",
    instructions=(
        "Given the current pipeline data and a list of operations, invoke any provided tools "
        "to compute intermediate results. Return only a single JSON object with double-quoted "
        "keys/values and no trailing text containing any newly derived fields."
    ),
    model="gpt-5-nano",
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
