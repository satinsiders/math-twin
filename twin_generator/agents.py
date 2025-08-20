"""Agent definitions used by the pipeline."""
from agents import Agent  # type: ignore

ParserAgent = Agent(
    name="ParserAgent",
    instructions=(
        "Input: raw problem text followed by its worked solution. Extract every variable, "
        "relation, constraint, visual requirement, and the expected answer form. Output: exactly "
        "one JSON object with double-quoted keys/values and no trailing text. The object should "
        "contain keys such as variables, relations, constraints, visual, and answer_form, providing all "
        "information needed for downstream steps."
    ),
    model="gpt-5-nano",
)

ConceptAgent = Agent(
    name="ConceptAgent",
    instructions=(
        "Input: JSON produced by ParserAgent. Identify the key mathematical concept(s) and outline "
        "the canonical solution path in ordered steps. Output: a plain-text string with the concept "
        "followed by numbered steps. Do not return JSON or extra commentary."
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
        "Input: JSON {parsed, concept}. Replace literals with symbolic parameters and supply their "
        "domains. Include fields: template, domains, answer_expression, operations[], and visual → "
        "{type: none|graph|table, data:{}}. Output: one JSON object with double-quoted keys/values and "
        "no trailing text."
    ),
    model="gpt-5-nano",
)

SampleAgent = Agent(
    name="SampleAgent",
    instructions=(
        "Input: JSON {template}. Generate a concrete parameter set that satisfies all domain "
        "constraints. Output: a single JSON object mapping each symbol to a plain number or "
        "SymPy-compatible numeric expression. Include only required parameters—no extra fields or commentary."
    ),
    model="gpt-5-nano",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Input: JSON {template, params, graph_path?, table_html?}. Substitute params into the template and craft a new "
        "SAT-style question `twin_stem` that tests the same concept. Generate an array `choices` of plausible answers with "
        "exactly one correct option and provide a brief `rationale` for that choice. Output: one JSON object with "
        "double-quoted keys/values and no trailing text."
    ),
    model="gpt-5-nano",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Input: JSON {twin_stem, choices, answer_value, rationale, graph_path?, table_html?}. Return a minified JSON object "
        "containing twin_stem, choices[], answer_index (0-based index of the correct choice), answer_value (matching the "
        "correct choice), rationale, and optional graph_path/table_html. Ensure answer_index and answer_value align with the "
        "choices. Output must be a single JSON object with double-quoted keys/values and no trailing text."
    ),
    model="gpt-5-nano",
)

QAAgent = Agent(
    name="QAAgent",
    instructions=(
        "Input: JSON produced by FormatterAgent. Verify strict JSON formatting and internal consistency—fields present, "
        "answer_index matches answer_value and choices, and any assets are valid. Output the plain string 'pass' if the data is "
        "sound; otherwise return a brief reason."
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
        "Input: JSON {data: {...}, operations: [...]}. Execute each operation—invoking tools when needed—to compute "
        "intermediate results or update parameters. Output: a single JSON object with double-quoted keys/values containing any "
        "newly derived fields or revised params, with no trailing text or commentary."
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
