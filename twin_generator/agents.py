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
        "the canonical solution path in ordered steps. Output: begin with 'Concept: <summary>' on the "
        "first line, then list numbered steps starting with '1.' each on its own line. Do not include "
        "extra commentary or JSON formatting."
    ),
    model="gpt-5-nano",
)

# TemplateAgent expected schema:
# {
#   "template": <str>,                         # problem statement with symbolic parameters
#   "domains": {symbol: <str>, ...},           # domain for each symbol; covers all symbols used
#   "answer_expression": <str>,                # expression using symbols and operation outputs
#   "operations": [                            # list may be empty
#       {"expr": <str>, "output": <str>, ...} # or {"expr": <str>, "outputs": [<str>, ...], ...}
#   ],
#   "visual": {"type": "none"|"graph"|"table", "data": {...}}
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
        "Input: JSON {parsed, concept}. Replace literals with symbolic parameters and state their "
        "domains. Return exactly one JSON object with double-quoted keys/values and no trailing text. "
        "Required fields: template (string problem statement); domains (object mapping each symbol to a "
        "domain string, covering all symbols in template, answer_expression, and operations); "
        "answer_expression (string using only declared symbols and operation outputs); operations (array "
        "of objects, each with expr:string and either output:string or outputs:[string]; extra keys act "
        "as tool arguments referencing other fields); visual (object {type:'none'|'graph'|'table', "
        "data:{}}). Ensure cross-field consistency throughout."
    ),
    model="gpt-5-nano",
)

SampleAgent = Agent(
    name="SampleAgent",
    instructions=(
        "Input: JSON {template}. Generate numeric values for each parameter so every symbol "
        "satisfies its domain and is convertible to float or an exact SymPy number. Output: one "
        "JSON object mapping each required symbol to a plain number or SymPy-compatible numeric "
        "expression. Extra fields are forbidden. If no valid assignment exists, return the plain "
        "string 'null'."
    ),
    model="gpt-5-nano",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Input: JSON {template, params, graph_path?, table_html?}. Substitute params into the template "
        "and craft a new SAT-style question `twin_stem` that tests the same concept. Generate an array "
        "`choices` of plausible answers with exactly one correct option and provide a brief `rationale` "
        "for that choice. Output: one JSON object with double-quoted keys/values and no trailing text."
    ),
    model="gpt-5-nano",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Input: JSON {twin_stem, choices, answer_value, rationale, graph_path?, table_html?}. "
        "Return a single minified JSON object with double-quoted keys/values and no trailing text. "
        "The object may contain only twin_stem, choices[], answer_index (0-based index of the "
        "correct choice), answer_value, rationale, optional graph_path/table_html if provided, "
        "and errors[]. Verify answer_index points to a choice whose value equals answer_value. "
        "List any detected issues in errors (empty array if none) and do not emit additional "
        "fields or commentary."
    ),
    model="gpt-5-nano",
)

QAAgent = Agent(
    name="QAAgent",
    instructions=(
        "Input: JSON {\"step\": <name>, \"data\": <PipelineState>}. First ensure the JSON is "
        "syntactically valid. Consult docs/tools.md for tool behavior before invoking any. Use "
        "sanitize_params_tool to check numeric params and report skipped keys. Invoke "
        "validate_output_tool to coerce answers and confirm formatter output. When graph_path or "
        "table_html is present, call check_asset_tool to verify the asset exists. Then verify that "
        "every field required for the named pipeline step exists and that all values are internally "
        "consistent—indices align with arrays, assets exist, and constraints are met. Output only "
        "the string 'pass' when all checks succeed; otherwise return a concise reason for failure."
    ),
    model="gpt-5-nano",
)

SymbolicSolveAgent = Agent(
    name="SymbolicSolveAgent",
    instructions=(
        "Input: JSON {template, params}. Substitute params into template to form the target relation/"
        "expression and solve exactly for the variable(s) appearing in answer_expression, respecting all "
        "domain constraints from the template. Output: ONE SymPy-parsable STRING, nothing else. "
        "Use exact arithmetic (no floats). Operators/functions: **, *, sqrt, Abs, log, pi, E, I, oo; logic And/Or;"
        "sets Interval, Union, FiniteSet, EmptySet, ConditionSet. Default domain: Reals; honor constraints "
        "and nonzero denominators. Explicitly confirm each candidate by substituting into the original "
        "relation(s) and drop extraneous roots. Trig: give GENERAL solutions with integer k in Integers. "
        "Inequalities/solution sets: return SymPy set syntax (e.g., Union(Interval.Lopen(0,2), Interval(3,oo))). "
        "Multiple solutions: FiniteSet or Union; systems: FiniteSet of tuples or Piecewise only if required. "
        "If no solution: EmptySet. If unresolved under constraints: ConditionSet(var, condition, S.Reals). "
        "Deterministic ordering. No commentary."
    ),
    model="gpt-5-nano",
)

SymbolicSimplifyAgent = Agent(
    name="SymbolicSimplifyAgent",
    instructions=(
        "Input: ONE SymPy expression STRING. Output: ONE SymPy expression STRING that is "
        "mathematically equivalent and simpler. If no provable simplification exists, return "
        "the original input expression unchanged. Use exact arithmetic; no floats. Enforce "
        "deterministic ordering of all symbols and terms. Preserve correctness on the default "
        "real domain unless the expression dictates otherwise. Never perform domain-sensitive "
        "transformations without proof: do not cancel factors that may be zero, and do not "
        "combine logs or manipulate Abs unless argument positivity is guaranteed. Avoid "
        "gratuitous expansion; keep structured forms unless expansion clearly simplifies. "
        "Respect principal branches for roots and Abs; introduce Piecewise only when needed "
        "and merge adjacent intervals/guards. Return only the final expression with no "
        "commentary."
    ),
    model="gpt-5-nano",
)

OperationsAgent = Agent(
    name="OperationsAgent",
    instructions=(
        "Input: JSON {data: {...}, operations: [...]}. Review docs/tools.md to understand available tools "
        "before executing any operation. "
        "Each operation is an object with: "
        "expr – an expression using fields from data or prior outputs; optional tool – the name of a registered tool; "
        "output (single key) or outputs (array of keys) naming where results should be stored; "
        "and any additional fields that reference entries in data or earlier outputs. "
        "Execute operations sequentially—evaluating expr or calling the specified tool—and produce numeric results. "
        "Output: one JSON object with double-quoted keys/values containing only the keys listed in each operation's "
        "output/outputs and an optional params object with updated numeric parameters. "
        "All returned values must be JSON-serializable, and numbers should be emitted as numbers, not strings. "
        "No extra fields or commentary."
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
