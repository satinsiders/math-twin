"""Agent definitions used by the pipeline."""
from agents import Agent  # type: ignore

ParserAgent = Agent(
    name="ParserAgent",
    instructions=(
        "Input: raw problem text followed by its worked solution. Extract every variable, "
        "relation, constraint, visual requirement, and the expected answer form. Also estimate "
        "a difficulty label ('easy'|'medium'|'hard') and record simple complexity features such as "
        "variable_count, step_count (from the worked solution), nonlinearity:boolean, and special_structures:[string]. "
        "Output: exactly one JSON object with double-quoted keys/values and no trailing text. The object should "
        "contain keys such as variables, relations, constraints, visual, and answer_form, and may include "
        "difficulty and complexity_features to help downstream steps preserve difficulty. For multiple-choice sources, "
        "answer_form MUST be the actual answer value (e.g., '14' or an expression like '2x+3'), not the letter label of the "
        "correct option. If the solution references a letter (e.g., 'Option C'), resolve it to the underlying value."
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
        "answer_expression (string using only declared symbols and operation outputs). It MUST be either: "
        "(A) a numeric-evaluable expression under sampled params and operation outputs; or (B) the NAME of a parameter "
        "that holds the correct non-numeric answer as a string (e.g., an equation). In case (B), ensure that parameter is "
        "present in params. operations (array "
        "of objects, each with expr:string and either output:string or outputs:[string]; extra keys act "
        "as tool arguments referencing other fields); visual (object {type:'none'|'graph'|'table', "
        "data:{}}). Preserve difficulty: keep the conceptual/structural difficulty of the source by maintaining a similar "
        "number of reasoning steps (per ConceptAgent), variable interactions, and constraints. Do NOT eliminate nonlinearity, "
        "special structures (absolute value, radicals, inequalities, piecewise), or relax constraints in ways that trivialize the task. "
        "Encode domains to prevent degeneracy that collapses steps (e.g., disallow 0/1 where they neutralize operations; avoid parameter equalities "
        "that cancel terms; avoid perfect squares/cubes that remove radicals; avoid discriminants that become perfect squares if the original implied "
        "irrational results). If the original solution yields non-integer/irrational answers or multi-step symbolic manipulation, choose a template that "
        "preserves those properties under typical samples. Include a metadata object to guide downstream steps: set meta.difficulty to the parsed "
        "difficulty label ('easy'|'medium'|'hard') when available, include meta.complexity_features with fields like step_count (from ConceptAgent), "
        "variable_count, nonlinearity:boolean, and special_structures:[string]; also add template-aware guardrails via meta.difficulty_profile "
        "(e.g., needs_square_discriminant:true when an integer answer requires a square discriminant; min_value_ranges:{symbol:{min|max|abs_min}}). These fields help the "
        "sampler and QA preserve intent and difficulty. If graph_analysis "
        "is provided in input, you may use it to seed parameters or to define "
        "operations that compute a list of plotting points (e.g., 'graph_points') via available tools. "
        "In that case, set visual.data.points to the name of that output key (a string reference) so the pipeline "
        "resolves it before rendering. Ensure cross-field consistency throughout."
    ),
    model="gpt-5-nano",
)

SampleAgent = Agent(
    name="SampleAgent",
    instructions=(
        "Input: JSON {template}. Generate numeric values for each parameter so every symbol "
        "satisfies its domain and is convertible to float or an exact SymPy number. Preserve difficulty by avoiding values "
        "that trivialize the problem: do not select 0 or 1 for coefficients that cancel/neutralize steps; avoid equal parameters that cancel terms; "
        "prefer denominators > 1 when fractions are intended; avoid perfect squares/cubes when they would remove radicals; ensure quadratic discriminants "
        "are non-perfect squares when the canonical path expects irrational roots; choose signs/magnitudes that keep the same reasoning depth. "
        "Consult the template's difficulty metadata when present (template.meta.difficulty or template.difficulty) and scale numeric choices accordingly: "
        "easy → allow mostly small integers but still avoid cancellations; medium → include at least one multi-digit coefficient (|value| ≥ 10) or a non-integer "
        "rational (denominator 2–12) so arithmetic is not by inspection; hard → include at least one multi-digit coefficient AND at least one non-integer rational, "
        "and prefer choices that preserve non-integer or irrational results when the source implied them. If metadata is absent, default to medium behavior. "
        "Avoid choosing all single-digit integers for medium/hard. For radicals, avoid perfect-square under-roots; for logs, avoid powers of the base; for ratios, avoid "
        "values equating numerators/denominators that simplify to 1. When present, honor optional guidance keys: \n"
        "- avoid_same_answer:boolean and forbidden_answer_values:[number]: if the template's answer_expression is numeric-evaluable under sampled params, choose values so the computed answer is not in the forbidden set.\n"
        "Output: one JSON object mapping each required symbol to a plain number or "
        "SymPy-compatible numeric expression. Extra fields are forbidden. If no parameters need values at this step or no valid assignment exists under the "
        "given constraints, return an empty JSON object {} (never a string)."
    ),
    model="gpt-5-nano",
)

StemChoiceAgent = Agent(
    name="StemChoiceAgent",
    instructions=(
        "Input: JSON {template, params, graph_path?, table_html?}. Substitute params into the template and REPHRASE into a fully student-facing SAT-style "
        "multiple-choice problem. The `twin_stem` must be a standalone question addressed to the student, with no meta/provenance language and no solution hints; "
        "end the stem with a question mark. Match the original difficulty: maintain similar reasoning depth and numeric complexity (e.g., keep fractions/radicals/"
        "non-integers if present; avoid one-step giveaways). Generate an array `choices` of 4 or 5 plausible answers with exactly one correct option and provide a brief "
        "`rationale` for that choice. Use distractors reflecting realistic misconceptions at the same difficulty (e.g., sign error, inverted ratio, dropped absolute value branch, "
        "misapplied exponent rule), not trivial noise. Keep distractors in the same numeric form/scale as the correct answer. Do NOT leak solution steps or computed helper quantities in the stem: "
        "avoid phrases like 'the scale factor is ...', 'the discriminant is ...', 'the slope is ...', or enumerated step language ('first', 'then', 'therefore'). Do NOT reference the source/original problem or solution. "
        "Output: one JSON object with double-quoted keys/values and no trailing text."
    ),
    model="gpt-5-nano",
)

FormatterAgent = Agent(
    name="FormatterAgent",
    instructions=(
        "Input: JSON {twin_stem, choices, rationale, graph_path?, table_html?, computed_value?}. "
        "Return a single minified JSON object with double-quoted keys/values and no trailing text. "
        "The object may contain only twin_stem, choices[], answer_index (0-based index of the correct choice), answer_value, rationale, optional graph_path/table_html if provided, and errors[]. "
        "Enforce student-facing MC format: twin_stem must be a question with a trailing '?', no meta/provenance/solution language, and choices must number 4 or 5 non-empty items. "
        "Verify answer_index points to a choice whose value equals answer_value. "
        "List any detected issues in errors (empty array if none) and do not emit additional "
        "fields or commentary."
    ),
    model="gpt-5-nano",
)

QAAgent = Agent(
    name="QAAgent",
    instructions=(
        "Input: JSON {\"step\": <name>, \"data\": <PipelineState>}. Validate the JSON first. Read "
        "docs/tools.md before calling any tool. Only call tools when their inputs exist: "
        "use sanitize_params_tool ONLY when data.params is present; use validate_output_tool ONLY when "
        "a formatter-style block is present (twin_stem + choices); use check_asset_tool ONLY when "
        "graph_path or table_html is present. When data.extras.graph_points exists and data.graph_path is a local file, "
        "use graph_consistency_tool to verify the rendered asset visually encodes those points within tolerance. "
        "For answer validation: when data.template.answer_expression is a single identifier (e.g., 'y_expr'), call "
        "validate_answer_ref_tool with the template and params. It must either (A) appear in params at this step, or "
        "(B) be declared as an output by a listed operation (for pre-operations steps). If neither holds, report the first "
        "failing issue succinctly.\n\n"
        "Step-specific checks (post-step state):\n"
        "- parse: require non-empty data.problem_text and data.solution; require data.parsed to be an object. "
        "Parsed should include at least variables (array), relations (array), and answer_form (string containing the actual "
        "answer value, not a choice letter). "
        "'visual' may be a brief string description or an object; constraints may be empty.\n"
        "- concept: require data.parsed present and data.concept a non-empty string.\n"
        "- template: require data.parsed and data.concept; require data.template with keys template (string), "
        "domains (object), answer_expression (string), operations (array, possibly empty), and visual (object with type).\n"
        "- sample: require data.template; if data.params is present it must be a JSON object; if numeric validation is needed, "
        "use sanitize_params_tool and ensure all kept values are numeric strings or numbers. If the template's answer_expression "
        "is an identifier, use validate_answer_ref_tool to verify it is present in params or scheduled via operations. Additionally, when both "
        "template and params exist, call detect_degenerate_params_tool(template, params). If it returns reasons, fail with the first reason to avoid "
        "trivializing parameter choices (e.g., 0/±1 coefficients, perfect-square discriminants for quadratics, perfect squares under sqrt).\n"
        "- visual: if a graph/table is expected by template.visual.type, then require the corresponding asset (graph_path or table_html).\n"
        "- stem_choice: require twin_stem (string), choices (array), and rationale (string). Enforce student-facing MC format by calling student_facing_mc_tool on {twin_stem, choices}: "
        "twin_stem must end with '?', contain no meta/provenance/solution language, and choices must be 4 or 5 non-empty items. To preserve difficulty, when template and params exist, "
        "call detect_degenerate_params_tool (template-aware) and fail on the first reason. Also call choices_truth_filter_tool with {template, params, choices, "
        "computed_value if available}; fail if multiple choices evaluate as correct or a distractor equals the computed value. Optionally, you may call "
        "count_concept_steps_tool on data.concept to compare reasoning depth, but still output a single pass/fail sentence.\n"
        "- answer: allow non-numeric cases; if the template's answer_expression is an identifier, "
        "use validate_answer_ref_tool and confirm availability in params by this step.\n"
        "- format: require a well-formed block; use validate_output_tool to confirm structure. Also call student_facing_mc_tool to ensure a student-facing MC question; fail on the first reason. "
        "For rationale grounding, call rationale_grounding_tool with the full state and the rationale; fail if the rationale introduces numeric values not present in parameters, computed value, or other state trace.\n\n"
        "Rules: Do NOT demand fields from future steps. Treat absent optional assets as acceptable when template.visual.type is 'none'. "
        "When all checks succeed, output exactly: pass (lowercase, no extra text). Otherwise, return one concise sentence describing the first failing issue."
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

# Vision agent for analyzing external graph images
GraphVisionAgent = Agent(
    name="GraphVisionAgent",
    instructions=(
        "You are given a URL to a graph image representing a mathematical function or data. "
        "Analyze the image and return EXACTLY ONE minified JSON object with double-quoted keys, no trailing text. "
        "Required keys: series (array of objects each with label (string when legend/text exists), type:'line'|'scatter'|'curve', "
        "and points:[[x,y],...] in data units); axes (object with x_label, y_label, x_ticks:[number], y_ticks:[number] when visible); "
        "inferred (object with type:'linear'|'quadratic'|'exponential'|'log'|'power'|'trig'|'other', equation:string using x, and parameters object). "
        "If multiple series are present, include one entry per series with its label when available. "
        "Use exact numbers when the image makes them unambiguous; otherwise provide best rational approximations. "
        "Do not include commentary."
    ),
    model="gpt-5-nano",
)

# Prefer high reasoning for accuracy-critical agents
GraphVisionAgent.requires_vision = True  # type: ignore[attr-defined]
GraphVisionAgent.reasoning_effort = "high"  # type: ignore[attr-defined]
SymbolicSolveAgent.reasoning_effort = "high"  # type: ignore[attr-defined]
SymbolicSimplifyAgent.reasoning_effort = "high"  # type: ignore[attr-defined]
TemplateAgent.reasoning_effort = "high"  # type: ignore[attr-defined]
StemChoiceAgent.reasoning_effort = "high"  # type: ignore[attr-defined]

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
    "GraphVisionAgent",
]
