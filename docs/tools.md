# Available Tools

Agents must read this document before calling any tool.

If a QA check fails, the next agent run includes a ``qa_feedback`` field in its
JSON payload (or appended text for plain prompts) containing the failure
message. Agents should incorporate this feedback to fix the prior issues before
responding again.

## calc_answer_tool
Evaluates a mathematical expression with given parameters using SymPy. Use for exact numeric computation of expressions and intermediate results.

## render_graph_tool
Renders a graph described by a JSON specification to a PNG file and returns the file path. Use when a problem requires a plotted graph visual.

## make_html_table_tool
Converts a JSON table specification into an HTML `<table>` string with escaped values. Use to generate table visuals for problems.

## sanitize_params_tool
Sanitizes a JSON mapping of parameters, keeping only those convertible to numeric SymPy expressions and reporting skipped keys. Use for parameter validation.

## validate_output_tool
Coerces answer fields and validates the formatter output for structural consistency. Use in QA steps to ensure final JSON correctness.

## validate_answer_ref_tool
Checks that `template.answer_expression` is either a composite, numeric-like expression or, if it is a single identifier (e.g., `y_expr`), that the identifier exists in current `params` or is declared as an output in `template.operations`. Returns `{ ok, is_identifier, ref, in_params, in_operations, detail }`.

## check_asset_tool
Verifies that a referenced graph file exists or that table HTML is non-empty. Use to confirm generated assets are present.

## graph_consistency_tool
Best-effort check that a local `graph_path` visually matches a provided set of `points`. Re-renders an image from the points using the same renderer and compares via pixel difference; returns `{ ok, score }` where lower scores are better. Requires optional `Pillow` and `numpy` to perform the visual comparison; degrades gracefully when unavailable.

## detect_degenerate_params_tool
Detects parameter assignments that likely trivialize the problem's difficulty. Inputs: `template_json` (stringified JSON object from TemplateAgent) and `params_json` (stringified JSON mapping). Honors `template.meta.difficulty_profile` when present (e.g., `needs_square_discriminant`).

Heuristics include:
- 0/±1 values used directly in the template or `answer_expression` (when referenced by name)
- Perfect-square discriminants for quadratic forms (when `x^2` or `quadratic` detected and `a,b,c` present). If `template.meta.difficulty_profile.needs_square_discriminant` is true, a perfect square is allowed (and non‑square may be flagged instead).
- Perfect square parameters used under `sqrt(...)`
- Magnitude/format checks to preserve difficulty when metadata is present (`template.meta.difficulty`, optional `template.meta.complexity_features.step_count`):
  - For medium/hard: flags when all used parameters are single-digit integers
  - For hard: also flags when there is no multi-digit or fractional parameter, or when all used parameters are integers (no non-integers)
  - When `step_count ≥ 3`: flags when all used parameters are single-digit integers

Returns `{ ok: boolean, reasons: [string], details: {...} }`.

## check_invariants_tool
Checks stem invariants emitted by the template to prevent ask drift. Inputs: `template_json`, `twin_stem`. Enforces `template.meta.invariants` keys: `ask`, `forbid_asks`, `require_phrases`, `forbid_phrases`. Returns `{ ok, reasons }`.

## choices_truth_filter_tool
Simple truth filter to avoid duplicate correct distractors. Inputs: `choices_json`, optional `computed_value`, and optionally `template_json` + `params_json` to compute the expected numeric answer when `computed_value` is not given. Counts how many choices equal the expected value (numeric/string equality within tight tolerance). Returns `{ ok, correct_count }` and fails when more than one matches.

## rationale_grounding_tool
Prevents rationale free‑writing new numbers. Inputs: full `state_json` (serialized PipelineState) and `rationale`. Extracts numbers from the rationale and ensures they appear in known state values (params, computed_value/answer, choices), and numbers present in `symbolic_solution`/`symbolic_simplified` strings if available. Returns `{ ok, unknown: [numbers] }`.

## count_concept_steps_tool
Counts the number of numbered lines in ConceptAgent output to approximate solution step depth. Input: `concept_text` string. Returns `{ steps: number }`. Use optionally in QA to compare intended reasoning depth with the twin prompt's apparent complexity.

## symbolic_solve_tool
Solves symbolic equations for specified variables and returns simplified solutions as JSON. Use to obtain exact symbolic results within operations.

## sample_function_points_tool
Generates `points` for a function expression of `x` given optional parameters. Inputs: `expr` (string), optional `params_json` (JSON mapping of symbol→value), optional `x_values` array or `n`, `x_min`, `x_max`. Returns `{ "points": [[x,y], ...] }` suitable for graph visuals.

## fit_function_tool
Fits function families (polynomial, exponential, logarithmic, power, trigonometric) to observed points and returns the best-fit equation, parameters, and `suggested_points`. Inputs: `points_json` (array of pairs, object with `points`, or `{ series:[{label?, points:[]}, ...] }`), optional `family` (`auto` default), optional `families` (explicit list to consider), and `max_degree` for polynomials. For multi-series input, returns `{ series:[{label, fit, suggested_points}, ...] }`.
