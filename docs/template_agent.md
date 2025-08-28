# TemplateAgent Schema

The `TemplateAgent` produces a JSON object describing a parameterized math problem.
All keys and string values must be double-quoted, and the agent should return **only**
this JSON object with no trailing explanation.

## Required Fields

- `template` (string): problem statement with symbolic parameters.
- `domains` (object): maps each symbol to a domain string. Every symbol used in
  `template`, `answer_expression`, or any operation's `expr` must appear here.
- `answer_expression` (string): expression yielding the final answer using only
  declared symbols or outputs from `operations`.
- `operations` (array): each item is an object with
  - `expr` (string)
  - `output` (string) **or** `outputs` (array of strings)
  - optional additional keys supplying tool arguments or referencing other
    fields from the pipeline state.
- `visual` (object): `{ "type": "none" | "graph" | "table", "data": {...} }`.

## Recommended Metadata

- `meta` (object): carries forward difficulty/complexity signals and guardrails for downstream steps.
  - `difficulty`: `'easy' | 'medium' | 'hard'` (copied from ParserAgent if available)
  - `complexity_features` (object): include features like `step_count` (from ConceptAgent),
    `variable_count`, `nonlinearity` (boolean), and `special_structures` (array of strings)
    to help the sampler maintain the intended cognitive load.
  - `difficulty_profile` (object, optional): template‑specific heuristics that downstream QA honors when
    evaluating parameter choices. Examples:
    - `needs_square_discriminant` (boolean): when true, QA will not flag a perfect‑square quadratic discriminant
      as trivializing; instead it will require the discriminant to be a perfect square when applicable (e.g., integer‑answer twins).
    - `min_value_ranges` (object): map of symbol → constraints object. Supported keys: `min`, `max`, `abs_min`. QA flags
      parameter values outside these ranges. Example: `{ "C": {"abs_min": 2}, "M": {"min": 5} }`.
  - `invariants` (object, optional): stem/ask constraints to prevent target drift. Examples:
    - `ask`: canonical ask tag like `"smaller_integer"`, `"value_of_f"`, `"solve_for_x"`.
    - `forbid_asks`: array of tags to forbid (e.g., `["ordered_pair", "range"]`).
    - `require_phrases`/`forbid_phrases`: arrays of literal phrases expected or disallowed in the final stem.

```json
{
  "template": "problem statement with symbolic parameters",
  "domains": {"symbol": "domain description"},
  "answer_expression": "expression using the symbols",
  "operations": [{"expr": "...", "output": "..."}],
  "visual": {"type": "none|graph|table", "data": {}}
  ,
  "meta": {
    "difficulty": "medium",
    "complexity_features": {"step_count": 3, "variable_count": 2, "nonlinearity": false, "special_structures": []},
    "difficulty_profile": {"needs_square_discriminant": true, "min_value_ranges": {"C": {"abs_min": 2}}},
    "invariants": {"ask": "smaller_integer", "forbid_asks": ["ordered_pair"]}
  }
}
```

## Example

```json
{
  "template": "Solve for x: a*x + b = c",
  "domains": {"a": "nonzero real", "b": "real", "c": "real"},
  "answer_expression": "(c - b) / a",
  "operations": [{"expr": "a*x + b", "output": "lhs"}],
  "visual": {"type": "none", "data": {}}
}
```
