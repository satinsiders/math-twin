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

```json
{
  "template": "problem statement with symbolic parameters",
  "domains": {"symbol": "domain description"},
  "answer_expression": "expression using the symbols",
  "operations": [{"expr": "...", "output": "..."}],
  "visual": {"type": "none|graph|table", "data": {}}
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
