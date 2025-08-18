# TemplateAgent Schema

The `TemplateAgent` produces a JSON object describing a parameterized math problem.
All keys and string values must be double-quoted, and the agent should return **only**
this JSON object with no trailing explanation.

## Expected Fields

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
