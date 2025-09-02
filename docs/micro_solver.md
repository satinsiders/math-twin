Micro‑Solver (Ultra‑Granular Math Solving)
=========================================

Overview
--------
The micro‑solver implements a multi‑agent architecture that decomposes math problem solving into tiny, verifiable micro‑steps aligned with a cognitive breakdown:

- Recognition: encode the problem into a structured internal representation.
- Reasoning: retrieve schemas/strategies and schedule operators based on progress.
- Calculation: execute algebraic/arithmetical steps, verify, and format the result.

Each micro‑step is handled by a single agent with a narrow responsibility, strict input/output contracts, and micro‑QA after every step. This minimizes per‑agent cognitive load and reduces cascading errors.

Key Concepts
------------
- Blackboard state (`micro_solver.state.MicroState`): shared, typed working memory updated by each step.
- Micro‑agents (`micro_solver.agents`): single‑purpose agents with instructions that enforce minimal outputs.
- Orchestrator (`micro_solver.orchestrator.MicroRunner`): executes steps sequentially and runs Micro‑QA after each.
- Micro‑QA (`MicroQAAgent`): checks minimal post‑conditions (presence, types, shape), not solution correctness.

Stage → Micro‑Steps
-------------------
1) Recognition
- Normalize: lightweight text normalization (local function).
- Tokenize: `TokenizerAgent` → {sentences[], tokens[]}.
- Entities: `EntityExtractorAgent` → {variables[], constants[], quantities[]}.
- Relations: `RelationExtractorAgent` → {relations[]} (equations/inequalities/definitions).
- Goal: `GoalInterpreterAgent` → {goal} (e.g., "solve for x").
- Classify: `TypeClassifierAgent` → {problem_type} (e.g., linear/quadratic/... ).
- Canonical Representation: `RepresentationAgent` → minimal canonical JSON {symbols, given, constraints, target, type}.

2) Reasoning
- Schema Retrieve: `SchemaRetrieverAgent` → {schemas[]} (named canonical schemas).
- Strategy Enumerate: `StrategyEnumeratorAgent` → {strategies[]} (micro‑plans).
- Progress Scheduler: `scheduler.solve_with_defaults` applies operators based on progress signals.

3) Calculation
- Execute Plan: handled by the progress scheduler above.
- (Optional) Substitute / Simplify: `SubstituteAgent`, `SimplifyAgent` for isolated, non‑global rewrites.
- Candidate Extract: detect candidate answer by pattern (e.g., x = expr).
- SymPy Simplify: `_micro_simplify_candidate_sympy` uses SymPy to canonicalize the candidate expression deterministically.
- Verify: `_micro_verify_sympy` prefers SymPy substitution checks; falls back to `VerifyAgent` if symbolic check is inconclusive.

Why This Helps
--------------
- Reduces per‑agent load: each instruction is tiny and avoids multi‑step reasoning.
- QA at every hop: quick detection of malformed outputs before errors compound.
- Traceability: `MicroState.A['symbolic']['intermediate']` records atomic operations with inputs/outputs.
- Extensibility: add new micro‑agents (e.g., geometry sub‑pipelines) without altering the orchestrator.

Usage (Programmatic)
--------------------
```python
from micro_solver import MicroState, MicroGraph, MicroRunner
from micro_solver.steps import DEFAULT_MICRO_STEPS

graph = MicroGraph(steps=DEFAULT_MICRO_STEPS)
runner = MicroRunner(graph)
state = MicroState(problem_text="Solve 2x + 3 = 11 for x.")
out = runner.run(state)
print(out.A["symbolic"]["final"])
```

CLI
---
- Module form: `python -m micro_solver.cli "Solve 2x + 3 = 11 for x."`
- Console script (after installing the package): `micro-solve "Solve 2x + 3 = 11 for x."`

Options:
- `--verbose`: enables structured logs, including step names and per-step QA results.

- Iteration: `scheduler.solve_with_defaults` iteratively applies operators guided by progress metrics; a stall counter guards against infinite loops.
- SymPy integration: `micro_solver/sym_utils.py` provides `simplify_expr` and `verify_candidate`. These helpers are defensive and degrade gracefully if SymPy is not available.
- Constraint analysis: `micro_solver.constraint_analysis` computes numeric Jacobians, flags redundant constraints, and attempts simple rank repairs before invoking expensive solves.

Plan Lint (Policy Tester)
-------------------------
Use the plan-lint tool to validate that `plan_steps` are planning-only (no precomputed results) before running:

```bash
micro-plan-lint plan.json        # where plan.json contains {"plan_steps": [...]} or just [...]
# or
python -m micro_solver.plan_lint plan.json
```

The linter rejects steps that include disallowed result keys (e.g., `result`, `sum`, `candidates`) or numeric-only lists (e.g., `operands: [1, 11, 37]`).

Notes
-----
- The default agents reference the OpenAI Responses API via a minimal runner; in offline/testing contexts, monkeypatch `AgentsRunner.run_sync` to return deterministic outputs.
- For exact arithmetic and expression evaluation, prefer the existing SymPy‑backed tools in `twin_generator.tools` (e.g., `calc_answer_tool`) inside custom micro‑steps.
- The micro‑solver is additive and does not modify the twin‑generator pipeline.
