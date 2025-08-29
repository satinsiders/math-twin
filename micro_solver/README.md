Micro‑Solver (Ultra‑Granular Math Solving)
=========================================

What It Is
----------
The micro‑solver is an additive, ultra‑granular multi‑agent pipeline that solves math problems by decomposing them into very small, verifiable steps aligned with a cognitive model:

- Recognition: encode the problem into a structured internal representation.
- Reasoning: choose a schema/strategy and produce an atomic action plan.
- Calculation: execute one atomic action at a time, verify, and finish.

It emphasizes strict I/O contracts, deterministic post‑conditions, and per‑step QA. The goal is to minimize per‑agent cognitive load and reduce compounding errors.

Run It
------
- CLI: `python -m micro_solver.cli "Solve 2x + 3 = 11 for x." --verbose`
- Console script (after install): `micro-solve "…" --verbose`

Logging (with `--verbose`):
- Step start, quick human‑readable summary, QA result (with attempt index)
- Iterative plan execution logs per atomic rewrite

High‑Level Pipeline
-------------------
Default graph (simplified names):

1) normalize → 2) tokenize → 3) entities → 4) relations → 5) goal → 6) classify → 7) repr →
8) schema → 9) strategies → 10) choose_strategy → 11) decompose → 12) execute_plan →
13) solve_sympy → 14) extract_candidate → 15) simplify_candidate_sympy → 16) verify_sympy

- Early exit: the runner exits as soon as `final_answer` is set and QA passes.
- Retries: each step retries with QA feedback up to a small budget.

Agents (By Stage)
-----------------
Recognition
- `TokenizerAgent`: sentences[], tokens[]
- `EntityExtractorAgent`: variables[], constants[], quantities[]
- `RelationExtractorAgent`: relations[] (equations/inequalities/definitions)
- `GoalInterpreterAgent`: goal (e.g., “solve for x”, “find area”)
- `TypeClassifierAgent`: problem_type (e.g., linear/quadratic/proportion/…)
- `RepresentationAgent`: canonical_repr (symbols/given/constraints/target/type)

Reasoning
- `SchemaRetrieverAgent`: schemas[] (named canonical patterns)
- `StrategyEnumeratorAgent`: strategies[] (micro‑plan names)
- `PreconditionCheckerAgent`: ok/reasons per strategy
- `StepDecomposerAgent`: plan_steps[{id, action, args}] (planning‑only; no computed results)
- `NextActionAgent`: next_step (one atomic action at a time)

Calculation
- `RewriteAgent`: applies one atomic algebraic action and returns new_relations
- `SubstituteAgent`/`SimplifyAgent`: isolated transforms when needed
- `VerifyAgent`: checks a candidate by substitution

Micro‑QA
- `MicroQAAgent`: step‑specific minimal post‑condition checks (shape/types/presence)
- Orchestrator pre‑checks (deterministic) for plan policy violations
- Retries: QA failure reason is injected as `qa_feedback` for the next attempt

Deterministic Helpers
---------------------
- `simplify_expr`, `evaluate_numeric`: SymPy‑backed simplification and numeric finishing
- `verify_candidate`: equality/inequality substitution checks for a candidate
- `rewrite_relations`: atomic “both‑sides” operations (add/sub/mul/div), substitute, expand, factor, simplify
- `solve_for`: direct SymPy solve for the inferred target variable

Plan Lint (Policy Tester)
-------------------------
- Library: `micro_solver.plan_policy.lint_plan(steps)` → `{ok, issues}`
- CLI: `micro-plan-lint plan.json` or `python -m micro_solver.plan_lint plan.json`
- Enforces “planning‑only” decomposition (no computed results or numeric‑only lists in args)

Why Steps 13–16 If Execute‑Plan Computes?
----------------------------------------
- Execute‑plan executes the atomic plan and updates relations, but it may not fully isolate the target or normalize a final value.
- The finishing steps act as a robust tail:
  - `solve_sympy`: try a direct symbolic solve for the target; if no clear target, attempt solving for any symbol that becomes fully determined (language‑agnostic).
  - `extract_candidate`: language‑agnostic selection that prioritizes numeric‑evaluable expressions from the latest relations; falls back to equality RHS or last relation.
  - `simplify_candidate_sympy`: canonicalize and numerically evaluate when fully determined.
  - `verify_sympy`: confirm the candidate satisfies the relations; fall back to `VerifyAgent` if needed.
- Early exit is enabled: if `final_answer` is already set after execute‑plan (or any later step), the runner stops immediately, so the finishing steps run only when needed.

Design Goals & Limits
---------------------
- Goals: transparency, atomicity, retries with feedback, deterministic fallbacks, minimal per‑agent burden
- Known limits: deep calculus/geometry/DEs/optimization still need specialized micro‑ops and assumptions; text‑only geometry often needs a coordinate embedding step

Extend It
---------
- Add micro‑ops: create new deterministic rewrites or agent tools (e.g., gcd/divisors/mod operations) and reference them in plan steps.
- Tighten QA: add more step‑specific pre‑checks where deterministic verification is possible.
- Parsing: add LaTeX→SymPy fallback to the tokenizer/relations path for richer input formats.
