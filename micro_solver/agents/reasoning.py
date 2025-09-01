from __future__ import annotations

from agents import Agent  # type: ignore

# ----------------------------- Reasoning ------------------------------
SchemaRetrieverAgent = Agent(
    name="SchemaRetrieverAgent",
    instructions=(
        "Input: JSON {type, relations, target}. Output: EXACTLY ONE JSON with keys: schemas:[string]. "
        "List 1–3 named canonical schemas applicable (e.g., 'linear_isolation', 'quadratic_formula', 'proportion_cross_multiply'). No commentary. "
        "If the goal implies counting ('count', 'number of') with 'exactly k' constraints, prefer exact-count schemas such as 'state_dp_counting', 'transfer_matrix_counting', 'casework_counting', or 'graph_degree_constraint_counting'. Avoid parity-only or GF(2) reductions when the text says 'exactly'."
    ),
    model="gpt-5-nano",
)
SchemaRetrieverAgent.reasoning_effort = "high"

StrategyEnumeratorAgent = Agent(
    name="StrategyEnumeratorAgent",
    instructions=(
        "Input: JSON {schemas, relations, target}. Output: EXACTLY ONE JSON with keys: strategies:[string]. "
        "Each strategy name describes a micro‑plan (e.g., 'isolate_x_by_add_sub', 'apply_quadratic_formula'). "
        "For counting with 'exactly k' constraints, include strategies that maintain exactness (e.g., 'dynamic_programming_on_edges', 'transfer_matrix_for_grid', 'casework_by_boundary_states', 'construct_recurrence_and_sum'). Do NOT propose parity/GF(2) strategies unless the text explicitly allows even/odd only."
    ),
    model="gpt-5-nano",
)
StrategyEnumeratorAgent.reasoning_effort = "high"

PreconditionCheckerAgent = Agent(
    name="PreconditionCheckerAgent",
    instructions=(
        "Input: JSON {strategy, relations}. Output: EXACTLY ONE JSON with keys: ok:boolean, reasons:[string]. "
        "Check minimal preconditions to apply the strategy; do not fix; do not suggest. "
        "If the relations/goal indicate 'exactly k' constraints, reject strategies that rely only on parity/GF(2) reductions because they drop cardinality information."
    ),
    model="gpt-5-nano",
)

AtomicPlannerAgent = Agent(
    name="AtomicPlannerAgent",
    instructions=(
        "Input: JSON {relations:[string], goal?:string, canonical_target?:string, env?:object, history?:[{action:string, ok?:boolean, reason?:string}]}. Output: EXACTLY ONE JSON with either {step:{action:string, args:object}} or {steps:[{action:string, args:object}..]}. "
        "Propose the next SINGLE atomic action (or up to 3 candidates) that maximizes progress toward the target. Progress correlates with: more bound symbols in env, fewer free symbols in relations, more numeric‑evaluable expressions, more equality constraints, newly‑solvable symbols (admit numeric solutions), and making the canonical target evaluable. Avoid repeating actions from history that failed for the same reason. Do NOT propose 'assign' unless RHS is numeric‑evaluable under env and structurally justified by the relations. "
        "Allowed actions: 'substitute_env' (apply all numeric env bindings into relations), 'substitute' (with explicit replacements), 'normalize'/'simplify', 'isolate_symbol' (symbol), 'eliminate_symbol' (symbol), 'bind_numeric' (target, expr) when expr becomes numeric. Prefer 'substitute_env' when env is non‑empty. If the target is a combination (e.g., m+n), plan to bind those target symbols first (e.g., bind m and n) using isolate+substitute_env or eliminate. Always emit one of the allowed actions with explicit args."
    ),
    model="gpt-5-nano",
)
AtomicPlannerAgent.reasoning_effort = "high"

