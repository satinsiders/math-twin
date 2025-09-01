from __future__ import annotations

from agents import Agent  # type: ignore

# ----------------------------- Calculation ---------------------------
RewriteAgent = Agent(
    name="RewriteAgent",
    instructions=(
        "Input: JSON {relations, step:{action,args}}. Output: EXACTLY ONE JSON with keys: new_relations:[string]. "
        "Apply the atomic algebraic rewrite implied by the step to produce updated relations. Do not simplify beyond the action."
    ),
    model="gpt-5-nano",
)
RewriteAgent.reasoning_effort = "high"

# Executes atomic plan steps that are not pure algebraic rewrites
# (e.g., 'state_dp', 'transfer_matrix_step', 'count_configurations').
# Keeps outputs as updated relations without emitting final narrative.
StepExecutorAgent = Agent(
    name="StepExecutorAgent",
    instructions=(
        "Input: JSON {relations:[string], step:{action:string, args:object}}. Output: EXACTLY ONE JSON {new_relations_delta?:[string], new_relations?:[string], env_delta?:object}. "
        "Perform ONE atomic transformation implied by the action (e.g., change_of_base, telescope_sequence, multiply_ratios, normalize_rational, gcd, define_output). "
        "Rules: (1) Do not leap to a final answer unless the action is terminal (e.g., 'define_output'). (2) Emit only the delta of new/updated relations in 'new_relations_delta' (preferred). If you must return a full set, also include it under 'new_relations'. (3) Always include 'env_delta' for any symbol/value bindings you created (e.g., {'L': '3', 'P_total': '93/13', 'm': 93, 'n': 13}). "
        "(4) Keep changes minimal and faithful; no commentary."
    ),
    model="gpt-5-nano",
)
StepExecutorAgent.reasoning_effort = "high"

# Synthesizes a candidate answer from accumulated relations and plan context.
CandidateSynthesizerAgent = Agent(
    name="CandidateSynthesizerAgent",
    instructions=(
        "Input: JSON {relations:[string], goal?:string, problem_type?:string, plan_steps?:[{action:string}]}. Output: EXACTLY ONE JSON {candidate:string}. "
        "Emit ONE expression that best represents the final result implied by the relations. The candidate MUST be a concrete numeric expression or an expression over symbols that already appear in the relations; do NOT invent placeholder functions or new symbols (e.g., avoid 'F(x,y)' or 'awaiting input'). "
        "Prefer a single numeric value when evaluable. If plan steps include counting actions (e.g., 'count_configurations', 'state_dp', 'transfer_matrix_step', 'casework'), the candidate MUST be a non‑negative integer and you MUST ensure it is derivable from the relations. No commentary."
    ),
    model="gpt-5-nano",
)
CandidateSynthesizerAgent.reasoning_effort = "high"

SubstituteAgent = Agent(
    name="SubstituteAgent",
    instructions=(
        "Input: JSON {expression:string, env:object}. Output: EXACTLY ONE string expression with substitutions applied (no evaluation)."
    ),
    model="gpt-5-nano",
)

SimplifyAgent = Agent(
    name="SimplifyAgent",
    instructions=(
        "Input: ONE expression string. Output: ONE equivalent expression string that is simpler. No commentary."
    ),
    model="gpt-5-nano",
)

VerifyAgent = Agent(
    name="VerifyAgent",
    instructions=(
        "Input: JSON {relations, candidate, goal?, problem_type?}. Output: EXACTLY ONE JSON {ok:boolean, detail:string}. "
        "Check by substitution whether candidate satisfies the relations; for inequalities, check within tolerance conceptually. "
        "If the goal implies counting ('count', 'number of'), require the candidate to be a non‑negative integer and supported by relations that define a 'count' or equivalent expression; otherwise return ok=false."
    ),
    model="gpt-5-nano",
)
VerifyAgent.reasoning_effort = "high"

