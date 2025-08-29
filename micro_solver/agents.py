"""Micro‑agents: tiny, single‑responsibility agents with strict IO.

Each agent performs one minimal action and outputs either a minified JSON
object with double‑quoted keys/values and no trailing text, or a plain string
when explicitly noted. These are designed to reduce cognitive load and enable
deterministic micro‑QA after every step.
"""
from __future__ import annotations

from agents import Agent  # type: ignore


# ----------------------------- Recognition -----------------------------
TokenizerAgent = Agent(
    name="TokenizerAgent",
    instructions=(
        "Input: raw problem text. Output: EXACTLY ONE minified JSON with keys: "
        "sentences:[string], tokens:[string]. Rules: split sentences conservatively ('.','?','!'), "
        "keep math tokens (variables, numbers, operators) separate (e.g., '2x+3=11' → ['2','x','+','3','=','11']). "
        "No commentary; double‑quoted keys/values only."
    ),
    model="gpt-5-nano",
)

EntityExtractorAgent = Agent(
    name="EntityExtractorAgent",
    instructions=(
        "Input: JSON {sentences, tokens}. Output: EXACTLY ONE JSON with keys: "
        "variables:[string], constants:[string], quantities:[{value:string, unit?:string, sentence_idx:int}]. "
        "Rules: variables are symbolic placeholders (x,y,n,k); constants are named constants (pi,e) or fixed labels; "
        "quantities are literal numbers in text with optional units; indexes refer to input sentences."
    ),
    model="gpt-5-nano",
)

RelationExtractorAgent = Agent(
    name="RelationExtractorAgent",
    instructions=(
        "Input: JSON {sentences, tokens}. Output: EXACTLY ONE JSON with keys: relations:[string]. "
        "Extract explicit equations/inequalities and definitions present in the text verbatim or with minimal normalization, "
        "e.g., '2x + 3 = 11', 'x > 0', 'A = pi r^2'."
    ),
    model="gpt-5-nano",
)

GoalInterpreterAgent = Agent(
    name="GoalInterpreterAgent",
    instructions=(
        "Input: JSON {sentences}. Output: EXACTLY ONE JSON with keys: goal:string. "
        "Goal states the task as an action + target, e.g., 'solve for x', 'find area', 'compute probability'."
    ),
    model="gpt-5-nano",
)

TypeClassifierAgent = Agent(
    name="TypeClassifierAgent",
    instructions=(
        "Input: JSON {relations, goal}. Output: EXACTLY ONE JSON with keys: problem_type:string. "
        "Choose the most specific from: linear, quadratic, rational, radical, absolute, system_linear, proportion, percent, rate, combinatorics, probability, geometry_similarity, geometry_pythagorean, geometry_circle, sequence, other."
    ),
    model="gpt-5-nano",
)

RepresentationAgent = Agent(
    name="RepresentationAgent",
    instructions=(
        "Input: JSON {variables, constants, quantities, relations, goal, problem_type}. "
        "Output: EXACTLY ONE JSON canonical representation with keys: symbols:[string], given:[string], "
        "constraints:[string], target:string, type:string. Keep minimal, faithful strings and avoid solving."
    ),
    model="gpt-5-nano",
)


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

StrategyEnumeratorAgent = Agent(
    name="StrategyEnumeratorAgent",
    instructions=(
        "Input: JSON {schemas, relations, target}. Output: EXACTLY ONE JSON with keys: strategies:[string]. "
        "Each strategy name describes a micro‑plan (e.g., 'isolate_x_by_add_sub', 'apply_quadratic_formula'). "
        "For counting with 'exactly k' constraints, include strategies that maintain exactness (e.g., 'dynamic_programming_on_edges', 'transfer_matrix_for_grid', 'casework_by_boundary_states', 'construct_recurrence_and_sum'). Do NOT propose parity/GF(2) strategies unless the text explicitly allows even/odd only."
    ),
    model="gpt-5-nano",
)

PreconditionCheckerAgent = Agent(
    name="PreconditionCheckerAgent",
    instructions=(
        "Input: JSON {strategy, relations}. Output: EXACTLY ONE JSON with keys: ok:boolean, reasons:[string]. "
        "Check minimal preconditions to apply the strategy; do not fix; do not suggest. "
        "If the relations/goal indicate 'exactly k' constraints, reject strategies that rely only on parity/GF(2) reductions because they drop cardinality information."
    ),
    model="gpt-5-nano",
)

StepDecomposerAgent = Agent(
    name="StepDecomposerAgent",
    instructions=(
        "Input: JSON {strategy, relations, target}. Output: EXACTLY ONE JSON with keys: plan_steps:[{id:string, action:string, args:object}]. "
        "STRICT rules (planning only): do NOT compute results or enumerate numeric lists. Do NOT include precomputed outputs (forbidden keys: 'result', 'results') or any numeric-only arrays. Keys like 'operands', 'candidates', 'sum', 'total' are allowed when they reference symbolic names (e.g., 'operands': 'valid_colorings_candidates'), not concrete numbers. "
        "Args must be symbolic references (identifiers/expressions) to operate ON, not the computed outcomes. If a step needs a set (e.g., divisors), reference it symbolically (e.g., 'divisors_of': '39') rather than listing numbers. "
        "Each step must be atomic (e.g., 'add both sides', 'divide both sides', 'expand', 'factor', 'substitute', 'gcd', 'state_dp', 'transfer_matrix_step', 'define_boundary_state', 'recurrence', 'substitute', 'sum', 'count_configurations'), with args specifying only inputs by name. Avoid 'modulo_reduction' when the goal demands exact counts. No solving."
    ),
    model="gpt-5-nano",
)

NextActionAgent = Agent(
    name="NextActionAgent",
    instructions=(
        "Input: JSON {plan_steps, current_idx}. Output: EXACTLY ONE JSON with keys: next_step:{id, action, args}. "
        "Return the next atomic step only. No commentary."
    ),
    model="gpt-5-nano",
)


# ----------------------------- Calculation ---------------------------
RewriteAgent = Agent(
    name="RewriteAgent",
    instructions=(
        "Input: JSON {relations, step:{action,args}}. Output: EXACTLY ONE JSON with keys: new_relations:[string]. "
        "Apply the atomic algebraic rewrite implied by the step to produce updated relations. Do not simplify beyond the action."
    ),
    model="gpt-5-nano",
)

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


# ----------------------------- Micro QA -------------------------------
MicroQAAgent = Agent(
    name="MicroQAAgent",
    instructions=(
        "You are a micro‑QA checker. Input: JSON {step:string, data:object, out:any}. "
        "Check ONLY minimal post‑conditions for the given step. Output exactly 'pass' or a one‑sentence failure reason. "
        "General rules: ensure output JSON is present when expected, required keys exist and have correct primitive types, arrays non‑empty when applicable, and no extraneous commentary. "
        "Additional guards for counting tasks (goal mentions 'count' or 'number of'): "
        "- choose_strategy: reject parity/GF(2) strategies when text says 'exactly k' since they drop cardinality. "
        "- decompose: require at least one counting‑appropriate action name (e.g., 'state_dp', 'transfer_matrix_step', 'recurrence', 'count_configurations'). "
        "- extract_candidate/verify: require candidate to be a non‑negative integer and supported by relations naming the count; reject trivial 0 with no justification."
    ),
    model="gpt-5-nano",
)

__all__ = [
    # Recognition
    "TokenizerAgent",
    "EntityExtractorAgent",
    "RelationExtractorAgent",
    "GoalInterpreterAgent",
    "TypeClassifierAgent",
    "RepresentationAgent",
    # Reasoning
    "SchemaRetrieverAgent",
    "StrategyEnumeratorAgent",
    "PreconditionCheckerAgent",
    "StepDecomposerAgent",
    "NextActionAgent",
    # Calculation
    "RewriteAgent",
    "SubstituteAgent",
    "SimplifyAgent",
    "VerifyAgent",
    # QA
    "MicroQAAgent",
]
