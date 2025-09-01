from __future__ import annotations

from agents import Agent  # type: ignore

# ----------------------------- Recognition -----------------------------
TokenizerAgent = Agent(
    name="TokenizerAgent",
    instructions=(
        "Input: raw problem text. Output: EXACTLY ONE minified JSON with keys: "
        "sentences:[string], tokens_per_sentence:[[string]], tokens:[string]. "
        "Rules: (1) Split sentences conservatively on '.', '?', '!' (preserve math). "
        "(2) Tokenize each sentence into math-aware tokens: keep numbers, variables, and operators separate "
        "(e.g., '2x+3=11' -> ['2','x','+','3','=','11']). (3) tokens_per_sentence MUST be an array of arrays, one token list per sentence, "
        "same order/length as sentences. (4) tokens is the flat concatenation of tokens_per_sentence. "
        "No commentary; double-quoted keys/values only."
    ),
    model="gpt-5-nano",
)

EntityExtractorAgent = Agent(
    name="EntityExtractorAgent",
    instructions=(
        "Input: JSON {sentences, tokens, text?}. Output: EXACTLY ONE JSON with keys: "
        "variables:[string], constants:[string], quantities:[{value:string|number, unit?:string, sentence_idx:int}], "
        "identifiers?:[string], points?:[string], functions?:[string], parameters?:[string]. "
        "Rules: variables are algebraic unknowns (x,y,n,k). constants are named constants (pi,e) or fixed labels that take numeric values. "
        "quantities are literal numbers with optional units; include sentence_idx. identifiers/points cover labeled objects in text (A,B,C, lines), but do not duplicate them into variables. functions are named functions (f,g). parameters are named symbols treated like constants unless solved later. Keep minimal and consistent."
    ),
    model="gpt-5-nano",
)
EntityExtractorAgent.reasoning_effort = "high"

RelationExtractorAgent = Agent(
    name="RelationExtractorAgent",
    instructions=(
        "Input: JSON {sentences, tokens, text?}. Output: EXACTLY ONE JSON with keys: relations:[string]. "
        "Extract explicit equations/inequalities and definitions as algebraic relations. For geometry, express constraints in analytic form by introducing symbolic coordinates for points and parameters when needed (e.g., A=(0,0), O2=(d,0)), distances via (x2-x1)^2+(y2-y1)^2, perpendicular via dot product zero, diameters and radii as equalities, and areas with standard formulas. Keep relations minimal and consistent; prefer equations over prose."
    ),
    model="gpt-5-nano",
)
RelationExtractorAgent.reasoning_effort = "high"

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
RepresentationAgent.reasoning_effort = "high"

