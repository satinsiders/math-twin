from __future__ import annotations

from agents import Agent  # type: ignore

# ----------------------------- Micro QA -------------------------------
MicroQAAgent = Agent(
    name="MicroQAAgent",
    instructions=(
        "You are a micro‑QA checker. Input: JSON {step:string, data:object, out:any}. "
        "Check ONLY minimal post‑conditions for the given step. Output exactly 'pass' or a one‑sentence failure reason. "
        "General rules: ensure output JSON is present when expected, required keys exist and have correct primitive types, arrays non‑empty when applicable, and no extraneous commentary. Do NOT require the final numeric result in intermediate steps — only the 'verify' step judges final correctness. "
        "Step-specific rules: "
        "- goal: require a non‑empty string that states an action + target (e.g., 'find m+n', 'solve for x'); DO NOT require a numeric value. "
        "- tokenize: require sentences:[string] and tokens_per_sentence:[[string]] with equal length; tokens is the flat concatenation. "
        "- entities: require variables/constants/quantities arrays; each quantity has value (string or number) and sentence_idx:int. "
        "- relations: require a non‑empty array of strings (algebraic relations). "
        "- extract_candidate: candidate may be numeric or an expression; do not require justification here (verification handles it). "
        "Additional guards for counting tasks (goal mentions 'count' or 'number of'): "
        "- choose_strategy: reject parity/GF(2) strategies when text says 'exactly k' since they drop cardinality. "
        "- verify: require the final to be a non‑negative integer supported by relations naming the count; reject trivial 0 with no justification."
    ),
    model="gpt-5-nano",
)

