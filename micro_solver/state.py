from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MicroState:
    """Blackboard state for micro‑solver.

    The state captures recognition artifacts (tokens, variables, relations),
    reasoning artifacts (candidate schemas, strategies, plan), and calculation
    artifacts (intermediate expressions, numeric evaluations, final answer).

    Each micro‑step updates a small subset of fields and is followed by a
    micro‑QA check to ensure pre/post‑conditions hold.
    """

    # Raw inputs
    problem_text: str = ""

    # Recognition outputs (atomic)
    normalized_text: Optional[str] = None
    sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    tokens_per_sentence: list[list[str]] = field(default_factory=list)
    quantities: list[dict[str, Any]] = field(default_factory=list)  # {value, unit?, sentence_idx}
    variables: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    points: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)  # e.g., "2x+3=11", "x>0"
    goal: Optional[str] = None  # e.g., "solve for x"
    problem_type: Optional[str] = None  # e.g., "linear", "quadratic", "ratio", "geometry"...
    canonical_repr: Optional[dict[str, Any]] = None  # structured representation

    # Reasoning artifacts (micro‑planned)
    schemas: list[str] = field(default_factory=list)  # matched known schemas by name
    strategies: list[str] = field(default_factory=list)
    chosen_strategy: Optional[str] = None
    plan_steps: list[dict[str, Any]] = field(default_factory=list)  # [{id, action, target, args}]
    current_step_idx: int = 0

    # Working memory (calculation context)
    env: dict[str, Any] = field(default_factory=dict)  # symbol table / bindings
    equations: list[str] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)

    # Meta‑reasoning stats
    eq_count: int = 0
    ineq_count: int = 0
    degrees_of_freedom: int = 0
    needs_replan: bool = False

    # Results
    intermediate: list[dict[str, Any]] = field(default_factory=list)  # trace of {op, in, out}
    candidate_answers: list[Any] = field(default_factory=list)
    final_answer: Optional[Any] = None
    final_explanation: Optional[str] = None

    # Control / diagnostics
    qa_feedback: Optional[str] = None
    error: Optional[str] = None

    # Orchestration hints
    skip_qa: bool = False
    next_steps: Optional[list] = None  # injected micro‑steps for dynamic plans
