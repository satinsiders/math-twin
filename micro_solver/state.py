from __future__ import annotations

"""State model for the micro-solver.

This refactors the previous flat :class:`MicroState` into a hierarchy of
five dictionaries representing ``R`` (representations), ``C`` (constraints),
``V`` (variables and working memory), ``A`` (answers) and ``M`` (metrics).

Legacy attribute access is preserved through property adapters so existing
callers that expect the old flat fields continue to work.  New code should use
the nested dictionaries directly, e.g. ``state.R['symbolic']['tokens']`` or
``state.C['symbolic']``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MicroState:
    """Blackboard state for the micro‑solver.

    The state captures recognition artifacts (tokens, variables, relations),
    reasoning artifacts (candidate schemas, strategies, plan), and calculation
    artifacts (intermediate expressions, numeric evaluations, final answer).

    State data is grouped into five top level buckets following the R/C/V/A/M
    plan:

    ``R`` – representations for symbolic/numeric/alternative views
    ``C`` – constraints/relations for each representation
    ``V`` – variables, environment and derived data
    ``A`` – candidate and final answers
    ``M`` – solver metrics

    ``problem_text`` and a handful of orchestration hints remain at the top
    level for convenience.
    """

    # ------------------------------------------------------------------
    # raw inputs
    problem_text: str = ""

    # ------------------------------------------------------------------
    # R/C/V/A/M containers with dual (symbolic/numeric/alt) representations
    R: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"symbolic": {}, "numeric": {}, "alt": {}}
    )
    C: Dict[str, List[str]] = field(
        default_factory=lambda: {"symbolic": [], "numeric": [], "alt": []}
    )
    V: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            rep: {
                "variables": [],
                "constants": [],
                "identifiers": [],
                "points": [],
                "functions": [],
                "parameters": [],
                "quantities": [],
                "env": {},
                "derived": {},
            }
            for rep in ("symbolic", "numeric", "alt")
        }
    )
    A: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            rep: {
                "candidates": [],
                "final": None,
                "explanation": None,
                "intermediate": [],
                "certificate": None,
            }
            for rep in ("symbolic", "numeric", "alt")
        }
    )
    M: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Solver control and reasoning artifacts (unchanged layout)
    representations: List[str] = field(default_factory=lambda: ["symbolic", "numeric"])
    representation: str = "symbolic"
    numeric_seed: float = 0.0
    case_splits: List[List[str]] = field(default_factory=list)
    active_case: int = 0

    goal: Optional[str] = None  # e.g. "solve for x"
    problem_type: Optional[str] = None  # e.g. "linear", "quadratic"

    schemas: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    chosen_strategy: Optional[str] = None
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step_idx: int = 0

    # Orchestration / diagnostics
    qa_feedback: Optional[str] = None
    error: Optional[str] = None
    skip_qa: bool = False
    next_steps: Optional[List] = None

    # ------------------------------------------------------------------
    # Property adapters for legacy flat fields
    # Representations ---------------------------------------------------
    @property
    def normalized_text(self) -> Optional[str]:
        return self.R["symbolic"].get("normalized_text")

    @normalized_text.setter
    def normalized_text(self, value: Optional[str]) -> None:
        self.R["symbolic"]["normalized_text"] = value

    @property
    def sentences(self) -> List[str]:
        return self.R["symbolic"].setdefault("sentences", [])

    @sentences.setter
    def sentences(self, value: List[str]) -> None:
        self.R["symbolic"]["sentences"] = value

    @property
    def tokens(self) -> List[str]:
        return self.R["symbolic"].setdefault("tokens", [])

    @tokens.setter
    def tokens(self, value: List[str]) -> None:
        self.R["symbolic"]["tokens"] = value

    @property
    def tokens_per_sentence(self) -> List[List[str]]:
        return self.R["symbolic"].setdefault("tokens_per_sentence", [])

    @tokens_per_sentence.setter
    def tokens_per_sentence(self, value: List[List[str]]) -> None:
        self.R["symbolic"]["tokens_per_sentence"] = value

    @property
    def canonical_repr(self) -> Optional[Dict[str, Any]]:
        return self.R["symbolic"].get("canonical_repr")

    @canonical_repr.setter
    def canonical_repr(self, value: Optional[Dict[str, Any]]) -> None:
        self.R["symbolic"]["canonical_repr"] = value

    # Variables ---------------------------------------------------------
    def _v(self, key: str) -> Any:
        return self.V["symbolic"].setdefault(key, [])

    def _set_v(self, key: str, value: Any) -> None:
        self.V["symbolic"][key] = value

    @property
    def quantities(self) -> List[Dict[str, Any]]:
        return self.V["symbolic"].setdefault("quantities", [])

    @quantities.setter
    def quantities(self, value: List[Dict[str, Any]]) -> None:
        self.V["symbolic"]["quantities"] = value

    @property
    def variables(self) -> List[str]:
        return self._v("variables")

    @variables.setter
    def variables(self, value: List[str]) -> None:
        self._set_v("variables", value)

    @property
    def constants(self) -> List[str]:
        return self._v("constants")

    @constants.setter
    def constants(self, value: List[str]) -> None:
        self._set_v("constants", value)

    @property
    def identifiers(self) -> List[str]:
        return self._v("identifiers")

    @identifiers.setter
    def identifiers(self, value: List[str]) -> None:
        self._set_v("identifiers", value)

    @property
    def points(self) -> List[str]:
        return self._v("points")

    @points.setter
    def points(self, value: List[str]) -> None:
        self._set_v("points", value)

    @property
    def functions(self) -> List[str]:
        return self._v("functions")

    @functions.setter
    def functions(self, value: List[str]) -> None:
        self._set_v("functions", value)

    @property
    def parameters(self) -> List[str]:
        return self._v("parameters")

    @parameters.setter
    def parameters(self, value: List[str]) -> None:
        self._set_v("parameters", value)

    @property
    def env(self) -> Dict[str, Any]:
        return self.V["symbolic"].setdefault("env", {})

    @env.setter
    def env(self, value: Dict[str, Any]) -> None:
        self.V["symbolic"]["env"] = value

    @property
    def derived(self) -> Dict[str, Any]:
        return self.V["symbolic"].setdefault("derived", {})

    @derived.setter
    def derived(self, value: Dict[str, Any]) -> None:
        self.V["symbolic"]["derived"] = value

    # Constraints -------------------------------------------------------
    @property
    def relations(self) -> List[str]:
        return self.C["symbolic"]

    @relations.setter
    def relations(self, value: List[str]) -> None:
        self.C["symbolic"] = value

    @property
    def equations(self) -> List[str]:
        return self.C["symbolic"]

    @equations.setter
    def equations(self, value: List[str]) -> None:
        self.C["symbolic"] = value

    # Answers -----------------------------------------------------------
    @property
    def intermediate(self) -> List[Dict[str, Any]]:
        return self.A["symbolic"].setdefault("intermediate", [])

    @intermediate.setter
    def intermediate(self, value: List[Dict[str, Any]]) -> None:
        self.A["symbolic"]["intermediate"] = value

    @property
    def candidate_answers(self) -> List[Any]:
        return self.A["symbolic"].setdefault("candidates", [])

    @candidate_answers.setter
    def candidate_answers(self, value: List[Any]) -> None:
        self.A["symbolic"]["candidates"] = value

    @property
    def final_answer(self) -> Any:
        return self.A["symbolic"].get("final")

    @final_answer.setter
    def final_answer(self, value: Any) -> None:
        self.A["symbolic"]["final"] = value

    @property
    def final_explanation(self) -> Optional[str]:
        return self.A["symbolic"].get("explanation")

    @final_explanation.setter
    def final_explanation(self, value: Optional[str]) -> None:
        self.A["symbolic"]["explanation"] = value

    @property
    def certificate(self) -> Optional[Dict[str, Any]]:
        return self.A["symbolic"].get("certificate")

    @certificate.setter
    def certificate(self, value: Optional[Dict[str, Any]]) -> None:
        self.A["symbolic"]["certificate"] = value

    # Metrics -----------------------------------------------------------
    def _m_get(self, key: str, default: Any = 0) -> Any:
        return self.M.get(key, default)

    def _m_set(self, key: str, value: Any) -> None:
        self.M[key] = value

    @property
    def eq_count(self) -> int:
        return self._m_get("eq_count", 0)

    @eq_count.setter
    def eq_count(self, value: int) -> None:
        self._m_set("eq_count", value)

    @property
    def ineq_count(self) -> int:
        return self._m_get("ineq_count", 0)

    @ineq_count.setter
    def ineq_count(self, value: int) -> None:
        self._m_set("ineq_count", value)

    @property
    def jacobian_rank(self) -> int:
        return self._m_get("jacobian_rank", 0)

    @jacobian_rank.setter
    def jacobian_rank(self, value: int) -> None:
        self._m_set("jacobian_rank", value)

    @property
    def degrees_of_freedom(self) -> int:
        return self._m_get("degrees_of_freedom", 0)

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, value: int) -> None:
        self._m_set("degrees_of_freedom", value)

    @property
    def needs_replan(self) -> bool:
        return self._m_get("needs_replan", False)

    @needs_replan.setter
    def needs_replan(self, value: bool) -> None:
        self._m_set("needs_replan", value)

    @property
    def progress_score(self) -> float:
        return self._m_get("progress_score", 0.0)

    @progress_score.setter
    def progress_score(self, value: float) -> None:
        self._m_set("progress_score", value)

    @property
    def stalls(self) -> int:
        return self._m_get("stalls", 0)

    @stalls.setter
    def stalls(self, value: int) -> None:
        self._m_set("stalls", value)

    @property
    def violations(self) -> int:
        return self._m_get("violations", 0)

    @violations.setter
    def violations(self, value: int) -> None:
        self._m_set("violations", value)

    # ------------------------------------------------------------------
    # Migration helpers -------------------------------------------------
    @classmethod
    def from_legacy(cls, data: Dict[str, Any]) -> "MicroState":
        """Construct a state from a flat dictionary of legacy fields."""

        ms = cls()
        for k, v in data.items():
            if hasattr(ms, k):
                try:
                    setattr(ms, k, v)
                except Exception:
                    pass
        return ms

    def to_legacy(self) -> Dict[str, Any]:
        """Return a flattened representation for backwards compatibility."""

        return {
            "problem_text": self.problem_text,
            "normalized_text": self.normalized_text,
            "sentences": self.sentences,
            "tokens": self.tokens,
            "tokens_per_sentence": self.tokens_per_sentence,
            "quantities": self.quantities,
            "variables": self.variables,
            "constants": self.constants,
            "identifiers": self.identifiers,
            "points": self.points,
            "functions": self.functions,
            "parameters": self.parameters,
            "relations": self.relations,
            "goal": getattr(self, "goal", None),
            "problem_type": getattr(self, "problem_type", None),
            "canonical_repr": self.canonical_repr,
            "env": self.env,
            "derived": self.derived,
            "candidate_answers": self.candidate_answers,
            "final_answer": self.final_answer,
            "final_explanation": self.final_explanation,
        }

