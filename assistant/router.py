"""Model router: a small, cheap model for simple turns, the frontier model for hard ones.

The score combines cheap text features with the input classifier's complexity label
(when the classifier ran). Every decision records its reasons in the trace and prompt log,
which is what you tune the threshold against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from assistant.guardrails.classifier import Classification
from assistant.llm import ModelRef

Tier = Literal["small", "frontier"]

_REASONING = re.compile(
    r"\b(?:compare|comparison|difference between|pros and cons|trade-?offs?|calculate|compute|pro-?rat\w*"
    r"|what if|scenario|explain why|analy[sz]e|draft|write (?:a|an|me)|step[- ]by[- ]step|plan(?:ning)?"
    r"|reconcile|conflict\w*|exception|edge case|both .+ and)\b"
    r"|\bif (?:i|we|my)\b.{0,80}\bhow (?:many|much|long)\b|\bhow (?:many|much|long)\b.{0,80}\bif (?:i|we|my)\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    tier: Tier
    model: ModelRef
    score: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"tier": self.tier, "model": str(self.model), "score": self.score, "reasons": self.reasons}


class ModelRouter:
    def __init__(self, small: ModelRef, frontier: ModelRef, threshold: float = 0.5):
        self.small = small
        self.frontier = frontier
        self.threshold = threshold

    def model_for(self, tier: Tier) -> ModelRef:
        return self.small if tier == "small" else self.frontier

    def alternate(self, decision: RouteDecision) -> ModelRef:
        return self.frontier if decision.tier == "small" else self.small

    def route(
        self,
        query: str,
        *,
        classification: Classification | None = None,
        distinct_docs: int = 0,
        history_turns: int = 0,
        force_tier: Tier | None = None,
    ) -> RouteDecision:
        if force_tier:
            return RouteDecision(force_tier, self.model_for(force_tier), 1.0, ["forced"])

        score, reasons = 0.0, []
        words = len(query.split())
        if words > 60:
            score += 0.25
            reasons.append(f"long message ({words} words)")
        elif words > 30:
            score += 0.1
            reasons.append(f"medium-length message ({words} words)")
        if query.count("?") >= 2:
            score += 0.15
            reasons.append("several questions")
        if _REASONING.search(query):
            score += 0.3
            reasons.append("reasoning or drafting language")
        if distinct_docs >= 3:
            score += 0.15
            reasons.append(f"spans {distinct_docs} policies")
        if history_turns >= 6:
            score += 0.1
            reasons.append("long conversation")
        if classification is not None:
            # The classifier's judgment alone is enough to reach the default threshold.
            if classification.complexity == "complex":
                score += 0.5
                reasons.append("classifier: complex")
            else:
                score -= 0.1
                reasons.append("classifier: simple")

        score = round(min(max(score, 0.0), 1.0), 3)
        tier: Tier = "frontier" if score >= self.threshold else "small"
        return RouteDecision(tier, self.model_for(tier), score, reasons)
