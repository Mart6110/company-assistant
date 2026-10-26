"""Layered guardrails.

Input:  1. normalize + length cap   2. PII/secret redaction   3. heuristic jailbreak/injection
        4. LLM classifier (safety + intent + complexity), run in parallel with retrieval
Context: retrieved chunks and tool outputs are scanned as untrusted data (indirect injection)
Output: system-prompt leakage (canary + n-gram overlap)
Actions: side-effecting tools never run without explicit user confirmation (orchestrator)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal

from assistant.config import Settings
from assistant.guardrails.classifier import Classification, InputClassifier
from assistant.guardrails.injection import InjectionVerdict, normalize, score_injection
from assistant.guardrails.pii import PIIMatch, PIIVault


@dataclass
class GuardrailEvent:
    layer: str
    action: Literal["allow", "flag", "redact", "drop", "block"]
    detail: str
    score: float | None = None

    def as_dict(self) -> dict:
        return {"layer": self.layer, "action": self.action, "detail": self.detail, "score": self.score}


@dataclass
class InputScreen:
    text: str  # normalized and redacted; the only form of the message used downstream
    heuristic: InjectionVerdict
    pii: list[PIIMatch]
    events: list[GuardrailEvent] = field(default_factory=list)
    block_reason: str | None = None
    flagged: bool = False  # suspicious but below the block threshold

    @property
    def blocked(self) -> bool:
        return self.block_reason is not None


class Guardrails:
    def __init__(self, settings: Settings, classifier: InputClassifier | None = None):
        self._settings = settings
        self._classifier = classifier

    @staticmethod
    def canary(seed: str) -> str:
        return "cnry-" + hashlib.sha256(seed.encode()).hexdigest()[:20]

    def screen_input(self, raw: str, vault: PIIVault) -> InputScreen:
        events: list[GuardrailEvent] = []
        text = normalize(raw).strip()
        if len(text) > self._settings.max_input_chars:
            text = text[: self._settings.max_input_chars]
            events.append(GuardrailEvent("length", "flag", f"truncated to {self._settings.max_input_chars} chars"))

        text, pii = vault.redact(text)
        if pii:
            kinds = sorted({m.kind for m in pii})
            events.append(GuardrailEvent("pii", "redact", ", ".join(kinds)))
            if "SECRET" in kinds:
                events.append(GuardrailEvent("pii", "flag", "message contained a credential or API key"))

        verdict = score_injection(text)
        screen = InputScreen(text=text, heuristic=verdict, pii=pii, events=events)
        if verdict.score >= self._settings.injection_block_threshold:
            screen.block_reason = "jailbreak_heuristic"
            events.append(GuardrailEvent("injection_heuristic", "block", ",".join(verdict.signals), verdict.score))
        elif verdict.signals:
            screen.flagged = verdict.score >= self._settings.injection_flag_threshold
            events.append(GuardrailEvent("injection_heuristic", "flag", ",".join(verdict.signals), verdict.score))
        return screen

    async def classify(self, text: str) -> Classification | None:
        if self._classifier is None or not self._settings.guardrails_llm_classifier:
            return None
        return await self._classifier.classify(text)

    def classification_event(self, c: Classification) -> GuardrailEvent:
        return GuardrailEvent(
            "input_classifier",
            "allow" if c.safe else "block",
            f"safety={c.safety} intent={c.intent} complexity={c.complexity}: {c.reason}",
        )

    def screen_untrusted(self, text: str, source: str) -> tuple[bool, GuardrailEvent | None]:
        """Scan retrieved or tool-returned content for embedded instructions. True = suspicious."""
        verdict = score_injection(text)
        if verdict.score >= self._settings.untrusted_content_threshold:
            return True, GuardrailEvent("indirect_injection", "drop", f"{source}: {','.join(verdict.signals)}", verdict.score)
        return False, None

    def check_output(self, text: str, system_prompt: str) -> GuardrailEvent | None:
        canary = self.canary(self._settings.canary_seed)
        if canary in text:
            return GuardrailEvent("output", "block", "canary token from the system prompt appeared in the answer")
        overlap = _shingle_overlap(text, system_prompt)
        if overlap >= 3:
            return GuardrailEvent("output", "block", f"answer reproduces the system prompt ({overlap} shared 8-grams)")
        return None


_WORD = re.compile(r"[a-z0-9']+")


def _shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _shingle_overlap(answer: str, system_prompt: str) -> int:
    return len(_shingles(answer) & _shingles(system_prompt))
