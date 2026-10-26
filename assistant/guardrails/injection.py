"""Heuristic jailbreak / prompt-injection scoring.

This is the cheap first layer: it runs on every user message, every retrieved chunk and
every tool output. Paraphrased attacks that slip past it are the job of the LLM input
classifier (`classifier.py`); the confirmation gate on side-effecting tools is the backstop
that holds even when both miss.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")

_FLAGS = re.IGNORECASE | re.DOTALL
_ROLE_NOUNS = r"(?:instructions?|prompts?|rules|guidelines|directives|programming|guardrails|restrictions|content\s+polic(?:y|ies))"
_SIGNATURES: list[tuple[str, re.Pattern[str], float]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass|skip)\s+(?:all\s+|any\s+)?(?:of\s+)?"
            r"(?:the\s+|your\s+|these\s+|those\s+|previous\s+|prior\s+|above\s+|earlier\s+|preceding\s+"
            rf"|original\s+|system\s+|developer\s+)+{_ROLE_NOUNS}\b"
            r"|\b(?:ignore|disregard|forget)\s+(?:everything|all)\s+(?:above|before|previously|you\s+were\s+told)",
            _FLAGS,
        ),
        0.7,
    ),
    (
        "new_instructions",
        re.compile(r"\b(?:new|updated|real|actual|true)\s+(?:instructions?|system\s+prompt|rules)\s*[:\-]", _FLAGS),
        0.5,
    ),
    (
        "reveal_prompt",
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|dump|leak|translate|summari[sz]e|paraphrase|encode"
            r"|tell\s+me|what\s+(?:is|are|were|would))\b.{0,40}"
            r"\b(?:system|initial|hidden|original|developer|secret|confidential)\s+"
            r"(?:prompt|instructions?|message|rules|configuration|config)\b"
            r"|\brepeat\s+(?:the\s+)?(?:words|text)\s+above\b",
            _FLAGS,
        ),
        0.6,
    ),
    (
        "prompt_extraction",
        re.compile(
            r"\bstarting\s+with\s+['\"]?you\s+are\b|\b(?:word\s+for\s+word|verbatim)\b"
            r"|\binclude\s+everything\b.{0,40}\babove\b",
            _FLAGS,
        ),
        0.4,
    ),
    (
        "persona_override",
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on,?\s+you|act\s+as|pretend\s+(?:to\s+be|you\s+are)|roleplay\s+as"
            r"|simulate)\b.{0,80}(?:(?-i:\bDAN\b)|\b(?:unrestricted|unfiltered|uncensored|jailbr\w*|no\s+(?:rules"
            r"|restrictions|limits|filters|guidelines)|evil|developer\s+mode|without\s+(?:rules|restrictions|limits)))",
            _FLAGS,
        ),
        0.8,
    ),
    ("dan", re.compile(r"(?-i:\bDAN\b)|\b(?:do\s+anything\s+now|god\s+mode|jailbreak(?:ed)?)\b", _FLAGS), 0.5),
    # Below the flag threshold on its own: there are legitimate questions about developer mode.
    ("developer_mode", re.compile(r"\bdeveloper\s+mode\b", _FLAGS), 0.3),
    (
        "role_spoof",
        re.compile(
            r"(?:^|\n)\s*(?:system|assistant|developer)\s*:\s*\S|<\|?(?:system|im_start|im_end|endoftext)\|?>"
            r"|\[/?(?:INST|SYS)\]|</?(?:system|instructions?|system_prompt)>",
            re.IGNORECASE,
        ),
        0.6,
    ),
    (
        "privilege_claim",
        re.compile(
            r"\b(?:the\s+)?(?:user|employee|requester)\s+is\s+(?:an?\s+)?(?:admin|administrator|superuser|root"
            r"|authori[sz]ed\s+to)\b|\b(?:i\s+am|i'm)\s+(?:the\s+)?(?:ceo|cfo|system\s+administrator)\b.{0,60}"
            r"\b(?:override|bypass|show|give)\b",
            _FLAGS,
        ),
        0.4,
    ),
    (
        "no_restrictions",
        re.compile(
            r"\b(?:without|ignoring|no)\s+(?:any\s+)?(?:rules|restrictions|limitations|filters|safety|guardrails"
            r"|guidelines|censorship|content\s+polic(?:y|ies))\b",
            _FLAGS,
        ),
        0.4,
    ),
    (
        "hypothetical_framing",
        re.compile(
            r"\b(?:hypothetically|imagine|suppose|pretend|in\s+a\s+(?:fictional|hypothetical)\s+(?:world|scenario))\b"
            r".{0,60}\b(?:you|an?\s+AI|an?\s+assistant)\b.{0,40}\b(?:no|without)\s+(?:rules|guidelines"
            r"|restrictions|filters|limits)",
            _FLAGS,
        ),
        0.5,
    ),
    (
        "emotional_roleplay",
        re.compile(
            r"\b(?:act|pretend|roleplay|play)\s+(?:as|to\s+be)\s+my\s+(?:late\s+|deceased\s+|dead\s+)?"
            r"(?:grand(?:ma|mother|father|pa)|mother|father|mom|dad)\b",
            _FLAGS,
        ),
        0.4,
    ),
    (
        "credential_request",
        re.compile(
            r"\b(?:admin|root|system|database|service|hr\s+system)\s+(?:passwords?|credentials|api\s+keys?)\b"
            r"|\b(?:passwords?|credentials|api\s+keys?|tokens?)\s+(?:for|of)\s+(?:the\s+)?(?:admin|root|other"
            r"|another|every|all)\b",
            _FLAGS,
        ),
        0.5,
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(?:send|post|forward|email|upload|exfiltrate|list|dump|export)\b.{0,60}\b(?:all|every|entire|full)\b"
            r".{0,30}\b(?:employees?|salar(?:y|ies)|passwords?|credentials|records|personal\s+data|conversations?"
            r"|chat\s+history|tokens?|api\s+keys?)\b",
            _FLAGS,
        ),
        0.5,
    ),
    (
        "tool_coercion",
        re.compile(
            r"\b(?:call|invoke|use|execute|run|trigger)\s+(?:the\s+)?(?:\w+\s+)?(?:tool|function|action)\b.{0,60}"
            r"\b(?:immediately|right\s+now|without\s+(?:asking|confirm\w*|telling))\b",
            _FLAGS,
        ),
        0.5,
    ),
    (
        "addressed_to_ai",
        re.compile(
            r"\b(?:AI|assistant|chatbot|LLM|language\s+model)\s*(?:reading\s+this|,)?\s*[:,]?\s*"
            r"(?:you\s+must|please\s+now|ignore|disregard|important\s+instruction)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (
        "decode_and_follow",
        re.compile(
            r"\b(?:decode|decrypt|deobfuscate|base64[- ]?decode)\b.{0,40}\b(?:do|follow|execute|run|obey)\b"
            r".{0,20}\b(?:it|this|what\s+it\s+says|the\s+instructions?)\b",
            _FLAGS,
        ),
        0.5,
    ),
    ("encoded_payload", re.compile(r"[A-Za-z0-9+/]{160,}={0,2}"), 0.3),
]


@dataclass
class InjectionVerdict:
    score: float
    signals: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """NFKC-fold look-alike characters and strip zero-width / bidi control characters."""
    return _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))


def score_injection(text: str) -> InjectionVerdict:
    """Combine matched signatures as independent evidence: 1 - prod(1 - w)."""
    text = normalize(text)
    remaining = 1.0
    signals: list[str] = []
    for name, pattern, weight in _SIGNATURES:
        if pattern.search(text):
            signals.append(name)
            remaining *= 1.0 - weight
    return InjectionVerdict(score=round(1.0 - remaining, 3), signals=signals)
