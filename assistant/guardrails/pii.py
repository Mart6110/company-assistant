"""PII and secret detection with reversible redaction.

Personal data is replaced by placeholders such as `<EMAIL_1>` before text reaches a model
provider, a log, or the database. The per-conversation vault maps placeholders back, so
tool calls receive real values and the user sees real values in answers.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    value: str


def _luhn_ok(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        checksum += d
    return checksum % 10 == 0


def _iban_ok(value: str) -> bool:
    s = value.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def _cpr_ok(value: str) -> bool:
    digits = value.replace("-", "")
    day, month, yy = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
    try:
        dt.date(2000 + yy if yy < 37 else 1900 + yy, month, day)
    except ValueError:
        return False
    return True


_DATE_LIKE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")


def _phone_ok(value: str) -> bool:
    if _DATE_LIKE.match(value.strip()):
        return False
    return 8 <= sum(c.isdigit() for c in value) <= 15


# Order matters: earlier detectors claim their span first.
_DETECTORS: list[tuple[str, re.Pattern[str], Callable[[str], bool] | None]] = [
    (
        "SECRET",
        re.compile(
            r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[abposr]-[A-Za-z0-9-]{10,}"
            r"|gh[pousr]_[A-Za-z0-9]{36,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
        ),
        None,
    ),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), _iban_ok),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _luhn_ok),
    ("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    ("DK_CPR", re.compile(r"\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}-?\d{4}\b"), _cpr_ok),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
    (
        "PHONE",
        re.compile(r"(?<![\w-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]?\d{2,4}){2,4}(?![\w-])"),
        _phone_ok,
    ),
]

_PLACEHOLDER = re.compile(r"<([A-Z_]+)_(\d+)>")


def find_pii(text: str) -> list[PIIMatch]:
    claimed: list[tuple[int, int]] = []
    found: list[PIIMatch] = []
    for kind, pattern, validator in _DETECTORS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < e and end > s for s, e in claimed):
                continue
            if validator and not validator(m.group()):
                continue
            claimed.append((start, end))
            found.append(PIIMatch(kind, start, end, m.group()))
    return sorted(found, key=lambda x: x.start)


class PIIVault:
    """Placeholder <-> value mapping for one conversation."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self._by_placeholder: dict[str, str] = dict(mapping or {})
        self._by_value = {v: k for k, v in self._by_placeholder.items()}

    def to_dict(self) -> dict[str, str]:
        return dict(self._by_placeholder)

    def __len__(self) -> int:
        return len(self._by_placeholder)

    def _placeholder_for(self, kind: str, value: str) -> str:
        if value in self._by_value:
            return self._by_value[value]
        n = 1 + sum(1 for p in self._by_placeholder if p.startswith(f"<{kind}_"))
        placeholder = f"<{kind}_{n}>"
        self._by_placeholder[placeholder] = value
        self._by_value[value] = placeholder
        return placeholder

    def redact(self, text: str) -> tuple[str, list[PIIMatch]]:
        matches = find_pii(text)
        out, pos = [], 0
        for m in matches:
            out.append(text[pos : m.start])
            out.append(self._placeholder_for(m.kind, m.value))
            pos = m.end
        out.append(text[pos:])
        return "".join(out), matches

    def restore(self, text: str) -> str:
        return _PLACEHOLDER.sub(lambda m: self._by_placeholder.get(m.group(0), m.group(0)), text)

    def restore_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.restore(obj)
        if isinstance(obj, dict):
            return {k: self.restore_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.restore_obj(v) for v in obj]
        return obj
