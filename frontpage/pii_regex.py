# Minimal, deterministic PII redaction helpers for Daft pipelines.
#
# This is intentionally regex-first (no LLMs), so it is cheap, reproducible, and easy to reason about.

from __future__ import annotations

import re

import daft

# Note: these are intentionally conservative patterns; they aim to catch common formats with low complexity.
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    flags=re.IGNORECASE,
)

# US-centric phone pattern: +1 optional, area code optional parens, separators space/dot/hyphen.
PHONE_RE = re.compile(
    r"\b(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
)

SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit card candidates (13-19 digits with optional spaces/hyphens). We'll Luhn-check before redacting.
CC_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s)


def _luhn_ok(num: str) -> bool:
    """Return True if `num` (digits-only string) passes the Luhn checksum."""
    if not (13 <= len(num) <= 19) or not num.isdigit():
        return False

    total = 0
    parity = len(num) % 2
    for i, ch in enumerate(num):
        d = ord(ch) - ord("0")
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii_regex(text: str | None) -> tuple[str, list[str]]:
    """Redact a few common PII formats and return (safe_text, pii_types_found)."""
    if text is None:
        return "", []

    safe_text = text
    pii: set[str] = set()

    if EMAIL_RE.search(safe_text):
        pii.add("email")
        safe_text = EMAIL_RE.sub("[EMAIL]", safe_text)

    if SSN_RE.search(safe_text):
        pii.add("ssn")
        safe_text = SSN_RE.sub("[SSN]", safe_text)

    def _cc_sub(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = _digits_only(raw)
        if _luhn_ok(digits):
            pii.add("credit_card")
            return "[CREDIT_CARD]"
        return raw

    safe_text = CC_CANDIDATE_RE.sub(_cc_sub, safe_text)

    if PHONE_RE.search(safe_text):
        pii.add("phone")
        safe_text = PHONE_RE.sub("[PHONE]", safe_text)

    return safe_text, sorted(pii)


RedactionStruct = daft.DataType.struct(
    {
        "safe_text": daft.DataType.string(),
        "pii_types": daft.DataType.list(daft.DataType.string()),
        "has_pii": daft.DataType.bool(),
    }
)


@daft.func(return_dtype=RedactionStruct)
def regex_redact(text: str):
    safe_text, pii_types = redact_pii_regex(text)
    return {"safe_text": safe_text, "pii_types": pii_types, "has_pii": len(pii_types) > 0}

