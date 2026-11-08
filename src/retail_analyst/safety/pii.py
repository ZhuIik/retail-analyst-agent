"""Output PII regex (D14): a safety net, not the main protection.

The model never sees real PII (scoped tables have no PII columns, D10). This catches anything
that slips through anyway — in answers, saved reports and trace payloads.

Patterns are deliberately conservative: reports are full of large numbers (revenue, order
counts, customer-id hashes), and redacting those would break answers. Phones must look like
phones (separators or a leading +), card numbers must pass the Luhn check.
"""

from __future__ import annotations

import re

REDACTED = "[redacted]"

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<![\w#-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]\d{2,4}(?:[\s.-]\d{2})?(?![\w.])"
)
_CARD_CANDIDATE = re.compile(r"(?<![\w#-])(?:\d{4}[ -]){3}\d{1,7}(?!\w)|(?<![\w#-])\d{13,19}(?!\w)")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _redact_cards(text: str) -> tuple[str, int]:
    hits = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal hits
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits += 1
            return REDACTED
        return m.group(0)

    return _CARD_CANDIDATE.sub(repl, text), hits


def redact_pii(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_hits)."""
    text, n_email = _EMAIL.subn(REDACTED, text)
    text, n_phone = _PHONE.subn(REDACTED, text)
    text, n_card = _redact_cards(text)
    return text, n_email + n_phone + n_card
