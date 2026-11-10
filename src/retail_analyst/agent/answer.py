"""User-facing answer text and the non-blocking number check (D15).

The model submits structured fields (`submit_answer`); code assembles the text, so the layout
is consistent, assumptions are always visible and hypotheses can never pass as facts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from retail_analyst.contracts import AnswerSubmission


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def format_answer(answer: AnswerSubmission) -> str:
    parts = [answer.message.strip()]
    if answer.kind == "answer":
        if answer.facts:
            facts = [f.text + (f" [{', '.join(f.query_ids)}]" if f.query_ids else "") for f in answer.facts]
            parts.append("**Facts**\n" + _bullets(facts))
        if answer.hypotheses:
            header = "**Hypotheses** (possible explanations, not verified by the data)"
            parts.append(header + "\n" + _bullets(answer.hypotheses))
        if answer.recommendations:
            parts.append("**Action items**\n" + _bullets(answer.recommendations))
    if answer.assumptions:
        parts.append("**Assumptions**\n" + _bullets(answer.assumptions))
    if answer.offer_save:
        title = f' "{answer.suggested_title}"' if answer.suggested_title else ""
        parts.append(f"Save this as a report{title}? (yes/no)")
    return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------------------
# Number check: are the numbers in facts present in the cited query results?
# --------------------------------------------------------------------------------------

# 1,234.5 | 1234 | 12.5% | $3.4M | 2.1 billion. The look-behind skips ids like "q1", "#12" and dates' parts.
_NUMBER = re.compile(r"(?<![\w.#])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|[kKmMbB]\b|thousand\b|million\b|billion\b)?")
_SCALE = {
    "k": 10**3, "thousand": 10**3, "m": 10**6, "million": 10**6, "b": 10**9, "billion": 10**9,
}  # fmt: skip


def _candidates(match: re.Match[str]) -> list[tuple[Decimal, Decimal]]:
    """(value, tolerance) pairs this token may stand for, allowing for rounding as written."""
    integer, fraction, suffix = match.group(1).replace(",", ""), match.group(2) or "", match.group(3)
    value = Decimal(integer + fraction)
    decimals = len(fraction) - 1 if fraction else 0
    tolerance = Decimal(5) / Decimal(10) ** (decimals + 1)  # half of the last shown digit
    suffix = (suffix or "").lower()
    if suffix == "%":
        # 12.5% may be stored as 12.5 or as the ratio 0.125
        return [(value, tolerance), (value / 100, tolerance / 100)]
    scale = _SCALE.get(suffix, 1)
    return [(value * scale, tolerance * scale)]


def _numeric_values(rows: Iterable[Iterable[Any]]) -> list[Decimal]:
    out: list[Decimal] = []
    for row in rows:
        for cell in row:
            if isinstance(cell, bool) or cell is None:
                continue
            try:
                value = Decimal(str(cell).replace(",", ""))
            except InvalidOperation:
                continue
            if value.is_finite():
                out.append(abs(value))
    return out


# Quoted text is a label from the data ('Under 25', "25-34"), not a number the model reports.
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def _is_year(match: re.Match[str]) -> bool:
    text = match.group(0).strip()
    return text.isdigit() and len(text) == 4 and 1900 <= int(text) <= 2100


def _is_prose_count(match: re.Match[str]) -> bool:
    """'top 3 categories', 'last 6 months': small bare integers are wording, not data."""
    text = match.group(0).strip()
    return text.isdigit() and int(text) <= 12


def check_numbers(answer: AnswerSubmission, queries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Numbers in facts that do not appear (within rounding) in the rows of the cited queries.

    Non-blocking by design (D15): the result goes to the trace only, so false alarms (a number
    the model derived, a period in the text) cost nothing. A fact without citations is checked
    against all queries of the conversation; so is the summary message (seen live: a typo like
    "$24,55.03" in the summary while the fact said $24,555.03).
    """
    unverified: list[str] = []
    texts = [(fact.text, fact.query_ids) for fact in answer.facts]
    if answer.kind == "answer":
        texts.append((answer.message, []))
    for text, query_ids in texts:
        cited = query_ids or list(queries)
        values = _numeric_values(row for q in cited if q in queries for row in queries[q].get("rows", []))
        for match in _NUMBER.finditer(_QUOTED.sub(" ", text)):
            if _is_year(match) or _is_prose_count(match):
                continue
            if not any(abs(v - c) <= tol for c, tol in _candidates(match) for v in values):
                unverified.append(match.group(0).strip())
    return unverified
