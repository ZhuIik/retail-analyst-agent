"""format_answer and the non-blocking number check (D15)."""

from __future__ import annotations

from retail_analyst.agent.answer import check_numbers, format_answer
from retail_analyst.contracts import AnswerSubmission, Fact


def _answer(*facts: Fact, **kw) -> AnswerSubmission:
    return AnswerSubmission(
        kind=kw.pop("kind", "answer"), message=kw.pop("message", "Summary."), facts=list(facts), **kw
    )


def test_answer_sections_in_order_and_hypotheses_marked():
    text = format_answer(
        _answer(
            Fact(text="Revenue was 10.", query_ids=["q1"]),
            hypotheses=["Seasonality."],
            recommendations=["Stock more."],
            assumptions=["Completed orders only."],
            offer_save=True,
            suggested_title="Revenue",
        )
    )
    order = [
        "Summary.",
        "**Facts**",
        "- Revenue was 10. [q1]",
        "**Hypotheses** (possible explanations, not verified",
        "- Seasonality.",
        "**Action items**",
        "**Assumptions**",
        'Save this as a report "Revenue"? (yes/no)',
    ]
    positions = [text.index(part) for part in order]
    assert positions == sorted(positions)


def test_other_kinds_show_message_and_assumptions_only():
    text = format_answer(
        _answer(
            Fact(text="ignored"), kind="clarification", message="Which period?", hypotheses=["h"], assumptions=["a"]
        )
    )
    assert text == "Which period?\n\n**Assumptions**\n- a"
    assert format_answer(_answer(kind="refusal", message="I can't help with that.")) == "I can't help with that."


def test_number_check_tolerates_rounding_formats_and_percent():
    queries = {"q1": {"rows": [["Texas", 1234.4567, 0.1249], ["Ohio", "2500000", None]]}, "q2": {"rows": [[777]]}}
    ok = _answer(
        Fact(text="Texas: 1,234.46 (12.5% share), Ohio $2.5M, in 2024.", query_ids=["q1"]),
        Fact(text="Total 777 orders."),  # no citation -> checked against all queries
    )
    assert check_numbers(ok, queries) == []
    bad = _answer(Fact(text="Texas: 1,240 and 777.", query_ids=["q1"]))  # 777 is in q2, not the cited q1
    assert check_numbers(bad, queries) == ["1,240", "777"]


def test_number_check_ignores_query_ids_and_missing_queries():
    assert check_numbers(_answer(Fact(text="See q1 and q12.", query_ids=["q9"])), {}) == []
    assert check_numbers(_answer(Fact(text="Value 50.", query_ids=["q9"])), {}) == ["50"]


def test_number_check_ignores_small_prose_counts():
    fact = Fact(text="The top 3 categories over the last 6 months", query_ids=["q1"])
    assert check_numbers(_answer(fact), {"q1": {"rows": [[1]]}}) == []


def test_number_check_ignores_quoted_labels():
    fact = Fact(text="The top age group is '25-34' and \"Under 25\" is second.", query_ids=["q1"])
    assert check_numbers(_answer(fact), {"q1": {"rows": [[1]]}}) == []
