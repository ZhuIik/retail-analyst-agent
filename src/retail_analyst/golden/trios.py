"""Golden Bucket trios: Question -> SQL -> Analyst Report, written by human analysts (requirement 1).

The file format is documented at the top of config/golden_trios.yaml. Loading validates the
format strictly (unknown fields, duplicate ids and numbers in reports are errors): a broken
example in the prompt is worse than none, and the caller decides how to degrade.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# Examples are inserted into the prompt inside <data> tags; a field containing the tag could
# close the block early and turn the rest of the example into instructions.
_DATA_TAG = re.compile(r"</?\s*data\b", re.IGNORECASE)
# Money, percentages, decimals, thousands and 4+ digit numbers (amounts, counts, years) are findings.
_NUMERIC_FACT = re.compile(r"[$€£]\s*\d|\d\s*%|\d+\.\d+|\d{1,3}(?:,\d{3})+|\d{4,}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnalystReport(_Strict):
    """How the analyst answered: the approach, never the findings."""

    interpretation: str = Field(min_length=1)
    approach: list[str] = Field(min_length=1)
    definitions: list[str] = []  # metric names from config/glossary.yaml
    what_to_look_for: list[str] = []
    action_item_patterns: list[str] = []

    def texts(self) -> list[str]:
        return [self.interpretation, *self.approach, *self.definitions, *self.what_to_look_for,
                *self.action_item_patterns]  # fmt: skip

    @model_validator(mode="after")
    def _no_numeric_facts(self) -> AnalystReport:
        # A report holds the approach, never the findings: a past result (money, a percentage, a
        # count, a year) would be stale now and could be repeated by the model as a current fact.
        # Small parameters of the approach stay allowed: "90 days", "top 10", "Q1".
        for text in self.texts():
            if match := _NUMERIC_FACT.search(text):
                raise ValueError(f"report must not contain numeric findings ({match.group(0)!r}): {text[:80]!r}")
        return self


class GoldenTrio(_Strict):
    id: str
    question: str = Field(min_length=1)
    tags: list[str] = []
    split: Literal["example", "test"]  # test trios are held out for evaluation, never shown to the model
    sql: str | None  # null when the question needs no query (e.g. "what data is available")
    report: AnalystReport

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"id must be a lowercase slug (letters, digits, dashes): {v!r}")
        return v

    @field_validator("sql")
    @classmethod
    def _strip_sql(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("sql must be null or a non-empty query")
        return v.strip() if v is not None else None

    @model_validator(mode="after")
    def _no_data_tags(self) -> GoldenTrio:
        for text in [self.question, *self.tags, self.sql or "", *self.report.texts()]:
            if _DATA_TAG.search(text):
                raise ValueError(f"trio {self.id}: <data> tags are not allowed in trio text")
        return self


class GoldenBucket(_Strict):
    format_version: Literal[1]
    trios: list[GoldenTrio] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldenBucket:
        seen: set[str] = set()
        for trio in self.trios:
            if trio.id in seen:
                raise ValueError(f"duplicate trio id: {trio.id}")
            seen.add(trio.id)
        return self

    @property
    def examples(self) -> list[GoldenTrio]:
        return [t for t in self.trios if t.split == "example"]


def load_trios(path: Path) -> GoldenBucket:
    """Read and validate the bucket. Raises (OSError, yaml.YAMLError, pydantic.ValidationError)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GoldenBucket.model_validate(raw)
