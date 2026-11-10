"""Deterministic lexical retrieval of golden examples (BM25 over question + tags).

Why BM25 and not embeddings in the prototype: the bucket is small, questions share business
vocabulary (churn, revenue, quarter), and BM25 needs no model call, no network and no new
dependency, so retrieval is free, instant and reproducible in tests. The production design is
hybrid (BM25 + embeddings, fused by rank) to catch paraphrases with no shared words; tags are
the prototype's substitute: analysts add the synonyms users actually type.

Score normalisation (so one threshold works for every query): the raw BM25 score is divided by
the score an *ideal* document would get - one of average length containing every query term
once, which is simply the sum of the query terms' IDF - and capped at 1. The result reads as
"the IDF-weighted share of the query this example covers". Query words the bucket has never
seen get the highest IDF, so a question about something the bucket does not cover ("write a
poem") scores near zero instead of matching on one common word.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from retail_analyst.golden.trios import GoldenTrio

K1 = 1.2  # standard BM25 term-frequency saturation
B = 0.75  # standard BM25 length normalisation

# Words that carry no topic in analytics questions; removed so chatty phrasing does not dilute scores.
_STOPWORD_TEXT = """
    a about after all also an and any are as at be been before by can could did do does for from
    get give had has have how i in into is it its me my of on or our over per please show so tell than
    that the their them then there these they this those to us vs was we were what when which who why
    will with would you your
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
# "why" is a stopword: it says *that* the user wants an explanation, not *what about* - the
# example's topic words decide the match.


def tokenize(text: str) -> list[str]:
    """Lowercase words without stopwords, with a crude plural strip (customers -> customer)."""
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _STOPWORDS or len(word) < 2:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


@dataclass(frozen=True)
class GoldenMatch:
    trio: GoldenTrio
    score: float  # normalised to 0..1


class GoldenRetriever:
    """BM25 index over `split: example` trios only: test trios cannot leak into the prompt."""

    def __init__(self, trios: list[GoldenTrio]) -> None:
        self._trios = [t for t in trios if t.split == "example"]
        self._docs = [Counter(tokenize(" ".join([t.question, *t.tags]))) for t in self._trios]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._docs else 1.0
        df = Counter(term for doc in self._docs for term in doc)
        self._n = len(self._docs)
        self._df = df

    def _idf(self, term: str) -> float:
        # Lucene's non-negative BM25 IDF; an unseen term gets the maximum.
        n = self._df.get(term, 0)
        return math.log(1 + (self._n - n + 0.5) / (n + 0.5))

    def score(self, question: str) -> list[GoldenMatch]:
        """Every example with its normalised score, best first; ties broken by id (deterministic)."""
        terms = set(tokenize(question))
        ideal = sum(self._idf(t) for t in terms)
        if not terms or not self._docs or ideal <= 0:
            return []
        matches = []
        for trio, doc, length in zip(self._trios, self._docs, self._lengths, strict=True):
            raw = 0.0
            for term in terms:
                if tf := doc.get(term, 0):
                    norm = 1 - B + B * length / self._avg_len
                    raw += self._idf(term) * tf * (K1 + 1) / (tf + K1 * norm)
            matches.append(GoldenMatch(trio, min(1.0, raw / ideal)))
        return sorted(matches, key=lambda m: (-m.score, m.trio.id))

    def retrieve(self, question: str, top_k: int, min_score: float) -> list[GoldenMatch]:
        """Up to top_k examples scoring at least min_score. Below the threshold nothing is
        returned: a loosely related example would steer the model wrong more than help it."""
        return [m for m in self.score(question) if m.score >= min_score][:top_k]
