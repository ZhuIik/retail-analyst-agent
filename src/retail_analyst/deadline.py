"""A shared monotonic deadline, including retries and BigQuery submission."""

from contextvars import ContextVar
from time import monotonic

turn_deadline: ContextVar[float | None] = ContextVar("turn_deadline", default=None)


def remaining_timeout(cap: float) -> float:
    end = turn_deadline.get()
    if end is None:
        return cap
    remaining = end - monotonic()
    if remaining <= 0:
        raise TimeoutError("The turn deadline is exhausted")
    return min(cap, remaining)
