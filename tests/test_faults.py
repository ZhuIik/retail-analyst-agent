"""Fault injection raises the exception types the real libraries raise."""

from __future__ import annotations

import pytest

from retail_analyst.bq.client import classify_bq_error
from retail_analyst.contracts import ErrorKind
from retail_analyst.faults import FAULT_MODES, FaultInjector
from retail_analyst.llm import classify_llm_error


def test_unknown_mode_lists_valid_modes() -> None:
    with pytest.raises(ValueError) as info:
        FaultInjector({"llm_429", "meteor"})
    message = str(info.value)
    assert "meteor" in message
    assert all(mode in message for mode in FAULT_MODES)


def test_no_modes_is_a_no_op() -> None:
    faults = FaultInjector(set())
    faults.before_bq_call()
    faults.before_llm_call("any-model")
    assert not faults.empty_result()


def test_bq_timeout_fails_first_two_attempts_only() -> None:
    faults = FaultInjector({"bq_timeout"})
    for _ in range(2):
        with pytest.raises(TimeoutError) as info:
            faults.before_bq_call()
        assert classify_bq_error(info.value) is ErrorKind.TRANSIENT
    faults.before_bq_call()  # third attempt succeeds


def test_bq_quota_is_quota_exhausted() -> None:
    with pytest.raises(Exception) as info:
        FaultInjector({"bq_quota"}).before_bq_call()
    assert classify_bq_error(info.value) is ErrorKind.QUOTA_EXHAUSTED


def test_llm_429_hits_only_the_primary() -> None:
    faults = FaultInjector({"llm_429"})
    with pytest.raises(Exception) as info:
        faults.before_llm_call("primary")
    assert type(info.value).__name__ == "GoogleRateLimitError"
    assert classify_llm_error(info.value) is ErrorKind.TRANSIENT  # per-minute limit -> fallback-worthy
    faults.before_llm_call("fallback", is_primary=False)


def test_llm_down_hits_every_model() -> None:
    faults = FaultInjector({"llm_down"})
    for is_primary in (True, False):
        with pytest.raises(Exception) as info:
            faults.before_llm_call("m", is_primary=is_primary)
        assert classify_llm_error(info.value) is ErrorKind.TRANSIENT


def test_empty_and_active() -> None:
    faults = FaultInjector({"empty"})
    assert faults.empty_result() and faults.active("empty") and not faults.active("llm_429")
