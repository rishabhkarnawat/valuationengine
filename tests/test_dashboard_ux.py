"""Tests for dashboard UX helpers."""

import pytest

from valuationengine.adapters import dashboard_ux as ux


def test_parse_tickers_normalizes_case_and_whitespace():
    tickers, warnings = ux.parse_tickers(" dpz , Dpz , cmG ")
    assert tickers == ["DPZ", "CMG"]
    assert any("Duplicate" in w for w in warnings)


def test_parse_tickers_deduplicates():
    tickers, warnings = ux.parse_tickers("DPZ, DPZ, CMG")
    assert tickers == ["DPZ", "CMG"]
    assert len(warnings) == 1


def test_parse_tickers_max_five():
    tickers, warnings = ux.parse_tickers("A,B,C,D,E,F,G")
    assert len(tickers) == 5
    assert any("first **5**" in w for w in warnings)


def test_classify_fetch_error_not_found():
    exc = ValueError("Could not fetch data for ticker 'FAKE'. Check the symbol.")
    assert ux.classify_fetch_error(exc, "FAKE") == ux.MSG_TICKER_NOT_FOUND


def test_classify_fetch_error_api_failure():
    assert ux.classify_fetch_error(ConnectionError("reset"), "X") == ux.MSG_API_FAILURE
    assert ux.classify_fetch_error(RuntimeError("timeout waiting"), "X") == ux.MSG_API_FAILURE


def test_fetch_companies_batch_partial_success():
    def fetch_fn(ticker: str):
        if ticker == "BAD":
            raise ValueError("Revenue history is empty for ticker 'BAD'.")
        return {"ticker": ticker}

    messages: list[str] = []
    companies, errors = ux.fetch_companies_batch(
        ["OK", "BAD"], fetch_fn, status_update=messages.append
    )
    assert companies == {"OK": {"ticker": "OK"}}
    assert errors["BAD"] == ux.MSG_TICKER_NOT_FOUND
    assert len(messages) == 2
