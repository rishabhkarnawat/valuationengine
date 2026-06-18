"""Input validation, fetch error handling, and loading UI for the dashboard."""

from __future__ import annotations

import html
from typing import Callable

MAX_COMPARE_TICKERS = 5

MSG_TICKER_NOT_FOUND = "Ticker not found. Please check spelling and try again."
MSG_API_FAILURE = "Unable to fetch data. Try again in a moment."


def parse_tickers(raw: str, max_tickers: int = MAX_COMPARE_TICKERS) -> tuple[list[str], list[str]]:
    """
    Parse comma-separated tickers: trim, uppercase, deduplicate, cap count.

    Returns:
        (tickers, warnings) where warnings are user-facing info messages.
    """
    seen: set[str] = set()
    tickers: list[str] = []
    warnings: list[str] = []

    for part in raw.split(","):
        ticker = part.strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            warnings.append(f"Duplicate ticker **{ticker}** removed.")
            continue
        seen.add(ticker)
        tickers.append(ticker)

    if len(tickers) > max_tickers:
        dropped = tickers[max_tickers:]
        tickers = tickers[:max_tickers]
        warnings.append(
            f"Showing the first **{max_tickers}** tickers only "
            f"(dropped {', '.join(dropped)})."
        )

    return tickers, warnings


def classify_fetch_error(exc: Exception, ticker: str) -> str:
    """Map exceptions to user-friendly dashboard messages."""
    if isinstance(exc, ValueError):
        msg = str(exc).lower()
        not_found_signals = (
            "could not fetch data for ticker",
            "revenue history is empty",
            "could not determine",
            "invalid",
            "not found",
        )
        if any(signal in msg for signal in not_found_signals):
            return MSG_TICKER_NOT_FOUND

    transient_types = (ConnectionError, TimeoutError, OSError)
    try:
        import requests

        transient_types = transient_types + (requests.RequestException,)
    except ImportError:
        pass

    if isinstance(exc, transient_types):
        return MSG_API_FAILURE

    # yfinance / network glitches often surface as generic exceptions
    generic_msg = str(exc).lower()
    if any(
        token in generic_msg
        for token in ("timeout", "connection", "network", "ssl", "429", "503", "502")
    ):
        return MSG_API_FAILURE

    # Default: treat unknown errors as transient API issues
    return MSG_API_FAILURE


def fetch_companies_batch(
    tickers: list[str],
    fetch_fn: Callable[[str], object],
    status_update: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    """
    Fetch multiple tickers with optional status updates.

    Returns:
        (companies_by_ticker, errors_by_ticker)
    """
    companies: dict[str, object] = {}
    errors: dict[str, str] = {}
    total = len(tickers)

    for index, ticker in enumerate(tickers, start=1):
        if status_update:
            status_update(f"Fetching **{index}** of **{total}**: **{ticker}**…")
        try:
            companies[ticker] = fetch_fn(ticker)
        except Exception as exc:
            errors[ticker] = classify_fetch_error(exc, ticker)

    return companies, errors
