"""Fundamentals fetcher via yfinance"""

from __future__ import annotations

import yfinance as yf

from core.models import Company


def fetch_company(ticker: str, history_years: int = 5) -> Company:
    """
    Pull fundamentals from yfinance and return a populated Company.

    Args:
        ticker: Equity ticker symbol.
        history_years: Number of most recent fiscal years to include (oldest first).

    Returns:
        Company populated from yfinance financial statements and market data.
    """
    if history_years <= 0:
        raise ValueError(f"history_years must be positive; got {history_years}.")

    symbol = ticker.strip().upper()
    yf_ticker = yf.Ticker(symbol)
    info = yf_ticker.info or {}

    if not info:
        raise ValueError(
            f"Could not fetch data for ticker '{symbol}'. Check the symbol or try again."
        )

    name = info.get("longName") or info.get("shortName") or symbol
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if current_price is None or current_price <= 0:
        raise ValueError(f"Could not determine a valid current price for '{symbol}'.")

    beta = info.get("beta")
    beta = 1.0 if beta is None else float(beta)

    shares_outstanding = info.get("sharesOutstanding")
    market_cap = info.get("marketCap")
    if shares_outstanding is None:
        if market_cap is None:
            raise ValueError(f"Could not determine shares outstanding for '{symbol}'.")
        shares_outstanding = market_cap / current_price
    shares_outstanding = float(shares_outstanding)
    market_cap = float(market_cap if market_cap is not None else shares_outstanding * current_price)

    financials = yf_ticker.financials
    cashflow = yf_ticker.cashflow
    balance_sheet = yf_ticker.balance_sheet

    revenue = _series_history(financials, ["Total Revenue", "Revenue"], history_years)
    if not revenue:
        raise ValueError(f"Revenue history is empty for ticker '{symbol}'.")

    ebit = _series_history(financials, ["EBIT", "Operating Income"], history_years, len(revenue))
    ebitda_raw = _series_history(financials, ["EBITDA", "Normalized EBITDA"], history_years, len(revenue))
    da = _series_history(
        cashflow,
        ["Depreciation And Amortization", "Depreciation", "Amortization"],
        history_years,
        len(revenue),
    )
    ebitda = _align_ebitda(ebit, ebitda_raw, da, len(revenue))

    tax_provision = _series_history(
        financials, ["Tax Provision", "Income Tax Expense"], history_years, len(revenue)
    )
    pretax_income = _series_history(
        financials,
        ["Pretax Income", "Income Before Tax", "Earnings Before Tax"],
        history_years,
        len(revenue),
    )
    effective_tax_rate = _effective_tax_rate(tax_provision, pretax_income)

    capex = [
        abs(v)
        for v in _series_history(
            cashflow,
            ["Capital Expenditure", "Capital Expenditures"],
            history_years,
            len(revenue),
        )
    ]
    change_in_nwc = [
        -v
        for v in _series_history(
            cashflow,
            ["Change In Working Capital", "Changes In Working Capital"],
            history_years,
            len(revenue),
        )
    ]

    cash = _latest_balance(
        balance_sheet,
        ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"],
    )
    total_debt = _latest_balance(
        balance_sheet,
        ["Total Debt", "Long Term Debt And Capital Lease Obligation"],
    )
    if total_debt is None:
        long_term = _latest_balance(balance_sheet, ["Long Term Debt"]) or 0.0
        current = _latest_balance(balance_sheet, ["Current Debt", "Short Long Term Debt"]) or 0.0
        total_debt = long_term + current

    return Company(
        ticker=symbol,
        name=name,
        revenue=revenue,
        ebit=ebit,
        ebitda=ebitda,
        depreciation_amortization=da,
        capex=capex,
        change_in_nwc=change_in_nwc,
        effective_tax_rate=effective_tax_rate,
        cash=float(cash or 0.0),
        total_debt=float(total_debt or 0.0),
        shares_outstanding=shares_outstanding,
        current_price=float(current_price),
        market_cap=market_cap,
        beta=beta,
    )


def _series_history(
    frame,
    row_names: list[str],
    years: int,
    length: int | None = None,
) -> list[float]:
    target_len = length or years
    if frame is None or frame.empty:
        return [0.0] * target_len

    for name in row_names:
        if name in frame.index:
            values = [float(v) for v in frame.loc[name].tolist() if _is_finite(v)]
            values.reverse()
            if values:
                return _pad_or_trim(values[-years:], target_len)

    return [0.0] * target_len


def _align_ebitda(
    ebit: list[float],
    ebitda_raw: list[float],
    da: list[float],
    length: int,
) -> list[float]:
    if any(v != 0 for v in ebitda_raw):
        return ebitda_raw
    return [e + d for e, d in zip(ebit, da)]


def _latest_balance(frame, row_names: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    for name in row_names:
        if name in frame.index:
            for value in frame.loc[name].tolist():
                if _is_finite(value):
                    return float(value)
    return None


def _effective_tax_rate(tax: list[float], pretax: list[float]) -> float:
    rates = []
    for t, p in zip(tax, pretax):
        if p and p != 0:
            rates.append(max(0.0, min(0.40, t / p)))
    return sum(rates) / len(rates) if rates else 0.25


def _pad_or_trim(values: list[float], length: int) -> list[float]:
    if len(values) >= length:
        return values[-length:]
    pad = [values[0]] * (length - len(values)) if values else [0.0] * length
    return pad + values


def _is_finite(value) -> bool:
    try:
        return value == value and value is not None
    except TypeError:
        return False
