"""FastMCP server exposing valuation tools to any LLM client"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np
import pandas as pd
from fastmcp import FastMCP

from core import dcf as dcf_module
from core import lbo as lbo_module
from core import reverse as reverse_module
from core import scenario as scenario_module
from core import sensitivity as sensitivity_module
from core.models import Assumptions
from data.fetcher import fetch_company

mcp = FastMCP("valuationengine")


@mcp.tool()
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch current fundamentals for a public company by ticker symbol.

    Returns key metrics: name, current price, market cap, shares outstanding,
    latest revenue and EBITDA, net debt, beta, historical revenue CAGR,
    and average operating margin. Use this before any valuation call if the
    user wants to see the underlying numbers.
    """
    try:
        c = fetch_company(ticker)
        return {
            "ticker": c.ticker,
            "name": c.name,
            "current_price": c.current_price,
            "market_cap": c.market_cap,
            "shares_outstanding": c.shares_outstanding,
            "latest_revenue": c.latest_revenue,
            "latest_ebitda": c.latest_ebitda,
            "net_debt": c.net_debt,
            "beta": c.beta,
            "historical_revenue_cagr": c.historical_revenue_cagr,
            "avg_operating_margin": c.avg_operating_margin,
            "historical_revenue": c.revenue,
            "historical_ebit": c.ebit,
            "historical_ebitda": c.ebitda,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_dcf(
    ticker: str,
    revenue_growth: float = 0.08,
    operating_margin: float = 0.20,
    terminal_growth: float = 0.025,
    risk_free_rate: float = 0.045,
    equity_risk_premium: float = 0.055,
    use_exit_multiple: bool = False,
    exit_ev_ebitda_multiple: float = 10.0,
    projection_years: int = 5,
) -> dict:
    """
    Run a discounted cash flow valuation on a public company.

    Projects unlevered free cash flow forward, discounts at WACC, computes
    terminal value (Gordon growth by default, or exit multiple), and returns
    the intrinsic equity value per share alongside current market price and
    implied upside or downside. Use when the user asks 'what is X worth' or
    'is X overvalued'.

    All percentage inputs are decimals (0.10 means 10%).
    """
    try:
        company = fetch_company(ticker)
        a = Assumptions(
            revenue_growth=revenue_growth,
            operating_margin=operating_margin,
            terminal_growth=terminal_growth,
            risk_free_rate=risk_free_rate,
            equity_risk_premium=equity_risk_premium,
            use_exit_multiple=use_exit_multiple,
            exit_ev_ebitda_multiple=exit_ev_ebitda_multiple,
            projection_years=projection_years,
            tax_rate=company.effective_tax_rate,
        )
        r = dcf_module.run(company, a)
        return {
            "ticker": ticker,
            "value_per_share": r.value_per_share,
            "current_price": company.current_price,
            "upside_pct": r.upside * 100,
            "wacc": r.wacc,
            "enterprise_value": r.enterprise_value,
            "equity_value": r.equity_value,
            "terminal_value": r.terminal_value,
            "projection": r.projection.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_lbo(
    ticker: str,
    entry_ev_ebitda_multiple: float = 10.0,
    exit_ev_ebitda_multiple: float = 10.0,
    debt_pct_purchase: float = 0.60,
    interest_rate: float = 0.08,
    revenue_growth: float = 0.05,
    operating_margin: float = 0.20,
    hold_period_years: int = 5,
) -> dict:
    """
    Run a leveraged buyout model on a public company as if a sponsor took it private.

    Builds sources and uses at the entry multiple, projects EBITDA through the
    hold period, services debt with mandatory amortization plus a cash sweep,
    exits at the exit multiple, and returns sponsor IRR and MOIC. Use when the
    user asks about take-private deals, PE returns, or LBO economics.
    """
    try:
        company = fetch_company(ticker)
        a = Assumptions(
            entry_ev_ebitda_multiple=entry_ev_ebitda_multiple,
            exit_lbo_ev_ebitda_multiple=exit_ev_ebitda_multiple,
            debt_pct_purchase=debt_pct_purchase,
            lbo_debt_interest_rate=interest_rate,
            revenue_growth=revenue_growth,
            operating_margin=operating_margin,
            hold_period_years=hold_period_years,
            tax_rate=company.effective_tax_rate,
        )
        r = lbo_module.run(company, a)
        return {
            "ticker": ticker,
            "sources_and_uses": r.sources_and_uses,
            "irr_pct": r.irr * 100,
            "moic": r.moic,
            "exit": r.exit,
            "projection": r.projection.to_dict(orient="records"),
            "debt_schedule": r.debt_schedule.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_reverse_dcf(
    ticker: str,
    field: str = "revenue_growth",
    target: str = "market_cap",
) -> dict:
    """
    Reverse-engineer the assumption the market is currently pricing in.

    Takes the current market price and solves for the value of `field`
    (e.g. 'revenue_growth' or 'operating_margin' or 'terminal_growth')
    that makes a DCF match the market. Returns the implied value with a
    human-readable interpretation. Use this when the user asks 'what is
    the market pricing in' or 'what growth rate would justify this price'.
    """
    try:
        company = fetch_company(ticker)
        return reverse_module.solve(
            company, Assumptions(tax_rate=company.effective_tax_rate), field=field, target=target
        )
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_sensitivity(
    ticker: str,
    x_field: str,
    x_min: float,
    x_max: float,
    x_steps: int,
    y_field: str,
    y_min: float,
    y_max: float,
    y_steps: int,
    output: str = "value_per_share",
) -> dict:
    """
    Run a 2D sensitivity table varying two assumption fields.

    For each combination, computes the chosen output ('value_per_share',
    'upside', 'irr', or 'moic'). Use this when the user asks how the
    valuation changes across ranges of assumptions.
    """
    try:
        company = fetch_company(ticker)
        x_values = list(np.linspace(x_min, x_max, x_steps))
        y_values = list(np.linspace(y_min, y_max, y_steps))
        table = sensitivity_module.run(
            company,
            Assumptions(tax_rate=company.effective_tax_rate),
            x_field,
            x_values,
            y_field,
            y_values,
            output=output,
        )
        return {
            "ticker": ticker,
            "x_field": x_field,
            "y_field": y_field,
            "output": output,
            "x_values": x_values,
            "y_values": y_values,
            "grid": table.values.tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_scenario(
    ticker: str,
    growth_delta: float = 0.03,
    margin_delta: float = 0.03,
) -> dict:
    """
    Run bull, base, and bear DCF scenarios.

    Builds three assumption sets by adjusting revenue growth and operating
    margin by the deltas, runs the DCF on each, and returns the resulting
    intrinsic values and upsides side by side.
    """
    try:
        company = fetch_company(ticker)
        scenarios = scenario_module.build_bull_base_bear(
            Assumptions(tax_rate=company.effective_tax_rate),
            growth_delta=growth_delta,
            margin_delta=margin_delta,
        )
        results = scenario_module.run(company, scenarios)
        return {
            "ticker": ticker,
            "current_price": company.current_price,
            "scenarios": {
                name: {
                    "value_per_share": r.value_per_share,
                    "upside_pct": r.upside * 100,
                    "wacc": r.wacc,
                }
                for name, r in results.items()
            },
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
