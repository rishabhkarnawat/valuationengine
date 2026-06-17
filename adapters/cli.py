"""Click-based command-line interface"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import click
import pandas as pd

from core import dcf as dcf_module
from core import lbo as lbo_module
from core import reverse as reverse_module
from core import scenario as scenario_module
from core import sensitivity as sensitivity_module
from core.models import Assumptions
from data.fetcher import fetch_company


@click.group()
def cli():
    """Open-source DCF and LBO valuation toolkit."""
    pass


@cli.command()
@click.argument("ticker")
@click.option("--growth", type=float, default=None, help="Revenue growth rate as decimal (e.g. 0.10).")
@click.option("--margin", type=float, default=None, help="Operating (EBIT) margin as decimal.")
@click.option("--wacc-rf", type=float, default=None, help="Risk-free rate.")
@click.option("--wacc-erp", type=float, default=None, help="Equity risk premium.")
@click.option("--terminal-growth", type=float, default=None, help="Terminal growth rate.")
@click.option("--use-exit-multiple", is_flag=True, help="Use exit EV/EBITDA multiple for terminal value.")
@click.option("--exit-multiple", type=float, default=None, help="Terminal exit EV/EBITDA multiple.")
@click.option("--years", type=int, default=None, help="Projection years.")
@click.pass_context
def dcf(
    ctx,
    ticker,
    growth,
    margin,
    wacc_rf,
    wacc_erp,
    terminal_growth,
    use_exit_multiple,
    exit_multiple,
    years,
):
    """Run a DCF on TICKER."""
    try:
        company = fetch_company(ticker)
        assumptions = _build_assumptions(
            company,
            growth=growth,
            margin=margin,
            wacc_rf=wacc_rf,
            wacc_erp=wacc_erp,
            terminal_growth=terminal_growth,
            use_exit_multiple=use_exit_multiple,
            exit_multiple=exit_multiple,
            years=years,
        )
        result = dcf_module.run(company, assumptions)
        click.echo(result.summary())
        click.echo("")
        click.echo("Projection:")
        click.echo(result.projection.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    except ValueError as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--entry-multiple", type=float, default=None)
@click.option("--exit-multiple", type=float, default=None)
@click.option("--debt-pct", type=float, default=None)
@click.option("--interest-rate", type=float, default=None)
@click.option("--hold", type=int, default=None)
@click.pass_context
def lbo(ctx, ticker, entry_multiple, exit_multiple, debt_pct, interest_rate, hold):
    """Run an LBO on TICKER."""
    try:
        company = fetch_company(ticker)
        a = Assumptions(tax_rate=company.effective_tax_rate)
        if entry_multiple is not None:
            a.entry_ev_ebitda_multiple = entry_multiple
        if exit_multiple is not None:
            a.exit_lbo_ev_ebitda_multiple = exit_multiple
        if debt_pct is not None:
            a.debt_pct_purchase = debt_pct
        if interest_rate is not None:
            a.lbo_debt_interest_rate = interest_rate
        if hold is not None:
            a.hold_period_years = hold
        result = lbo_module.run(company, a)
        click.echo(result.summary())
        click.echo("")
        click.echo("Debt schedule:")
        click.echo(result.debt_schedule.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    except ValueError as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--field", default="revenue_growth", help="Assumption field to solve for.")
@click.option("--target", default="market_cap", type=click.Choice(["market_cap", "current_price"]))
@click.pass_context
def reverse(ctx, ticker, field, target):
    """Run a reverse DCF on TICKER: back-solve market-implied assumptions."""
    try:
        company = fetch_company(ticker)
        result = reverse_module.solve(
            company, Assumptions(tax_rate=company.effective_tax_rate), field=field, target=target
        )
        click.echo(f"\nReverse DCF for {ticker}")
        click.echo(f"Solving for: {result['field']}")
        click.echo(f"Target ({result['target']}): {result['target_value']:,.2f}")
        click.echo(f"Implied value: {result['implied_value']:.4f}")
        click.echo(f"\n{result['interpretation']}")
    except ValueError as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--x-field", required=True)
@click.option("--x-min", type=float, required=True)
@click.option("--x-max", type=float, required=True)
@click.option("--x-steps", type=int, default=5)
@click.option("--y-field", required=True)
@click.option("--y-min", type=float, required=True)
@click.option("--y-max", type=float, required=True)
@click.option("--y-steps", type=int, default=5)
@click.option("--output", default="value_per_share")
@click.pass_context
def sensitivity(
    ctx,
    ticker,
    x_field,
    x_min,
    x_max,
    x_steps,
    y_field,
    y_min,
    y_max,
    y_steps,
    output,
):
    """Run a 2D sensitivity table on TICKER."""
    import numpy as np

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
        click.echo(f"\nSensitivity table for {ticker} ({output})")
        click.echo(f"Rows: {y_field}, Columns: {x_field}\n")
        click.echo(table.to_string(float_format=lambda x: f"{x:,.2f}"))
    except ValueError as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


@cli.command()
@click.argument("ticker")
@click.option("--growth-delta", type=float, default=0.03)
@click.option("--margin-delta", type=float, default=0.03)
@click.pass_context
def scenario(ctx, ticker, growth_delta, margin_delta):
    """Run bull / base / bear scenario analysis on TICKER."""
    try:
        company = fetch_company(ticker)
        scenarios = scenario_module.build_bull_base_bear(
            Assumptions(tax_rate=company.effective_tax_rate),
            growth_delta=growth_delta,
            margin_delta=margin_delta,
        )
        results = scenario_module.run(company, scenarios)
        click.echo(f"\nScenario analysis for {ticker}\n")
        rows = []
        for name, r in results.items():
            rows.append(
                {
                    "scenario": name,
                    "value_per_share": r.value_per_share,
                    "current_price": company.current_price,
                    "upside_pct": r.upside * 100,
                }
            )
        click.echo(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    except ValueError as e:
        click.echo(str(e), err=True)
        ctx.exit(1)


def _build_assumptions(
    company,
    growth=None,
    margin=None,
    wacc_rf=None,
    wacc_erp=None,
    terminal_growth=None,
    use_exit_multiple=False,
    exit_multiple=None,
    years=None,
) -> Assumptions:
    a = Assumptions(tax_rate=company.effective_tax_rate)
    if growth is not None:
        a.revenue_growth = growth
    if margin is not None:
        a.operating_margin = margin
    if wacc_rf is not None:
        a.risk_free_rate = wacc_rf
    if wacc_erp is not None:
        a.equity_risk_premium = wacc_erp
    if terminal_growth is not None:
        a.terminal_growth = terminal_growth
    if use_exit_multiple:
        a.use_exit_multiple = True
    if exit_multiple is not None:
        a.exit_ev_ebitda_multiple = exit_multiple
    if years is not None:
        a.projection_years = years
    return a


if __name__ == "__main__":
    cli()
