"""Tests for LBO model."""

import pytest

from valuationengine.core import lbo
from valuationengine.core.models import Assumptions


def test_sources_and_uses_balance(sample_company, base_assumptions):
    """Equity check plus debt minus fees equals purchase price."""
    result = lbo.run(sample_company, base_assumptions)
    su = result.sources_and_uses
    assert su["equity_check"] + su["debt"] == pytest.approx(
        su["purchase_price"] + su["fees"], rel=1e-4
    )


def test_purchase_price_equals_multiple_times_ebitda(sample_company, base_assumptions):
    """Purchase EV equals latest EBITDA times entry multiple."""
    base_assumptions.entry_ev_ebitda_multiple = 10.0
    result = lbo.run(sample_company, base_assumptions)
    assert result.sources_and_uses["purchase_price"] == pytest.approx(
        sample_company.latest_ebitda * 10.0
    )


def test_projection_length_matches_hold_period(sample_company, base_assumptions):
    """LBO projection has hold_period_years rows."""
    base_assumptions.hold_period_years = 5
    result = lbo.run(sample_company, base_assumptions)
    assert len(result.projection) == 5


def test_debt_schedule_length_matches_hold_period(sample_company, base_assumptions):
    """Debt schedule has hold_period_years rows."""
    base_assumptions.hold_period_years = 5
    result = lbo.run(sample_company, base_assumptions)
    assert len(result.debt_schedule) == 5


def test_irr_and_moic_consistent(sample_company, base_assumptions):
    """IRR derived from MOIC matches the reported IRR over the hold period."""
    result = lbo.run(sample_company, base_assumptions)
    implied_irr = result.moic ** (1.0 / base_assumptions.hold_period_years) - 1.0
    assert result.irr == pytest.approx(implied_irr, rel=1e-3)


def test_lower_entry_multiple_yields_higher_irr(sample_company, base_assumptions):
    """Buying cheaper produces a higher sponsor IRR, all else equal."""
    cheap = lbo.run(
        sample_company,
        Assumptions(entry_ev_ebitda_multiple=8.0, exit_lbo_ev_ebitda_multiple=10.0),
    )
    expensive = lbo.run(
        sample_company,
        Assumptions(entry_ev_ebitda_multiple=12.0, exit_lbo_ev_ebitda_multiple=10.0),
    )
    assert cheap.irr > expensive.irr


def test_exit_equity_equals_exit_ev_minus_exit_debt(sample_company, base_assumptions):
    """Exit equity equals exit EV minus remaining debt at exit."""
    result = lbo.run(sample_company, base_assumptions)
    assert result.exit["exit_equity"] == pytest.approx(
        result.exit["exit_ev"] - result.exit["exit_debt"]
    )
