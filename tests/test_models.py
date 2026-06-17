"""Tests for Company, Assumptions, and computed properties."""

import math

import pytest

from valuationengine.core.models import Assumptions, Company


def test_company_net_debt(sample_company):
    """Net debt equals total_debt minus cash."""
    assert sample_company.net_debt == pytest.approx(50.0)


def test_company_latest_values(sample_company):
    """Latest values are the last entries in the historical lists."""
    assert sample_company.latest_revenue == pytest.approx(146.41)
    assert sample_company.latest_ebit == pytest.approx(29.282)
    assert sample_company.latest_ebitda == pytest.approx(35.1384)


def test_company_historical_cagr(sample_company):
    """CAGR of 100 to 146.41 over 4 years is 10%."""
    assert sample_company.historical_revenue_cagr == pytest.approx(0.10, rel=1e-3)


def test_company_avg_operating_margin(sample_company):
    """Average EBIT margin across history is 20%."""
    assert sample_company.avg_operating_margin == pytest.approx(0.20, rel=1e-3)


def test_company_avg_ratios(sample_company):
    """Capex, D&A, and NWC as share of revenue are 5%, 4%, 2%."""
    assert sample_company.avg_capex_pct_revenue == pytest.approx(0.05, rel=1e-3)
    assert sample_company.avg_da_pct_revenue == pytest.approx(0.04, rel=1e-3)
    assert sample_company.avg_nwc_pct_revenue == pytest.approx(0.02, rel=1e-3)


def test_assumptions_defaults_reasonable(base_assumptions):
    """Default assumptions are sensible for a US large-cap."""
    assert 0.0 < base_assumptions.revenue_growth < 0.30
    assert 0.0 < base_assumptions.operating_margin < 0.50
    assert base_assumptions.terminal_growth < (
        base_assumptions.risk_free_rate + base_assumptions.equity_risk_premium
    )
    assert base_assumptions.projection_years >= 3


def test_get_growth_path_broadcasts_scalar(base_assumptions):
    """Scalar growth is broadcast to a list of length projection_years."""
    base_assumptions.revenue_growth = 0.08
    base_assumptions.projection_years = 5
    path = base_assumptions.get_growth_path()
    assert len(path) == 5
    assert all(g == pytest.approx(0.08) for g in path)


def test_get_growth_path_passes_through_list(base_assumptions):
    """When growth is already a list, return it unchanged."""
    base_assumptions.revenue_growth = [0.10, 0.08, 0.06, 0.04, 0.03]
    path = base_assumptions.get_growth_path()
    assert path == [0.10, 0.08, 0.06, 0.04, 0.03]


def test_get_margin_path_broadcasts(base_assumptions):
    """Scalar margin is broadcast to a list of length projection_years."""
    base_assumptions.operating_margin = 0.18
    base_assumptions.projection_years = 4
    path = base_assumptions.get_margin_path()
    assert len(path) == 4
    assert all(m == pytest.approx(0.18) for m in path)


def test_calibrated_for_uses_company_history(sample_company):
    """calibrated_for defaults growth and margin to the company's own historical averages."""
    a = Assumptions.calibrated_for(sample_company)
    assert a.revenue_growth == pytest.approx(sample_company.historical_revenue_cagr, rel=1e-3)
    assert a.operating_margin == pytest.approx(sample_company.avg_operating_margin, rel=1e-3)
    assert a.capex_pct_revenue == pytest.approx(sample_company.avg_capex_pct_revenue, rel=1e-3)
    assert a.da_pct_revenue == pytest.approx(sample_company.avg_da_pct_revenue, rel=1e-3)
    assert a.nwc_pct_revenue == pytest.approx(sample_company.avg_nwc_pct_revenue, rel=1e-3)


def test_calibrated_for_explicit_override_wins(sample_company):
    """An explicitly passed value overrides the company-calibrated default."""
    a = Assumptions.calibrated_for(sample_company, operating_margin=0.42)
    assert a.operating_margin == pytest.approx(0.42)
    assert a.revenue_growth == pytest.approx(sample_company.historical_revenue_cagr, rel=1e-3)


def test_calibrated_for_does_not_touch_terminal_growth(sample_company):
    """terminal_growth is never calibrated to company history."""
    a = Assumptions.calibrated_for(sample_company)
    assert a.terminal_growth == pytest.approx(Assumptions().terminal_growth)


def test_calibrated_for_invalid_field_raises(sample_company):
    """Passing a kwarg that is not a real Assumptions field raises ValueError."""
    with pytest.raises(ValueError):
        Assumptions.calibrated_for(sample_company, not_a_real_field=0.1)


def test_calibrated_for_handles_bad_history_gracefully():
    """If a company's historical data is degenerate, fall back to the generic default instead of propagating bad data."""
    bad_company = Company(
        ticker="BAD",
        name="Bad Data Co",
        revenue=[0.0, 0.0],
        ebit=[0.0, 0.0],
        ebitda=[0.0, 0.0],
        depreciation_amortization=[0.0, 0.0],
        capex=[0.0, 0.0],
        change_in_nwc=[0.0, 0.0],
        effective_tax_rate=0.25,
        cash=0.0,
        total_debt=0.0,
        shares_outstanding=1.0,
        current_price=1.0,
        market_cap=1.0,
        beta=1.0,
    )
    a = Assumptions.calibrated_for(bad_company)
    assert math.isfinite(a.revenue_growth)
    assert math.isfinite(a.operating_margin)
