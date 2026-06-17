"""Tests for Company, Assumptions, and computed properties."""

import pytest


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
