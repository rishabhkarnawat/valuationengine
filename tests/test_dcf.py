"""Tests for DCF valuation engine."""

import pytest

from valuationengine.core import dcf
from valuationengine.core.models import Assumptions


def test_wacc_default(sample_company, base_assumptions):
    """WACC for default assumptions with beta=1.0 is approximately 8.35%."""
    wacc = dcf.compute_wacc(base_assumptions, beta=sample_company.beta)
    assert wacc == pytest.approx(0.0835, abs=1e-4)


def test_wacc_scales_with_beta(base_assumptions):
    """Higher beta should produce higher WACC."""
    low = dcf.compute_wacc(base_assumptions, beta=0.5)
    high = dcf.compute_wacc(base_assumptions, beta=1.5)
    assert high > low


def test_projection_length(sample_company, base_assumptions):
    """Projection has exactly projection_years rows."""
    base_assumptions.projection_years = 7
    proj = dcf.project_fcf(sample_company, base_assumptions)
    assert len(proj) == 7


def test_projection_columns_present(sample_company, base_assumptions):
    """Projection has all required columns."""
    proj = dcf.project_fcf(sample_company, base_assumptions)
    for col in ["year", "revenue", "ebit", "nopat", "da", "capex", "change_nwc", "fcf"]:
        assert col in proj.columns


def test_projection_first_revenue(sample_company, base_assumptions):
    """Year 1 revenue equals latest revenue times (1 + growth)."""
    base_assumptions.revenue_growth = 0.10
    proj = dcf.project_fcf(sample_company, base_assumptions)
    assert proj.iloc[0]["revenue"] == pytest.approx(146.41 * 1.10, rel=1e-4)


def test_dcf_run_returns_positive_value(sample_company, base_assumptions):
    """DCF run produces a positive intrinsic value per share."""
    result = dcf.run(sample_company, base_assumptions)
    assert result.value_per_share > 0
    assert result.enterprise_value > 0
    assert result.equity_value == pytest.approx(result.enterprise_value - sample_company.net_debt)


def test_dcf_equity_per_share(sample_company, base_assumptions):
    """value_per_share equals equity_value divided by shares_outstanding."""
    result = dcf.run(sample_company, base_assumptions)
    assert result.value_per_share == pytest.approx(
        result.equity_value / sample_company.shares_outstanding
    )


def test_dcf_upside_consistent(sample_company, base_assumptions):
    """Upside equals (value_per_share - current_price) / current_price."""
    result = dcf.run(sample_company, base_assumptions)
    expected = (result.value_per_share - sample_company.current_price) / sample_company.current_price
    assert result.upside == pytest.approx(expected)


def test_terminal_growth_above_wacc_raises(sample_company, base_assumptions):
    """Gordon growth with terminal_growth >= WACC must raise ValueError."""
    base_assumptions.terminal_growth = 0.20
    base_assumptions.use_exit_multiple = False
    with pytest.raises(ValueError):
        dcf.run(sample_company, base_assumptions)


def test_exit_multiple_path_runs(sample_company, base_assumptions):
    """Using exit multiple produces a positive valuation without error."""
    base_assumptions.use_exit_multiple = True
    base_assumptions.exit_ev_ebitda_multiple = 8.0
    result = dcf.run(sample_company, base_assumptions)
    assert result.value_per_share > 0
    assert result.terminal_value > 0


def test_higher_growth_yields_higher_value(sample_company, base_assumptions):
    """Higher revenue growth produces higher intrinsic value, all else equal."""
    low = dcf.run(sample_company, Assumptions(revenue_growth=0.05))
    high = dcf.run(sample_company, Assumptions(revenue_growth=0.15))
    assert high.value_per_share > low.value_per_share
