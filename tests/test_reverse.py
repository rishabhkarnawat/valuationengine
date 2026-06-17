"""Tests for reverse DCF solver."""

import copy

import pytest

from valuationengine.core import dcf, reverse


def test_reverse_recovers_known_growth(sample_company, base_assumptions):
    """If we compute the equity value at a known growth, reverse DCF recovers that growth."""
    known_assumptions = copy.deepcopy(base_assumptions)
    known_assumptions.revenue_growth = 0.12
    target_result = dcf.run(sample_company, known_assumptions)
    planted = copy.deepcopy(sample_company)
    planted.market_cap = target_result.equity_value
    planted.current_price = target_result.value_per_share
    solved = reverse.solve(planted, base_assumptions, field="revenue_growth", target="market_cap")
    assert solved["implied_value"] == pytest.approx(0.12, abs=1e-3)


def test_reverse_returns_interpretation_string(sample_company, base_assumptions):
    """Solver output contains a non-empty interpretation field."""
    result = reverse.solve(
        sample_company, base_assumptions, field="revenue_growth", target="market_cap"
    )
    assert "interpretation" in result
    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 0


def test_reverse_unknown_field_raises(sample_company, base_assumptions):
    """Unknown field name raises ValueError."""
    with pytest.raises(ValueError):
        reverse.solve(sample_company, base_assumptions, field="not_a_field")


def test_reverse_works_for_margin(sample_company, base_assumptions):
    """Solver runs successfully when solving for operating_margin."""
    result = reverse.solve(
        sample_company,
        base_assumptions,
        field="operating_margin",
        target="market_cap",
        bracket=(0.01, 0.60),
    )
    assert "implied_value" in result
    assert 0.01 < result["implied_value"] < 0.60
