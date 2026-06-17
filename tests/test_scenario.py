"""Tests for bull/base/bear scenario analysis."""

import pytest

from valuationengine.core import scenario


def test_build_bull_base_bear_keys(base_assumptions):
    """Helper returns the three expected scenario names."""
    scenarios = scenario.build_bull_base_bear(base_assumptions)
    assert set(scenarios.keys()) == {"bull", "base", "bear"}


def test_bull_growth_above_base_above_bear(base_assumptions):
    """Bull growth is highest, bear growth is lowest."""
    scenarios = scenario.build_bull_base_bear(
        base_assumptions, growth_delta=0.03, margin_delta=0.03
    )
    assert scenarios["bull"].revenue_growth > scenarios["base"].revenue_growth
    assert scenarios["base"].revenue_growth > scenarios["bear"].revenue_growth


def test_base_unchanged_from_input(base_assumptions):
    """The base scenario equals the input assumptions."""
    scenarios = scenario.build_bull_base_bear(base_assumptions)
    assert scenarios["base"].revenue_growth == base_assumptions.revenue_growth


def test_input_not_mutated(base_assumptions):
    """Helper must deepcopy; the input assumptions are unchanged."""
    original = base_assumptions.revenue_growth
    scenario.build_bull_base_bear(base_assumptions, growth_delta=0.03)
    assert base_assumptions.revenue_growth == original


def test_scenario_run_returns_three_results(sample_company, base_assumptions):
    """Running three scenarios returns three results."""
    scenarios = scenario.build_bull_base_bear(base_assumptions)
    results = scenario.run(sample_company, scenarios)
    assert set(results.keys()) == {"bull", "base", "bear"}


def test_bull_value_above_bear(sample_company, base_assumptions):
    """Bull intrinsic value exceeds bear intrinsic value."""
    scenarios = scenario.build_bull_base_bear(base_assumptions)
    results = scenario.run(sample_company, scenarios)
    assert results["bull"].value_per_share > results["bear"].value_per_share
