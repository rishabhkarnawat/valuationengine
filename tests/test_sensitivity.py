"""Tests for 2D sensitivity analysis."""

import pytest

from valuationengine.core import sensitivity


def test_grid_shape(sample_company, base_assumptions):
    """Sensitivity grid shape matches the count of x and y values."""
    x_vals = [0.05, 0.08, 0.10]
    y_vals = [0.15, 0.20, 0.25]
    table = sensitivity.run(
        sample_company,
        base_assumptions,
        x_field="revenue_growth",
        x_values=x_vals,
        y_field="operating_margin",
        y_values=y_vals,
    )
    assert table.shape == (3, 3)


def test_grid_indexed_by_y_columns_x(sample_company, base_assumptions):
    """Rows are y_values, columns are x_values."""
    x_vals = [0.05, 0.10]
    y_vals = [0.15, 0.20, 0.25]
    table = sensitivity.run(
        sample_company,
        base_assumptions,
        x_field="revenue_growth",
        x_values=x_vals,
        y_field="operating_margin",
        y_values=y_vals,
    )
    assert list(table.index) == pytest.approx(y_vals)
    assert list(table.columns) == pytest.approx(x_vals)


def test_assumptions_not_mutated(sample_company, base_assumptions):
    """Running sensitivity must not mutate the base assumptions."""
    original_growth = base_assumptions.revenue_growth
    original_margin = base_assumptions.operating_margin
    sensitivity.run(
        sample_company,
        base_assumptions,
        x_field="revenue_growth",
        x_values=[0.05, 0.10],
        y_field="operating_margin",
        y_values=[0.15, 0.25],
    )
    assert base_assumptions.revenue_growth == original_growth
    assert base_assumptions.operating_margin == original_margin


def test_monotonic_in_growth(sample_company, base_assumptions):
    """Across rows of constant margin, value rises with growth."""
    x_vals = [0.05, 0.08, 0.12]
    table = sensitivity.run(
        sample_company,
        base_assumptions,
        x_field="revenue_growth",
        x_values=x_vals,
        y_field="operating_margin",
        y_values=[0.20],
    )
    row = list(table.iloc[0])
    assert row[0] < row[1] < row[2]


def test_invalid_field_raises(sample_company, base_assumptions):
    """Unknown field name raises ValueError."""
    with pytest.raises(ValueError):
        sensitivity.run(
            sample_company,
            base_assumptions,
            x_field="not_a_field",
            x_values=[0.05],
            y_field="operating_margin",
            y_values=[0.20],
        )
