"""Tests for dashboard HTML table renderers."""

import pytest

from valuationengine.adapters import dashboard_tables as tables
from valuationengine.core import dcf, scenario


def test_fundamentals_table_contains_key_fields(sample_company):
    html = tables.render_fundamentals_table(sample_company)
    assert "ve-table" in html
    assert sample_company.ticker in html
    assert sample_company.name in html
    assert "Operating Margin %" in html
    assert "Revenue CAGR %" in html


def test_dcf_valuation_table_highlights_fair_value(sample_company, base_assumptions):
    result = dcf.run(sample_company, base_assumptions)
    html = tables.render_dcf_valuation_table(result)
    assert "Fair Value Per Share" in html
    assert "ve-highlight" in html
    assert f"${result.value_per_share:,.2f}" in html


def test_scenarios_table_color_classes(sample_company, base_assumptions):
    scenarios = scenario.build_bull_base_bear(base_assumptions, growth_delta=0.03, margin_delta=0.03)
    results = scenario.run(sample_company, scenarios)
    html = tables.render_scenarios_table(results, 0.03, 0.03)
    assert "ve-scenario-bull" in html
    assert "ve-scenario-base" in html
    assert "ve-scenario-bear" in html
    assert "+3% growth" in html


def test_reverse_dcf_table_shows_historical_comparison(sample_company, base_assumptions):
    implied = tables.solve_reverse_fields(sample_company, base_assumptions)
    html = tables.render_reverse_dcf_table(sample_company, base_assumptions, implied)
    assert "Revenue Growth" in html
    assert "Operating Margin" in html
    assert "Historical" in html


def test_lbo_returns_two_leverage_rows(sample_company, base_assumptions):
    html = tables.render_lbo_returns_table(sample_company, base_assumptions)
    assert "60%" in html
    assert "75%" in html
    assert "MOIC" in html


def test_projection_table_merges_dcf_and_lbo(sample_company, base_assumptions):
    dcf_result = dcf.run(sample_company, base_assumptions)
    lbo_result = tables.run_lbo_at_leverage(sample_company, base_assumptions, 0.60)
    html = tables.render_projection_table(dcf_result, lbo_result)
    assert "Debt Paydown" in html
    assert "Free Cash Flow" in html
    assert "Year" in html


def test_fmt_currency_and_pct():
    assert tables.fmt_currency(1234.5) == "$1,234.50"
    assert tables.fmt_pct(0.123, signed=True) == "+12.3%"


def test_sensitivity_grid_shape(sample_company, base_assumptions):
    table, growth, margin = tables.build_sensitivity_grid(sample_company, base_assumptions)
    assert table.shape == (11, 10)
    assert growth == tables.SENSITIVITY_GROWTH_VALUES
    assert margin == tables.SENSITIVITY_MARGIN_VALUES


def test_sensitivity_cell_colors():
    assert tables.sensitivity_cell_color(-0.25) == "#e74c3c"
    assert tables.sensitivity_cell_color(-0.05) == "#f1c40f"
    assert tables.sensitivity_cell_color(0.10) == "#a8e6a1"
    assert tables.sensitivity_cell_color(0.30) == "#1e8449"


def test_sensitivity_table_renders_highlights(sample_company, base_assumptions):
    table, growth, margin = tables.build_sensitivity_grid(sample_company, base_assumptions)
    implied = tables.solve_reverse_fields(sample_company, base_assumptions)
    html = tables.render_sensitivity_table(
        table,
        growth,
        margin,
        sample_company.current_price,
        float(base_assumptions.revenue_growth),
        float(base_assumptions.operating_margin),
        implied.get("revenue_growth"),
        implied.get("operating_margin"),
    )
    assert "3498db" in html
    assert "ve-sens-ref" in html
    assert "Revenue Growth" in html


def test_sensitivity_caption_includes_required_text():
    text = tables.render_sensitivity_caption(0.08, 0.20, 0.077, 0.232)
    assert "blue border" in text
    assert "Reverse DCF" in text or "DCF & Scenarios" in text
    assert "7.7%" in text or "7.7" in text


def test_sensitivity_caption_no_false_higher_pricing_claim():
    text = tables.render_sensitivity_caption(0.08, 0.20, 0.05, 0.15)
    assert "lower growth and margins" in text


def test_fetch_status_escapes_html():
    html_out = tables.render_fetch_status("<img src=x onerror=alert(1)>")
    assert "<img" not in html_out
    assert "&lt;img" in html_out


def test_peer_comparison_table_sorts_and_highlights():
    peers = [
        {
            "ticker": "AAA",
            "name": "A Co",
            "current_price": 10.0,
            "market_cap": 100.0,
            "operating_margin": 0.10,
            "revenue_cagr": 0.05,
            "dcf_fair_value": 12.0,
            "dcf_upside": 0.20,
            "lbo_irr": 0.15,
        },
        {
            "ticker": "BBB",
            "name": "B Co",
            "current_price": 20.0,
            "market_cap": 200.0,
            "operating_margin": 0.15,
            "revenue_cagr": 0.08,
            "dcf_fair_value": 19.0,
            "dcf_upside": -0.05,
            "lbo_irr": 0.25,
        },
    ]
    peers_sorted = sorted(peers, key=lambda p: p["dcf_upside"], reverse=True)
    html = tables.render_peer_comparison_table(peers_sorted)
    assert html.index("AAA") < html.index("BBB")  # columns ordered by upside
    assert "background-color" in html  # upside heat colors
    assert "font-weight:700" in html  # best metric bolding


def test_summary_interpretation_premium(sample_company, base_assumptions):
    result = dcf.run(sample_company, base_assumptions)
    implied = tables.solve_reverse_fields(sample_company, base_assumptions)
    text = tables.build_summary_interpretation(sample_company, result, implied)
    assert "Trading at" in text
    assert "fair value" in text.lower()


def test_summary_card_renders_metrics(sample_company, base_assumptions):
    from datetime import datetime, timezone

    result = dcf.run(sample_company, base_assumptions)
    lbo = tables.run_lbo_at_leverage(sample_company, base_assumptions, 0.60)
    implied = tables.solve_reverse_fields(sample_company, base_assumptions)
    html = tables.render_summary_card(
        sample_company, result, lbo, implied, datetime.now(timezone.utc)
    )
    assert "ve-summary-card" in html
    assert sample_company.ticker in html
    assert "ve-metric-value" in html


def test_valuation_summary_csv(sample_company, base_assumptions):
    from datetime import datetime, timezone

    result = dcf.run(sample_company, base_assumptions)
    lbo = tables.run_lbo_at_leverage(sample_company, base_assumptions, 0.60)
    implied = tables.solve_reverse_fields(sample_company, base_assumptions)
    csv = tables.build_valuation_summary_csv(
        sample_company, result, lbo, implied, datetime.now(timezone.utc)
    )
    assert sample_company.ticker in csv
    assert "DCF Fair Value" in csv


def test_empty_state_and_skeleton():
    initial = tables.render_empty_state("initial")
    assert "Enter a stock ticker" in initial
    compare = tables.render_empty_state("compare")
    assert "two or more tickers" in compare
    skeleton = tables.render_skeleton("full")
    assert "ve-skeleton" in skeleton


def test_summary_card_includes_tooltips(sample_company, base_assumptions):
    from datetime import datetime, timezone

    result = dcf.run(sample_company, base_assumptions)
    lbo = tables.run_lbo_at_leverage(sample_company, base_assumptions, 0.60)
    implied = tables.solve_reverse_fields(sample_company, base_assumptions)
    html = tables.render_summary_card(
        sample_company, result, lbo, implied, datetime.now(timezone.utc)
    )
    assert "ve-tip" in html
    assert "title=" in html
