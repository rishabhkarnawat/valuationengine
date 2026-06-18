"""HTML table renderers for the Streamlit valuation dashboard."""

from __future__ import annotations

import html
from copy import deepcopy
from datetime import datetime
from io import StringIO
from typing import Callable

import pandas as pd

from core import lbo as lbo_module
from core import reverse as reverse_module
from core import sensitivity as sensitivity_module
from core.models import Assumptions, Company, DCFResult, LBOResult

from adapters.dashboard_ux import classify_fetch_error

SENSITIVITY_GROWTH_VALUES = [i / 100 for i in range(-2, 9)]
SENSITIVITY_MARGIN_VALUES = [i / 100 for i in range(15, 25)]

# Tailwind-inspired palette, layout chrome, and shared table styles
_LAYOUT_CSS = """
<style>
.ve-layout { padding: 16px 0 24px; }
.ve-header-block { margin-bottom: 24px; }
.ve-header-ticker { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin: 0; line-height: 1.1; }
.ve-header-name { font-size: 1rem; color: #64748b; margin: 4px 0 0; }
.ve-timestamp { font-size: 0.8rem; color: #94a3b8; margin-top: 8px; }
.ve-summary-card {
  background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,.08), 0 2px 4px -2px rgba(0,0,0,.06);
  padding: 24px; margin: 16px 0 24px;
}
.ve-summary-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
}
.ve-metric-box {
  text-align: center; padding: 16px; background: #f8fafc; border-radius: 8px;
  border: 1px solid #e2e8f0;
}
.ve-metric-label {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: #64748b; font-weight: 600;
}
.ve-metric-value { font-size: 1.75rem; font-weight: 700; margin-top: 8px; color: #0f172a; }
.ve-metric-upside-pos { color: #166534 !important; }
.ve-metric-upside-neg { color: #b91c1c !important; }
.ve-interpretation {
  margin-top: 16px; padding-top: 16px; border-top: 1px solid #e2e8f0;
  color: #475569; font-size: 0.95rem; line-height: 1.5;
}
.ve-footer {
  margin-top: 32px; padding: 16px 24px; text-align: center;
  font-size: 0.8rem; color: #94a3b8; border-top: 1px solid #e2e8f0;
}
.ve-empty-state {
  text-align: center; padding: 48px 24px; margin: 24px 0;
  background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px;
  color: #64748b; font-size: 1rem; line-height: 1.6;
}
.ve-empty-state strong { color: #334155; }
.ve-skeleton-wrap { margin: 16px 0 24px; display: flex; flex-direction: column; gap: 12px; }
.ve-skeleton {
  background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
  background-size: 200% 100%; animation: ve-shimmer 1.4s ease-in-out infinite;
  border-radius: 8px; border: 1px solid #e2e8f0;
}
.ve-skeleton-header { height: 72px; }
.ve-skeleton-card { height: 140px; }
.ve-skeleton-table { height: 200px; }
@keyframes ve-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
.ve-tip { cursor: help; border-bottom: 1px dotted #94a3b8; }
.ve-tip-icon { font-size: 0.75rem; color: #94a3b8; margin-left: 2px; }
.ve-fetch-status {
  font-size: 0.875rem; color: #64748b; margin: 8px 0 16px; padding: 8px 12px;
  background: #f8fafc; border-radius: 6px; border-left: 3px solid #3b82f6;
}
@media (max-width: 900px) {
  .ve-summary-grid { grid-template-columns: repeat(2, 1fr); }
  .ve-header-ticker { font-size: 2rem; }
}
@media (max-width: 480px) {
  .ve-summary-grid { grid-template-columns: 1fr; }
  .ve-header-ticker { font-size: 1.75rem; }
}
</style>
"""

_TABLE_CSS = """
<style>
.ve-table-wrap {
  overflow-x: auto; margin: 16px 0 24px;
  border-radius: 12px; border: 1px solid #e2e8f0;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,.08), 0 2px 4px -2px rgba(0,0,0,.06);
}
.ve-table {
  width: 100%; border-collapse: collapse; font-size: 0.875rem; line-height: 1.5;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.ve-table th {
  background-color: #f1f5f9; color: #0f172a; font-weight: 700;
  padding: 0.625rem 0.75rem; border: 1px solid #e2e8f0; text-align: left;
}
.ve-table td {
  padding: 0.625rem 0.75rem; border: 1px solid #e2e8f0; color: #1e293b;
}
.ve-table .ve-num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ve-table .ve-row-alt { background-color: #f8fafc; }
.ve-table .ve-highlight { background-color: #eff6ff; font-weight: 600; }
.ve-table .ve-upside-pos { color: #166534; font-weight: 600; }
.ve-table .ve-upside-neg { color: #b91c1c; font-weight: 600; }
.ve-scenario-bull { background-color: #dcfce7; }
.ve-scenario-base { background-color: #f1f5f9; }
.ve-scenario-bear { background-color: #fee2e2; }
.ve-sens-legend { display: flex; flex-wrap: wrap; gap: 1rem; margin: 0.75rem 0 1rem; font-size: 0.875rem; }
.ve-sens-legend span { display: inline-flex; align-items: center; gap: 0.375rem; }
.ve-sens-swatch { width: 1rem; height: 1rem; border-radius: 0.125rem; border: 1px solid #cbd5e1; }
.ve-sens-ref { margin: 0.5rem 0 0.75rem; font-size: 0.95rem; }
.ve-sens-ref strong { font-variant-numeric: tabular-nums; }
.ve-sens-cell { text-align: center; font-weight: 600; font-variant-numeric: tabular-nums; }
@media (max-width: 640px) {
  .ve-table thead { display: none; }
  .ve-table tr { display: block; margin-bottom: 0.75rem; border: 1px solid #e2e8f0; border-radius: 0.375rem; }
  .ve-table td, .ve-table th[scope="row"] {
    display: flex; justify-content: space-between; align-items: center;
    border: none; border-bottom: 1px solid #f1f5f9;
  }
  .ve-table td::before, .ve-table th[scope="row"]::before {
    content: attr(data-label); font-weight: 600; color: #64748b; margin-right: 1rem;
  }
  .ve-table td:last-child, .ve-table th[scope="row"]:last-child { border-bottom: none; }
}
</style>
"""

TABLE_CSS = _LAYOUT_CSS + _TABLE_CSS

TOOLTIPS: dict[str, str] = {
    "Fair Value": (
        "Intrinsic value based on 5-year DCF projection using historical "
        "growth and margin assumptions"
    ),
    "Fair Value Per Share": (
        "Intrinsic value based on 5-year DCF projection using historical "
        "growth and margin assumptions"
    ),
    "Upside/Downside": (
        "Percent difference between fair value and current market price"
    ),
    "Upside/Downside %": (
        "Percent difference between fair value and current market price"
    ),
    "MOIC": (
        "Money multiple. A 1.5x MOIC means you got back 1.5x your invested capital"
    ),
    "IRR": (
        "Annualized return on the LBO equity investment"
    ),
    "LBO IRR": (
        "Annualized return on the LBO equity investment"
    ),
}


def _tip_label(label: str) -> str:
    """Render a label with an HTML title tooltip when defined."""
    tip = TOOLTIPS.get(label)
    if not tip:
        return label
    safe_tip = html.escape(tip, quote=True)
    return (
        f'<span class="ve-tip" title="{safe_tip}">{html.escape(label)}'
        f'<span class="ve-tip-icon" aria-hidden="true"> ⓘ</span></span>'
    )


def render_empty_state(mode: str = "initial") -> str:
    """Helpful prompts when no data is available."""
    if mode == "compare":
        body = (
            "Enter <strong>two or more tickers</strong> separated by commas "
            "(e.g., <strong>DPZ, CMG, SBUX</strong>) to compare valuations side by side."
        )
    elif mode == "compare_failed":
        body = (
            "No tickers in your comparison could be loaded. "
            "Check spelling and try again, or refresh if the data source is temporarily unavailable."
        )
    else:
        body = (
            'Enter a stock ticker above (e.g., <strong>DPZ</strong>) to analyze valuation.'
        )
    return f'{_LAYOUT_CSS}<div class="ve-empty-state">{body}</div>'


def render_skeleton(section: str = "full") -> str:
    """Animated placeholder blocks shown while data loads."""
    if section == "summary":
        blocks = '<div class="ve-skeleton ve-skeleton-header"></div><div class="ve-skeleton ve-skeleton-card"></div>'
    elif section == "table":
        blocks = '<div class="ve-skeleton ve-skeleton-table"></div>'
    else:
        blocks = (
            '<div class="ve-skeleton ve-skeleton-header"></div>'
            '<div class="ve-skeleton ve-skeleton-card"></div>'
            '<div class="ve-skeleton ve-skeleton-table"></div>'
        )
    return f'{_LAYOUT_CSS}<div class="ve-skeleton-wrap">{blocks}</div>'


def render_fetch_status(message: str) -> str:
    return f'{_LAYOUT_CSS}<div class="ve-fetch-status">{message}</div>'


def inject_layout_css() -> None:
    """Inject global dashboard layout styles once."""
    import streamlit as st

    st.markdown(_LAYOUT_CSS, unsafe_allow_html=True)


def fmt_currency(value: float, decimals: int = 2) -> str:
    return f"${value:,.{decimals}f}"


def fmt_large_currency(value: float) -> str:
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs_val >= 1e6:
        return f"${value / 1e6:,.2f}M"
    return fmt_currency(value, 0)


def fmt_pct(value: float, decimals: int = 1, signed: bool = False) -> str:
    if signed:
        return f"{value * 100:+.{decimals}f}%"
    return f"{value * 100:.{decimals}f}%"


def wrap_table(body: str, caption: str | None = None) -> str:
    cap = f'<caption class="sr-only">{caption}</caption>' if caption else ""
    return f'{_TABLE_CSS}<div class="ve-table-wrap"><table class="ve-table">{cap}{body}</table></div>'


def _kv_row(
    label: str,
    value: str,
    alt: bool = False,
    row_class: str = "",
    value_class: str = "",
    tooltip_label: str | None = None,
) -> str:
    alt_class = " ve-row-alt" if alt else ""
    extra = f" {row_class}" if row_class else ""
    vc = f" ve-num {value_class}".strip()
    label_html = _tip_label(tooltip_label or label) if (tooltip_label or label) in TOOLTIPS else label
    return (
        f'<tr class="{alt_class}{extra}">'
        f'<th scope="row" data-label="{label}">{label_html}</th>'
        f'<td class="{vc}" data-label="{label}">{value}</td>'
        f"</tr>"
    )


def render_fundamentals_table(company: Company) -> str:
    op_margin = company.avg_operating_margin
    cagr = company.historical_revenue_cagr
    rows = [
        ("Ticker", company.ticker, False, ""),
        ("Company Name", company.name, True, ""),
        ("Current Price", fmt_currency(company.current_price), False, ""),
        ("Market Cap", fmt_large_currency(company.market_cap), True, ""),
        ("Latest Revenue", fmt_large_currency(company.latest_revenue), False, ""),
        ("Latest EBITDA", fmt_large_currency(company.latest_ebitda), True, ""),
        ("Operating Margin %", fmt_pct(op_margin), False, ""),
        ("Beta", f"{company.beta:.2f}", True, ""),
        ("Revenue CAGR %", fmt_pct(cagr), False, ""),
    ]
    body = "<tbody>" + "".join(_kv_row(l, v, alt, "", vc) for l, v, alt, vc in rows) + "</tbody>"
    return wrap_table(body, "Company fundamentals")


def render_dcf_valuation_table(result: DCFResult) -> str:
    upside = result.upside
    upside_class = "ve-upside-pos" if upside >= 0 else "ve-upside-neg"
    rows = [
        ("Fair Value Per Share", fmt_currency(result.value_per_share), False, "ve-highlight"),
        ("Current Price", fmt_currency(result.company.current_price), True, ""),
        ("Upside/Downside %", fmt_pct(upside, signed=True), False, upside_class),
        ("WACC %", fmt_pct(result.wacc, decimals=2), True, ""),
        ("Enterprise Value", fmt_large_currency(result.enterprise_value), False, ""),
    ]
    body = "<tbody>" + "".join(_kv_row(l, v, alt, vc) for l, v, alt, vc in rows) + "</tbody>"
    return wrap_table(body, "DCF valuation summary")


def _scenario_delta_label(name: str, growth_delta: float, margin_delta: float) -> str:
    if name == "bull":
        return f"+{fmt_pct(growth_delta, decimals=0)} growth, +{fmt_pct(margin_delta, decimals=0)} margin"
    if name == "bear":
        return f"-{fmt_pct(growth_delta, decimals=0)} growth, -{fmt_pct(margin_delta, decimals=0)} margin"
    return "Base assumptions"


def render_scenarios_table(
    results: dict[str, DCFResult],
    growth_delta: float,
    margin_delta: float,
) -> str:
    order = [("bull", "ve-scenario-bull"), ("base", "ve-scenario-base"), ("bear", "ve-scenario-bear")]
    header = (
        "<thead><tr>"
        "<th>Scenario</th><th>Assumption Deltas</th>"
        '<th class="ve-num">Fair Value Per Share</th>'
        '<th class="ve-num">Upside/Downside %</th>'
        "</tr></thead>"
    )
    body_rows = []
    for name, row_class in order:
        r = results[name]
        upside = r.upside
        upside_class = "ve-upside-pos" if upside >= 0 else "ve-upside-neg"
        label = name.capitalize()
        deltas = _scenario_delta_label(name, growth_delta, margin_delta)
        body_rows.append(
            f'<tr class="{row_class}">'
            f'<td data-label="Scenario">{label}</td>'
            f'<td data-label="Assumption Deltas">{deltas}</td>'
            f'<td class="ve-num" data-label="Fair Value Per Share">{fmt_currency(r.value_per_share)}</td>'
            f'<td class="ve-num {upside_class}" data-label="Upside/Downside %">{fmt_pct(upside, signed=True)}</td>'
            f"</tr>"
        )
    return wrap_table(header + "<tbody>" + "".join(body_rows) + "</tbody>", "Scenario analysis")


def _reverse_interpretation(field: str, implied: float | None, company: Company) -> str:
    if implied is None:
        return "Could not solve — current price may require adjusting multiple assumptions."
    if field == "revenue_growth":
        historical = company.historical_revenue_cagr
        diff_pp = (implied - historical) * 100
        return (
            f"Market implies {fmt_pct(implied)} vs {fmt_pct(historical)} historical "
            f"({diff_pp:+.1f} pp)."
        )
    if field == "operating_margin":
        historical = company.avg_operating_margin
        diff_pp = (implied - historical) * 100
        return (
            f"Market implies {fmt_pct(implied)} vs {fmt_pct(historical)} historical "
            f"({diff_pp:+.1f} pp)."
        )
    return f"Implied {field.replace('_', ' ')}: {fmt_pct(implied)}"


def solve_reverse_fields(
    company: Company, assumptions: Assumptions, target: str = "current_price"
) -> dict[str, float | None]:
    """Solve revenue growth and operating margin implied by the market."""
    out: dict[str, float | None] = {}
    for field, bracket in (
        ("revenue_growth", (-0.10, 0.50)),
        ("operating_margin", (0.01, 0.60)),
    ):
        try:
            out[field] = reverse_module.solve(
                company, assumptions, field=field, target=target, bracket=bracket
            )["implied_value"]
        except ValueError:
            out[field] = None
    return out


def render_reverse_dcf_table(
    company: Company,
    assumptions: Assumptions,
    implied: dict[str, float | None],
    target: str = "current_price",
) -> str:
    target_label = "current price" if target == "current_price" else "market cap"
    header = (
        "<thead><tr>"
        "<th>Field</th>"
        '<th class="ve-num">Implied Value</th>'
        "<th>Interpretation</th>"
        '<th class="ve-num">Historical</th>'
        "</tr></thead>"
    )
    fields = [
        ("Revenue Growth", "revenue_growth", company.historical_revenue_cagr),
        ("Operating Margin", "operating_margin", company.avg_operating_margin),
    ]
    rows = []
    for idx, (label, key, historical) in enumerate(fields):
        value = implied.get(key)
        implied_str = fmt_pct(value) if value is not None else "N/A"
        interp = _reverse_interpretation(key, value, company)
        alt = " ve-row-alt" if idx % 2 else ""
        rows.append(
            f'<tr class="{alt.strip()}">'
            f'<td data-label="Field">{label}</td>'
            f'<td class="ve-num" data-label="Implied Value">{implied_str}</td>'
            f'<td data-label="Interpretation">At {target_label}: {interp}</td>'
            f'<td class="ve-num" data-label="Historical">{fmt_pct(historical)}</td>'
            f"</tr>"
        )
    caption = f"Reverse DCF vs {target_label}"
    return wrap_table(header + "<tbody>" + "".join(rows) + "</tbody>", caption)


def run_lbo_at_leverage(company: Company, assumptions: Assumptions, debt_pct: float) -> LBOResult:
    lbo_assumptions = deepcopy(assumptions)
    lbo_assumptions.debt_pct_purchase = debt_pct
    return lbo_module.run(company, lbo_assumptions)


def render_lbo_returns_table(
    company: Company,
    assumptions: Assumptions,
    leverage_levels: tuple[float, ...] = (0.60, 0.75),
) -> str:
    header = (
        "<thead><tr>"
        "<th>Entry Leverage</th>"
        f'<th class="ve-num">{_tip_label("IRR")}</th>'
        f'<th class="ve-num">{_tip_label("MOIC")}</th>'
        '<th class="ve-num">Exit Equity Value</th>'
        "</tr></thead>"
    )
    rows = []
    for idx, leverage in enumerate(leverage_levels):
        result = run_lbo_at_leverage(company, assumptions, leverage)
        alt = " ve-row-alt" if idx % 2 else ""
        rows.append(
            f'<tr class="{alt.strip()}">'
            f'<td data-label="Entry Leverage">{fmt_pct(leverage, decimals=0)}</td>'
            f'<td class="ve-num" data-label="IRR %">{fmt_pct(result.irr, decimals=1)}</td>'
            f'<td class="ve-num" data-label="MOIC">{result.moic:.2f}x</td>'
            f'<td class="ve-num" data-label="Exit Equity Value">{fmt_large_currency(result.exit["exit_equity"])}</td>'
            f"</tr>"
        )
    return wrap_table(header + "<tbody>" + "".join(rows) + "</tbody>", "LBO returns by leverage")


def render_projection_table(dcf_result: DCFResult, lbo_result: LBOResult) -> str:
    dcf_proj = dcf_result.projection
    lbo_proj = lbo_result.projection
    if len(dcf_proj) != len(lbo_proj):
        years = min(len(dcf_proj), len(lbo_proj))
        dcf_proj = dcf_proj.iloc[:years]
        lbo_proj = lbo_proj.iloc[:years]

    header = (
        "<thead><tr>"
        "<th>Year</th>"
        '<th class="ve-num">Revenue</th>'
        '<th class="ve-num">EBITDA</th>'
        '<th class="ve-num">Free Cash Flow</th>'
        '<th class="ve-num">Debt Paydown</th>'
        "</tr></thead>"
    )
    rows = []
    for idx, (_, dcf_row) in enumerate(dcf_proj.iterrows()):
        year = int(dcf_row["year"])
        ebitda = float(dcf_row["ebit"]) + float(dcf_row["da"])
        lbo_row = lbo_proj.iloc[idx]
        debt_paydown = float(lbo_row["debt_paydown"])
        alt = " ve-row-alt" if idx % 2 else ""
        rows.append(
            f'<tr class="{alt.strip()}">'
            f'<td data-label="Year">{year}</td>'
            f'<td class="ve-num" data-label="Revenue">{fmt_large_currency(float(dcf_row["revenue"]))}</td>'
            f'<td class="ve-num" data-label="EBITDA">{fmt_large_currency(ebitda)}</td>'
            f'<td class="ve-num" data-label="Free Cash Flow">{fmt_large_currency(float(dcf_row["fcf"]))}</td>'
            f'<td class="ve-num" data-label="Debt Paydown">{fmt_large_currency(debt_paydown)}</td>'
            f"</tr>"
        )
    return wrap_table(header + "<tbody>" + "".join(rows) + "</tbody>", "Five-year projection")


def sensitivity_cell_color(upside: float) -> str:
    """Map upside vs current price to heatmap background color."""
    if upside <= -0.20:
        return "#e74c3c"
    if upside < 0:
        return "#f1c40f"
    if upside <= 0.20:
        return "#a8e6a1"
    return "#1e8449"


def nearest_grid_cell(
    growth_values: list[float],
    margin_values: list[float],
    target_growth: float | None,
    target_margin: float | None,
) -> tuple[int, int] | None:
    """Return grid indices closest to the target growth/margin pair."""
    if target_growth is None or target_margin is None:
        return None
    best_idx: tuple[int, int] | None = None
    best_dist = float("inf")
    for row_idx, growth in enumerate(growth_values):
        for col_idx, margin in enumerate(margin_values):
            dist = (growth - target_growth) ** 2 + (margin - target_margin) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = (row_idx, col_idx)
    return best_idx


def build_sensitivity_grid(
    company: Company,
    base_assumptions: Assumptions,
    growth_values: list[float] | None = None,
    margin_values: list[float] | None = None,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    """Run DCF across a fixed revenue growth x operating margin grid."""
    growth_values = growth_values or SENSITIVITY_GROWTH_VALUES
    margin_values = margin_values or SENSITIVITY_MARGIN_VALUES
    table = sensitivity_module.run(
        company,
        base_assumptions,
        x_field="operating_margin",
        x_values=margin_values,
        y_field="revenue_growth",
        y_values=growth_values,
        output="value_per_share",
    )
    return table, growth_values, margin_values


def render_sensitivity_reference(current_price: float) -> str:
    return (
        f'<p class="ve-sens-ref">Current stock price (reference): '
        f"<strong>{fmt_currency(current_price)}</strong></p>"
    )


def render_sensitivity_legend() -> str:
    items = [
        ("#e74c3c", "20%+ downside"),
        ("#f1c40f", "0–20% downside"),
        ("#a8e6a1", "0–20% upside"),
        ("#1e8449", "20%+ upside"),
        ("#3498db", "Base case border"),
        ("#e67e22", "Market-implied border"),
    ]
    spans = "".join(
        f'<span><span class="ve-sens-swatch" style="background:{color};"></span>{label}</span>'
        for color, label in items
    )
    return f'<div class="ve-sens-legend">{spans}</div>'


def render_sensitivity_table(
    table: pd.DataFrame,
    growth_values: list[float],
    margin_values: list[float],
    current_price: float,
    base_growth: float,
    base_margin: float,
    market_growth: float | None,
    market_margin: float | None,
) -> str:
    """Build the 2D sensitivity heatmap with base and market-implied highlights."""
    base_cell = nearest_grid_cell(growth_values, margin_values, base_growth, base_margin)
    market_cell = nearest_grid_cell(growth_values, margin_values, market_growth, market_margin)
    grid = table.to_numpy()

    header = (
        "<thead><tr>"
        "<th>Revenue Growth \\ Op. Margin</th>"
        + "".join(f'<th class="ve-num">{margin * 100:.0f}%</th>' for margin in margin_values)
        + "</tr></thead>"
    )

    rows_html: list[str] = []
    for row_idx, growth in enumerate(growth_values):
        cells = f'<th scope="row">{growth * 100:+.0f}%</th>'
        for col_idx, _margin in enumerate(margin_values):
            value = float(grid[row_idx, col_idx])
            upside = (value - current_price) / current_price
            bg = sensitivity_cell_color(upside)
            borders: list[str] = []
            if base_cell == (row_idx, col_idx):
                borders.append("3px solid #3498db")
            if market_cell == (row_idx, col_idx):
                borders.append("3px solid #e67e22")
            if borders:
                border_css = "; ".join(f"box-shadow: inset 0 0 0 {b}" for b in borders)
            else:
                border_css = "border: 1px solid #e2e8f0"
            cells += (
                f'<td class="ve-sens-cell" style="background-color:{bg}; color:#111; '
                f'{border_css};">{fmt_currency(value)}</td>'
            )
        rows_html.append(f"<tr>{cells}</tr>")

    body = "<tbody>" + "".join(rows_html) + "</tbody>"
    return (
        f"{TABLE_CSS}{render_sensitivity_reference(current_price)}"
        f"{render_sensitivity_legend()}"
        f'<div class="ve-table-wrap"><table class="ve-table">{header}{body}</table></div>'
    )


def render_sensitivity_caption(
    base_growth: float,
    base_margin: float,
    market_growth: float | None,
    market_margin: float | None,
) -> str:
    """Footnotes for base case and market-implied assumptions."""
    lines = [
        f"**Base case** (blue border): {fmt_pct(base_growth)} growth, "
        f"{fmt_pct(base_margin)} margin (historical averages).",
    ]
    if market_growth is not None and market_margin is not None:
        lines.append(
            f"**Market-implied** (orange border): {fmt_pct(market_growth)} growth, "
            f"{fmt_pct(market_margin)} margin (justifies current price via reverse DCF)."
        )
    elif market_margin is not None:
        lines.append(
            f"**Market-implied margin** (orange border nearest cell): "
            f"{fmt_pct(market_margin)} margin required at current price."
        )
    elif market_growth is not None:
        lines.append(
            f"**Market-implied growth** (orange border nearest cell): "
            f"{fmt_pct(market_growth)} growth required at current price."
        )
    lines.append(
        "The base case (blue border) assumes historical growth and margins. "
        "The market is pricing in higher growth and margins (see Reverse DCF above). "
        "This table shows all paths to today's valuation."
    )
    return "\n\n".join(lines)


def build_summary_interpretation(
    company: Company,
    dcf_result: DCFResult,
    reverse_implied: dict[str, float | None],
) -> str:
    """One-line valuation interpretation for the summary card."""
    upside = dcf_result.upside
    if upside >= 0:
        lead = f"Trading at {abs(upside) * 100:.0f}% discount to fair value."
    else:
        lead = f"Trading at {abs(upside) * 100:.0f}% premium to fair value."

    hist = company.historical_revenue_cagr
    market_g = reverse_implied.get("revenue_growth")
    if market_g is not None:
        return f"{lead} Market assumes {fmt_pct(market_g)} growth vs {fmt_pct(hist)} historical."
    return f"{lead} Historical revenue CAGR is {fmt_pct(hist)}."


def render_summary_card(
    company: Company,
    dcf_result: DCFResult,
    lbo_result: LBOResult,
    reverse_implied: dict[str, float | None],
    fetched_at: datetime,
) -> str:
    """Top-of-page summary with four key metrics and interpretation."""
    upside = dcf_result.upside
    upside_class = "ve-metric-upside-pos" if upside >= 0 else "ve-metric-upside-neg"
    interpretation = build_summary_interpretation(company, dcf_result, reverse_implied)
    ts = fetched_at.strftime("%b %d, %Y %H:%M UTC")

    metrics = [
        ("Current Price", fmt_currency(company.current_price), "", None),
        ("Fair Value", fmt_currency(dcf_result.value_per_share), "", "Fair Value"),
        ("Upside/Downside", fmt_pct(upside, signed=True), upside_class, "Upside/Downside"),
        ("LBO IRR", fmt_pct(lbo_result.irr), "", "LBO IRR"),
    ]
    grid = "".join(
        f'<div class="ve-metric-box">'
        f'<div class="ve-metric-label">{_tip_label(tip) if tip else label}</div>'
        f'<div class="ve-metric-value {cls}">{value}</div>'
        f"</div>"
        for label, value, cls, tip in metrics
    )

    return (
        f'{_LAYOUT_CSS}<div class="ve-header-block">'
        f'<div class="ve-header-ticker">{company.ticker}</div>'
        f'<div class="ve-header-name">{company.name}</div>'
        f'<div class="ve-timestamp">Data fetched {ts}</div>'
        f"</div>"
        f'<div class="ve-summary-card"><div class="ve-summary-grid">{grid}</div>'
        f'<div class="ve-interpretation">{interpretation}</div></div>'
    )


def build_valuation_summary_csv(
    company: Company,
    dcf_result: DCFResult,
    lbo_result: LBOResult,
    reverse_implied: dict[str, float | None],
    fetched_at: datetime,
) -> str:
    """CSV export of the main valuation summary."""
    implied_g = reverse_implied.get("revenue_growth")
    implied_m = reverse_implied.get("operating_margin")
    row = {
        "Ticker": company.ticker,
        "Company Name": company.name,
        "Fetched At": fetched_at.isoformat(),
        "Current Price": company.current_price,
        "DCF Fair Value": dcf_result.value_per_share,
        "Upside/Downside %": round(dcf_result.upside * 100, 2),
        "WACC %": round(dcf_result.wacc * 100, 2),
        "Enterprise Value": dcf_result.enterprise_value,
        "Equity Value": dcf_result.equity_value,
        "Operating Margin %": round(company.avg_operating_margin * 100, 2),
        "Revenue CAGR %": round(company.historical_revenue_cagr * 100, 2),
        "Market Implied Growth %": round(implied_g * 100, 2) if implied_g is not None else "",
        "Market Implied Margin %": round(implied_m * 100, 2) if implied_m is not None else "",
        "LBO IRR %": round(lbo_result.irr * 100, 2),
        "LBO MOIC": round(lbo_result.moic, 2),
        "Market Cap": company.market_cap,
        "Beta": company.beta,
    }
    buf = StringIO()
    pd.DataFrame([row]).to_csv(buf, index=False)
    return buf.getvalue()


def render_debt_schedule_table(lbo_result: LBOResult) -> str:
    """HTML table for the LBO debt schedule."""
    header = (
        "<thead><tr>"
        "<th>Year</th>"
        '<th class="ve-num">Beginning Balance</th>'
        '<th class="ve-num">Interest</th>'
        '<th class="ve-num">Mandatory Paydown</th>'
        '<th class="ve-num">Sweep Paydown</th>'
        '<th class="ve-num">Ending Balance</th>'
        "</tr></thead>"
    )
    rows = []
    for idx, (_, row) in enumerate(lbo_result.debt_schedule.iterrows()):
        alt = " ve-row-alt" if idx % 2 else ""
        rows.append(
            f'<tr class="{alt.strip()}">'
            f'<td data-label="Year">{int(row["year"])}</td>'
            f'<td class="ve-num" data-label="Beginning Balance">{fmt_large_currency(float(row["beginning_balance"]))}</td>'
            f'<td class="ve-num" data-label="Interest">{fmt_large_currency(float(row["interest_expense"]))}</td>'
            f'<td class="ve-num" data-label="Mandatory Paydown">{fmt_large_currency(float(row["mandatory_paydown"]))}</td>'
            f'<td class="ve-num" data-label="Sweep Paydown">{fmt_large_currency(float(row["sweep_paydown"]))}</td>'
            f'<td class="ve-num" data-label="Ending Balance">{fmt_large_currency(float(row["ending_balance"]))}</td>'
            f"</tr>"
        )
    return wrap_table(header + "<tbody>" + "".join(rows) + "</tbody>", "Debt schedule")


def render_footer() -> str:
    return (
        '<div class="ve-footer">'
        "Data from Valuation Engine. Fair value estimates based on DCF with historical "
        "growth and margin assumptions. Not investment advice."
        "</div>"
    )


def build_peer_rows(
    tickers: list[str],
    fetch_fn,
    projection_years: int,
    risk_free_rate: float,
    equity_risk_premium: float,
    target_debt_weight: float,
    terminal_growth: float,
    use_exit_multiple: bool,
    exit_ev_ebitda_multiple: float,
    entry_ev_ebitda_multiple: float,
    debt_pct_purchase: float,
    lbo_interest_rate: float,
    hold_period_years: int,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    """Fetch and value multiple tickers for peer comparison."""
    from core import dcf as dcf_module

    peer_rows: list[dict] = []
    errors: list[str] = []
    total = len(tickers)
    for index, t in enumerate(tickers, start=1):
        if status_callback:
            status_callback(f"Valuing **{index}** of **{total}**: **{t}**…")
        try:
            c = fetch_fn(t)
            base = Assumptions.calibrated_for(
                c,
                projection_years=projection_years,
                risk_free_rate=risk_free_rate,
                equity_risk_premium=equity_risk_premium,
                target_debt_weight=target_debt_weight,
                terminal_growth=terminal_growth,
                use_exit_multiple=use_exit_multiple,
                exit_ev_ebitda_multiple=exit_ev_ebitda_multiple,
                entry_ev_ebitda_multiple=entry_ev_ebitda_multiple,
                debt_pct_purchase=debt_pct_purchase,
                lbo_debt_interest_rate=lbo_interest_rate,
                hold_period_years=hold_period_years,
                tax_rate=c.effective_tax_rate,
            )
            d = dcf_module.run(c, base)
            l = lbo_module.run(c, base)
            peer_rows.append(compute_peer_metrics(c, base, d, l))
        except Exception as e:
            errors.append(f"**{t}**: {classify_fetch_error(e, t)}")
    peer_rows.sort(key=lambda r: float(r["dcf_upside"]), reverse=True)
    return peer_rows, errors


def show_table(html: str) -> None:
    """Render an HTML table block in Streamlit."""
    import streamlit as st

    st.markdown(html, unsafe_allow_html=True)


def compute_peer_metrics(
    company: Company,
    assumptions: Assumptions,
    dcf_result: DCFResult,
    lbo_result: LBOResult,
) -> dict:
    return {
        "ticker": company.ticker,
        "name": company.name,
        "current_price": company.current_price,
        "market_cap": company.market_cap,
        "operating_margin": company.avg_operating_margin,
        "revenue_cagr": company.historical_revenue_cagr,
        "dcf_fair_value": dcf_result.value_per_share,
        "dcf_upside": dcf_result.upside,
        "lbo_irr": lbo_result.irr,
    }


def _best_keys(peers: list[dict]) -> dict[str, str]:
    """Return the ticker that is best for each metric."""
    if not peers:
        return {}

    def best_by(key: str) -> str:
        return max(peers, key=lambda p: float(p.get(key, float("-inf"))))["ticker"]

    return {
        "operating_margin": best_by("operating_margin"),
        "revenue_cagr": best_by("revenue_cagr"),
        "dcf_upside": best_by("dcf_upside"),
        "lbo_irr": best_by("lbo_irr"),
    }


def render_peer_comparison_table(peers: list[dict]) -> str:
    """
    Side-by-side peer comparison table.

    peers: list of dicts produced by compute_peer_metrics, already sorted as desired.
    """
    if not peers:
        return wrap_table("<tbody></tbody>", "Peer comparison")

    best = _best_keys(peers)

    header = (
        "<thead><tr>"
        "<th>Metric</th>"
        + "".join(f"<th>{p['ticker']}</th>" for p in peers)
        + "</tr></thead>"
    )

    def cell(value: str, is_best: bool = False, bg: str | None = None, extra_class: str = "") -> str:
        best_css = "font-weight:700;" if is_best else ""
        bg_css = f"background-color:{bg};" if bg else ""
        cls = f've-num {extra_class}'.strip()
        return f'<td class="{cls}" style="{bg_css}{best_css}">{value}</td>'

    rows_html: list[str] = []

    # Non-numeric first row (no right align)
    rows_html.append(
        "<tr>"
        '<th scope="row">Company Name</th>'
        + "".join(f'<td data-label="Company Name">{p["name"]}</td>' for p in peers)
        + "</tr>"
    )

    metrics: list[tuple[str, str, callable]] = [
        ("Current Price", "current_price", lambda v: fmt_currency(float(v))),
        ("Market Cap", "market_cap", lambda v: fmt_large_currency(float(v))),
        ("Operating Margin %", "operating_margin", lambda v: fmt_pct(float(v))),
        ("Revenue CAGR %", "revenue_cagr", lambda v: fmt_pct(float(v))),
        ("DCF Fair Value", "dcf_fair_value", lambda v: fmt_currency(float(v))),
        ("Upside/Downside %", "dcf_upside", lambda v: fmt_pct(float(v), signed=True)),
        ("LBO IRR", "lbo_irr", lambda v: fmt_pct(float(v))),
    ]

    for idx, (label, key, fmt) in enumerate(metrics, start=1):
        alt = " ve-row-alt" if idx % 2 else ""
        tds: list[str] = []
        for p in peers:
            is_best = best.get(key) == p["ticker"]
            if key == "dcf_upside":
                bg = sensitivity_cell_color(float(p[key]))
                tds.append(cell(fmt(p[key]), is_best=is_best, bg=bg))
            else:
                tds.append(cell(fmt(p[key]), is_best=is_best))
        rows_html.append(f'<tr class="{alt.strip()}"><th scope="row">{label}</th>' + "".join(tds) + "</tr>")

    body = "<tbody>" + "".join(rows_html) + "</tbody>"
    return f'{TABLE_CSS}<div class="ve-table-wrap"><table class="ve-table">{header}{body}</table></div>'


def summarize_peers(peers: list[dict]) -> dict[str, dict] | dict:
    """Return best candidates for the requested summary bullets."""
    if not peers:
        return {}
    most_undervalued = max(peers, key=lambda p: float(p["dcf_upside"]))
    best_lbo = max(peers, key=lambda p: float(p["lbo_irr"]))
    best_growth = max(peers, key=lambda p: float(p["revenue_cagr"]))
    return {
        "most_undervalued": most_undervalued,
        "best_lbo": best_lbo,
        "best_growth": best_growth,
    }
