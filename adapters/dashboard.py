"""Streamlit dashboard for interactive valuation"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import dcf as dcf_module
from core import lbo as lbo_module
from core import reverse as reverse_module
from core import scenario as scenario_module
from core import sensitivity as sensitivity_module
from core.models import Assumptions
from data.fetcher import fetch_company

st.set_page_config(page_title="Valuation Engine", layout="wide", page_icon="📊")


@st.cache_data(show_spinner="Fetching fundamentals...")
def _fetch(ticker: str):
    return fetch_company(ticker)


st.title("Open-Source Valuation Engine")
st.caption("DCF, LBO, sensitivity, scenarios, and reverse DCF on any public company.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AMZN").strip().upper()

    st.subheader("Operating assumptions")
    revenue_growth = st.slider("Revenue growth", -0.10, 0.40, 0.08, 0.01, format="%.2f")
    operating_margin = st.slider("Operating margin (EBIT)", 0.0, 0.50, 0.20, 0.01, format="%.2f")
    projection_years = st.slider("Projection years", 3, 10, 5)

    st.subheader("WACC")
    risk_free_rate = st.slider("Risk-free rate", 0.0, 0.10, 0.045, 0.005, format="%.3f")
    equity_risk_premium = st.slider("Equity risk premium", 0.0, 0.10, 0.055, 0.005, format="%.3f")
    target_debt_weight = st.slider("Target debt weight", 0.0, 0.80, 0.30, 0.05, format="%.2f")

    st.subheader("Terminal value")
    use_exit_multiple = st.checkbox("Use exit multiple instead of Gordon growth")
    if use_exit_multiple:
        exit_ev_ebitda_multiple = st.slider("Exit EV/EBITDA", 4.0, 25.0, 10.0, 0.5)
        terminal_growth = 0.025
    else:
        terminal_growth = st.slider("Terminal growth", 0.0, 0.05, 0.025, 0.005, format="%.3f")
        exit_ev_ebitda_multiple = 10.0

    st.subheader("LBO")
    entry_ev_ebitda_multiple = st.slider("Entry EV/EBITDA", 4.0, 25.0, 10.0, 0.5)
    debt_pct_purchase = st.slider("Debt % of purchase", 0.0, 0.80, 0.60, 0.05)
    lbo_interest_rate = st.slider("LBO debt interest rate", 0.0, 0.15, 0.08, 0.005, format="%.3f")
    hold_period_years = st.slider("Hold period (years)", 3, 7, 5)

try:
    company = _fetch(ticker)
except Exception as e:
    st.error(f"Could not fetch data for {ticker}: {e}")
    st.stop()

assumptions = Assumptions(
    revenue_growth=revenue_growth,
    operating_margin=operating_margin,
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
    tax_rate=company.effective_tax_rate,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Company", company.name)
col2.metric("Price", f"${company.current_price:,.2f}")
col3.metric("Market Cap", f"${company.market_cap / 1e9:,.2f}B")
col4.metric("Beta", f"{company.beta:.2f}")

tab_dcf, tab_lbo, tab_reverse, tab_sens, tab_scen = st.tabs(
    ["DCF", "LBO", "Reverse DCF", "Sensitivity", "Scenarios"]
)

with tab_dcf:
    result = dcf_module.run(company, assumptions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Intrinsic value / share", f"${result.value_per_share:,.2f}")
    c2.metric("Current price", f"${company.current_price:,.2f}")
    c3.metric("Upside", f"{result.upside * 100:+.1f}%")
    st.metric("WACC", f"{result.wacc * 100:.2f}%")
    st.subheader("FCF projection")
    st.dataframe(result.projection, use_container_width=True)

with tab_lbo:
    result = lbo_module.run(company, assumptions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sponsor IRR", f"{result.irr * 100:.1f}%")
    c2.metric("MOIC", f"{result.moic:.2f}x")
    c3.metric("Equity check", f"${result.sources_and_uses['equity_check'] / 1e9:,.2f}B")
    st.subheader("Projection")
    st.dataframe(result.projection, use_container_width=True)
    st.subheader("Debt schedule")
    st.dataframe(result.debt_schedule, use_container_width=True)

with tab_reverse:
    field = st.selectbox("Solve for", ["revenue_growth", "operating_margin", "terminal_growth"])
    target = st.selectbox("Match to", ["market_cap", "current_price"])
    if st.button("Run reverse DCF"):
        try:
            res = reverse_module.solve(company, assumptions, field=field, target=target)
            st.success(res["interpretation"])
            st.json(res)
        except Exception as e:
            st.error(str(e))

with tab_sens:
    c1, c2 = st.columns(2)
    with c1:
        x_field = st.selectbox(
            "X axis field",
            ["revenue_growth", "operating_margin", "terminal_growth", "risk_free_rate"],
            key="xf",
        )
        x_min = st.number_input("X min", value=0.04, step=0.01, key="xmin")
        x_max = st.number_input("X max", value=0.12, step=0.01, key="xmax")
        x_steps = st.number_input("X steps", value=5, min_value=2, max_value=15, key="xs")
    with c2:
        y_field = st.selectbox(
            "Y axis field",
            ["operating_margin", "revenue_growth", "terminal_growth", "risk_free_rate"],
            key="yf",
        )
        y_min = st.number_input("Y min", value=0.15, step=0.01, key="ymin")
        y_max = st.number_input("Y max", value=0.25, step=0.01, key="ymax")
        y_steps = st.number_input("Y steps", value=5, min_value=2, max_value=15, key="ys")
    if st.button("Run sensitivity"):
        x_values = list(np.linspace(x_min, x_max, int(x_steps)))
        y_values = list(np.linspace(y_min, y_max, int(y_steps)))
        table = sensitivity_module.run(company, assumptions, x_field, x_values, y_field, y_values)
        st.dataframe(
            table.style.format("{:,.2f}").background_gradient(cmap="RdYlGn"),
            use_container_width=True,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(table.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([f"{v:.2f}" for v in x_values])
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([f"{v:.2f}" for v in y_values])
        ax.set_xlabel(x_field)
        ax.set_ylabel(y_field)
        plt.colorbar(im)
        st.pyplot(fig)

with tab_scen:
    growth_delta = st.slider("Growth delta", 0.0, 0.10, 0.03, 0.01)
    margin_delta = st.slider("Margin delta", 0.0, 0.10, 0.03, 0.01)
    scenarios = scenario_module.build_bull_base_bear(
        assumptions, growth_delta=growth_delta, margin_delta=margin_delta
    )
    results = scenario_module.run(company, scenarios)
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "scenario": name,
                "value_per_share": r.value_per_share,
                "upside_pct": r.upside * 100,
                "wacc": r.wacc * 100,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format(
            {
                "value_per_share": "${:,.2f}",
                "upside_pct": "{:+.1f}%",
                "wacc": "{:.2f}%",
            }
        ),
        use_container_width=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["scenario"], df["value_per_share"], color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.axhline(company.current_price, color="black", linestyle="--", label="Current price")
    ax.set_ylabel("Intrinsic value per share ($)")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.caption(
    "Open-source valuation toolkit. Data via yfinance. "
    "No part of this output constitutes investment advice."
)
