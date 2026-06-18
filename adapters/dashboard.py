"""Streamlit dashboard for interactive valuation"""

import sys
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core import dcf as dcf_module
from core import lbo as lbo_module
from core import scenario as scenario_module
from core import sensitivity as sensitivity_module
from core.models import Assumptions
from data.fetcher import fetch_company

from adapters import dashboard_tables as tables
from adapters import dashboard_ux as ux
from adapters import openai_tools_client as oai

st.set_page_config(page_title="Valuation Engine", layout="wide", page_icon="📊")
# If custom CSS ever breaks interactivity in a browser, you can force plain UI by
# visiting `http://localhost:8501/?plain=1`.
plain_ui = str(st.query_params.get("plain", "0")) == "1"
if not plain_ui:
    tables.inject_layout_css()
else:
    st.info("Plain UI mode enabled (`?plain=1`). Remove the query param to restore themed UI.")


def _style_chart(fig, ax) -> None:
    """Apply dashboard palette to matplotlib figures."""
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")
    ax.title.set_color("#0f172a")
    ax.grid(axis="y", color="#e2e8f0", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


@st.cache_data(show_spinner=False)
def _fetch(ticker: str):
    return fetch_company(ticker)


def _historical_base_assumptions(
    company,
    projection_years: int,
    risk_free_rate: float,
    equity_risk_premium: float,
    target_debt_weight: float,
    terminal_growth: float,
    use_exit_multiple: bool,
    exit_ev_ebitda_multiple: float,
) -> Assumptions:
    """Calibrated operating profile from company history; macro from sidebar."""
    return Assumptions.calibrated_for(
        company,
        projection_years=projection_years,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        target_debt_weight=target_debt_weight,
        terminal_growth=terminal_growth,
        use_exit_multiple=use_exit_multiple,
        exit_ev_ebitda_multiple=exit_ev_ebitda_multiple,
        tax_rate=company.effective_tax_rate,
    )


def _assumptions_cache_key(a: Assumptions) -> tuple:
    """Hashable snapshot of assumptions for cache invalidation."""
    growth = a.revenue_growth if isinstance(a.revenue_growth, float) else tuple(a.revenue_growth)
    margin = a.operating_margin if isinstance(a.operating_margin, float) else tuple(a.operating_margin)
    return (
        growth,
        margin,
        a.projection_years,
        a.risk_free_rate,
        a.equity_risk_premium,
        a.target_debt_weight,
        a.terminal_growth,
        a.use_exit_multiple,
        a.exit_ev_ebitda_multiple,
        a.tax_rate,
        a.capex_pct_revenue,
        a.da_pct_revenue,
        a.nwc_pct_revenue,
    )


@st.cache_data(show_spinner=False)
def _growth_margin_sensitivity_grid(ticker: str, assumptions_key: tuple):
    """Run fixed growth x margin DCF grid."""
    company = _fetch(ticker)
    base_assumptions = Assumptions(
        revenue_growth=assumptions_key[0],
        operating_margin=assumptions_key[1],
        projection_years=assumptions_key[2],
        risk_free_rate=assumptions_key[3],
        equity_risk_premium=assumptions_key[4],
        target_debt_weight=assumptions_key[5],
        terminal_growth=assumptions_key[6],
        use_exit_multiple=assumptions_key[7],
        exit_ev_ebitda_multiple=assumptions_key[8],
        tax_rate=assumptions_key[9],
        capex_pct_revenue=assumptions_key[10],
        da_pct_revenue=assumptions_key[11],
        nwc_pct_revenue=assumptions_key[12],
    )
    return tables.build_sensitivity_grid(company, base_assumptions)


def _build_assumptions(
    company,
    revenue_growth: float,
    operating_margin: float,
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
) -> Assumptions:
    return Assumptions.calibrated_for(
        company,
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


def _clear_data_cache() -> None:
    _fetch.clear()
    _growth_margin_sensitivity_grid.clear()


# ── Header ──────────────────────────────────────────────────────────────────

st.markdown(tables.render_page_hero(), unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh data", help="Clear cached data and re-fetch from the API", use_container_width=True):
        _clear_data_cache()
        st.rerun()

    st.divider()
    st.subheader("OpenAI")
    st.caption("Optional tool-calling layer. Key stays in this session only.")
    use_openai = st.toggle("Enable tool-calling", value=False)
    openai_key_input = st.text_input(
        "API key",
        value="",
        type="password",
        placeholder="sk-…",
        help="Or set OPENAI_API_KEY in your environment.",
        disabled=not use_openai,
    )

search_col, export_col = st.columns([5, 1])
with search_col:
    tickers_input = st.text_input(
        "Analyze ticker(s)",
        value="",
        placeholder="DPZ  ·  DPZ, CMG, SBUX",
        help="One ticker for full analysis, or up to 5 comma-separated tickers to compare.",
    )

tickers, parse_warnings = ux.parse_tickers(tickers_input)
for warning in parse_warnings:
    st.info(warning)

if not tickers:
    tables.show_table(tables.render_empty_state("initial"))
    st.markdown(tables.render_footer(), unsafe_allow_html=True)
    st.stop()

api_key = oai.get_api_key(openai_key_input if use_openai else None)
if use_openai and not api_key:
    st.error("OpenAI API key missing. Add it in the sidebar or set OPENAI_API_KEY.")
    tables.show_table(tables.render_empty_state("initial"))
    st.markdown(tables.render_footer(), unsafe_allow_html=True)
    st.stop()

# ── Fetch with loading UI ───────────────────────────────────────────────────

fetched_at = datetime.now(timezone.utc)
status_slot = st.empty()
skeleton_slot = st.empty()
skeleton_slot.markdown(tables.render_skeleton("full"), unsafe_allow_html=True)


def _update_status(msg: str) -> None:
    status_slot.markdown(tables.render_fetch_status(msg), unsafe_allow_html=True)


companies, fetch_errors = ux.fetch_companies_batch(tickers, _fetch, _update_status)
skeleton_slot.empty()
status_slot.empty()

successful_tickers = [t for t in tickers if t in companies]

if not successful_tickers:
    if len(tickers) == 1:
        st.error(fetch_errors.get(tickers[0], ux.MSG_API_FAILURE))
    else:
        st.error("None of the tickers could be loaded.")
        for t, msg in fetch_errors.items():
            st.error(f"**{t}**: {msg}")
        tables.show_table(tables.render_empty_state("compare_failed"))
    st.markdown(tables.render_footer(), unsafe_allow_html=True)
    st.stop()

if fetch_errors:
    for t, msg in fetch_errors.items():
        st.warning(f"**{t}**: {msg}")

company = companies[successful_tickers[0]]

# ── Sidebar assumptions ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Assumptions")
    if len(successful_tickers) > 1:
        active_ticker = st.selectbox("Detailed view", successful_tickers, key="active_ticker")
        company = companies[active_ticker]

    st.caption(f"Operating defaults from {company.ticker}'s 5-year history.")
    default_growth = float(np.clip(company.historical_revenue_cagr, -0.10, 0.40))
    default_margin = float(np.clip(company.avg_operating_margin, 0.01, 0.50))
    revenue_growth = st.slider(
        "Revenue growth", -0.10, 0.40, default_growth, 0.01, format="%.2f",
        key=f"growth_{company.ticker}",
    )
    operating_margin = st.slider(
        "Operating margin (EBIT)", 0.0, 0.50, default_margin, 0.01, format="%.2f",
        key=f"margin_{company.ticker}",
    )
    projection_years = st.slider("Projection years", 3, 10, 5, key=f"years_{company.ticker}")

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

macro = dict(
    projection_years=projection_years,
    risk_free_rate=risk_free_rate,
    equity_risk_premium=equity_risk_premium,
    target_debt_weight=target_debt_weight,
    terminal_growth=terminal_growth,
    use_exit_multiple=use_exit_multiple,
    exit_ev_ebitda_multiple=exit_ev_ebitda_multiple,
    entry_ev_ebitda_multiple=entry_ev_ebitda_multiple,
    debt_pct_purchase=debt_pct_purchase,
    lbo_interest_rate=lbo_interest_rate,
    hold_period_years=hold_period_years,
)

assumptions = _build_assumptions(company, revenue_growth, operating_margin, **macro)

openai_result_json = None
if use_openai:
    def _tool_exec(name: str, args: dict) -> dict:
        from adapters import mcp_server as mcp_tools

        if name == "valuationengine_fetch_fundamentals":
            return mcp_tools.fetch_fundamentals(**args)
        if name == "valuationengine_run_dcf":
            return mcp_tools.run_dcf(**args)
        if name == "valuationengine_run_scenario":
            return mcp_tools.run_scenario(**args)
        if name == "valuationengine_run_reverse_dcf":
            return mcp_tools.run_reverse_dcf(**args)
        if name == "valuationengine_run_lbo":
            return mcp_tools.run_lbo(**args)
        return {"error": f"Unknown tool: {name}"}

    prompt = (
        f"Analyze the valuation of {company.ticker}. "
        "Call valuationengine_fetch_fundamentals, valuationengine_run_dcf, "
        "valuationengine_run_lbo, and valuationengine_run_reverse_dcf. "
        "Return JSON with keys: fundamentals, dcf, lbo, reverse."
    )
    with st.spinner("Running OpenAI tool-calling…"):
        try:
            openai_result_json = oai.run_tool_calling_session(api_key, prompt, _tool_exec)
            st.success("OpenAI tool-calling completed.")
        except oai.OpenAIRateLimit as e:
            st.warning(str(e))
        except oai.OpenAITokenLimit as e:
            st.warning(str(e))
        except oai.OpenAIToolsUnavailable as e:
            st.warning(str(e))
        except oai.OpenAIError as e:
            st.warning(f"OpenAI tool-calling failed: {e}")
        except Exception as e:
            st.warning(f"OpenAI tool-calling failed: {e}")

dcf_result = dcf_module.run(company, assumptions)
lbo_base_result = lbo_module.run(company, assumptions)
reverse_implied = tables.solve_reverse_fields(company, assumptions)

# ── Summary card + export ───────────────────────────────────────────────────

tables.show_table(
    tables.render_summary_card(company, dcf_result, lbo_base_result, reverse_implied, fetched_at)
)

with export_col:
    st.write("")
    st.write("")
    csv_data = tables.build_valuation_summary_csv(
        company, dcf_result, lbo_base_result, reverse_implied, fetched_at
    )
    st.download_button(
        label="Export CSV",
        data=csv_data,
        file_name=f"{company.ticker}_valuation_summary.csv",
        mime="text/csv",
    )

# ── Tabs ────────────────────────────────────────────────────────────────────

st.markdown(tables.render_tab_nav_hint(), unsafe_allow_html=True)

tab_labels = [
    "Fundamentals",
    "DCF & Scenarios",
    "Sensitivity Grid",
    "LBO Analysis",
]
if len(successful_tickers) > 1:
    tab_labels.append("Compare Peers")

tabs = st.tabs(tab_labels)
tab_fund, tab_val, tab_sens, tab_lbo = tabs[:4]
tab_compare = tabs[4] if len(successful_tickers) > 1 else None

with tab_fund:
    st.markdown(tables.render_section_title("Company fundamentals", "Trailing history and market data"), unsafe_allow_html=True)
    tables.show_table(tables.render_fundamentals_table(company))

with tab_val:
    st.markdown(tables.render_section_title("DCF valuation", "Intrinsic value from projected free cash flows"), unsafe_allow_html=True)
    tables.show_table(tables.render_dcf_valuation_table(dcf_result))

    st.markdown(tables.render_section_title("Scenario analysis", "Bull, base, and bear cases"), unsafe_allow_html=True)
    growth_delta = st.slider("Growth delta", 0.0, 0.10, 0.03, 0.01, key="scen_growth_delta")
    margin_delta = st.slider("Margin delta", 0.0, 0.10, 0.03, 0.01, key="scen_margin_delta")
    scenarios = scenario_module.build_bull_base_bear(
        assumptions, growth_delta=growth_delta, margin_delta=margin_delta
    )
    scenario_results = scenario_module.run(company, scenarios)
    tables.show_table(
        tables.render_scenarios_table(scenario_results, growth_delta, margin_delta)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Bull", "Base", "Bear"]
    values = [scenario_results[k].value_per_share for k in ("bull", "base", "bear")]
    colors = ["#059669", "#6366f1", "#dc2626"]
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.axhline(company.current_price, color="#0f172a", linestyle="--", linewidth=1.2, label="Current price")
    ax.set_ylabel("Intrinsic value per share ($)")
    _style_chart(fig, ax)
    ax.legend(frameon=False, loc="upper right")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${val:,.0f}",
                ha="center", va="bottom", fontsize=9, color="#334155")
    st.pyplot(fig)

    st.markdown(tables.render_section_title("Reverse DCF", "What the market is pricing in"), unsafe_allow_html=True)
    target = st.selectbox(
        "Match to",
        ["current_price", "market_cap"],
        format_func=lambda x: "Current price" if x == "current_price" else "Market cap",
        key="reverse_target",
    )
    if target != "current_price":
        reverse_implied = tables.solve_reverse_fields(company, assumptions, target=target)
    tables.show_table(
        tables.render_reverse_dcf_table(company, assumptions, reverse_implied, target=target)
    )

with tab_sens:
    st.markdown(
        tables.render_section_title(
            "Growth × margin sensitivity",
            "Fair value per share across revenue growth (rows) and operating margin (columns)",
        ),
        unsafe_allow_html=True,
    )
    historical_base = _historical_base_assumptions(company, **{k: macro[k] for k in (
        "projection_years", "risk_free_rate", "equity_risk_premium", "target_debt_weight",
        "terminal_growth", "use_exit_multiple", "exit_ev_ebitda_multiple",
    )})
    cache_key = _assumptions_cache_key(historical_base)
    sens_skeleton = st.empty()
    sens_skeleton.markdown(tables.render_skeleton("table"), unsafe_allow_html=True)
    sens_status = st.empty()
    sens_status.markdown(
        tables.render_fetch_status(f"Building sensitivity grid for **{company.ticker}**…"),
        unsafe_allow_html=True,
    )
    sens_table, growth_values, margin_values = _growth_margin_sensitivity_grid(
        company.ticker, cache_key
    )
    sens_skeleton.empty()
    sens_status.empty()
    market_growth = reverse_implied.get("revenue_growth")
    market_margin = reverse_implied.get("operating_margin")
    tables.show_table(
        tables.render_sensitivity_table(
            sens_table,
            growth_values,
            margin_values,
            company.current_price,
            float(historical_base.revenue_growth),
            float(historical_base.operating_margin),
            market_growth,
            market_margin,
        )
    )
    st.markdown(
        tables.render_sensitivity_caption(
            float(historical_base.revenue_growth),
            float(historical_base.operating_margin),
            market_growth,
            market_margin,
        )
    )
    with st.expander("Custom sensitivity (advanced)"):
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
        if st.button("Run custom sensitivity"):
            x_values = list(np.linspace(x_min, x_max, int(x_steps)))
            y_values = list(np.linspace(y_min, y_max, int(y_steps)))
            custom_table = sensitivity_module.run(
                company, assumptions, x_field, x_values, y_field, y_values
            )
            st.dataframe(
                custom_table.style.format("{:,.2f}").background_gradient(cmap="RdYlGn"),
                use_container_width=True,
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(custom_table.values, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels([f"{v:.2f}" for v in x_values])
            ax.set_yticks(range(len(y_values)))
            ax.set_yticklabels([f"{v:.2f}" for v in y_values])
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            _style_chart(fig, ax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

with tab_lbo:
    st.markdown(tables.render_section_title("LBO returns", "Sponsor IRR and MOIC at 60% and 75% leverage"), unsafe_allow_html=True)
    tables.show_table(tables.render_lbo_returns_table(company, assumptions))
    st.markdown(tables.render_section_title("5-year projection", "Operating and cash flow forecast"), unsafe_allow_html=True)
    tables.show_table(tables.render_projection_table(dcf_result, lbo_base_result))
    st.markdown(tables.render_section_title("Debt schedule", "Mandatory paydown and cash sweep"), unsafe_allow_html=True)
    tables.show_table(tables.render_debt_schedule_table(lbo_base_result))

if tab_compare is not None:
    with tab_compare:
        if len(successful_tickers) < 2:
            tables.show_table(tables.render_empty_state("compare"))
        else:
            st.markdown(
                tables.render_section_title("Peer comparison", "Historical base per company, shared macro inputs"),
                unsafe_allow_html=True,
            )
            compare_skeleton = st.empty()
            compare_skeleton.markdown(tables.render_skeleton("table"), unsafe_allow_html=True)
            compare_status = st.empty()

            def _compare_status(msg: str) -> None:
                compare_status.markdown(tables.render_fetch_status(msg), unsafe_allow_html=True)

            peer_rows, peer_errors = tables.build_peer_rows(
                successful_tickers, _fetch, status_callback=_compare_status, **macro
            )
            compare_skeleton.empty()
            compare_status.empty()

            if peer_errors:
                for err in peer_errors:
                    st.warning(err)
            if not peer_rows:
                tables.show_table(tables.render_empty_state("compare_failed"))
            else:
                tables.show_table(tables.render_peer_comparison_table(peer_rows))
                summary = tables.summarize_peers(peer_rows)
                if summary:
                    st.markdown(tables.render_peer_insights(summary), unsafe_allow_html=True)

st.markdown(tables.render_footer(), unsafe_allow_html=True)
