# valuationengine

**Open-source DCF and LBO valuation toolkit.** Shipped as a Python library, a command-line tool, an interactive Streamlit dashboard, and an MCP server that plugs into Claude, ChatGPT, and any other LLM client via the [Model Context Protocol](https://modelcontextprotocol.io/).

One engine. Four interfaces. Built for analysts, students, and AI agents that want to run real valuation models instead of guessing.

---

## What it does

`valuationengine` runs the kind of valuation work an equity analyst or private-equity associate would do in Excel, but in seconds and from any interface:

- **Discounted Cash Flow (DCF)** with Gordon growth or exit-multiple terminal value
- **Leveraged Buyout (LBO)** with sources and uses, debt schedules, sponsor IRR and MOIC
- **Reverse DCF** that back-solves the market-implied assumptions baked into the current price
- **Sensitivity tables** across any two assumption fields
- **Scenario analysis** with built-in bull, base, and bear cases
- **Auto-fetched fundamentals** via yfinance with no manual data entry

Every interface (Python, CLI, dashboard, MCP) is a thin wrapper around the same pure-Python engine, so a result computed in one is identical to a result computed in another.

---

## Installation

```bash
git clone https://github.com/rishabhkarnawat/valuationengine.git
cd valuationengine
pip install -r requirements.txt
```

Requires Python 3.10 or higher.

---

## Quick start

### As a Python library

```python
from valuationengine.data.fetcher import fetch_company
from valuationengine.core.models import Assumptions
from valuationengine.core import dcf, reverse

amzn = fetch_company("AMZN")
result = dcf.run(amzn, Assumptions())
print(result.summary())

# Reverse DCF: what is the market currently pricing in?
implied = reverse.solve(amzn, Assumptions(), field="revenue_growth")
print(implied["interpretation"])
```

### From the command line

```bash
python -m valuationengine.adapters.cli dcf AMZN --growth 0.10 --margin 0.20
python -m valuationengine.adapters.cli reverse AMZN
python -m valuationengine.adapters.cli lbo AMZN --entry-multiple 10 --debt-pct 0.6
python -m valuationengine.adapters.cli scenario AMZN
python -m valuationengine.adapters.cli sensitivity AMZN \
    --x-field revenue_growth --x-min 0.04 --x-max 0.12 --x-steps 5 \
    --y-field operating_margin --y-min 0.15 --y-max 0.25 --y-steps 5
```

### As a Streamlit dashboard

```bash
streamlit run valuationengine/adapters/dashboard.py
```

Type a ticker, move the sliders, watch every tab update live: DCF, LBO, reverse DCF, sensitivity, and scenarios.

### As an MCP server (Claude, ChatGPT, Cursor, Continue, anything that speaks MCP)

The MCP server exposes six tools that any Model Context Protocol client can call:

| Tool | What it does |
|------|--------------|
| `fetch_fundamentals` | Pulls revenue, EBITDA, debt, beta, and history for a ticker |
| `run_dcf` | Full DCF with WACC, projection, terminal value, intrinsic value per share |
| `run_lbo` | LBO with sources/uses, debt schedule, sponsor IRR and MOIC |
| `run_reverse_dcf` | Back-solves the market-implied growth, margin, or terminal rate |
| `run_sensitivity` | 2D sensitivity grid across any two assumption fields |
| `run_scenario` | Bull, base, and bear DCF cases side by side |

**Claude Desktop.** Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "valuationengine": {
      "command": "python",
      "args": ["-m", "valuationengine.adapters.mcp_server"],
      "cwd": "/absolute/path/to/valuationengine"
    }
  }
}
```

**Cursor, ChatGPT Desktop, Continue, or any MCP-aware client.** The same JSON block in their respective MCP config files. Point at the local repo.

Once connected, conversations like these work end to end:

> *"What revenue growth rate is the market pricing into Nvidia right now?"*
>
> *"Run a sponsor LBO on Domino's at a 10x entry multiple and 60% debt. What's the IRR?"*
>
> *"Compare DCF intrinsic value to current price for the Mag 7 and rank by upside."*

The LLM calls the tools, receives real numbers from a real model, and reasons about them. No hallucinated valuations.

---

## Architecture

```
valuationengine/
├── core/                pure-Python valuation engine (no I/O)
│   ├── models.py            Company, Assumptions, DCFResult, LBOResult
│   ├── dcf.py               WACC, FCF projection, terminal value, DCF
│   ├── lbo.py               sources/uses, projection, sponsor IRR and MOIC
│   ├── debt.py              amortization schedule with mandatory paydown and cash sweep
│   ├── sensitivity.py       2D grid across any two assumption fields
│   ├── scenario.py          bull, base, bear runner
│   └── reverse.py           reverse DCF solver via scipy brentq
├── data/
│   └── fetcher.py           auto-fetched fundamentals via yfinance
├── adapters/
│   ├── cli.py               Click CLI
│   ├── mcp_server.py        FastMCP server for any LLM client
│   └── dashboard.py         Streamlit dashboard
├── examples/                case study notebooks
└── tests/                   pytest suite, no network calls
```

The core engine never touches the network, never reads files, and never knows about UI. Every adapter calls the same set of functions, which is why the dashboard, the CLI, and the MCP server always produce identical results from identical inputs.

---

## Why reverse DCF

A standard DCF asks: *given these assumptions, what is this company worth?* That question has no clean answer because the assumptions are made up.

A reverse DCF flips it: *given the current price, what assumptions must you believe?* That question has exactly one answer. Compare it to your own view of what is realistic.

`valuationengine` treats reverse DCF as a first-class capability. Solve for implied revenue growth, operating margin, or terminal growth. The MCP server exposes it directly so LLMs can answer "what is the market pricing in?" with a real number instead of a guess.

---

## Case studies

Real-world walkthroughs in `examples/`:

- [**Amazon**](examples/amzn_case_study.ipynb) — what growth does AMZN's current price require, and is that defensible?
- [**Nvidia**](examples/nvda_case_study.ipynb) — a reverse DCF on the most-watched stock of the cycle.
- [**LBO screen**](examples/lbo_case_study.ipynb) — take-private analysis on a candidate, with debt schedule and entry-vs-exit sensitivity.

---

## Testing

```bash
pytest tests/
```

Full pytest suite covering the core engine and the data fetcher. All yfinance calls are mocked, so the suite runs offline and finishes in seconds.

---

## Roadmap

- Direct SEC EDGAR integration as a primary data source
- Trading-comps and transaction-comps modules
- Multi-currency support for international tickers
- Hosted Streamlit Cloud demo

---

## License

MIT. See [LICENSE](LICENSE).

---

## Author

Built by [Rishabh Karnawat](https://github.com/rishabhkarnawat). Issues and pull requests welcome.
