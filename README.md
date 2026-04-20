# options_trader
# Options Signal Agent

> **v1.0** — An AI-powered daily options briefing tool that scans the market from the top down: sector strength → category leadership → individual ticker momentum → AI signal generation.

---

## What It Does

Most options tools give you a ticker and ask you to figure out the context yourself. This agent works the other way — it starts with the full market, identifies which sectors and sub-categories are leading or lagging, surfaces the most relevant individual names, and then uses Claude AI to synthesize news, fundamentals, earnings call transcripts, and SEC filings into a directional signal with a recommended expiration and position size.

A typical run looks like this:

```
Phase 1  Rank all 11 GICS sectors by momentum vs SPY
Phase 2  Within top/bottom sectors, rank industry categories (e.g. Semiconductors vs Software)
Phase 3  Within leading categories, score individual holdings by momentum
Phase 4  For each surfaced ticker, fetch deep data and generate an AI signal
```

The output is a prioritized list of put and call candidates — with reasoning, expiration rationale, and a suggested dollar allocation based on your capital.

---

## Architecture

```
options-agent/
├── main.py                  Entry point, CLI, orchestration, terminal output
├── sector_analyzer.py       Ranks all 11 GICS sectors vs SPY by relative strength
├── category_analyzer.py     Groups sector holdings by industry, ranks categories
├── ticker_discovery.py      Surfaces individual tickers from leading categories
├── data_fetcher.py          Merges live data (yfinance) with deep data (defeatbeta)
├── defeatbeta_fetcher.py    Full news bodies, transcripts, SEC filings, fundamentals
├── agent.py                 Sends enriched data to Claude, parses signal JSON
├── config.py                Watchlist, position sizing, thresholds — edit this
├── requirements.txt
└── .env.example
```

### Data sources

| Source | What it provides | Update frequency |
|---|---|---|
| **yfinance** | Live price, volume, options chain, ETF holdings | Real-time |
| **defeatbeta-api** | Full news article bodies, earnings call transcripts, SEC filings, TTM EPS, revenue trend, earnings calendar | Weekly |
| **Anthropic API** | Signal generation, reasoning, expiration selection, position sizing | Per request |

> **Note on defeatbeta:** Data is refreshed weekly, not intraday. It is best used for fundamental context, transcript language, and SEC filings — not as a substitute for live price data.

---

## Setup

### Requirements

- Python 3.10 or higher
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Install

```bash
# Clone or unzip the project
cd options-agent

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure your API key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

No other API keys are required. defeatbeta pulls from Hugging Face publicly with no authentication.

---

## Usage

### Basic run (auto-discovers tickers from sector rankings)
```bash
python main.py
```
You will be prompted for your available trading capital.

### Pass capital and run immediately
```bash
python main.py --capital 5000
```

### Analyze specific tickers only
```bash
python main.py --tickers NVDA TSLA AMD META --capital 10000
```

### Widen the discovery net
```bash
# Pull from top 4 sectors, 5 tickers each
python main.py --capital 5000 --top-sectors 4 --tickers-per-sector 5
```

### Call candidates only (skip put sectors)
```bash
python main.py --capital 5000 --no-puts
```

### Save results to JSON
```bash
python main.py --capital 5000 --output signals.json
```

### All flags

| Flag | Description | Default |
|---|---|---|
| `--tickers` | Specific tickers to analyze. Skips auto-discovery if provided. | None (auto-discover) |
| `--capital` | Available trading capital in USD | Prompted |
| `--max-signals` | Maximum signals to display | 10 |
| `--min-confidence` | Only show signals at or above this confidence % | 55 |
| `--top-sectors` | How many top/bottom sectors to draw tickers from | 3 |
| `--tickers-per-sector` | Tickers to surface per sector | 3 |
| `--include-sectors` | Also analyze sector ETFs directly (XLK, XLE, etc.) | False |
| `--no-puts` | Skip put candidates from weak sectors | False |
| `--output` | Save results as JSON to this path | None |

---

## How Signals Are Generated

### 1. Sector ranking

All 11 GICS sectors are scored against SPY using a composite momentum formula across three windows:

- **5-day relative strength** (heaviest weight — medium-term trend)
- **20-day relative strength** (longer-term context)
- **1-day relative strength** (today's price action)
- **Volume ratio** (conviction modifier)

Sectors are ranked 1–11. The market bias (risk-on / risk-off / mixed) is derived from SPY direction and the breadth of bullish sectors.

### 2. Category ranking

Within each top/bottom sector, holdings are grouped by industry (e.g. Semiconductors, Software, IT Services within Technology). Each category is scored by the average momentum of its members. This prevents a single hot stock from masking weakness across the rest of its peer group.

### 3. Ticker discovery

Individual tickers are scored within their category by a weighted momentum score:

```
momentum = (5d % change × 0.6) + (1d % change × 0.4)
```

Tickers below the minimum average daily volume (3M shares/day by default) are excluded to ensure options liquidity.

### 4. AI signal generation

Each discovered ticker is sent to Claude with a full data package:

- Full news article bodies (not just headlines)
- Latest earnings call transcript excerpt — management guidance language, Q&A tone
- Recent SEC 8-K filings (material events) and Form 4 insider activity
- TTM EPS, quarterly revenue trend, net income
- Treasury yield curve (macro context)
- Sector rank and category rank
- Live price action and available options expiration dates

Claude synthesizes all of this into a structured JSON signal. Confidence is calibrated against data agreement across layers — a bullish news catalyst in the top-ranked sector with confirming price action and positive management guidance will score significantly higher than the same headline in a weak sector with flat price action.

### 5. Position sizing

Suggested allocation is calculated from your stated capital and the signal's confidence tier:

| Tier | Confidence | Allocation |
|---|---|---|
| High | ≥ 75% | 5% of capital |
| Medium | ≥ 60% | 3% of capital |
| Low | < 60% | 1% of capital |

These tiers are configurable in `config.py`.

---

## Signal Output

Each signal includes:

- **Direction** — CALL / PUT / NEUTRAL
- **Confidence** — 40–90%
- **Expiration** — chosen from the ticker's actual options chain (no hallucinated dates)
- **Strike guidance** — e.g. "slightly OTM, 1–2 strikes above current price"
- **Suggested allocation** — dollar amount based on your capital and confidence tier
- **Catalysts** — specific facts pulled from news or transcripts
- **Sector context** — how sector tailwind/headwind affects the signal
- **Category context** — which sub-industry is leading within the sector
- **Transcript signal** — management tone from latest earnings call
- **SEC signal** — any material 8-K or insider Form 4 activity
- **Macro context** — yield curve interpretation
- **Reasoning** — 3–4 sentence synthesis of all data layers
- **Risks** — what could invalidate the trade

---

## Configuration

All tunable parameters live in `config.py`:

```python
# Tickers scanned when no --tickers flag is passed
DEFAULT_WATCHLIST = [...]

# Sector ETFs used for ranking and category analysis
SECTOR_ETFS = {...}

# Position sizing tiers
POSITION_SIZING = {
    "high":   {"min_confidence": 75, "capital_pct": 0.05},
    "medium": {"min_confidence": 60, "capital_pct": 0.03},
    "low":    {"min_confidence": 0,  "capital_pct": 0.01},
}

# Claude model
CLAUDE_MODEL = "claude-sonnet-4-20250514"
```

Two additional constants live at the top of their respective files:

- `MAX_HOLDINGS_PER_SECTOR` in `ticker_discovery.py` — how deep into each ETF's holdings to look (default: 30)
- `MIN_AVG_VOLUME` in `ticker_discovery.py` — minimum daily volume for options liquidity gate (default: 3,000,000)
- `MAX_HOLDINGS_TO_CLASSIFY` in `category_analyzer.py` — how many holdings to classify per sector for category analysis (default: 30)

---

## Known Limitations (v1)

**defeatbeta data is weekly, not real-time.** Earnings call transcripts and SEC filings may lag by several days. Always verify material events against live sources before trading.

**No implied volatility data.** The agent does not have access to IV, IV rank, or options chain greeks. A technically correct directional signal can still be a losing trade if IV is sky-high and crush is imminent after a catalyst.

**No options chain depth.** Strike selection is guidance only — the agent cannot see bid/ask spreads, open interest, or actual premium levels.

**yfinance can be fragile.** ETF holdings data (used for category classification) depends on `funds_data.top_holdings` which can return incomplete results depending on the ETF. The agent falls back gracefully but some sectors may classify fewer holdings than others.

**Signals are not buy orders.** Confidence scores reflect data agreement, not win probability. A 78% confidence signal means the data layers align — not that the trade has a 78% chance of being profitable.

---

## Disclaimer

This tool is for **informational and educational purposes only**. It is not financial advice. Options trading involves substantial risk of loss and is not appropriate for all investors. Past signal quality does not guarantee future results. Always conduct your own due diligence and consult a licensed financial advisor before placing any trade.
