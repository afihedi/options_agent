"""
config.py — Edit this file to customize your watchlist, sectors, and risk settings.
"""

# ─── Watchlist ────────────────────────────────────────────────────────────────
DEFAULT_WATCHLIST = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMD",
    "TSLA", "AMZN", "META", "GOOGL",
    "JPM", "GS",
    "XOM", "CVX",
]

# ─── Sector ETFs ──────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology":             "XLK",
    "Energy":                 "XLE",
    "Financials":             "XLF",
    "Healthcare":             "XLV",
    "Consumer Discretionary": "XLY",
    "Industrials":            "XLI",
    "Utilities":              "XLU",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
    "Consumer Staples":       "XLP",
}

# ─── Signal Thresholds ────────────────────────────────────────────────────────
DEFAULT_MIN_CONFIDENCE = 55
DEFAULT_MAX_SIGNALS = 10

# ─── Position Sizing Rules ────────────────────────────────────────────────────
POSITION_SIZING = {
    "high":   {"min_confidence": 75, "capital_pct": 0.05},
    "medium": {"min_confidence": 60, "capital_pct": 0.03},
    "low":    {"min_confidence": 0,  "capital_pct": 0.01},
}

# ─── Data Fetch Settings ──────────────────────────────────────────────────────
NEWS_LIMIT = 6
PRICE_HISTORY_DAYS = 7

# ─── Claude Model ─────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 1200
