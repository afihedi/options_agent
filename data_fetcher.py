"""
data_fetcher.py — Merges two data sources into one enriched ticker dict.

  yfinance   → live price action, options chain, today's volume, ETF holdings
  defeatbeta → full news article bodies, earnings call transcripts, SEC filings,
               TTM fundamentals, earnings calendar (Nasdaq-sourced)

The combined output is passed to agent.py for Claude to analyze.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
from config import NEWS_LIMIT

from defeatbeta_fetcher import (
    fetch_news_full,
    fetch_latest_transcript_excerpt,
    fetch_recent_sec_filings,
    fetch_next_earnings,
    fetch_fundamentals,
    fetch_treasury_yields,
    format_news_for_prompt,
    format_transcript_for_prompt,
    format_sec_filings_for_prompt,
    format_fundamentals_for_prompt,
    format_yields_for_prompt,
)


def fetch_ticker_data(ticker: str) -> Optional[dict]:
    """
    Fetch a fully enriched data dict for a ticker.

    Live data (yfinance):
      - Current price, 1d/5d % change, volume ratio
      - Options expiration dates
      - Basic company info (fallback for sector/name)

    Deep data (defeatbeta):
      - Full news articles with body text
      - Latest earnings call transcript excerpt
      - Recent SEC 8-K and Form 4 filings
      - TTM EPS, revenue trend, net income
      - Next earnings date (Nasdaq-sourced)

    Returns None if the ticker cannot be resolved at all.
    """
    try:
        # ── yfinance (live) ───────────────────────────────────────────────────
        t = yf.Ticker(ticker)
        info = _safe_info(t)

        quote_type = info.get("quoteType", "")
        is_etf = quote_type in ("ETF", "MUTUALFUND") or ticker.upper() in (
            "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG",
            "XLK", "XLE", "XLF", "XLV", "XLY", "XLI", "XLU", "XLB", "XLRE", "XLC", "XLP",
        )

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not current_price:
            try:
                hist = t.history(period="2d")
                if not hist.empty:
                    current_price = round(float(hist["Close"].iloc[-1]), 2)
            except Exception:
                pass

        # ── defeatbeta (deep) ─────────────────────────────────────────────────
        # Run in parallel conceptually — each call is independent
        news_articles  = fetch_news_full(ticker, max_articles=4)
        transcript     = None if is_etf else fetch_latest_transcript_excerpt(ticker)
        sec_filings    = [] if is_etf else fetch_recent_sec_filings(ticker, days_back=30)
        fundamentals   = {} if is_etf else fetch_fundamentals(ticker)
        treasury       = fetch_treasury_yields()

        # Prefer defeatbeta earnings date (Nasdaq), fall back to yfinance
        earnings_date = None
        if not is_etf:
            earnings_date = fetch_next_earnings(ticker)
            if not earnings_date:
                earnings_date = _next_earnings_yf(t)

        # Sector: prefer defeatbeta profile, fall back to yfinance info
        sector = (
            fundamentals.get("sector")
            or info.get("sector")
            or ("ETF / Fund" if is_etf else "Unknown")
        )

        return {
            # Identity
            "ticker":        ticker.upper(),
            "name":          info.get("longName") or info.get("shortName", ticker),
            "sector":        sector,
            "industry":      fundamentals.get("industry") or info.get("industry", ""),
            "market_cap":    "ETF" if is_etf else _fmt_market_cap(info.get("marketCap")),
            "beta":          info.get("beta"),
            "is_etf":        is_etf,

            # Live price data (yfinance)
            "current_price":    current_price,
            "price_change_1d":  _price_change(t, days=1),
            "price_change_5d":  _price_change(t, days=5),
            "volume_ratio":     _volume_ratio(t, info),

            # Options chain (yfinance)
            "options_expiries": _options_expiries(t),
            "earnings_date":    earnings_date,

            # Rich news (defeatbeta — full article body)
            "news_articles":  news_articles,
            "news_formatted": format_news_for_prompt(news_articles),

            # Earnings call transcript (defeatbeta)
            "transcript":           transcript,
            "transcript_formatted": format_transcript_for_prompt(transcript),

            # SEC filings (defeatbeta)
            "sec_filings":           sec_filings,
            "sec_filings_formatted": format_sec_filings_for_prompt(sec_filings),

            # Fundamentals (defeatbeta)
            "fundamentals":           fundamentals,
            "fundamentals_formatted": format_fundamentals_for_prompt(fundamentals),

            # Macro context (defeatbeta)
            "treasury_yields":    treasury,
            "yields_formatted":   format_yields_for_prompt(treasury),
        }

    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}


# ─── yfinance helpers (live data only) ───────────────────────────────────────

def _safe_info(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}


def _price_change(t: yf.Ticker, days: int) -> Optional[str]:
    try:
        hist = t.history(period=f"{days + 3}d")
        if hist.empty or len(hist) < 2:
            return None
        close = hist["Close"]
        idx = min(days + 1, len(close) - 1)
        pct = ((close.iloc[-1] - close.iloc[-idx]) / close.iloc[-idx]) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.2f}%"
    except Exception:
        return None


def _volume_ratio(t: yf.Ticker, info: dict) -> Optional[str]:
    try:
        vol     = info.get("volume") or info.get("regularMarketVolume")
        avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        if vol and avg_vol and avg_vol > 0:
            return f"{vol / avg_vol:.2f}x avg"
        return None
    except Exception:
        return None


def _next_earnings_yf(t: yf.Ticker) -> Optional[str]:
    """yfinance earnings date fallback."""
    try:
        cal = t.calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
            if dates and len(dates) > 0:
                d = dates[0]
                return d.date().isoformat() if hasattr(d, "date") else str(d)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                return val.date().isoformat() if hasattr(val, "date") else str(val)
        return None
    except Exception:
        return None


def _options_expiries(t: yf.Ticker) -> list[str]:
    try:
        exps = t.options or []
        today = datetime.now().date().isoformat()
        return [e for e in exps if e >= today][:6]
    except Exception:
        return []


def _fmt_market_cap(cap: Optional[int]) -> Optional[str]:
    if cap is None:
        return None
    if cap >= 1e12: return f"${cap/1e12:.1f}T"
    if cap >= 1e9:  return f"${cap/1e9:.1f}B"
    if cap >= 1e6:  return f"${cap/1e6:.1f}M"
    return f"${cap:,}"
