"""
agent.py — Sends the fully enriched ticker data to Claude for signal generation.

With defeatbeta integration, Claude now receives:
  - Full news article bodies (not just headlines)
  - Earnings call transcript excerpts (management guidance, Q&A)
  - Recent SEC filings (8-K material events, Form 4 insider activity)
  - TTM EPS, revenue trend, net income
  - Treasury yield curve (macro context)
  - Sector strength rankings (from sector_analyzer)
  - Live price action and options expiry dates (from yfinance)
"""

import json
import anthropic
from datetime import date
from typing import Optional
from config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS, POSITION_SIZING
from sector_analyzer import SectorReport


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional options analyst generating daily trade signals.

You receive deeply enriched market data for each ticker including:
  - Full news article bodies (not just headlines) — read the actual content
  - Recent earnings call transcript excerpts — pay close attention to management
    guidance language, outlook statements, and analyst Q&A tone
  - Recent SEC filings — 8-K filings often contain material events before news
    covers them; Form 4 insider buying/selling is a strong directional signal
  - Fundamental metrics: TTM EPS, revenue trend, net income
  - Treasury yield curve — inverted curve = risk-off; normal curve = risk-on
  - Sector strength ranking — how this sector ranks vs all others vs SPY
  - Live price action (1d, 5d % change) and volume
  - Available options expiration dates (choose ONLY from this list)
  - The trader's available capital and position sizing tiers

Your analytical process:
1. Read the full news body text — extract specific facts, not just vibes
2. Check the earnings call transcript for management tone and guidance language.
   Phrases like "cautious on Q2", "headwinds", "revisiting targets" = bearish.
   "Accelerating", "raised guidance", "strong pipeline" = bullish.
3. Flag any 8-K filings — these are often the most time-sensitive signal.
   Insider Form 4 buying by executives = bullish. Selling = neutral to bearish
   (selling is common but large coordinated selling is notable).
4. Layer in sector strength — a bullish setup in the #1 sector is higher
   conviction than the same setup in the #9 sector.
5. Check the yield curve — if it's inverted, bias against aggressive calls.
6. Weigh all of this against price action and volume.

Return ONLY a valid JSON object. No markdown fences. No preamble.

JSON schema:
{
  "ticker": "NVDA",
  "signal": "call" | "put" | "neutral",
  "confidence": <integer 40-90>,
  "expiration": "<date from available_expiries only>",
  "expiration_rationale": "<one sentence>",
  "strike_guidance": "<e.g. 'slightly OTM, 1-2 strikes above current price'>",
  "suggested_capital_usd": <float>,
  "position_size_rationale": "<one sentence>",
  "catalysts": ["<specific fact from news or transcript>", "...", "..."],
  "sector_context": "<one sentence on sector tailwind/headwind>",
  "fundamental_context": "<one sentence on EPS/revenue trend>",
  "transcript_signal": "<one sentence on management tone, or 'No transcript available'>",
  "sec_signal": "<one sentence on any material 8-K or insider activity, or 'None'>",
  "macro_context": "<one sentence on yield curve and risk environment>",
  "reasoning": "<3-4 sentences synthesizing ALL data layers into a coherent thesis>",
  "risks": "<1-2 sentences on what could invalidate this trade>",
  "alert_priority": "high" | "medium" | "low",
  "watch_notes": "<optional: context worth monitoring>"
}

Confidence calibration:
  High conviction (75-88): Multiple data layers agree — news + transcript + price + sector all aligned
  Medium conviction (60-74): 2-3 layers agree, 1 is neutral or missing
  Low conviction (48-59): Mixed signals or thin data
  Neutral (<48 or signal=neutral): Contradictory signals or insufficient data to form a view

Expiration rules (CRITICAL — only choose from available_expiries):
  - 8-K material event or earnings within 14 days → expiry AFTER the event
  - Clear short-term catalyst (analyst day, product launch) → 7-14 DTE weekly
  - Fundamental trend + sector momentum (no near-term catalyst) → 21-40 DTE
  - Macro or speculative thesis → 30-50 DTE
"""


# ─── Public function ──────────────────────────────────────────────────────────

def generate_signal(
    ticker_data: dict,
    capital: float,
    client: anthropic.Anthropic,
    sector_report: Optional[SectorReport] = None,
    discovery_reason: Optional[str] = None,
    category_context: Optional[str] = None,
) -> Optional[dict]:
    """
    Send enriched ticker data to Claude and return a parsed signal dict.
    """
    prompt = _build_prompt(ticker_data, capital, sector_report, discovery_reason, category_context)

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _extract_text(response)
        signal = _parse_json(raw)
        if signal:
            signal["_raw_data"] = ticker_data
        return signal

    except anthropic.APIError as e:
        return {"ticker": ticker_data.get("ticker", "?"), "error": f"API error: {e}"}
    except Exception as e:
        return {"ticker": ticker_data.get("ticker", "?"), "error": str(e)}


# ─── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(
    data: dict,
    capital: float,
    sector_report: Optional[SectorReport],
    discovery_reason: Optional[str],
    category_context: Optional[str] = None,
) -> str:
    today = date.today().isoformat()
    expiries = data.get("options_expiries", [])
    expiries_str = ", ".join(expiries) if expiries else "None available"
    sizing_note = _sizing_note(capital)

    # Sector context block
    sector_block = ""
    if sector_report:
        sector_name = data.get("sector", "")
        sc = _find_sector_score(sector_report, sector_name, data.get("ticker", ""))
        sector_block = f"\n--- MARKET & SECTOR CONTEXT ---\n{sector_report.summary_for_agent()}"
        if sc:
            sector_block += (
                f"\n\nThis ticker's sector: {sector_name}"
                f"\nSector rank:  #{sc.rank} of {len(sector_report.sectors)}"
                f"\nSector trend: {sc.trend}"
                f"\nSector RS 5d: {sc.fmt_change(sc.rs_5d)} vs SPY"
                f"\nSector bias:  {sc.signal_bias}"
            )

    discovery_block = f"\nWhy surfaced: {discovery_reason}\n" if discovery_reason else ""
    cat_block = f"\nCategory context: {category_context}\n" if category_context else ""

    return f"""Today: {today}
Capital: ${capital:,.2f}
{sizing_note}
{sector_block}
{discovery_block}{cat_block}
--- TICKER ---
Ticker:        {data.get('ticker')}
Name:          {data.get('name')}
Sector:        {data.get('sector')}
Industry:      {data.get('industry', 'N/A')}
Market Cap:    {data.get('market_cap', 'N/A')}
Beta:          {data.get('beta', 'N/A')}
Price:         ${data.get('current_price', 'N/A')}
1d Change:     {data.get('price_change_1d', 'N/A')}
5d Change:     {data.get('price_change_5d', 'N/A')}
Volume:        {data.get('volume_ratio', 'N/A')}
Next Earnings: {data.get('earnings_date') or 'Unknown'}

Available expiration dates: {expiries_str}

--- FUNDAMENTALS ---
{data.get('fundamentals_formatted', 'No fundamental data.')}

--- MACRO: TREASURY YIELD CURVE ---
{data.get('yields_formatted', 'No yield data.')}

--- RECENT NEWS (full article content) ---
{data.get('news_formatted', 'No news available.')}

--- LATEST EARNINGS CALL TRANSCRIPT ---
{data.get('transcript_formatted', 'No transcript available.')}

--- RECENT SEC FILINGS ---
{data.get('sec_filings_formatted', 'No recent filings.')}

Generate the options signal JSON now."""


# ─── Private helpers ──────────────────────────────────────────────────────────

def _find_sector_score(report: SectorReport, sector_name: str, ticker: str):
    if sector_name:
        sc = report.for_sector_name(sector_name)
        if sc:
            return sc
    return report.by_etf(ticker)


def _sizing_note(capital: float) -> str:
    lines = ["Position sizing (apply to suggested_capital_usd):"]
    for tier, rule in POSITION_SIZING.items():
        usd = capital * rule["capital_pct"]
        lines.append(
            f"  {tier.capitalize()} (confidence >= {rule['min_confidence']}%): "
            f"${usd:,.0f} ({rule['capital_pct']*100:.0f}% of capital)"
        )
    return "\n".join(lines)


def _extract_text(response) -> str:
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()


def _parse_json(raw: str) -> Optional[dict]:
    clean = raw.replace("```json", "").replace("```", "").strip()
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None
