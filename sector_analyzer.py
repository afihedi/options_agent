"""
sector_analyzer.py — Scores and ranks all 11 GICS sector ETFs relative to SPY.

For each sector ETF it computes:
  - 1d, 5d, and 20d price % change
  - Relative strength vs SPY over each window (sector return minus SPY return)
  - Volume ratio (today vs 30d average)
  - A composite momentum score used for ranking

The output is a SectorReport used by ticker_discovery.py to identify which
sectors to focus on, and by agent.py to give Claude directional market context.
"""

import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from config import SECTOR_ETFS


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SectorScore:
    name: str
    etf: str
    change_1d: Optional[float]    # % change today
    change_5d: Optional[float]    # % change past week
    change_20d: Optional[float]   # % change past month
    rs_1d: Optional[float]        # relative strength vs SPY, 1d
    rs_5d: Optional[float]        # relative strength vs SPY, 5d
    rs_20d: Optional[float]       # relative strength vs SPY, 20d
    volume_ratio: Optional[float] # today vol / 30d avg vol
    momentum_score: float = 0.0   # composite score, higher = stronger
    rank: int = 0                 # 1 = strongest sector

    @property
    def trend(self) -> str:
        """Human-readable trend label based on momentum score."""
        if self.momentum_score >= 6:
            return "strong bullish"
        if self.momentum_score >= 3:
            return "bullish"
        if self.momentum_score >= -2:
            return "neutral"
        if self.momentum_score >= -5:
            return "bearish"
        return "strong bearish"

    @property
    def signal_bias(self) -> str:
        """call / put / neutral — directional hint for the agent."""
        if self.momentum_score >= 3:
            return "call"
        if self.momentum_score <= -3:
            return "put"
        return "neutral"

    def fmt_change(self, val: Optional[float]) -> str:
        if val is None:
            return "—"
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.2f}%"


@dataclass
class SectorReport:
    scored_at: str
    spy_change_1d: Optional[float]
    spy_change_5d: Optional[float]
    market_bias: str               # "risk-on" / "risk-off" / "mixed"
    sectors: list[SectorScore] = field(default_factory=list)

    @property
    def top_sectors(self) -> list[SectorScore]:
        """Top 3 strongest sectors by rank."""
        return [s for s in self.sectors if s.rank <= 3]

    @property
    def bottom_sectors(self) -> list[SectorScore]:
        """Bottom 3 weakest sectors by rank."""
        sorted_desc = sorted(self.sectors, key=lambda s: s.rank, reverse=True)
        return sorted_desc[:3]

    def by_etf(self, etf: str) -> Optional[SectorScore]:
        """Look up a sector score by its ETF ticker."""
        etf = etf.upper()
        return next((s for s in self.sectors if s.etf == etf), None)

    def for_sector_name(self, name: str) -> Optional[SectorScore]:
        """Look up a sector score by its name (e.g. 'Technology')."""
        return next((s for s in self.sectors if s.name == name), None)

    def summary_for_agent(self) -> str:
        """
        Compact text block injected into Claude's prompt.
        Gives sector context without blowing out the token budget.
        """
        lines = [
            f"Market bias: {self.market_bias}",
            f"SPY: 1d {_fmt(self.spy_change_1d)}  5d {_fmt(self.spy_change_5d)}",
            "",
            "Sector rankings (1 = strongest):",
        ]
        for s in self.sectors:
            lines.append(
                f"  #{s.rank:>2}  {s.etf:<5}  {s.name:<28}"
                f"  1d {s.fmt_change(s.change_1d):>8}"
                f"  5d {s.fmt_change(s.change_5d):>8}"
                f"  vs SPY 5d {s.fmt_change(s.rs_5d):>8}"
                f"  [{s.trend}]"
            )
        return "\n".join(lines)


# ─── Public function ──────────────────────────────────────────────────────────

def analyze_sectors() -> SectorReport:
    """
    Fetch data for SPY and all sector ETFs, score and rank them.
    Returns a SectorReport.
    """
    # Fetch SPY as the benchmark
    spy_1d, spy_5d, spy_20d = _fetch_returns("SPY")

    # Fetch and score each sector
    scores: list[SectorScore] = []
    for name, etf in SECTOR_ETFS.items():
        score = _score_sector(name, etf, spy_1d, spy_5d, spy_20d)
        scores.append(score)

    # Rank by composite momentum score (highest = rank 1)
    scores.sort(key=lambda s: s.momentum_score, reverse=True)
    for i, s in enumerate(scores):
        s.rank = i + 1

    # Derive market bias from SPY direction + breadth
    bullish_count = sum(1 for s in scores if s.momentum_score > 0)
    market_bias = _market_bias(spy_1d, spy_5d, bullish_count, len(scores))

    return SectorReport(
        scored_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        spy_change_1d=spy_1d,
        spy_change_5d=spy_5d,
        market_bias=market_bias,
        sectors=scores,
    )


# ─── Private helpers ──────────────────────────────────────────────────────────

def _fetch_returns(ticker: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (1d, 5d, 20d) % price changes for a ticker."""
    try:
        hist = yf.Ticker(ticker).history(period="30d")
        if hist.empty or len(hist) < 2:
            return None, None, None
        close = hist["Close"]
        r1d  = _pct(close, 1)
        r5d  = _pct(close, 5)
        r20d = _pct(close, 20)
        return r1d, r5d, r20d
    except Exception:
        return None, None, None


def _pct(series, n: int) -> Optional[float]:
    """% change between the last close and n periods ago."""
    if len(series) < n + 1:
        return None
    return ((series.iloc[-1] - series.iloc[-(n + 1)]) / series.iloc[-(n + 1)]) * 100


def _volume_ratio(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(ticker).info
        vol     = info.get("volume") or info.get("regularMarketVolume")
        avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        if vol and avg_vol and avg_vol > 0:
            return round(vol / avg_vol, 2)
        return None
    except Exception:
        return None


def _score_sector(
    name: str,
    etf: str,
    spy_1d: Optional[float],
    spy_5d: Optional[float],
    spy_20d: Optional[float],
) -> SectorScore:
    """
    Compute a composite momentum score for one sector.

    Scoring breakdown (max ~12 points, min ~-12):
      Relative strength 5d  → ±4 pts (heaviest weight — medium-term trend)
      Relative strength 20d → ±3 pts (longer-term context)
      Relative strength 1d  → ±2 pts (today's price action)
      Volume ratio          → ±1 pt  (conviction signal)
      Absolute 5d return    → ±2 pts (confirms or contradicts RS)
    """
    r1d, r5d, r20d = _fetch_returns(etf)
    vol_ratio = _volume_ratio(etf)

    rs_1d  = _diff(r1d,  spy_1d)
    rs_5d  = _diff(r5d,  spy_5d)
    rs_20d = _diff(r20d, spy_20d)

    score = 0.0

    # Relative strength (sector vs SPY)
    if rs_5d  is not None: score += _clamp(rs_5d  * 0.8, 4)
    if rs_20d is not None: score += _clamp(rs_20d * 0.6, 3)
    if rs_1d  is not None: score += _clamp(rs_1d  * 0.5, 2)

    # Absolute 5d return
    if r5d is not None: score += _clamp(r5d * 0.4, 2)

    # Volume confirmation
    if vol_ratio is not None:
        if vol_ratio >= 1.5:
            score += 1.0 if (r1d or 0) >= 0 else -1.0
        elif vol_ratio <= 0.7:
            score -= 0.5  # low conviction

    return SectorScore(
        name=name,
        etf=etf,
        change_1d=r1d,
        change_5d=r5d,
        change_20d=r20d,
        rs_1d=rs_1d,
        rs_5d=rs_5d,
        rs_20d=rs_20d,
        volume_ratio=vol_ratio,
        momentum_score=round(score, 2),
    )


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 3)


def _clamp(val: float, limit: float) -> float:
    return max(-limit, min(limit, val))


def _market_bias(
    spy_1d: Optional[float],
    spy_5d: Optional[float],
    bullish_count: int,
    total: int,
) -> str:
    bullish_pct = bullish_count / total if total else 0
    spy_ok = (spy_5d or 0) >= 0

    if bullish_pct >= 0.7 and spy_ok:
        return "risk-on"
    if bullish_pct <= 0.3 or (spy_5d or 0) <= -2:
        return "risk-off"
    return "mixed"


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"
