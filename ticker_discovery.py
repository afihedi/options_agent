"""
ticker_discovery.py — Surfaces tickers from leading categories within leading sectors.

Discovery pipeline:
  1. Sector rank       (XLK > XLE > XLF...)          [sector_analyzer.py]
  2. Category rank     (Semiconductors > Software...)  [category_analyzer.py]
  3. Ticker rank       (NVDA > AMD > AVGO...)          [scored here by momentum]

For a call candidate:
  - Sector must be top-ranked AND bullish
  - Category within that sector must be top-ranked
  - Individual ticker must be the momentum leader within that category

For a put candidate:
  - Sector must be bottom-ranked AND bearish
  - Category within that sector must be the weakest
  - Individual ticker must be the weakest within that category
"""

import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from sector_analyzer import SectorReport, SectorScore
from category_analyzer import (
    SectorCategoryReport,
    CategoryScore,
    analyze_sector_categories,
)


# ─── Configuration ────────────────────────────────────────────────────────────

MIN_AVG_VOLUME       = 3_000_000
MAX_HOLDINGS_PER_CAT = 10
FETCH_THREADS        = 6


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DiscoveredTicker:
    ticker: str
    sector: str
    sector_rank: int
    sector_score: float
    category: str
    category_rank: int
    category_score: float
    sector_bias: str
    suggested_direction: str
    individual_momentum: float
    reason: str


@dataclass
class DiscoveryResult:
    call_candidates: list[DiscoveredTicker] = field(default_factory=list)
    put_candidates:  list[DiscoveredTicker] = field(default_factory=list)
    category_reports: dict[str, SectorCategoryReport] = field(default_factory=dict)

    @property
    def all_tickers(self) -> list[str]:
        seen = set()
        out = []
        for d in self.call_candidates + self.put_candidates:
            if d.ticker not in seen:
                out.append(d.ticker)
                seen.add(d.ticker)
        return out

    def summary(self) -> str:
        lines = []
        if self.call_candidates:
            by_cat: dict[str, list[str]] = {}
            for d in self.call_candidates:
                key = f"{d.sector} › {d.category}"
                by_cat.setdefault(key, []).append(d.ticker)
            lines.append("Call candidates:")
            for cat, tickers in by_cat.items():
                lines.append(f"  {cat}: {', '.join(tickers)}")
        if self.put_candidates:
            by_cat = {}
            for d in self.put_candidates:
                key = f"{d.sector} › {d.category}"
                by_cat.setdefault(key, []).append(d.ticker)
            lines.append("Put candidates:")
            for cat, tickers in by_cat.items():
                lines.append(f"  {cat}: {', '.join(tickers)}")
        return "\n".join(lines) if lines else "No high-conviction setups identified."

    def category_context_for_ticker(self, ticker: str) -> Optional[str]:
        for d in self.call_candidates + self.put_candidates:
            if d.ticker.upper() == ticker.upper():
                return (
                    f"{d.sector} sector (#{d.sector_rank}) › "
                    f"{d.category} category (#{d.category_rank} in sector) › "
                    f"{ticker} (individual momentum: {d.individual_momentum:+.2f})"
                )
        return None


# ─── Public function ──────────────────────────────────────────────────────────

def discover_tickers(
    report: SectorReport,
    top_n: int = 3,
    tickers_per_sector: int = 3,
    include_put_sectors: bool = True,
) -> DiscoveryResult:
    """
    Discover call and put candidates using a 3-tier ranking:
    sector → category → individual ticker.
    """
    result = DiscoveryResult()
    cat_reports: dict[str, SectorCategoryReport] = {}

    # Collect sectors to analyze
    sectors_to_analyze = {}
    for sector in report.top_sectors[:top_n]:
        if sector.momentum_score >= 1.0:
            sectors_to_analyze[sector.name] = sector.etf
    if include_put_sectors:
        for sector in report.bottom_sectors[:top_n]:
            if sector.momentum_score <= -1.0:
                sectors_to_analyze[sector.name] = sector.etf

    # Run category analysis in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analyze_sector_categories, name, etf): name
            for name, etf in sectors_to_analyze.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                cat_reports[name] = future.result()
            except Exception:
                pass

    result.category_reports = cat_reports

    # Call candidates
    for sector in report.top_sectors[:top_n]:
        if sector.momentum_score < 1.0:
            continue
        cat_report = cat_reports.get(sector.name)
        if not cat_report or not cat_report.categories:
            continue
        result.call_candidates.extend(
            _pick_from_categories(sector, cat_report, "call", tickers_per_sector)
        )

    # Put candidates
    if include_put_sectors:
        for sector in report.bottom_sectors[:top_n]:
            if sector.momentum_score > -1.0:
                continue
            cat_report = cat_reports.get(sector.name)
            if not cat_report or not cat_report.categories:
                continue
            result.put_candidates.extend(
                _pick_from_categories(sector, cat_report, "put", tickers_per_sector)
            )

    return result


def sector_for_ticker(ticker: str, report: SectorReport) -> Optional[str]:
    for sector in report.sectors:
        holdings = _fetch_holdings(sector.etf)
        if ticker.upper() in [h.upper() for h in holdings]:
            return sector.name
    return None


# ─── Core picking logic ───────────────────────────────────────────────────────

def _pick_from_categories(
    sector: SectorScore,
    cat_report: SectorCategoryReport,
    direction: str,
    total_tickers: int,
) -> list[DiscoveredTicker]:
    if direction == "call":
        target_cats = [c for c in cat_report.categories if c.signal_bias == "call"]
        if not target_cats:
            target_cats = cat_report.categories[:2]
    else:
        target_cats = [c for c in cat_report.categories if c.signal_bias == "put"]
        if not target_cats:
            target_cats = cat_report.categories[-2:]

    if not target_cats:
        return []

    discovered = []
    seen = set()

    for cat in target_cats:
        if len(discovered) >= total_tickers:
            break
        scored = _score_tickers_in_category(cat.tickers, direction)
        for ticker, momentum in scored:
            if ticker in seen or len(discovered) >= total_tickers:
                break
            discovered.append(DiscoveredTicker(
                ticker=ticker,
                sector=sector.name,
                sector_rank=sector.rank,
                sector_score=sector.momentum_score,
                category=cat.name,
                category_rank=cat.rank,
                category_score=cat.avg_momentum,
                sector_bias=sector.signal_bias,
                suggested_direction=direction,
                individual_momentum=round(momentum, 2),
                reason=_build_reason(ticker, sector, cat, direction),
            ))
            seen.add(ticker)

    return discovered


def _score_tickers_in_category(
    tickers: list[str],
    direction: str,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    with ThreadPoolExecutor(max_workers=FETCH_THREADS) as executor:
        futures = {
            executor.submit(_score_one, t): t
            for t in tickers[:MAX_HOLDINGS_PER_CAT]
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    scored.append(result)
            except Exception:
                pass
    scored.sort(key=lambda x: x[1], reverse=(direction == "call"))
    return scored


def _score_one(ticker: str) -> Optional[tuple[str, float]]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day") or 0
        if avg_vol < MIN_AVG_VOLUME:
            return None
        hist = t.history(period="10d")
        if hist.empty or len(hist) < 2:
            return ticker, 0.0
        close = hist["Close"]
        c1 = _pct(close, 1) or 0.0
        c5 = _pct(close, 5) or 0.0
        return ticker, (c5 * 0.6) + (c1 * 0.4)
    except Exception:
        return ticker, 0.0


def _build_reason(ticker, sector, cat, direction) -> str:
    qualifier = "leading" if direction == "call" else "weakest"
    c5 = f"{cat.avg_change_5d:+.2f}%" if cat.avg_change_5d is not None else "—"
    return (
        f"{sector.name} is #{sector.rank} ranked sector ({sector.trend}). "
        f"Within {sector.name}, {cat.name} is the #{cat.rank} ranked category "
        f"(avg 5d: {c5}, {cat.trend}). "
        f"{ticker} is among the {qualifier} stocks within {cat.name}."
    )


def _fetch_holdings(etf_ticker: str) -> list[str]:
    try:
        t = yf.Ticker(etf_ticker)
        try:
            df = t.funds_data.top_holdings
            if df is not None and not df.empty:
                return [_clean(s) for s in df.index[:30] if _clean(s)]
        except Exception:
            pass
        info = t.info or {}
        return [_clean(h.get("symbol", "")) for h in info.get("holdings", [])[:30]]
    except Exception:
        return []


def _pct(series, n: int) -> Optional[float]:
    if len(series) < n + 1:
        return None
    return ((series.iloc[-1] - series.iloc[-(n + 1)]) / series.iloc[-(n + 1)]) * 100


def _clean(raw: str) -> str:
    if not raw:
        return ""
    clean = raw.split(".")[0].strip().upper()
    return clean if clean and clean.replace("-", "").isalpha() else ""
