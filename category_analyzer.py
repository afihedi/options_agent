"""
category_analyzer.py — Groups sector ETF holdings by industry category
and scores each category by aggregate momentum.

This sits between sector_analyzer (macro sector rankings) and
ticker_discovery (individual stock selection). The pipeline becomes:

  Sector rank (XLK > XLE > XLF...)
      ↓
  Category rank within sector (Semiconductors > Software > Hardware...)
      ↓
  Individual ticker rank within category (NVDA > AMD > AVGO...)

For each holding in a sector ETF, we fetch its industry classification
from yfinance and group holdings accordingly. Each category is then
scored by the average momentum of its members, weighted by ETF weight
where available.

This means the agent can say:
  "Technology is the #1 sector. Within Technology, Semiconductors are
   leading Software by 3.2% this week. NVDA is the top performer
   within Semiconductors."
"""

import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─── How many holdings to pull per ETF for category analysis ─────────────────
MAX_HOLDINGS_TO_CLASSIFY = 30

# Minimum number of stocks in a category for it to be considered valid.
# Avoids "categories" that are just one oddball stock.
MIN_CATEGORY_SIZE = 2

# Max parallel threads for fetching industry info per holding.
# Keep this conservative to avoid yfinance rate limiting.
FETCH_THREADS = 6


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CategoryScore:
    name: str                      # e.g. "Semiconductors"
    sector: str                    # parent sector, e.g. "Technology"
    tickers: list[str]             # holdings in this category
    avg_momentum: float            # average of individual momentum scores
    avg_change_5d: Optional[float] # average 5d % change across members
    avg_change_1d: Optional[float] # average 1d % change across members
    member_count: int
    rank: int = 0                  # rank within sector (1 = strongest)

    @property
    def trend(self) -> str:
        if self.avg_momentum >= 4:   return "strong bullish"
        if self.avg_momentum >= 1.5: return "bullish"
        if self.avg_momentum >= -1:  return "neutral"
        if self.avg_momentum >= -3:  return "bearish"
        return "strong bearish"

    @property
    def signal_bias(self) -> str:
        if self.avg_momentum >= 1.5: return "call"
        if self.avg_momentum <= -1.5: return "put"
        return "neutral"

    def fmt(self) -> str:
        c5 = f"{self.avg_change_5d:+.2f}%" if self.avg_change_5d is not None else "—"
        c1 = f"{self.avg_change_1d:+.2f}%" if self.avg_change_1d is not None else "—"
        return (
            f"#{self.rank} {self.name} ({self.member_count} stocks) "
            f"1d:{c1} 5d:{c5} [{self.trend}]"
        )


@dataclass
class SectorCategoryReport:
    sector_name: str
    sector_etf: str
    categories: list[CategoryScore] = field(default_factory=list)

    @property
    def top_category(self) -> Optional[CategoryScore]:
        return self.categories[0] if self.categories else None

    @property
    def bottom_category(self) -> Optional[CategoryScore]:
        return self.categories[-1] if self.categories else None

    def for_ticker(self, ticker: str) -> Optional[CategoryScore]:
        """Find which category a specific ticker belongs to."""
        for cat in self.categories:
            if ticker.upper() in [t.upper() for t in cat.tickers]:
                return cat
        return None

    def summary_for_agent(self) -> str:
        """Compact text block for Claude prompt injection."""
        if not self.categories:
            return f"  No category breakdown available for {self.sector_name}."
        lines = [f"  Categories within {self.sector_name} (ranked by momentum):"]
        for cat in self.categories:
            lines.append(f"    {cat.fmt()}")
            lines.append(f"      Members: {', '.join(cat.tickers)}")
        return "\n".join(lines)


# ─── Public function ──────────────────────────────────────────────────────────

def analyze_sector_categories(
    sector_name: str,
    sector_etf: str,
) -> SectorCategoryReport:
    """
    Fetch the top holdings of a sector ETF, classify each by industry,
    score each industry category by aggregate momentum, and rank them.

    Args:
        sector_name: Human-readable sector name (e.g. "Technology")
        sector_etf:  Sector ETF ticker (e.g. "XLK")

    Returns:
        SectorCategoryReport with ranked CategoryScore list
    """
    # Step 1: Get ETF holdings
    holdings = _fetch_holdings(sector_etf)
    if not holdings:
        return SectorCategoryReport(sector_name=sector_name, sector_etf=sector_etf)

    # Step 2: Classify each holding by industry + fetch momentum in parallel
    classified = _classify_holdings_parallel(holdings)

    # Step 3: Group by category
    groups: dict[str, list[dict]] = {}
    for item in classified:
        cat = item["category"]
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(item)

    # Step 4: Score each category
    scores: list[CategoryScore] = []
    for cat_name, members in groups.items():
        if len(members) < MIN_CATEGORY_SIZE:
            continue  # skip singleton categories

        momentums  = [m["momentum"] for m in members if m["momentum"] is not None]
        changes_5d = [m["change_5d"] for m in members if m["change_5d"] is not None]
        changes_1d = [m["change_1d"] for m in members if m["change_1d"] is not None]

        avg_momentum  = sum(momentums)  / len(momentums)  if momentums  else 0.0
        avg_change_5d = sum(changes_5d) / len(changes_5d) if changes_5d else None
        avg_change_1d = sum(changes_1d) / len(changes_1d) if changes_1d else None

        scores.append(CategoryScore(
            name=cat_name,
            sector=sector_name,
            tickers=[m["ticker"] for m in members],
            avg_momentum=round(avg_momentum, 2),
            avg_change_5d=round(avg_change_5d, 2) if avg_change_5d is not None else None,
            avg_change_1d=round(avg_change_1d, 2) if avg_change_1d is not None else None,
            member_count=len(members),
        ))

    # Step 5: Rank by avg_momentum descending
    scores.sort(key=lambda s: s.avg_momentum, reverse=True)
    for i, s in enumerate(scores):
        s.rank = i + 1

    return SectorCategoryReport(
        sector_name=sector_name,
        sector_etf=sector_etf,
        categories=scores,
    )


def analyze_multiple_sectors(
    sectors: dict[str, str],
) -> dict[str, SectorCategoryReport]:
    """
    Run category analysis for multiple sectors in parallel.

    Args:
        sectors: dict of {sector_name: etf_ticker}, e.g. {"Technology": "XLK"}

    Returns:
        dict of {sector_name: SectorCategoryReport}
    """
    results: dict[str, SectorCategoryReport] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analyze_sector_categories, name, etf): name
            for name, etf in sectors.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = SectorCategoryReport(
                    sector_name=name,
                    sector_etf=sectors[name],
                )

    return results


# ─── Private helpers ──────────────────────────────────────────────────────────

def _fetch_holdings(etf_ticker: str) -> list[str]:
    """Fetch top holdings of a sector ETF."""
    try:
        t = yf.Ticker(etf_ticker)
        try:
            df = t.funds_data.top_holdings
            if df is not None and not df.empty:
                tickers = [_clean(sym) for sym in df.index[:MAX_HOLDINGS_TO_CLASSIFY]]
                return [s for s in tickers if s]
        except Exception:
            pass

        info = t.info or {}
        holdings = info.get("holdings", [])
        if holdings:
            return [
                _clean(h.get("symbol", ""))
                for h in holdings[:MAX_HOLDINGS_TO_CLASSIFY]
            ]
        return []
    except Exception:
        return []


def _classify_holdings_parallel(holdings: list[str]) -> list[dict]:
    """
    Fetch industry classification and momentum for each holding in parallel.
    Returns list of dicts: {ticker, category, momentum, change_5d, change_1d}
    """
    results = []
    with ThreadPoolExecutor(max_workers=FETCH_THREADS) as executor:
        futures = {
            executor.submit(_classify_one, ticker): ticker
            for ticker in holdings
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                pass
    return results


def _classify_one(ticker: str) -> Optional[dict]:
    """
    Fetch industry + momentum for a single ticker.
    Returns None if the ticker can't be resolved.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Industry classification
        industry = info.get("industry") or info.get("sector") or "Other"
        industry = _normalize_industry(industry)

        # Momentum from price history
        hist = t.history(period="10d")
        change_1d = change_5d = momentum = None

        if not hist.empty and len(hist) >= 2:
            close = hist["Close"]
            change_1d = _pct(close, 1)
            change_5d = _pct(close, 5)
            if change_1d is not None and change_5d is not None:
                momentum = (change_5d * 0.6) + (change_1d * 0.4)

        return {
            "ticker":    ticker,
            "category":  industry,
            "momentum":  round(momentum, 3) if momentum is not None else 0.0,
            "change_5d": round(change_5d, 3) if change_5d is not None else None,
            "change_1d": round(change_1d, 3) if change_1d is not None else None,
        }
    except Exception:
        return None


def _normalize_industry(raw: str) -> str:
    """
    Normalize yfinance industry strings into cleaner category names.
    yfinance returns strings like "Semiconductors" or "Software—Application"
    which we clean into readable labels.
    """
    # Common normalizations
    mappings = {
        "semiconductors":                   "Semiconductors",
        "semiconductor equipment":          "Semiconductor Equipment",
        "software—application":             "Software",
        "software-application":             "Software",
        "software application":             "Software",
        "software—infrastructure":          "Software Infrastructure",
        "software-infrastructure":          "Software Infrastructure",
        "software infrastructure":          "Software Infrastructure",
        "information technology services":  "IT Services",
        "it services":                      "IT Services",
        "computer hardware":                "Hardware",
        "electronic components":            "Electronic Components",
        "communication equipment":          "Communication Equipment",
        "consumer electronics":             "Consumer Electronics",
        "internet content & information":   "Internet",
        "internet retail":                  "E-Commerce",
        "cloud computing":                  "Cloud",
        "data storage":                     "Data Storage",
        "electronic gaming & multimedia":   "Gaming",
        "oil & gas e&p":                    "E&P",
        "oil & gas integrated":             "Integrated Oil",
        "oil & gas midstream":              "Midstream",
        "oil & gas equipment & services":   "Oilfield Services",
        "oil & gas refining & marketing":   "Refining",
        "banks—diversified":                "Banks",
        "banks—regional":                   "Regional Banks",
        "insurance—diversified":            "Insurance",
        "capital markets":                  "Capital Markets",
        "asset management":                 "Asset Management",
        "credit services":                  "Credit",
        "biotechnology":                    "Biotech",
        "drug manufacturers—general":       "Large Pharma",
        "drug manufacturers—specialty":     "Specialty Pharma",
        "medical devices":                  "Medical Devices",
        "health information services":      "Health IT",
        "managed health care":              "Managed Care",
        "specialty retail":                 "Retail",
        "auto manufacturers":               "Auto",
        "auto parts":                       "Auto Parts",
        "restaurants":                      "Restaurants",
        "leisure":                          "Leisure",
        "aerospace & defense":              "Aerospace & Defense",
        "railroads":                        "Railroads",
        "trucking":                         "Trucking",
        "industrials":                      "Industrials",
        "utilities—regulated electric":     "Electric Utilities",
        "utilities—renewable":              "Renewable Utilities",
        "reit—industrial":                  "Industrial REIT",
        "reit—retail":                      "Retail REIT",
        "reit—residential":                 "Residential REIT",
        "reit—office":                      "Office REIT",
        "reit—diversified":                 "Diversified REIT",
    }
    key = raw.lower().strip()
    return mappings.get(key, raw.title())


def _pct(series, n: int) -> Optional[float]:
    if len(series) < n + 1:
        return None
    return ((series.iloc[-1] - series.iloc[-(n + 1)]) / series.iloc[-(n + 1)]) * 100


def _clean(raw: str) -> str:
    if not raw:
        return ""
    clean = raw.split(".")[0].strip().upper()
    if not clean or not clean.replace("-", "").isalpha():
        return ""
    return clean
