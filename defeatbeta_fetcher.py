"""
defeatbeta_fetcher.py — Fetches rich fundamental and news data from the
defeatbeta/yahoo-finance-data dataset hosted on Hugging Face.

Data is stored in Parquet format and queried via DuckDB's OLAP engine with
the cache_httpfs extension. Queries are fast (sub-second for filtered lookups)
and there are no rate limits.

IMPORTANT: This data is updated weekly, not in real-time. Use yfinance
(data_fetcher.py) for live price action, options chain, and today's volume.
Use this module for everything that benefits from depth over recency:
  - Full news article bodies (not just headlines)
  - Earnings call transcripts (management guidance, outlook language)
  - SEC 8-K filings (material events)
  - TTM EPS, P/E, and income statement trends
  - Earnings calendar (Nasdaq-sourced, more reliable than yfinance)

Install: pip install defeatbeta-api
"""

import duckdb
from datetime import datetime, timedelta
from typing import Optional

# ─── Hugging Face Parquet URLs ────────────────────────────────────────────────
# All tables live here. DuckDB fetches and caches them via cache_httpfs.
_BASE = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data"

URLS = {
    "news":          f"{_BASE}/stock_news.parquet",
    "transcripts":   f"{_BASE}/stock_earning_call_transcripts.parquet",
    "sec_filings":   f"{_BASE}/stock_sec_filing.parquet",
    "earnings_cal":  f"{_BASE}/stock_earning_calendar.parquet",
    "ttm_eps":       f"{_BASE}/stock_tailing_eps.parquet",
    "statement":     f"{_BASE}/stock_statement.parquet",
    "profile":       f"{_BASE}/stock_profile.parquet",
    "treasury":      f"{_BASE}/daily_treasury_yield.parquet",
}

# ─── DuckDB connection (singleton, shared across all queries) ─────────────────
_con: Optional[duckdb.DuckDBPyConnection] = None


def _get_con() -> duckdb.DuckDBPyConnection:
    """Return a cached DuckDB connection with httpfs enabled."""
    global _con
    if _con is None:
        _con = duckdb.connect()
        _con.execute("INSTALL httpfs; LOAD httpfs;")
        # cache_httpfs caches remote Parquet files locally for repeat queries
        try:
            _con.execute("INSTALL cache_httpfs FROM community; LOAD cache_httpfs;")
        except Exception:
            pass  # cache_httpfs may not be available in all environments; non-fatal
    return _con


def _query(sql: str) -> list[dict]:
    """Execute a SQL query and return results as a list of dicts."""
    try:
        con = _get_con()
        result = con.execute(sql).fetchdf()
        return result.to_dict(orient="records")
    except Exception as e:
        return []


# ─── News ─────────────────────────────────────────────────────────────────────

def fetch_news_full(ticker: str, max_articles: int = 4, days_back: int = 7) -> list[dict]:
    """
    Fetch full news articles for a ticker including body text.

    Returns a list of dicts with:
      - title, publisher, report_date, link
      - body: list of paragraph strings (up to 5 paragraphs per article)
      - highlight: the article's highlight/summary line
    """
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # stock_news uses related_symbols as an array; we use array_contains to filter
    sql = f"""
    SELECT
        title,
        publisher,
        report_date,
        link,
        unnest(news).highlight   AS highlight,
        unnest(news).paragraph   AS paragraph,
        unnest(news).paragraph_number AS para_num
    FROM '{URLS["news"]}'
    WHERE array_contains(related_symbols, '{ticker.upper()}')
      AND report_date >= '{cutoff}'
    ORDER BY report_date DESC
    LIMIT {max_articles * 8}
    """
    rows = _query(sql)
    if not rows:
        return []

    # Group paragraphs back into articles
    articles: dict[str, dict] = {}
    for row in rows:
        key = row.get("title", "") or row.get("link", "")
        if not key:
            continue
        if key not in articles:
            articles[key] = {
                "title":       row.get("title", ""),
                "publisher":   row.get("publisher", ""),
                "report_date": row.get("report_date", ""),
                "link":        row.get("link", ""),
                "highlight":   row.get("highlight", ""),
                "paragraphs":  [],
            }
        para = row.get("paragraph", "")
        if para and para not in articles[key]["paragraphs"]:
            articles[key]["paragraphs"].append(para)

    # Sort by date descending, cap to max_articles
    sorted_articles = sorted(
        articles.values(),
        key=lambda a: a["report_date"],
        reverse=True,
    )[:max_articles]

    # Format for prompt injection
    result = []
    for a in sorted_articles:
        result.append({
            "title":       a["title"],
            "publisher":   a["publisher"],
            "report_date": a["report_date"],
            "link":        a["link"],
            "highlight":   a["highlight"],
            "body":        a["paragraphs"][:5],  # first 5 paragraphs
        })

    return result


# ─── Earnings call transcripts ────────────────────────────────────────────────

def fetch_latest_transcript_excerpt(ticker: str, max_paragraphs: int = 12) -> Optional[dict]:
    """
    Fetch the most recent earnings call transcript for a ticker.

    Returns a dict with fiscal_year, fiscal_quarter, report_date,
    and a list of (speaker, content) tuples for management remarks
    and Q&A highlights (analyst questions + management responses).

    We skip boilerplate (operator instructions, safe harbor disclaimers)
    and focus on forward-looking language, guidance, and Q&A.
    """
    sql = f"""
    SELECT
        fiscal_year,
        fiscal_quarter,
        report_date,
        unnest(transcripts).speaker AS speaker,
        unnest(transcripts).content AS content,
        unnest(transcripts).paragraph_number AS para_num
    FROM '{URLS["transcripts"]}'
    WHERE symbol = '{ticker.upper()}'
    ORDER BY fiscal_year DESC, fiscal_quarter DESC
    LIMIT 200
    """
    rows = _query(sql)
    if not rows:
        return None

    # Take the most recent quarter's data
    latest_year    = rows[0]["fiscal_year"]
    latest_quarter = rows[0]["fiscal_quarter"]
    report_date    = rows[0]["report_date"]

    # Filter to latest quarter only
    latest_rows = [
        r for r in rows
        if r["fiscal_year"] == latest_year and r["fiscal_quarter"] == latest_quarter
    ]

    # Skip boilerplate paragraphs
    boilerplate_keywords = [
        "safe harbor", "forward-looking statements", "sec filings",
        "operator", "thank you for joining", "please go ahead",
        "your line is open", "we will now begin", "conference call",
    ]

    meaningful = []
    for row in latest_rows:
        content = (row.get("content") or "").strip()
        speaker = (row.get("speaker") or "").strip()
        if not content or len(content) < 60:
            continue
        if any(kw in content.lower() for kw in boilerplate_keywords):
            continue
        meaningful.append({"speaker": speaker, "content": content})
        if len(meaningful) >= max_paragraphs:
            break

    if not meaningful:
        return None

    return {
        "fiscal_year":    latest_year,
        "fiscal_quarter": latest_quarter,
        "report_date":    report_date,
        "excerpts":       meaningful,
    }


# ─── SEC filings ──────────────────────────────────────────────────────────────

def fetch_recent_sec_filings(ticker: str, days_back: int = 30) -> list[dict]:
    """
    Fetch recent SEC filings for a ticker.

    Focuses on material event forms:
      - 8-K / 8-K/A : material events (earnings, M&A, exec changes, etc.)
      - 4 / 4/A     : insider transactions (buying/selling by executives)
      - SC 13D/G    : large position changes by institutional investors

    Returns list of dicts with form_type, filing_date, description, url.
    """
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    relevant_forms = ("'8-K'", "'8-K/A'", "'4'", "'4/A'", "'SC 13D'", "'SC 13G'")

    sql = f"""
    SELECT
        form_type,
        form_type_description,
        filing_date,
        filing_url
    FROM '{URLS["sec_filings"]}'
    WHERE symbol = '{ticker.upper()}'
      AND filing_date >= '{cutoff}'
      AND form_type IN ({', '.join(relevant_forms)})
    ORDER BY filing_date DESC
    LIMIT 10
    """
    rows = _query(sql)
    return [
        {
            "form_type":   r.get("form_type", ""),
            "description": r.get("form_type_description", ""),
            "filing_date": r.get("filing_date", ""),
            "url":         r.get("filing_url", ""),
        }
        for r in rows
    ]


# ─── Earnings calendar ────────────────────────────────────────────────────────

def fetch_next_earnings(ticker: str) -> Optional[str]:
    """
    Fetch the next earnings report date from Nasdaq (via defeatbeta).
    More reliable than yfinance's calendar endpoint.
    Returns ISO date string or None.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    sql = f"""
    SELECT report_date
    FROM '{URLS["earnings_cal"]}'
    WHERE symbol = '{ticker.upper()}'
      AND report_date >= '{today}'
    ORDER BY report_date ASC
    LIMIT 1
    """
    rows = _query(sql)
    if rows:
        return str(rows[0].get("report_date", ""))
    return None


# ─── Fundamentals ─────────────────────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch key fundamental metrics for a ticker.

    Returns a dict with:
      - ttm_eps: trailing twelve month EPS
      - revenue_ttm: TTM total revenue
      - net_income_ttm: TTM net income
      - operating_cash_flow_ttm: TTM operating cash flow
      - revenue_trend: list of (date, revenue) for last 4 quarters
      - sector, industry, business_summary
    """
    result = {}

    # TTM EPS
    eps_sql = f"""
    SELECT tailing_eps, report_date
    FROM '{URLS["ttm_eps"]}'
    WHERE symbol = '{ticker.upper()}'
    ORDER BY report_date DESC
    LIMIT 1
    """
    eps_rows = _query(eps_sql)
    if eps_rows:
        result["ttm_eps"]          = eps_rows[0].get("tailing_eps")
        result["ttm_eps_date"]     = eps_rows[0].get("report_date")

    # Income statement — TTM revenue and net income
    income_sql = f"""
    SELECT item_name, item_value, report_date, period_type
    FROM '{URLS["statement"]}'
    WHERE symbol = '{ticker.upper()}'
      AND finance_type = 'income_statement'
      AND period_type = 'quarterly'
      AND item_name IN ('Total Revenue', 'Net Income', 'Operating Income')
    ORDER BY report_date DESC
    LIMIT 20
    """
    income_rows = _query(income_sql)
    if income_rows:
        result["recent_income"] = _summarize_income(income_rows)

    # Cash flow — operating cash flow trend
    cf_sql = f"""
    SELECT item_name, item_value, report_date
    FROM '{URLS["statement"]}'
    WHERE symbol = '{ticker.upper()}'
      AND finance_type = 'cash_flow'
      AND period_type = 'quarterly'
      AND item_name = 'Operating Cash Flow'
    ORDER BY report_date DESC
    LIMIT 4
    """
    cf_rows = _query(cf_sql)
    if cf_rows:
        result["operating_cash_flow_quarters"] = [
            {"date": r["report_date"], "value": r["item_value"]}
            for r in cf_rows
        ]

    # Company profile
    profile_sql = f"""
    SELECT sector, industry, long_business_summary
    FROM '{URLS["profile"]}'
    WHERE symbol = '{ticker.upper()}'
    LIMIT 1
    """
    profile_rows = _query(profile_sql)
    if profile_rows:
        p = profile_rows[0]
        result["sector"]           = p.get("sector", "")
        result["industry"]         = p.get("industry", "")
        result["business_summary"] = (p.get("long_business_summary") or "")[:300]

    return result


# ─── Treasury yields (macro context) ─────────────────────────────────────────

def fetch_treasury_yields() -> Optional[dict]:
    """
    Fetch the most recent daily treasury yield curve.
    Useful macro context — inverted curve signals risk-off; steep curve = risk-on.
    Returns dict with yield values for key tenors.
    """
    sql = f"""
    SELECT report_date, bc3_month, bc2_year, bc5_year, bc10_year, bc30_year
    FROM '{URLS["treasury"]}'
    ORDER BY report_date DESC
    LIMIT 1
    """
    rows = _query(sql)
    if not rows:
        return None
    r = rows[0]
    return {
        "date":       r.get("report_date"),
        "3m":         r.get("bc3_month"),
        "2y":         r.get("bc2_year"),
        "5y":         r.get("bc5_year"),
        "10y":        r.get("bc10_year"),
        "30y":        r.get("bc30_year"),
        "curve_note": _curve_note(r),
    }


# ─── Formatting helpers (for agent prompt injection) ─────────────────────────

def format_news_for_prompt(articles: list[dict]) -> str:
    """Format full news articles as a readable block for Claude."""
    if not articles:
        return "  No recent news found."

    lines = []
    for a in articles:
        lines.append(f"  [{a['report_date']}] {a['publisher']}: {a['title']}")
        if a.get("highlight"):
            lines.append(f"    Summary: {a['highlight']}")
        for i, para in enumerate(a.get("body", [])[:3]):
            if para:
                lines.append(f"    {para[:300]}")
        lines.append("")

    return "\n".join(lines).strip()


def format_transcript_for_prompt(transcript: Optional[dict]) -> str:
    """Format earnings call excerpts for Claude."""
    if not transcript:
        return "  No recent earnings call transcript available."

    lines = [
        f"  Q{transcript['fiscal_quarter']} {transcript['fiscal_year']} "
        f"earnings call ({transcript['report_date']}):"
    ]
    for ex in transcript.get("excerpts", []):
        speaker = ex.get("speaker", "Unknown")
        content = ex.get("content", "")[:400]
        lines.append(f"  [{speaker}]: {content}")

    return "\n".join(lines)


def format_sec_filings_for_prompt(filings: list[dict]) -> str:
    """Format recent SEC filings as a readable block."""
    if not filings:
        return "  No recent material filings."
    lines = []
    for f in filings:
        lines.append(
            f"  [{f['filing_date']}] {f['form_type']} — {f['description']}"
        )
    return "\n".join(lines)


def format_fundamentals_for_prompt(fundamentals: dict) -> str:
    """Format fundamental metrics as a compact block for Claude."""
    if not fundamentals:
        return "  No fundamental data available."

    lines = []
    if fundamentals.get("ttm_eps") is not None:
        lines.append(f"  TTM EPS: ${fundamentals['ttm_eps']:.2f}")
    if fundamentals.get("industry"):
        lines.append(f"  Industry: {fundamentals['industry']}")
    if fundamentals.get("business_summary"):
        lines.append(f"  Business: {fundamentals['business_summary'][:200]}")

    income = fundamentals.get("recent_income", {})
    if income.get("revenue"):
        lines.append(
            "  Revenue (last 4Q): " +
            ", ".join(f"{r['date']}: ${r['value']/1e6:.0f}M" for r in income["revenue"])
        )
    if income.get("net_income"):
        latest_ni = income["net_income"][0]["value"] if income["net_income"] else None
        if latest_ni is not None:
            ni_label = f"${latest_ni/1e6:.0f}M" if latest_ni >= 0 else f"-${abs(latest_ni)/1e6:.0f}M"
            lines.append(f"  Net Income (latest Q): {ni_label}")

    return "\n".join(lines) if lines else "  No fundamental data available."


def format_yields_for_prompt(yields: Optional[dict]) -> str:
    """Format treasury yield data as a compact macro context line."""
    if not yields:
        return ""
    return (
        f"  Treasury yields ({yields['date']}): "
        f"3m={yields['3m']}%  2y={yields['2y']}%  "
        f"10y={yields['10y']}%  30y={yields['30y']}%  "
        f"— {yields['curve_note']}"
    )


# ─── Private helpers ──────────────────────────────────────────────────────────

def _summarize_income(rows: list[dict]) -> dict:
    """Group income statement rows by item_name and return last 4 quarters each."""
    grouped: dict[str, list] = {}
    for r in rows:
        name = r.get("item_name", "")
        if name not in grouped:
            grouped[name] = []
        if len(grouped[name]) < 4:
            grouped[name].append({
                "date":  r.get("report_date", ""),
                "value": float(r.get("item_value") or 0),
            })

    return {
        "revenue":     grouped.get("Total Revenue", []),
        "net_income":  grouped.get("Net Income", []),
        "op_income":   grouped.get("Operating Income", []),
    }


def _curve_note(r: dict) -> str:
    """Derive a simple yield curve shape note."""
    try:
        spread = float(r.get("bc10_year") or 0) - float(r.get("bc3_month") or 0)
        if spread < -0.5:
            return "inverted curve (risk-off signal)"
        if spread < 0.2:
            return "flat curve (mixed outlook)"
        return "normal curve (risk-on)"
    except Exception:
        return ""
