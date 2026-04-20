"""
main.py — Entry point for the Options Signal Agent (v2 with sector strength).

New in v2:
  - Runs a sector scan first, ranking all 11 GICS sectors vs SPY
  - Prints a sector heat map before individual signals
  - Auto-discovers tickers from strongest/weakest sectors when no --tickers are passed
  - Passes sector context into every Claude prompt
  - Sector headwinds reduce confidence; tailwinds amplify it

Run:
    python main.py
    python main.py --tickers NVDA TSLA AMD --capital 5000
    python main.py --include-sectors --min-confidence 65 --output signals.json
    python main.py --capital 10000 --top-sectors 3 --tickers-per-sector 4
"""

import os
import sys
import json
import argparse
import anthropic
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from config import (
    DEFAULT_WATCHLIST,
    SECTOR_ETFS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MAX_SIGNALS,
)
from data_fetcher import fetch_ticker_data
from agent import generate_signal
from sector_analyzer import analyze_sectors, SectorReport
from ticker_discovery import discover_tickers, sector_for_ticker, DiscoveryResult

load_dotenv()
console = Console()


# ─── CLI args ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI options signal agent with sector strength analysis."
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Specific tickers to analyze. If omitted, tickers are auto-discovered from sector rankings.")
    parser.add_argument("--capital", type=float,
                        help="Available trading capital in USD")
    parser.add_argument("--max-signals", type=int, default=DEFAULT_MAX_SIGNALS,
                        help=f"Max signals to display (default: {DEFAULT_MAX_SIGNALS})")
    parser.add_argument("--min-confidence", type=int, default=DEFAULT_MIN_CONFIDENCE,
                        help=f"Min confidence to show a signal (default: {DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--include-sectors", action="store_true",
                        help="Also analyze sector ETFs directly (XLK, XLE, etc.)")
    parser.add_argument("--top-sectors", type=int, default=3,
                        help="How many top/bottom sectors to draw tickers from (default: 3)")
    parser.add_argument("--tickers-per-sector", type=int, default=3,
                        help="Tickers to surface per sector (default: 3)")
    parser.add_argument("--no-puts", action="store_true",
                        help="Skip put candidates from weak sectors")
    parser.add_argument("--output", type=str, metavar="FILE",
                        help="Save results to a JSON file")
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] ANTHROPIC_API_KEY not found. "
                      "Add it to your .env file or set it as an environment variable.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    capital = args.capital
    if not capital:
        try:
            capital = float(console.input(
                "[bold]Enter your available trading capital[/] (USD, e.g. 5000): $"
            ).replace(",", "").strip())
        except ValueError:
            console.print("[red]Invalid capital amount.[/]")
            sys.exit(1)

    _print_header(capital)

    # ── Phase 1: Sector scan ──────────────────────────────────────────────────
    console.print("[bold]Phase 1 — Sector analysis[/]")

    sector_report: SectorReport = None
    discovery: DiscoveryResult = None

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as p:
        p.add_task("Scanning all 11 GICS sectors vs SPY...")
        sector_report = analyze_sectors()

    _print_sector_heatmap(sector_report)

    # ── Phase 2: Ticker discovery (if no --tickers passed) ────────────────────
    if not args.tickers:
        console.print("[bold]Phase 2 — Ticker discovery[/]")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      console=console, transient=True) as p:
            p.add_task("Classifying sector holdings by category and scoring momentum...")
            discovery = discover_tickers(
                report=sector_report,
                top_n=args.top_sectors,
                tickers_per_sector=args.tickers_per_sector,
                include_put_sectors=not args.no_puts,
            )

        _print_discovery_summary(discovery)
        tickers = discovery.all_tickers

        if not tickers:
            console.print("[yellow]No high-conviction sectors found. "
                          "Falling back to default watchlist.[/]")
            tickers = list(DEFAULT_WATCHLIST)
    else:
        tickers = [t.upper() for t in args.tickers]

    if args.include_sectors:
        tickers += list(SECTOR_ETFS.values())

    tickers = list(dict.fromkeys(tickers))  # deduplicate

    # ── Phase 3: Fetch + analyze each ticker ──────────────────────────────────
    console.print(f"\n[bold]Phase 3 — Analyzing {len(tickers)} ticker(s)[/]")

    signals = []
    errors = []

    # Build a lookup: ticker → discovery reason + category context
    discovery_reasons: dict[str, str] = {}
    category_contexts: dict[str, str] = {}
    if discovery:
        for d in (discovery.call_candidates + discovery.put_candidates):
            discovery_reasons[d.ticker] = d.reason
            category_contexts[d.ticker] = (
                f"{d.sector} sector (#{d.sector_rank}) › "
                f"{d.category} category (#{d.category_rank} within sector, "
                f"avg 5d momentum: {d.category_score:+.2f}) › "
                f"{d.ticker} individual momentum: {d.individual_momentum:+.2f}"
            )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as p:
        task = p.add_task("Starting...", total=len(tickers))

        for ticker in tickers:
            p.update(task, description=f"Fetching {ticker}...")
            data = fetch_ticker_data(ticker)

            if not data:
                errors.append(f"{ticker}: could not resolve")
                p.advance(task)
                continue

            if "error" in data and not data.get("news"):
                errors.append(f"{ticker}: {data['error']}")
                p.advance(task)
                continue

            p.update(task, description=f"Analyzing {ticker} with AI...")
            signal = generate_signal(
                ticker_data=data,
                capital=capital,
                client=client,
                sector_report=sector_report,
                discovery_reason=discovery_reasons.get(ticker),
                category_context=category_contexts.get(ticker),
            )

            if signal and "error" not in signal:
                signals.append(signal)
            elif signal:
                errors.append(f"{ticker}: {signal.get('error', 'unknown')}")

            p.advance(task)

    # ── Phase 4: Filter, sort, display ───────────────────────────────────────
    filtered = [
        s for s in signals
        if s.get("signal") != "neutral"
        and s.get("confidence", 0) >= args.min_confidence
    ]
    filtered.sort(key=lambda s: (
        {"high": 0, "medium": 1, "low": 2}.get(s.get("alert_priority", "low"), 3),
        -s.get("confidence", 0)
    ))
    filtered = filtered[:args.max_signals]

    console.print()
    if not filtered:
        console.print("[yellow]No signals met the confidence threshold. "
                      "Try --min-confidence 45 or expand your sector coverage.[/]")
    else:
        _print_summary_table(filtered, sector_report)
        for sig in filtered:
            _print_signal_detail(sig, sector_report)

    neutral_count = sum(1 for s in signals if s.get("signal") == "neutral")
    if neutral_count:
        console.print(f"[dim]{neutral_count} ticker(s) returned neutral.[/]")
    if errors:
        console.print(f"[dim]Skipped: {', '.join(e.split(':')[0] for e in errors)}[/]")

    if args.output:
        _save_json(filtered, signals, sector_report, args.output, capital)

    console.print(
        "\n[dim italic]For informational purposes only. Not financial advice. "
        "Options trading involves substantial risk of loss.[/]"
    )


# ─── Sector heat map output ───────────────────────────────────────────────────

def _print_sector_heatmap(report: SectorReport):
    bias_color = {"risk-on": "green", "risk-off": "red", "mixed": "yellow"}.get(
        report.market_bias, "white"
    )
    console.print(
        f"  Market bias: [{bias_color}]{report.market_bias.upper()}[/]   "
        f"SPY 1d: {_fmt(report.spy_change_1d)}   "
        f"SPY 5d: {_fmt(report.spy_change_5d)}\n"
    )

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold",
                  padding=(0, 1))
    table.add_column("Rank", width=5)
    table.add_column("Sector", width=26)
    table.add_column("ETF",  width=6)
    table.add_column("1d",   width=8)
    table.add_column("5d",   width=8)
    table.add_column("vs SPY 5d", width=10)
    table.add_column("Trend", width=16)
    table.add_column("Bias", width=8)

    for s in report.sectors:
        trend_color = {
            "strong bullish": "green",
            "bullish": "green",
            "neutral": "dim",
            "bearish": "red",
            "strong bearish": "red",
        }.get(s.trend, "white")

        bias_style = "green" if s.signal_bias == "call" else "red" if s.signal_bias == "put" else "dim"

        table.add_row(
            f"#{s.rank}",
            s.name,
            s.etf,
            _colored_pct(s.change_1d),
            _colored_pct(s.change_5d),
            _colored_pct(s.rs_5d),
            f"[{trend_color}]{s.trend}[/]",
            f"[{bias_style}]{s.signal_bias.upper()}[/]",
        )

    console.print(table)


def _print_discovery_summary(discovery: DiscoveryResult):
    if not discovery.all_tickers:
        return

    if discovery.category_reports:
        table = Table(box=box.SIMPLE_HEAD, show_header=True,
                      header_style="bold", padding=(0, 1),
                      title="Category breakdown within sectors", title_style="bold")
        table.add_column("Sector",   width=26)
        table.add_column("Category", width=24)
        table.add_column("Rank",     width=6)
        table.add_column("1d avg",   width=9)
        table.add_column("5d avg",   width=9)
        table.add_column("Trend",    width=16)
        table.add_column("Bias",     width=8)
        table.add_column("Members",  width=32)

        for sector_name, cat_report in discovery.category_reports.items():
            for cat in cat_report.categories:
                trend_color = {
                    "strong bullish": "green", "bullish": "green",
                    "neutral": "dim", "bearish": "red", "strong bearish": "red",
                }.get(cat.trend, "white")
                bias_color = ("green" if cat.signal_bias == "call" else
                              "red"   if cat.signal_bias == "put"  else "dim")
                c1 = f"{cat.avg_change_1d:+.2f}%" if cat.avg_change_1d is not None else "—"
                c5 = f"{cat.avg_change_5d:+.2f}%" if cat.avg_change_5d is not None else "—"
                c1_color = "green" if (cat.avg_change_1d or 0) >= 0 else "red"
                c5_color = "green" if (cat.avg_change_5d or 0) >= 0 else "red"
                members_str = ", ".join(cat.tickers[:5])
                if len(cat.tickers) > 5:
                    members_str += f" +{len(cat.tickers)-5}"
                table.add_row(
                    sector_name, cat.name, f"#{cat.rank}",
                    f"[{c1_color}]{c1}[/]", f"[{c5_color}]{c5}[/]",
                    f"[{trend_color}]{cat.trend}[/]",
                    f"[{bias_color}]{cat.signal_bias.upper()}[/]",
                    f"[dim]{members_str}[/]",
                )

        console.print(table)
        console.print()

    console.print(f"  {discovery.summary()}\n")


# ─── Signal output ────────────────────────────────────────────────────────────

def _print_summary_table(signals: list, report: SectorReport):
    table = Table(title="Signal Summary", box=box.SIMPLE_HEAD,
                  show_header=True, header_style="bold", title_style="bold",
                  padding=(0, 1))
    table.add_column("Priority", width=8)
    table.add_column("Ticker",   width=8)
    table.add_column("Signal",   width=8)
    table.add_column("Conf",     width=6)
    table.add_column("Sector rank", width=12)
    table.add_column("Expiry",   width=12)
    table.add_column("Allocate", width=10)
    table.add_column("1d / 5d",  width=14)

    for s in signals:
        sig_val = s.get("signal", "?").upper()
        sig_color = "green" if sig_val == "CALL" else "red"
        priority = s.get("alert_priority", "low")
        icon = {"high": "[bold red]●[/]", "medium": "[bold yellow]●[/]",
                "low": "[dim]●[/]"}.get(priority, "")

        raw = s.get("_raw_data", {})
        sector_name = raw.get("sector", "")
        sector_rank = ""
        if report and sector_name:
            sc = report.for_sector_name(sector_name)
            if sc:
                sector_rank = f"#{sc.rank} / {len(report.sectors)}"

        table.add_row(
            icon,
            f"[bold]{s.get('ticker', '?')}[/]",
            f"[{sig_color}]{sig_val}[/]",
            f"{s.get('confidence', '?')}%",
            sector_rank or "—",
            s.get("expiration", "—"),
            f"${s.get('suggested_capital_usd', 0):,.0f}",
            f"{raw.get('price_change_1d', '—')} / {raw.get('price_change_5d', '—')}",
        )

    console.print(table)
    console.print()


def _print_signal_detail(s: dict, report: SectorReport):
    ticker   = s.get("ticker", "?")
    sig_val  = s.get("signal", "?").upper()
    conf     = s.get("confidence", "?")
    expiry   = s.get("expiration", "—")
    strike   = s.get("strike_guidance", "—")
    cap_usd  = s.get("suggested_capital_usd", 0)
    reasoning = s.get("reasoning", "")
    catalysts = s.get("catalysts", [])
    risks    = s.get("risks", "")
    exp_why  = s.get("expiration_rationale", "")
    pos_why  = s.get("position_size_rationale", "")
    sec_ctx  = s.get("sector_context", "")
    watch    = s.get("watch_notes", "")
    priority = s.get("alert_priority", "low")

    raw      = s.get("_raw_data", {})
    name     = raw.get("name", ticker)
    sector   = raw.get("sector", "")
    price    = raw.get("current_price")
    earnings = raw.get("earnings_date")
    volume   = raw.get("volume_ratio", "")

    sig_color = "green" if sig_val == "CALL" else "red"
    pri_color = {"high": "red", "medium": "yellow", "low": "white"}.get(priority, "white")

    # Sector rank for this ticker
    sector_rank_str = ""
    if report and sector:
        sc = report.for_sector_name(sector)
        if sc:
            sector_rank_str = (
                f"Sector rank #{sc.rank}/{len(report.sectors)} · "
                f"{sc.trend} · RS vs SPY 5d: {sc.fmt_change(sc.rs_5d)}"
            )

    title = Text()
    title.append(f"  {ticker}", style="bold")
    if name and name != ticker:
        title.append(f"  {name[:38]}", style="dim")
    title.append(f"   [{sig_val}]", style=f"bold {sig_color}")
    title.append(f"  {conf}% confidence", style="dim")
    title.append(f"  priority: {priority}", style=pri_color)

    lines = []

    price_parts = []
    if price:       price_parts.append(f"Price: ${price:.2f}")
    if raw.get("price_change_1d"): price_parts.append(f"1d: {raw['price_change_1d']}")
    if raw.get("price_change_5d"): price_parts.append(f"5d: {raw['price_change_5d']}")
    if volume:      price_parts.append(f"Vol: {volume}")
    if sector:      price_parts.append(f"Sector: {sector}")
    if price_parts: lines.append("  " + "   ".join(price_parts))
    if sector_rank_str: lines.append(f"  [dim]{sector_rank_str}[/]")

    lines.append("")
    lines.append("  [bold]Contract[/]")
    lines.append(f"    Expiration:  {expiry}  —  {exp_why}")
    lines.append(f"    Strike:      {strike}")
    if earnings: lines.append(f"    Earnings:    {earnings}")
    lines.append(f"    Allocate:    [bold]${cap_usd:,.0f}[/]  —  {pos_why}")

    lines.append("")
    lines.append("  [bold]Signal reasoning[/]")
    lines.append(f"    {reasoning}")

    if sec_ctx:
        lines.append("")
        lines.append(f"  [bold]Sector context[/]")
        lines.append(f"    {sec_ctx}")

    if catalysts:
        lines.append("")
        lines.append("  [bold]Key catalysts[/]")
        for c in catalysts:
            lines.append(f"    · {c}")

    if risks:
        lines.append("")
        lines.append(f"  [bold]Risks[/]  {risks}")

    if watch:
        lines.append("")
        lines.append(f"  [dim]Watch: {watch}[/]")

    news = raw.get("news", [])
    if news:
        lines.append("")
        lines.append("  [bold]Recent headlines[/]")
        for n in news[:4]:
            lines.append(f"    [{n['age']}] {n['publisher']}: {n['title'][:80]}")

    lines.append("")
    console.print(Panel("\n".join(lines), title=title, border_style="dim", padding=(0, 0)))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print_header(capital: float):
    now = datetime.now().strftime("%A, %B %d %Y  %I:%M %p")
    console.print()
    console.print(Panel(
        f"[bold]Options Signal Agent v2[/]  ·  {now}\n"
        f"Capital: [bold]${capital:,.2f}[/]  ·  Sector-aware mode enabled",
        border_style="dim", padding=(0, 2),
    ))
    console.print()


def _fmt(val) -> str:
    if val is None: return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _colored_pct(val) -> str:
    if val is None: return "[dim]—[/]"
    s = _fmt(val)
    color = "green" if val >= 0 else "red"
    return f"[{color}]{s}[/]"


def _save_json(filtered, all_signals, report, path, capital):
    output = {
        "generated_at": datetime.now().isoformat(),
        "capital": capital,
        "market_bias": report.market_bias if report else None,
        "spy_1d": report.spy_change_1d if report else None,
        "spy_5d": report.spy_change_5d if report else None,
        "sector_rankings": [
            {"rank": s.rank, "name": s.name, "etf": s.etf,
             "momentum_score": s.momentum_score, "trend": s.trend,
             "rs_5d": s.rs_5d}
            for s in (report.sectors if report else [])
        ],
        "signals": [{k: v for k, v in s.items() if k != "_raw_data"} for s in filtered],
        "total_analyzed": len(all_signals),
    }
    try:
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        console.print(f"\n[green]Saved to {path}[/]")
    except Exception as e:
        console.print(f"\n[red]Could not save: {e}[/]")


if __name__ == "__main__":
    main()
