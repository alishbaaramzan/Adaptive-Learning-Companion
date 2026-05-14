#!/usr/bin/env python3
"""
analyze.py — Drift Monitoring & Feedback Analysis
──────────────────────────────────────────────────
Usage (inside container or with DB path override):
    python analyze.py
    python analyze.py --db /path/to/feedback_log.db
    python analyze.py --export report.json

Outputs to stdout and optionally writes a JSON report file.
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path("/app/checkpoint_data/feedback_log.db")

# ── ANSI colours (gracefully disabled if not a tty) ──────────────────────────
USE_COLOUR = sys.stdout.isatty()

def c(code: str, text: str) -> str:
    if not USE_COLOUR:
        return text
    codes = {"bold": "\033[1m", "reset": "\033[0m",
             "green": "\033[92m", "red": "\033[91m",
             "yellow": "\033[93m", "cyan": "\033[96m", "grey": "\033[90m"}
    return codes.get(code, "") + text + codes["reset"]


def hr(char="─", width=60) -> str:
    return char * width


# ── DB helpers ────────────────────────────────────────────────────────────────

def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(c("red", f"[ERROR] Database not found: {db_path}"))
        print("       Run the API and collect some feedback first.")
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM feedback_log ORDER BY timestamp ASC"
    ).fetchall()


# ── Analysis functions ────────────────────────────────────────────────────────

def count_totals(rows: list) -> dict:
    total    = len(rows)
    negative = sum(1 for r in rows if r["feedback"] == "bad")
    positive = total - negative
    rate     = round(negative / total * 100, 1) if total else 0
    return {"total": total, "positive": positive, "negative": negative, "neg_rate": rate}


def top_failed_queries(rows: list, n: int = 3) -> list[dict]:
    """Return the N most-thumbed-down user queries."""
    bad_inputs = [r["user_input"] for r in rows if r["feedback"] == "bad"]
    counts = Counter(bad_inputs)
    return [{"query": q, "bad_count": cnt} for q, cnt in counts.most_common(n)]


def agent_breakdown(rows: list) -> dict[str, dict]:
    agents: dict[str, Counter] = {}
    for r in rows:
        a = r["agent_type"] or "unknown"
        if a not in agents:
            agents[a] = Counter()
        agents[a][r["feedback"]] += 1
    result = {}
    for a, ctr in agents.items():
        t = ctr["good"] + ctr["bad"]
        result[a] = {
            "total": t,
            "good": ctr["good"],
            "bad": ctr["bad"],
            "neg_rate": round(ctr["bad"] / t * 100, 1) if t else 0,
        }
    return result


def daily_trend(rows: list) -> list[dict]:
    """Aggregate feedback counts by calendar date."""
    by_day: dict[str, Counter] = {}
    for r in rows:
        day = r["timestamp"][:10]           # "YYYY-MM-DD"
        if day not in by_day:
            by_day[day] = Counter()
        by_day[day][r["feedback"]] += 1
    return [
        {"date": d, "good": by_day[d]["good"], "bad": by_day[d]["bad"]}
        for d in sorted(by_day)
    ]


def sample_bad_responses(rows: list, n: int = 3) -> list[dict]:
    """Return a few recent bad interactions for manual review."""
    bad = [r for r in reversed(rows) if r["feedback"] == "bad"][:n]
    return [
        {
            "user_input":     r["user_input"][:120],
            "agent_response": r["agent_response"][:200],
            "agent_type":     r["agent_type"],
            "timestamp":      r["timestamp"],
        }
        for r in bad
    ]


# ── Pretty-print report ───────────────────────────────────────────────────────

def print_report(totals: dict, top3: list, agents: dict,
                 trend: list, samples: list) -> None:

    print()
    print(c("bold", hr("═")))
    print(c("bold", "  ADAPTIVE LEARNING COMPANION — FEEDBACK ANALYSIS REPORT"))
    print(c("grey",  f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"))
    print(c("bold", hr("═")))

    # ── 1. Totals ─────────────────────────────────────────
    print()
    print(c("cyan", "  1 · OVERVIEW"))
    print(c("grey", hr()))
    print(f"  Total responses   : {c('bold', str(totals['total']))}")
    print(f"  Positive (👍)     : {c('green', str(totals['positive']))}")
    print(f"  Negative (👎)     : {c('red',   str(totals['negative']))}")
    neg_display = f"{totals['neg_rate']}%"
    colour = "red" if totals["neg_rate"] > 30 else "yellow" if totals["neg_rate"] > 10 else "green"
    print(f"  Negative rate     : {c(colour, neg_display)}")

    # ── 2. Top 3 failed queries ───────────────────────────
    print()
    print(c("cyan", "  2 · TOP 3 FAILED QUERIES (most thumbs-down)"))
    print(c("grey", hr()))
    if top3:
        for i, item in enumerate(top3, 1):
            q   = item["query"][:70] + ("…" if len(item["query"]) > 70 else "")
            cnt = item["bad_count"]
            print(f"  {i}. [{c('red', str(cnt))} bad]  {q}")
    else:
        print(c("green", "  ✓ No negative feedback recorded yet."))

    # ── 3. Per-agent breakdown ────────────────────────────
    print()
    print(c("cyan", "  3 · PER-AGENT BREAKDOWN"))
    print(c("grey", hr()))
    if agents:
        for name, stats in agents.items():
            bar_len = int(stats["neg_rate"] / 5)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {name:<18} total={stats['total']:>4}  "
                  f"👍{stats['good']:>3}  👎{stats['bad']:>3}  "
                  f"neg={c('yellow', bar)}  {stats['neg_rate']}%")
    else:
        print("  (no data)")

    # ── 4. Daily trend ────────────────────────────────────
    print()
    print(c("cyan", "  4 · DAILY TREND (last 7 days)"))
    print(c("grey", hr()))
    recent = trend[-7:]
    if recent:
        for day in recent:
            g, b = day["good"], day["bad"]
            total = g + b or 1
            neg_pct = round(b / total * 100)
            bar = c("green", "▓" * g) + c("red", "▓" * b)
            print(f"  {day['date']}  {bar}  👍{g} 👎{b}  ({neg_pct}% neg)")
    else:
        print("  (no data)")

    # ── 5. Sample bad interactions ────────────────────────
    print()
    print(c("cyan", "  5 · SAMPLE NEGATIVE INTERACTIONS (for manual review)"))
    print(c("grey", hr()))
    if samples:
        for i, s in enumerate(samples, 1):
            print(f"  [{i}] {c('grey', s['timestamp'][:19])} · agent={s['agent_type']}")
            print(f"      USER : {s['user_input'][:80]}")
            print(f"      AGENT: {s['agent_response'][:100]}")
            print()
    else:
        print(c("green", "  ✓ No negative interactions to show."))

    print(c("bold", hr("═")))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze feedback_log.db")
    parser.add_argument("--db",     default=str(DEFAULT_DB), help="Path to feedback_log.db")
    parser.add_argument("--export", default="",              help="Optional: write JSON report to this file")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn    = connect(db_path)
    rows    = fetch_all(conn)
    conn.close()

    totals  = count_totals(rows)
    top3    = top_failed_queries(rows, n=3)
    agents  = agent_breakdown(rows)
    trend   = daily_trend(rows)
    samples = sample_bad_responses(rows, n=3)

    print_report(totals, top3, agents, trend, samples)

    # Optional JSON export
    if args.export:
        report = {
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "overview":       totals,
            "top_failed_queries": top3,
            "agent_breakdown":    agents,
            "daily_trend":        trend,
        }
        out = Path(args.export)
        out.write_text(json.dumps(report, indent=2))
        print(f"  JSON report written to: {out}\n")


if __name__ == "__main__":
    main()