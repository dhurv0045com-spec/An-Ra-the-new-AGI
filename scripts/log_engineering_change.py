#!/usr/bin/env python3
"""Append a standardized entry to docs/engineering/ENGINEERING_LOG.md."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "docs" / "engineering" / "ENGINEERING_LOG.md"
SEPARATOR = "\n---\n\n"
HEADER_END_MARKER = "---\n\n## "


def _build_entry(
    *,
    date: str,
    author: str,
    component: str,
    change_type: str,
    title: str,
    summary: str,
    files: str,
    metrics: str,
    verification: str,
    risk: str,
    follow_up: str,
    detail: str,
) -> str:
    heading = f"## {date} — {change_type} — `{component}` — {title}\n\n"
    table = (
        "| Field | Value |\n"
        "|-------|-------|\n"
        f"| **Date** | {date} |\n"
        f"| **Author** | {author} |\n"
        f"| **Component** | `{component}` |\n"
        f"| **Type** | {change_type} |\n"
        f"| **Summary** | {summary} |\n"
        f"| **Files** | {files} |\n"
        f"| **Metrics** | {metrics} |\n"
        f"| **Verification** | {verification} |\n"
        f"| **Risk** | {risk} |\n"
        f"| **Follow-up** | {follow_up} |\n"
    )
    body = ""
    if detail.strip():
        body = f"\n### Detail\n{detail.strip()}\n"
    return heading + table + body + SEPARATOR


def append_entry(entry: str, *, dry_run: bool = False) -> None:
    if not LOG_FILE.exists():
        raise SystemExit(f"Log file missing: {LOG_FILE}")

    text = LOG_FILE.read_text(encoding="utf-8")
    marker = "## 20"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Could not find insertion point (expected dated ## headings).")

    new_text = text[:idx] + entry + text[idx:]
    if dry_run:
        print(new_text[:2000])
        print("\n... [dry-run: not written] ...")
        return

    LOG_FILE.write_text(new_text, encoding="utf-8")
    print(f"Appended entry to {LOG_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append to An-Ra ENGINEERING_LOG.md")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--author", default="human")
    parser.add_argument("--component", required=True, help="Registry id or docs/operator/engine/tests")
    parser.add_argument("--type", required=True, choices=["ADD", "CHANGE", "REMOVE", "FIX", "DOCS", "EVAL"])
    parser.add_argument("--title", required=True, help="Short title for heading")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--files", default="n/a")
    parser.add_argument("--metrics", default="n/a")
    parser.add_argument("--verify", default="n/a", help="Verification commands")
    parser.add_argument("--risk", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--follow-up", default="none")
    parser.add_argument("--detail", default="", help="Optional markdown bullets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entry = _build_entry(
        date=args.date,
        author=args.author,
        component=args.component,
        change_type=args.type,
        title=args.title,
        summary=args.summary,
        files=args.files,
        metrics=args.metrics,
        verification=args.verify,
        risk=args.risk,
        follow_up=args.follow_up,
        detail=args.detail,
    )
    append_entry(entry, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
