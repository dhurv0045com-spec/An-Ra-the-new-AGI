"""
sovereignty/reporter.py
=======================
Pass 4 of the nightly improvement pipeline: generates the nightly self-report.

Compiles results from Passes 1–3 into a structured plain-English report with:
  - Executive summary (health score: EXCELLENT/GOOD/FAIR/POOR)
  - Code quality table with delta arrows
  - Dead code findings numbered list
  - Performance benchmark table with REGRESSION/IMPROVEMENT flags
  - Recommended actions for the developer
  - 7-night historical trend with ASCII sparklines

Output:
  nightly_report_YYYYMMDD.txt

Relationship to other modules:
    improver.py calls ReportPass.run() during Pass 4.
    api.py serves the report via GET /report.
    logger.py receives the report text for queryability via GET /log.
"""

import json
import pathlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)

# Sparkline characters from lowest to highest
_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float]) -> str:
    """
    Generate an ASCII sparkline from a list of values.

    Parameters:
        values: List of numeric values (any scale).

    Returns:
        String of sparkline block characters, one per value.
    """
    if not values:
        return ""
    mn, mx = min(values), max(values)
    span = mx - mn or 1
    chars = _SPARKLINE_CHARS
    n = len(chars) - 1
    return "".join(
        chars[round((v - mn) / span * n)] for v in values
    )


def _health_score(audit: Dict, bench: Dict, dead: Dict) -> str:
    """
    Compute overall health: EXCELLENT / GOOD / FAIR / POOR.

    Scoring:
      - Start at 100
      - Each regression: -15
      - Each dead code issue: -2 (max -20)
      - Avg cyclomatic > 10: -10
      - Avg cognitive > 15: -10
      - <50% functions have docstrings: -10

    Parameters:
        audit: Aggregate dict from Pass 1.
        bench: Result dict from Pass 3.
        dead: Summary dict from Pass 2.

    Returns:
        One of 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'.
    """
    score = 100
    score -= bench.get("regressions", 0) * 15
    score -= min(dead.get("total", 0) * 2, 20)
    if audit.get("avg_cyclomatic", 0) > 10:
        score -= 10
    if audit.get("avg_cognitive", 0) > 15:
        score -= 10
    if audit.get("pct_with_docstring", 100) < 50:
        score -= 10

    if score >= 90:
        return "EXCELLENT"
    elif score >= 70:
        return "GOOD"
    elif score >= 50:
        return "FAIR"
    else:
        return "POOR"


def _load_json_safe(path: pathlib.Path) -> Dict:
    """Load JSON from path, returning empty dict on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ReportPass:
    """
    Orchestrates Pass 4: compiles all pass outputs into the nightly report.
    """

    def __init__(self, config: Config) -> None:
        """
        Parameters:
            config: Active Config instance.
        """
        self._config = config

    def run(
        self,
        audit_result: Optional[Dict] = None,
        dead_result: Optional[Dict] = None,
        bench_result: Optional[Dict] = None,
        date_str: Optional[str] = None,
    ) -> str:
        """
        Generate and write the nightly report.

        Parameters:
            audit_result: Output from AuditPass.run() (or None to load from disk).
            dead_result: Output from DeadCodePass.run() (or None to load from disk).
            bench_result: Output from BenchmarkPass.run() (or None to load from disk).
            date_str: Date label (YYYYMMDD). Defaults to today.

        Returns:
            The full report text.
        """
        date_str = date_str or datetime.now().strftime("%Y%m%d")
        data_dir = self._config.DATA_DIR

        # Load from passed results or fall back to reading JSON files
        if audit_result is None:
            audit_data = _load_json_safe(data_dir / f"audit_{date_str}.json")
            audit_agg = audit_data.get("aggregate", {})
            audit_deltas = audit_data.get("deltas", {})
        else:
            audit_agg = audit_result.get("aggregate", {})
            audit_deltas = audit_result.get("deltas", {})

        if dead_result is None:
            dead_data = _load_json_safe(data_dir / f"dead_code_{date_str}.json")
            dead_summary = dead_data.get("summary", {})
            dead_findings = dead_data.get("findings", [])
        else:
            dead_summary = dead_result.get("summary", {})
            dead_findings = dead_result.get("findings", [])

        if bench_result is None:
            bench_data = _load_json_safe(data_dir / f"benchmark_{date_str}.json")
            bench_results_list = bench_data.get("results", [])
            regressions = bench_data.get("regressions", 0)
        else:
            bench_results_list = bench_result.get("results", [])
            regressions = len(bench_result.get("regressions", []))

        bench_meta = {
            "regressions": regressions,
            "total": len(bench_results_list),
        }

        health = _health_score(audit_agg, bench_meta, dead_summary)
        now_str = datetime.now().strftime("%H:%M")
        total_issues = dead_summary.get("total", 0)

        report = self._build_report(
            date_str, now_str, health, total_issues,
            audit_agg, audit_deltas,
            dead_findings, dead_summary,
            bench_results_list, bench_meta,
        )

        # Write report file
        out_path = data_dir / f"nightly_report_{date_str}.txt"
        out_path.write_text(report, encoding="utf-8")

        log.info("Pass 4: Nightly report written to %s", out_path)
        # Log the report content for API queryability
        for line in report.splitlines():
            log.info("[REPORT] %s", line)

        return report

    def _build_report(
        self,
        date_str: str,
        now_str: str,
        health: str,
        total_issues: int,
        audit_agg: Dict,
        audit_deltas: Dict,
        dead_findings: List[Dict],
        dead_summary: Dict,
        bench_results: List[Dict],
        bench_meta: Dict,
    ) -> str:
        """Build the full report string."""
        sections = []

        # ── Section 1: Executive Summary ──────────────────────────────────
        sections.append(self._section_header("SECTION 1 — EXECUTIVE SUMMARY"))
        sections.append(
            f"Tonight's run completed at {now_str}. "
            f"{total_issues} code quality issues were found. "
            f"Overall system health is {health}."
        )

        # ── Section 2: Code Quality ────────────────────────────────────────
        sections.append(self._section_header("SECTION 2 — CODE QUALITY (Pass 1)"))
        if audit_deltas:
            rows = [f"  {'Metric':<30} {'Value':>10} {'Delta':>10} Dir"]
            rows.append("  " + "-" * 54)
            for key, info in audit_deltas.items():
                rows.append(
                    f"  {key:<30} {str(info.get('current', '?')):>10} "
                    f"{str(info.get('delta', '?')):>10}  {info.get('direction', '?')}"
                )
            sections.append("\n".join(rows))
        else:
            sections.append("  (No baseline available — first run establishes baseline)")

        if audit_agg.get("flagged_complexity"):
            sections.append("\n  Flagged for high complexity:")
            for fn in audit_agg["flagged_complexity"]:
                sections.append(f"    ⚠  {fn}")

        # ── Section 3: Dead Code Findings ─────────────────────────────────
        sections.append(self._section_header("SECTION 3 — DEAD CODE FINDINGS (Pass 2)"))
        if dead_findings:
            for i, f in enumerate(dead_findings[:20], 1):
                sections.append(f"  {i}. [{f['category']}] {f['description']}")
            if len(dead_findings) > 20:
                sections.append(f"  ... and {len(dead_findings) - 20} more (see dead_code_{date_str}.json)")
        else:
            sections.append("  No dead code or quality issues found.")

        # ── Section 4: Performance Benchmarks ─────────────────────────────
        sections.append(self._section_header("SECTION 4 — PERFORMANCE BENCHMARKS (Pass 3)"))
        if bench_results:
            rows = [f"  {'ID':<5} {'Name':<35} {'Median':>12} {'Unit':<8} {'Δ%':>7}  Flag"]
            rows.append("  " + "-" * 76)
            for r in bench_results:
                flag_str = ""
                if r.get("flag") == "REGRESSION":
                    flag_str = "⚠ REGRESSION"
                elif r.get("flag") == "IMPROVEMENT":
                    flag_str = "✓ IMPROVEMENT"
                rows.append(
                    f"  {r['id']:<5} {r['name']:<35} {r['median']:>12.4f} "
                    f"{r['unit']:<8} {r['delta_pct']:>+6.1f}%  {flag_str}"
                )
            sections.append("\n".join(rows))
        else:
            sections.append("  (No benchmark results available)")

        # ── Section 5: Recommended Actions ────────────────────────────────
        sections.append(self._section_header("SECTION 5 — RECOMMENDED ACTIONS"))
        actions = self._generate_actions(audit_agg, dead_findings, bench_results)
        if actions:
            for i, action in enumerate(actions, 1):
                sections.append(f"  {i}. {action}")
        else:
            sections.append("  No actions required — system is in excellent shape.")

        # ── Section 6: Historical Trend ────────────────────────────────────
        sections.append(self._section_header("SECTION 6 — HISTORICAL TREND (Last 7 Nights)"))
        sections.append(self._historical_trend(date_str))

        header = [
            "=" * 70,
            f"  SOVEREIGNTY — NIGHTLY SELF-REPORT",
            f"  Date: {date_str}   Health: {health}",
            "=" * 70,
            "",
        ]
        footer = ["", "=" * 70, "  END OF REPORT", "=" * 70]

        return "\n".join(header + sections + footer)

    def _section_header(self, title: str) -> str:
        """Format a section header."""
        return f"\n{'─' * 70}\n  {title}\n{'─' * 70}"

    def _generate_actions(
        self,
        audit_agg: Dict,
        dead_findings: List[Dict],
        bench_results: List[Dict],
    ) -> List[str]:
        """Generate concrete recommended actions for the developer."""
        actions = []

        # High complexity functions
        for fn in audit_agg.get("flagged_complexity", [])[:3]:
            actions.append(
                f"Reduce complexity of '{fn}' — consider splitting into smaller functions. "
                "Effort: Medium."
            )

        # Missing docstrings (sample)
        no_doc = audit_agg.get("flagged_no_docstring", [])
        if no_doc:
            sample = ", ".join(no_doc[:3])
            actions.append(
                f"Add docstrings to: {sample} (and {len(no_doc) - 3} more). "
                "Effort: Low."
            )

        # Dead code
        unused_imports = [f for f in dead_findings if f["category"] == "unused_import"]
        if unused_imports:
            actions.append(
                f"Remove {len(unused_imports)} unused import(s) — see suggested_removals file. "
                "Effort: Low."
            )

        # Benchmark regressions
        for r in bench_results:
            if r.get("flag") == "REGRESSION":
                actions.append(
                    f"Investigate {r['id']} ({r['name']}) regression of {r['delta_pct']:+.1f}%. "
                    "Effort: High."
                )

        return actions

    def _historical_trend(self, current_date_str: str) -> str:
        """
        Build 7-night trend tables and sparklines from stored audit/benchmark files.

        Parameters:
            current_date_str: Current date (YYYYMMDD).

        Returns:
            Formatted trend section string.
        """
        data_dir = self._config.DATA_DIR
        current_date = datetime.strptime(current_date_str, "%Y%m%d")
        dates = [
            (current_date - timedelta(days=i)).strftime("%Y%m%d")
            for i in range(6, -1, -1)
        ]

        complexities = []
        issue_counts = []
        regression_counts = []

        for d in dates:
            audit_data = _load_json_safe(data_dir / f"audit_{d}.json")
            agg = audit_data.get("aggregate", {})
            complexities.append(agg.get("avg_cyclomatic", 0.0))

            dead_data = _load_json_safe(data_dir / f"dead_code_{d}.json")
            issue_counts.append(dead_data.get("summary", {}).get("total", 0))

            bench_data = _load_json_safe(data_dir / f"benchmark_{d}.json")
            regression_counts.append(bench_data.get("regressions", 0))

        lines = [
            "  Date        AvgCyclo  Issues  Regressions",
            "  " + "-" * 44,
        ]
        for d, cyc, iss, reg in zip(dates, complexities, issue_counts, regression_counts):
            dt = datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
            lines.append(f"  {dt}  {cyc:>8.2f}  {iss:>6}  {reg:>11}")

        lines += [
            "",
            f"  Complexity  sparkline: {_sparkline(complexities)}",
            f"  Issues      sparkline: {_sparkline(issue_counts)}",
            f"  Regressions sparkline: {_sparkline(regression_counts)}",
        ]
        return "\n".join(lines)
