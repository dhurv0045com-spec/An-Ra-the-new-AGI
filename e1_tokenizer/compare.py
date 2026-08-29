"""Pareto comparison for audited E1 candidates; avoids arbitrary weighted scores."""

from __future__ import annotations

from typing import Any


def pareto_front(reports: list[dict[str, Any]]) -> list[str]:
    if not reports:
        raise ValueError("at least one audit report is required")
    for report in reports:
        if report.get("status") != "PASS":
            raise ValueError("only passing identity audits enter comparison")
    points = {
        report["candidate"]: (
            float(report["metrics"]["tokens_per_byte"]),
            int(report["vocabulary_size"]),
        )
        for report in reports
    }
    winners: list[str] = []
    for name, point in points.items():
        dominated = any(
            other != name
            and other_point[0] <= point[0]
            and other_point[1] <= point[1]
            and other_point != point
            for other, other_point in points.items()
        )
        if not dominated:
            winners.append(name)
    return sorted(winners)
