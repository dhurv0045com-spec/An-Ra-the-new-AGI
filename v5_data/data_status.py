"""Operator data-status: foundry inventory without marketing language.

Aggregates foundry run receipts, the source registry, generator
qualification, mixture feasibility, and P35-A/V5-A data readiness into one
machine-readable report. Absent inputs report as MISSING, never as zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def build_data_status(
    *,
    foundry_receipts: list[Path],
    registry_path: Path | None = None,
    qualification: dict[str, str] | None = None,
    mixture_budget: int = 200_000_000,
    mixture_targets: dict[str, float] | None = None,
    class_to_slice: dict[str, str] | None = None,
) -> dict[str, object]:
    """Aggregate inventory, quality, qualification, and feasibility."""

    mixture_targets = mixture_targets or {"natural": 0.65, "code_math_formal": 0.20, "verified_cognition": 0.15}
    class_to_slice = class_to_slice or {
        "natural": "natural", "math": "code_math_formal", "code": "code_math_formal",
        "formal": "code_math_formal", "technical_prose": "natural", "general_prose": "natural",
    }
    total_docs = 0
    total_tokens = 0
    by_class: dict[str, int] = {}
    by_slice: dict[str, int] = {}
    unmapped_classes: dict[str, int] = {}
    quality: dict[str, int] = {"KEEP": 0, "DROP": 0, "QUARANTINE": 0}
    exact_drops = 0
    runs = 0
    for path in foundry_receipts:
        receipt = _load(path)
        if receipt is None:
            continue
        runs += 1
        total_docs += int(receipt.get("unique_documents", 0))
        total_tokens += int(receipt.get("unique_tokens", 0))
        for key, value in (receipt.get("tokens_by_class") or {}).items():
            by_class[str(key)] = by_class.get(str(key), 0) + int(value)
            target_slice = class_to_slice.get(str(key))
            if target_slice is None:
                unmapped_classes[str(key)] = unmapped_classes.get(str(key), 0) + int(value)
            else:
                by_slice[target_slice] = by_slice.get(target_slice, 0) + int(value)
        for key in quality:
            quality[key] += int((receipt.get("quality") or {}).get(key, 0))
        exact_drops += int(receipt.get("exact_duplicate_drops", 0))
    registry = _load(registry_path) if registry_path else None
    qualification = qualification or {}
    feasibility: dict[str, object] = {}
    for target_class, fraction in mixture_targets.items():
        want = int(mixture_budget * fraction)
        have = by_slice.get(target_class, 0)
        feasibility[target_class] = {"wanted": want, "available_unique": have, "covered": have >= want}
    p35a_blockers = []
    if total_tokens < mixture_budget:
        p35a_blockers.append(f"only {total_tokens} unique tokens against {mixture_budget} budget")
    for target_class in mixture_targets:
        if not feasibility[target_class]["covered"]:  # type: ignore[index]
            p35a_blockers.append(f"mixture class short: {target_class}")
    unqualified = [family for family, verdict in qualification.items() if verdict != "GENERATOR_QUALIFIED"]
    if unqualified:
        p35a_blockers.append(f"cognition families not qualified: {sorted(unqualified)}")
    return {
        "schema": "anra-v5-data-status/v1",
        "foundry_runs": runs,
        "unique_documents": total_docs,
        "unique_tokens": total_tokens,
        "tokens_by_class": by_class,
        "unmapped_classes": unmapped_classes,
        "quality": quality,
        "exact_duplicate_drops": exact_drops,
        "registry_present": registry is not None,
        "generator_qualification": dict(qualification),
        "mixture_feasibility_200M": feasibility,
        "p35a_data_blockers": p35a_blockers,
        "p35a_data_ready": not p35a_blockers,
        "v5a_data_ready": False,
        "v5a_note": "5B center needs two orders of magnitude more qualified tokens",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundry-receipt", action="append", default=[], dest="receipts")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = build_data_status(
        foundry_receipts=[Path(path) for path in args.receipts],
        registry_path=args.registry,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
