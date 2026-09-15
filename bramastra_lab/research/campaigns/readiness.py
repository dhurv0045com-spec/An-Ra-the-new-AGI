"""Evidence-bound implementation readiness for the K8 campaign (F21).

The former static reviewed disposition (always `ready=False`) is superseded
by FINAL-K8 section 22: readiness is DERIVED from a build verification
report produced by `python -m bramastra_lab.research.campaigns.k8
verify-build --data <bundle> --report-dir <new-dir> --no-updates`. The gate
revalidates the report's integrity (schema, zero-commit certification,
24 requirement verdicts, body identity) and its currency (the current
source closure must match the verified closure). Missing, stale, tampered
or incomplete evidence is fail-closed. Runtime qualification (G01-G04)
remains a separate owner-E0 requirement.

This module intentionally keeps the historical entry points
(`implementation_readiness`, `require_implementation_ready`,
`ImplementationReadinessError`) so existing consumers fail-closed
identically when evidence is absent.
"""
from __future__ import annotations

from typing import Any

from bramastra_lab.research.campaigns.verify_build import (
    BUILD_VERIFICATION_SCHEMA,
    ImplementationReadinessError,
    implementation_readiness,
    require_implementation_ready,
)

__all__ = [
    "BUILD_VERIFICATION_SCHEMA",
    "ImplementationReadinessError",
    "implementation_readiness",
    "require_implementation_ready",
]


def _unused() -> Any:  # pragma: no cover - typing anchor only
    return None
