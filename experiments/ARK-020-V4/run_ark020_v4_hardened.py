"""Operator entry point for ARK-020 V4 + durability amendment A1.

Scientific tasks/treatments/verdicts remain in the frozen V4 runner. This entry
point installs the evidence-preserving durability overlay and its last-mile
fail-closed guardrails before exposing any mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ark020_v4_durability as A1  # noqa: E402
import ark020_v4_a1_guardrails as A1_GUARDRAILS  # noqa: E402
import run_ark020_v4 as R  # noqa: E402

A1.install(R)
A1_GUARDRAILS.install(R)


if __name__ == "__main__":
    raise SystemExit(R.main())
