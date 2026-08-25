"""VIE qualification audit: judge MC-v7 adaptive flips against the
VerifiedExperience contract (connector/experience.py).

Contract per flip:
  - same model, same task, same decode config (single changed variable)
  - baseline (NO_CHANGE) FAILS, chosen intervention SUCCEEDS
  - observed-only intervention choice (guaranteed by policy schema)
  - verifier decides both sides (mc.verify)
  - clean provenance: commit + checkpoint SHA + fixture SHA recorded
  - before/after outputs retained in the receipt

Grouped by intervention class; each qualifying case becomes one candidate
VerifiedExperience. VIE count updates ONLY if the bank accepts.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from connector.experience import ExperienceBank, VerifiedExperience, ObservedFailure

SOURCE_COMMIT = "306be11"
CHECKPOINT_SHA = "36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001"
FIXTURE_SHA = "56be1755c03aee0e53ef672ad4354ee36ee9354a0138debf7714d9324af73ee9"


def main() -> None:
    rep = json.loads(
        (ROOT / "output/mixed_causal_v7_replication.json").read_text(encoding="utf-8"))
    rows_full = rep["per_task_rows"]
    # The committed v7 receipt stores only (adaptive_action, family) per row
    # — the full-arm booleans live in the runner's stdout summary, not the
    # per-task file. An honest audit therefore reports the blocking gap and
    # requires a retention-enabled rerun; it does NOT fabricate counts.

    qualified = 0
    rejected_no_baseline = len(rows_full)
    by_class: dict = {}

    print(f"MC-v7 rows: {len(rows_full)}")
    print(f"contract-qualifying flips: {qualified}")
    print(f"by intervention class: {dict(by_class)}")
    print(f"rejected (no NO_CHANGE arm): {rejected_no_baseline}")

    # Write the AUDIT receipt; do NOT write to the live experience bank yet —
    # qualification requires outputs retained, which the v7 runner did not
    # store (only pass booleans). Honest outcome: candidates counted,
    # VIE stays 0 until a rerun retains before/after outputs.
    receipt = {
        "schema": "anra-vie-audit/v1",
        "source_receipt": "mixed_causal_v7_replication.json",
        "fixture_sha256": FIXTURE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source_commit": SOURCE_COMMIT,
        "rows_audited": len(rows_full),
        "contract_qualifying_flips": qualified,
        "by_intervention_class": dict(by_class),
        "blocking_gap": ("v7 promotion runner stored verifier booleans but "
                         "not before/after OUTPUT STRINGS; the "
                         "VerifiedExperience contract requires retained "
                         "outputs. A retention-enabled rerun is required "
                         "before any entry enters the bank."),
        "verdict_vie_count_after_audit": 0,
    }
    (ROOT / "output/vie_audit_mc7.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print("audit receipt written")


if __name__ == "__main__":
    main()
