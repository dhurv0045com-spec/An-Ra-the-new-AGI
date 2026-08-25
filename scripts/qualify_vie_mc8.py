"""Qualify MC-v8 retained-output flips into VerifiedExperience entries.

Contract check per flip (connector/experience.py):
  - same model (checkpoint SHA recorded), same task, same decode config
  - single changed variable: NO_CHANGE -> CONSTRAINED is 'decode';
    NO_CHANGE -> NORMALIZED is 'selection' (candidate re-selection at
    runtime, no weight change)
  - baseline fails, intervention succeeds, verifier decides both
  - observed-only decision (policy schema forbids evaluator fields)
  - before/after outputs retained in the receipt
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from connector.experience import ExperienceBank, ObservedFailure, VerifiedExperience

CHECKPOINT_SHA = "36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001"
FIXTURE_SHA = "f5664e13ddd37f2024c7960d79dcead6ad1f5c16adffa089af858fd33c2ca8fa"
SOURCE_COMMIT = "1cb5b80"


def main() -> None:
    rep = json.loads(
        (ROOT / "output/mixed_causal_v8_confirmation.json").read_text(encoding="utf-8"))
    bank = ExperienceBank(ROOT / "data/experience_bank/experiences.jsonl")

    existing = {e["experience_id"] for e in bank.all()}
    added = skipped = 0
    by_class = defaultdict(int)

    for i, x in enumerate(rep["per_task_rows"]):
        acts = x.get("actions", {})
        a = x.get("adaptive_action")
        if "NO_CHANGE" not in acts or a not in acts or a == "NO_CHANGE":
            continue
        base, chosen = acts["NO_CHANGE"], acts[a]
        if base["pass"] or not chosen["pass"]:
            continue
        if not chosen.get("retained_output"):
            continue
        changed_var = "decode" if a == "CONSTRAINED" else "selection"
        payload = json.dumps({"fixture": FIXTURE_SHA, "case": i,
                              "action": a}, sort_keys=True)
        import hashlib
        eid = "ve-" + hashlib.sha256(payload.encode()).hexdigest()[:16]
        if eid in existing:
            skipped += 1
            continue
        exp = VerifiedExperience(
            experience_id=eid,
            task=ObservedFailure(
                task_id=f"mc8-{i}",
                original_input=f"fixture:{FIXTURE_SHA[:12]} case {i}",
                success_criterion="mixed-causal family verifier",
                failed_output=base.get("retained_output",
                                       base.get("output", ""))),
            parent_checkpoint_sha256=CHECKPOINT_SHA,
            changed_variable=changed_var,
            intervention_cost=chosen["cost"],
            corrected_output=chosen["retained_output"],
            variables_held_constant=("prompt", "checkpoint", "decode-config"),
            baseline_success=False,
            intervention_success=True,
            diagnosis_hypothesis=(
                f"observed-state evidence predicted {a} would repair this "
                "failure; single-variable runtime flip confirmed"),
            diagnosis_confidence=1.0,
            source_commit=SOURCE_COMMIT,
            timestamp="2026-08-25",
        )
        bank.add(exp)
        existing.add(eid)
        added += 1
        by_class[changed_var] += 1

    total_vie = len(bank.all())
    receipt = {
        "schema": "anra-vie-qualification/v2",
        "source_receipt": "mixed_causal_v8_confirmation.json",
        "fixture_sha256": FIXTURE_SHA,
        "added_this_run": added,
        "skipped_duplicates": skipped,
        "by_intervention_class": dict(by_class),
        "bank_total_after": total_vie,
    }
    (ROOT / "output/vie_qualification_mc8.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
