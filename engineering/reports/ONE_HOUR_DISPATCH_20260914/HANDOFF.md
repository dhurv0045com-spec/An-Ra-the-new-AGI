# Chief handoff — clear assignment and immediate scoring repair

## Assignment and status

The owner asked why the agent was not executing and requested a concrete one-hour assignment. Reviewed source: ba038c6 after 79dcb18. The agent did push implementation work, but the prior chief dispatch described a 30–50-hour program and the entry points contained multiple competing current instructions. That dispatch structure was a chief-engineering problem; this change removes it.

Chief assignment: complete. H01 implementation: assigned, not completed by this update. Owner experiment: not ready. Total elapsed chief effort was not measured; individual test durations are recorded. No optimizer updates or GPU runs.

## What changed

- Replaced the accumulated notices in AGENTS.md, engineering/README.md, engineering/STATUS.md and the K8 agent prompt with one active H01 assignment. Kept old specifications available as optional references and historical notices in Git at ba038c6.
- Wrote ONE_HOUR_EXECUTION.md: three production fixes, exact file ownership, 60-minute execution sequence, acceptance criteria, commands and a commit/push handoff. It does not require reading the entire historical roadmap.
- Added h01_acceptance.py with seven executable regression contracts. Known failures are outside the default test-discovery tree and are explicitly baseline failures, not passing tests or implemented fixes.
- Assigned a bounded Luna implementation and accepted its scoring-device patch: candidate inputs, padding, spans, legal masks, value inputs and transition inputs/targets now use the model parameter device. Invalid masks and overlong value context fail before forward execution. Existing gradient and gated hidden paths are preserved.

## Verification

The old integrated audit aborts at the current submission codec, recorded in AUDIT_COMMAND.json. No complete AUDIT.json was produced and no result is inferred for later checks.

The new H01 baseline ran seven tests and failed with two assertion failures and eight subtest/errors; the exact output is in H01_BASELINE.json. It demonstrates that the assigned compiler/codec/consumer work is still needed. These are expected baseline failures, not a regression introduced by the scorer patch.

Chief verification of the new scorer tests: **4 passed, 2 CUDA tests skipped in 1.82 seconds**, recorded in SCORING_TESTS.json. Luna additionally reported **21 passed, 2 skipped** across the new scorer file plus test_research_k8.py, and one selected foundation world-backward test passed. No optimizer step was used. CUDA behavior is covered by tests but not executed on this CPU host; do not call it GPU-qualified.

Luna also identified an existing missing torch import in the foundation F6 test. H01 explicitly permits that import repair while preserving assertions so the agent can execute the full targeted integration checks.

## Acceptance and next action

Accepted: device-aware scoring implementation and CPU verification, simplified active instructions, executable one-hour assignment. Not accepted: broader cognition/RSI readiness, H01 fixes not yet implemented, or any AGI claim. Give the external implementation agent the H01 prompt; it should begin code changes, finish the three named repairs and return a tested scoped push.
