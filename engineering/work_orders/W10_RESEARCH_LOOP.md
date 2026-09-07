# W10 — Integrated learning and experiment loop

**Status:** blocked on accepted component interfaces/evidence from W01–W08 and W11. **Effort:** 3–5 hours for initial integration with those prerequisites; additional campaigns separately budgeted. **Role:** integration engineer.

Read all engineering specifications and accepted component handoffs. Own `research/orchestration/`, integration tests, campaign CLI and `engineering/reports/W10/`. Global package registration changes require chief coordination.

## Deliverable

One executable campaign: select declared training tasks; collect policy-driven experience; consolidate a child; checkpoint it; compare with its parent through the independent examiner; preserve evidence; accept/reject under a frozen rule. Start with one bounded round. Extend to three only after one round restores and reproduces correctly.

The first orchestration policy is fixed and visible. The learned part is inquiry/experience selection supplied by W04, not the fact that a Python loop launches jobs. A learned method-selection controller is a subsequent experiment, not a naming change.

## Required controls

- Fixed versus learned experience collection at equal real interaction and optimizer budgets.
- Same parent, data interface, inference budget and examiner for compared children.
- Rejected children retained as evidence without replacing the champion.
- Parent and original retained-skill reference evaluated on fresh paired tasks.
- Full search/training/evaluation/recovery cost included in campaign accounting.

## Acceptance evidence

- One CLI command runs a complete small CPU campaign from random initialization with a unique manifest.
- Restart resumes from the last committed state without duplicate acquisitions or skipped decisions.
- Every promotion links actual parent/child/checkpoint/data/evaluator identities.
- Tests cover insufficient evidence, retention rejection, runtime interruption and a valid controlled improvement fixture.
- A real development run reports outcomes honestly even if no child is promoted.
- No candidate can overwrite examiner configuration, answer artifacts or prior evidence through the supported interface.
- Documentation lists exactly which parts are learned, fixed, teacher-assisted and unimplemented.

No self-modifying host code in the initial integration. Later code-search candidates require the isolation and capability contract described in the learning specification. Completing this packet establishes an integrated experimental system; it does not establish AGI or indefinitely improving intelligence.
