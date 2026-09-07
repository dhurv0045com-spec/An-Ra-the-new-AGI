# W06 — Acquisition, replay and retention

**Status:** depends on W01/W03/W11 and qualified tasks; learned-selection extension depends on W04. **Effort:** 3–5 hours. **Role:** continual-learning engineer. **Compute:** bounded CPU acquisition pilots; larger campaigns separately allocated.

Read LEARNING_ALGORITHMS.md F and D06/D07 registry entries. Own `research/learning/consolidation/`, retention tests and `engineering/reports/W06/`.

## Problem

The initial discovery campaign shows that new-family acquisition can coincide with substantial family regressions. Improving an overall average is insufficient. Build a consolidation comparison that measures and controls this loss rather than assuming replay fixes it.

## Deliverable

Implement parent freezing, new-only acquisition, equal prior/new replay and family-balanced replay. Match optimizer updates and declared interaction budgets. Specify moment reset/preservation consistently. Connect the independent examiner through a read-only candidate interface.

Start with one acquisition. Add a three-round extension only after the single-round accounting and checkpoint lineage are correct. Preserve the initial skill reference as well as each immediate parent.

## Acceptance evidence

- Children start from identical parent tensors; parent remains unchanged.
- Replay draws no development/confirmation cases and restores its distribution/cursor correctly.
- Each arm's new/prior exposure, updates, costs and optimizer policy are recorded.
- Reports show per-family parent/child accuracy, Brier scores, paired uncertainty and worst-family changes at fixed inquiry/inference budgets.
- Cumulative forgetting is measured against the historical reference, not hidden by moving parents.
- A bounded experiment reproduces or resolves a retention weakness with all arms and seeds retained.
- No automated promotion occurs merely from a positive mean; the examiner applies the declared constraints.

The learned-selection extension compares experience selected by W04 with fixed collection under identical update/interaction limits. It must distinguish more data from more useful data. Do not introduce unmeasured regularizers, expanding capacity and replay changes in the same purported single-factor comparison.
