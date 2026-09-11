# ARK-020 V4 — DESIGN REASONING

Repairs were constrained to the seven confirmed blockers; no architecture churn.

- **Lifecycle over after-the-fact accounting**: retention events are recorded at the
  observation moment with explicit eligibility, so no later analysis can confuse
  acquisition with forgetting. This is the single most important scientific fix:
  under V3 accounting, every Guardian arm would have been blamed for "forgetting" the
  skill it was actively acquiring, guaranteeing a false interference gate.
- **Cadence semantics are the science**: R2's dose findings (1/64 vs 1/32) are only
  meaningful if the names mean real exposure. The scheduler computes per-capability
  eligibility first (state + deterministic parity), then risk-orders only eligible
  capabilities — preserving dose meaning under multi-capability contention.
- **Provenance by receipts, not trust**: historical V4 artifacts are consumed
  read-only after hash verification, or regenerated locally; dose provenance lives in
  an immutable campaign-local receipt. A resume must never depend on another
  experiment's mutable directory, and must never mutate history.
- **Identity is dual**: source model and acquired parent are different objects with
  different hashes; every receipt records both, and the scan checks the one that the
  checkpoint actually inherits.
- **Operator UX is part of fail-closed design**: the scan is a subprocess with a
  machine-parsable marker line; the notebook aborts unless the scan itself succeeds.
