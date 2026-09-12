# NEXT EXPERIMENT DECISION TREE

**Phase 2 · 2026-09-13.** Machine source: [`NEXT_EXPERIMENT_DECISION_TREE.json`](NEXT_EXPERIMENT_DECISION_TREE.json). An agent facing "what do we run next" walks this tree using only current evidence states — no strategy may be invented after seeing outcomes. World labels (A/B/C) match `RESEARCH_DECISION_MODEL.json`.

```
ROOT: Is R1C launcher provenance clean (no origin-absent pinned commit)?
├─ NO  → ACTION-REBIND-R1C-LAUNCHER (amendment only; frozen science untouched) → Q-GUARDIAN-BUNDLE
└─ YES → Q-GUARDIAN-BUNDLE

Q-GUARDIAN-BUNDLE: Can the ARK-019 V4 raw bundle be recovered + byte-audited (zero GPU)?
├─ yes, audit PASSES → U02 positive: B17 → SUPPORTED; AD21 conditional-unlocks → Q-R1C-STATUS
├─ yes, audit FAILS  → U02 negative: B17 → CONTRADICTED (unless rerun); K02 partial → Q-R1C-STATUS
└─ no bundle         → queue GRD-VALID-001 Stage 1 rerun behind the science spine → Q-COMPUTE-BUDGET

Q-R1C-STATUS: Has EXEC-R1C executed + been interpreted?
├─ no  → ACTION-RUN-R1C (frozen campaign unchanged, ~22 T4-h) → Q-R1C-VERDICT
└─ yes → Q-R1C-VERDICT

Q-R1C-VERDICT: which frozen verdict fired?
├─ FUNCTIONAL_AND_STRUCTURAL_SUPPORTED → WORLD-A (competition dominant)
│    beliefs: B05→SUPPORTED, B04→STRONGLY_SUPPORTED
│    decisions: AD16 designable, AD15 designable, AD13 still transfer-blocked
│    next: CS-TRANSFER-001 WITH mechanism arm (MASK partition at 8L/256w)
│    unnecessary: broader vocab sweeps; tied-geometry dissection as first choice
├─ STRUCTURAL_SUPPORTED_OUTPUT_CALIBRATION_LIMITED or MIXED → WORLD-B (partial)
│    beliefs: B05→WEAKLY_SUPPORTED
│    decisions: AD16 PROVISIONAL (calibration caveat); AD15 still blocked
│    next: CS-TRANSFER-001 with BOTH mechanism + tied-geometry diagnostic arms
│    unnecessary: production tokenizer change
└─ NOT_SUFFICIENT → WORLD-C (competition insufficient)
     beliefs: B05→CONTRADICTED; K01 fires
     decisions: AD16 REJECTED (scope); AD15 blocked on tied-geometry
     next: CS-TRANSFER-001 with tied-geometry dissection arm (untied low-rank output vs tied, matched params)
     unnecessary: any output-space intervention design

Q-COMPUTE-BUDGET (reached when the V4 bundle is unavailable):
├─ GPU available  → ACTION-RUN-R1C (science spine first; Guardian rerun queues behind)
└─ no GPU         → ACTION-ZERO-GPU-PACKAGE:
                     1. V4 bundle recovery + audit attempt
                     2. CIT-C0 extended scorer screen design (untried policy families)
                     3. regenerated arithmetic corpus spec passing tools/audit_tiered_corpus.py
                        (CITADEL-DATA-001: shortcut 1.000/tier, 530 leaks, 13.5% dup, 0.004× supply)
                     4. export unreachable history (SENORA 30a8fa7 chain, CYR-006 smoke in stash,
                        CYR-005 frozen executable) to a kept ref/bundle before any gc
                   → Q-GUARDIAN-BUNDLE

Standing rules across every path:
- no experiment from EXPERIMENTS_TO_CANCEL_OR_DEFER.md may enter via any branch
- kill criteria (RESEARCH_KILL_CRITERIA.md) fire without renegotiation
- every world transition must be recorded in the evidence ledger with artifact provenance
```
