# ARCHITECTURE DECISION TREE (human summary)

**Written before outcomes are known** (§28). Machine source: [`ARCHITECTURE_DECISION_TREE.json`](ARCHITECTURE_DECISION_TREE.json). Worlds match NEXT_CORE_SPEC.json `output_space_worlds`.

```
Q-R1C: EXEC-R1C verdict?
├─ functional AND structural supported        → WORLD A → Q-TRANSFER
├─ structural-only / mixed                    → WORLD B → Q-TRANSFER
└─ NOT_SUFFICIENT                             → WORLD C → WORLD-C-GEOMETRY
                                                  ├─ geometry dissection reproduces → DECIDE-OUTPUT-SPACE
                                                  └─ fails → KEEP-DEFAULT-OUTPUT-SPACE (A permanent)

Q-TRANSFER: CS-TRANSFER-001 beyond arithmetic-micro?
├─ gap ≥ 0.30 (production-tokenizer task, ≥2/3 seed-pairs) → DECIDE-OUTPUT-SPACE
├─ 0.10 < gap < 0.30                                        → natural-text mini-probe → DECIDE or K04
└─ gap ≤ 0.10 both tasks                                    → KEEP-DEFAULT-OUTPUT-SPACE (K03)

DECIDE-OUTPUT-SPACE:
├─ partition treatment sufficient (World A) → ADOPT-TRAINING-TREATMENT
│    (matrix retained; dual structural/functional reporting permanent; still V5.1)
└─ matrix size itself binds (World C)       → ADOPT-GEOMETRY-CHANGE
     (function-preservation test + new receipt + checkpoint compatibility verdict; V5.2-class)

Q-GUARDIAN: GRD-VALID-001?
├─ Guardian beats static on retention-at-dose, hidden task features, ≥3/4 sets
│    → KEEP-EXTERNAL-CONTROLLER-CANDIDATE (Core unchanged; ARK-021 unblocks)
└─ no → DO-NOT-INVEST-IN-INTERNALIZATION (K02)

Q-CITADEL-EVALUATION:
├─ NOT_READY (today) → NO-COGNITION-PROMOTION (regardless of training loss)
└─ READY             → PROMOTION-PATH-OPEN (candidate-free + sealed still mandatory)

Standing rules: kill criteria (K01-K04 here) fire without renegotiation; no branch
may invent strategy after seeing outcomes; every transition is recorded in the
evidence ledger with artifact provenance.
```
