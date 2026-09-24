# NEXT EXPERIMENT DECISION TREE

**Generated:** 2026-09-24
**Re-freeze phase:** 3
**Historical phase-2 snapshot:** 2026-09-13 (preserved in machine metadata)
**Machine source:** [`NEXT_EXPERIMENT_DECISION_TREE.json`](NEXT_EXPERIMENT_DECISION_TREE.json)
**Authority:** [`EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`](EVIDENCE_SOURCE_MANIFEST_2026-09-24.json), SHA-256 `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`
**Pre-consolidation tip:** `research/evidence-consolidation-2026-09-25` @ `90f77b7fa6ffd99f5a982263f03b2908298805ec`; parent of the consolidated head.

The tree starts from the current evidence state, not from a stale plan to run R1C.

```text
ROOT: R1C COMPLETE + CS-TRANSFER-001 COMPLETE + Formation-Mux S5 floor-limited
├─ yes → ACTION-FMUX-CONTROL-METRIC-PREFLIGHT
│        └─ Q-FMUX-PREFLIGHT
│           ├─ capability/metric unresolved → ACTION-FMUX-CUSTODY-AND-METRIC-REVIEW
│           └─ capability/metric pass → Q-FMUX-RECOVERY
│                ├─ original Output absent → ACTION-RECOVER-FMUX-OUTPUT
│                └─ exact custody pass → Q-FRONTIER-CONTINUATION
│                     ├─ frozen continuation prerequisites pass → Q-ROLE-TRANSFER-READINESS
│                     └─ still confounded → WORLD-FLOOR-LIMITED
│                          └─ blockers remain → ACTION-ROLE-TRANSFER-IMPLEMENT
│                                             └─ all pass → ACTION-ROLE-TRANSFER-REVIEW
└─ metadata/row state unresolved → ACTION-RECONCILE-EVIDENCE → ROOT
```

## Recorded WORLD-C path

```text
CYR-GPU-014-R1C COMPLETE 24/24
  → K01 FIRED
  → CS-TRANSFER-001 COMPLETE / PARTIAL_OR_INTERACTION
  → FMUX-CONTROL-METRIC-PREFLIGHT
  → exact original-Output recovery and conditional frozen-frontier completion
  → preregistered execution-blocked ROLE-TRANSFER-001 after all readiness gates
```

- **B05:** `CONTRADICTED` within the tested scope, not globally disproved.
- **B04:** `SUPPORTED` only for a non-monotonic, seed-sensitive development-scale class-space effect.
- **B07:** `OPEN`; CS does not establish natural-language or larger-scale transfer.
- **AD16:** masked/intermediate softmax is rejected as a tested production remedy; V24576 optimality is not claimed.
- **AD15/AD13:** tied-row geometry, parameterization/initialization, optimizer/weight-decay/denominator interactions, and transfer remain open.

## Formation-Mux floor branch

`FORMATION-MUX-001-S5-V8` is COMPLETE_BUT_FLOOR_LIMITED. Its formal `CS-MECH-002` and `REP-FORM-003A` NULLs remain authoritative, but the near-zero identity baseline prevents mechanism exoneration. The later `FORMATION-MUX-001-V12-FRONTIER-PARTIAL` has 24/24 S5 development arms plus 2/24 TIE-role frontier arms and no sealed, final, or checkpoint payload; it is not a scientific verdict. Recovery-preflight engineering passed remotely, but the original Output is absent and no recovery execution occurred.

No broad expensive mechanism campaign starts from that floor. First run the capability/metric preflight; then recover the original Output and continue the frozen frontier only if every gate passes. `ROLE-TRANSFER-001` is the preregistered blocked successor of record. The old tied-row placeholder is superseded and must not run beside it.

## Corpus and scale branch

Corpus regeneration remains required. It must pass contamination, shortcut, leakage, and sealed-fixture screens before any natural-language or larger-scale transfer. A future transfer design is not automatically authorized and does not authorize PRE500M, 250M, 500M, cognition, or AGI.

## Guardian and ARK-020 branch

```text
ARK-019 V4 raw Guardian bundle recovery/byte audit
  ├─ absent/fails → ARK-020 remains DO_NOT_RUN
  └─ passes → readiness repair/validation
                  └─ only then a separately authorized interpretation review
```

ARK-020-V4 is `DO_NOT_RUN/NOT_EXECUTED` with confirmed phase-boundary resume, partial-identity, missing-checkpoint, controller-coverage, and readiness-provenance defects. All authorization flags remain false.

## Standing prohibitions

- No rerun of R1C as a mask-only campaign.
- No production vocabulary/tokenizer change.
- No Formation-Mux mechanism lock or exoneration from a floor-limited/partial record.
- No Role-Transfer official arm, sealed evaluation, or authorization from preregistration/protocol CI alone; never run the superseded tied-row placeholder beside it.
- No cognition, AGI, tool-learning, RSI, TPU-qualification, PRE500M, 250M, or 500M authorization.
