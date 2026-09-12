# NEXT 3 EXPERIMENTS (designed, NOT executed)

**Phase 2 · 2026-09-13.** Three experiments chosen to maximize **joint** information gain. EXP-2's design branches on EXP-1's outcome; EXP-3 branches on both. These are planning preregistrations: freezing them now prevents post-hoc design after results arrive. Nothing here modifies frozen runners (R1C and ARK-020 artifacts are untouched).

Scoring context (see `RESEARCH_DECISION_MODEL.json` `candidate_experiments`): EXP-1 VALUE 4.5, EXP-2 3.4, EXP-3 3.1 (decision-model estimates, not probabilities).

---

## EXP-1: EXEC-R1C — execute the frozen softmax-competition dissection

| Field | Specification |
|---|---|
| ID | EXEC-R1C (implements the already-frozen CYR-GPU-014-R1C) |
| OBJECTIVE | Decide whether training-time inactive-class softmax competition is sufficient to change formation on a fixed 24,576-row tied model (U01) |
| DEPENDENCY | Launcher provenance amendment only: rebind `ENGINEERING_FIX` from origin-absent `6653b4ce` to the pushed equivalent (`f2c27a6` carries byte-identical repair content + identical RUN_READINESS_V4 blob `a62d18d`); preregistration, arms, seeds, endpoints untouched |
| HYPOTHESIS | MASK_4096 beats FULL_24576 on the preregistered paired endpoints |
| NULL | No paired advantage (verdict `SOFTMAX_COMPETITION_NOT_SUFFICIENT`) |
| TREATMENTS | frozen six arms: FULL_24576 / MASK_19 / MASK_4096 / MASK_8192 / MASK_16384 / OFFSET_EQ4096 |
| CONTROLS | FULL_24576 per seed; ACTIVE_ONLY_DIAGNOSTIC separates structural vs calibration failure |
| FROZEN VARIABLES | physical vocab 24,576 all arms; active IDs 0..18; batch 64; AdamW LR 1e-3 (0.9, 0.95); 3,000 updates/arm; split `0dd93056…`; model seeds 3711–3714; order seeds 6001–6004 |
| MODEL | Cymek V5 RESEARCH_SMALL 4L/128w |
| DATA | ARK-002B T2 no-carry addition (frozen split) |
| SEEDS | 4 matched pairs (frozen) |
| PRIMARY METRIC | frozen: FORMATION_AUC paired gaps + sustained-G50 counts, structural AND functional |
| SECONDARY | dose response MASK_19→16384; OFFSET rescue; failure decomposition (calibration vs structural) |
| SEALED EVALUATION | per frozen plan (full structural batteries at 0/1500/3000, diagnostics only) |
| DECISION RULE | frozen four-verdict taxonomy in `docs/cymek/experiments/CYR-GPU-014-R1C/PLAN.md` — not restated, not altered |
| STOP RULE | frozen multi-session exact-resume; no outcome-based early stopping |
| CHECKPOINT RULE | frozen: Drive every 500 updates + pre-timebox; byte-identical resume smoke gate |
| EXPECTED GPU HOURS | ~22 T4-h (4 sessions × ~5.5 h) |
| EXPECTED INFORMATION GAIN | very high (converts the program's largest discovery into a mechanism claim or kills it) |
| UNLOCKS | AD13, AD14, AD15, AD16; unblocks ARK-025; shapes EXP-2 |

## EXP-2: CS-TRANSFER-001 — class-space transfer probe (conditional on EXP-1)

| Field | Specification |
|---|---|
| ID | CS-TRANSFER-001 |
| OBJECTIVE | Test whether the class-space formation effect persists at 8× scale and under production-style tokenization (U03, B07) |
| DEPENDENCY | EXP-1 outcome selects the arm set (below); regenerated arithmetic surface for task B (post CITADEL-DATA-001 screens) |
| HYPOTHESIS (H) | intermediate-vs-extreme formation gap persists at 8L/256w and on production-tokenizer numbers |
| NULL (H0) | gap < 0.10 in both tasks → effect is micro-arithmetic-specific (belief B07 falsified; kill criterion K03 fires) |
| TREATMENTS — **if EXP-1 confirms competition** | A1: 8L/256w V4096 vs V24576 (physical); A2: 8L/256w fixed-24,576 matrix with MASK_4096-style training partition (mechanism transfer test); A3: production-tokenizer numeric-rendering task, V4096-class vs V24576 output space |
| TREATMENTS — **if EXP-1 is null/partial** | A1' as above; A2' becomes tied-geometry dissection (untied low-rank output vs tied at matched parameter count); A3 unchanged |
| CONTROLS | matched seeds/streams/init per task; copy-first + query-blind policy baselines on task B; contamination + shortcut screens pre-run |
| FROZEN VARIABLES | batch 64; AdamW (0.9,0.95) LR 1e-3; fixed endpoints (12,000 updates task A; 6,000 task B); no early stopping |
| MODEL | 8L/256w dense decoder (same family/norm/precision contracts) |
| DATA | task A: scaled ARK-002B-style generator (regenerated, screened); task B: numeric-rendering rows from the regenerated corpus surface |
| SEEDS | 3 matched model/order seed pairs per task (two mandatory curves minimum; fail closed if the wall cannot fit them) |
| PRIMARY METRIC | INTERMEDIATE_GAP (best intermediate − best extreme) on held-out candidate-free exact-with-valid-EOS at the fixed endpoint |
| SECONDARY | formation timing; seed variance; ACTIVE_ONLY diagnostics (if A2 runs); policy-baseline gaps on task B |
| SEALED EVALUATION | sealed rows drawn at generator build time, hash-bound, consumed once at endpoint |
| DECISION RULE | gap ≥ 0.30 in ≥ 2/3 seed-pairs on a task → transfer SUPPORTED on that task; 0.10 < gap < 0.30 → partial; ≤ 0.10 both tasks → K03 fires |
| STOP RULE | fixed endpoint; fail-closed wall calibration before any update |
| CHECKPOINT RULE | every 1,000 updates + pre-timebox; exact-resume smoke mandatory |
| EXPECTED GPU HOURS | ~12–16 T4-h |
| EXPECTED INFORMATION GAIN | high (decides tokenizer's status in the 500M plan) |
| UNLOCKS | AD12/AD13/AD14 (with EXP-1); K03 kill decision; 500M data-plan shape |

## EXP-3: GRD-VALID-001 — Guardian validity under audit + task-ID-hidden evaluation (conditional on EXP-1/2 sequencing, not outcomes)

| Field | Specification |
|---|---|
| ID | GRD-VALID-001 (Stage 0 is the ARK-019 V4 bundle audit; Stage 1 is the discriminating contrast) |
| OBJECTIVE | Settle U02: does the dynamic Guardian beat matched static replay on retention-at-replay-cost without task-ID leakage? |
| DEPENDENCY | Stage 0: recover + byte-audit the ARK-019 V4 bundle (zero GPU). Stage 1 requires a science-preserving parent set (exists if the bundle audits; else Stage 1 reruns V4's parent-construction stage) |
| HYPOTHESIS (H) | Guardian arms reach ≥ static-replay retention with < 50% of static replay dose, with controller features task-ID-hidden |
| NULL (H0) | Guardian advantage disappears with hidden features or is matched by static replay (kill criterion K02 fires) |
| TREATMENTS | GUARDIAN_REPLAY vs GUARDIAN_HYBRID vs STATIC_REPLAY_1OF64 vs STATIC_REPLAY_1OF32 vs PLASTIC_HIGH (reference) |
| CONTROLS | matched sets; controller observation features masked of task identity (the discriminating intervention); formation gate: reference must acquire SKILL_B at the V4-selected dose or the verdict is INCONCLUSIVE (not a controller failure) |
| FROZEN VARIABLES | doses, phases, thresholds from `PREREGISTRATION_V4.json`; observation cadence tightened 100→25 updates for prevention power (pre-registered change, documented as an amendment BEFORE outcomes) |
| MODEL | ~20–25M parents from ARK-018 SCIENCE_ONLY line |
| DATA | peS2o science + SKILL_A + SKILL_B streams (V4 definitions) |
| SEEDS | 4 matched sets × 2 SKILL_B streams |
| PRIMARY METRIC | final SKILL_A robust-min AND cumulative replay dose per set; new-skill qualification count |
| SECONDARY | prevention-vs-recovery decomposition; escalation frequency; science NLL cost |
| SEALED EVALUATION | sealed skill rows consumed once at finalization |
| DECISION RULE | H requires: Guardian retention ≥ static within 0.02 AND dose < 50% of static in ≥ 3/4 sets AND advantage persists with hidden features |
| STOP RULE | formation-first gate can stop Stage 1 before arm comparison (INCONCLUSIVE_NO_VIABLE_DOSE) |
| CHECKPOINT RULE | every 200 updates; exact-resume smoke (A1 amendment applied if running V4 code) |
| EXPECTED GPU HOURS | 0 (Stage 0) / ~6–9 T4-h (Stage 1 rerun path) |
| EXPECTED INFORMATION GAIN | high (decides the continual-learning program's direction) |
| UNLOCKS | AD21, AD20 dose policy; ARK-021 design; internalization decision |

**Joint-information rationale:** EXP-1 answers *mechanism*; EXP-2 answers *transfer* using the mechanism EXP-1 established; EXP-3 settles the only other line whose verdict could redirect the program, and its Stage 0 costs zero GPU. If compute allows only one: run EXP-1.
