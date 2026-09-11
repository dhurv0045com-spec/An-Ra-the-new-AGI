# ARK-020 — PRE-EXECUTION AUDIT (static, before any GPU run)

**Audit date:** 2026-09-12
**Auditor:** Arkenstone agent (ARK-020 mission)
**Verdict: READY FOR OPERATOR COLAB CUDA EXECUTION (with the runtime-gate sequence below)**

## 1. Evidence basis audit

| item | finding |
|---|---|
| ARK-019 V4 result | **NOT in the repository** (`RUN_READINESS_V4.json`: `scientific_result_status: NOT_EXECUTED` at audit time). The believed `GUARDIAN_CONTINUAL_PROXY_CANDIDATE` is treated as unverified and is NOT a design input. |
| Live HEAD at design | `e2e6e09` lineage (post `da19cb6` ledger re-baseline); no outcome data existed for ARK-020 design. |
| Evidence actually used | ARK-017/R2 (audited `BOTH_LEVERS_SUFFICIENT`), ARK-019 V3.1 (audited `CONTROLLER_NOT_SUPPORTED` / formation bottleneck), ARK-018 V4 (audited assimilation/interiorization/plasticity results). All read from committed audits, not summaries. |

## 2. Design-vs-preregistration consistency

- `PREREGISTRATION.json` frozen (commit `e2e6e09`) before implementation; implementation
  matches it: 4 skills / token groups of 12 / five-way splits (A, B, D seeds 524218-20;
  C cycle seed 524221), 7 arms, phases 2000/1500/1500, thresholds, streaks, warning
  signal (margin < 0.95 OR rate ≤ −0.03/100 over last 3 CONTROL obs), emergency rule
  (≥2 failures under REPLAY32), verdict rules incl. efficiency gate vs STATIC_REPLAY_1OF32.
- Deviation note (declared): for skill C the generic third metric slot is realized as a
  distractor-context measurement — `order_only` = 1 distractor, `distractor` = 3
  distractors, both on CONTROL keys. Thresholds unchanged. C validation = 3 held-out
  trained keys; C sealed = 3 never-trained keys (induction, measurement-only).

## 3. Leakage / confound checks

- **SEALED firewall**: controller entry points (`observe_capability`, `update_controller`,
  `treatment`) accept CONTROL-derived state only; tests pin this structurally. Validation
  metrics (used for confirmation) come from VALIDATION splits; SEALED is evaluated only in
  the measurement battery.
- **C sealed induction keys** are never trained and never in control/validation sets
  (test: disjointness; cycle closes over all 12 tokens).
- **Task exposure parity**: task slots per phase are arm-invariant (dose_b for B; 12 for
  C/D); replay replaces real-text slots only; real_slots > 0 fail-closed.
- **Identity confounds**: all arms of a matched set restore the same parent snapshot
  (model+optimizer+scaler+RNG identity-checked); per-set CAP16X calibrated by the V4
  LOW-LR shadow procedure; per-phase order seeds frozen in preregistration.
- **Determinism**: rendering/stream helpers use SHA-256-derived RNG only; the one
  process-randomized hazard (`hash()`) was removed in favor of a frozen capability index
  map. Exact-resume smoke (10 = 5 + save + reload + 5) is a required gate before main.
- **Token gate**: 48 eligible single-token words required; explicit fail-closed error
  otherwise; overlap between skill vocabularies forbidden and asserted.
- **Telemetry**: `dedupe_trajectory` applied at completion (V3.1 lesson); PARTIAL receipts
  at every timebox.

## 4. Compute plan (honest)

- Per arm: 5000 continuation updates + batteries (control every 25 over ≤4 capabilities;
  sealed+science every 100; diagnostics every 100).
- Full campaign: 4 matched sets × 7 arms × 5000 = 140,000 updates + parents + dose pilots
  + calibrations. The runner measures per-update and battery seconds on the actual T4 and
  reports `estimated_main_sessions` (225-min sessions, 1.30 safety factor). **Runtime
  estimation may not change the protocol.**
- Frozen fallback if operator time is prohibitive: a prospective addendum reducing to 2
  matched sets must be committed BEFORE any outcome data exists. Not pre-authorized here.

## 5. Failure modes and handling

| mode | handling |
|---|---|
| < 48 eligible tokens | entry-gate RuntimeError, no partial state written beyond receipt |
| parent joint gate fails | `INCONCLUSIVE_PARENT_GATE_FAILED`, campaign stops before dose stage |
| no viable B dose | `INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE` (V4 pilot rule, deadline 1800) |
| session death | exact Drive checkpoint + PARTIAL zip; rerun resumes; 6-hour advisory lock prevents concurrent writers |
| plastic cannot form C or D | formation gate in `decide` → `INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_{C,D}` |
| low interference | `INCONCLUSIVE_LOW_INTERFERENCE` (Guardian question moot) |

## 6. Residual risks (accepted, declared)

- C's induction sealed keys measure generalization of a 12-token permutation cycle; a
  model could partially solve it by co-activation rather than rule induction — the
  distractor modes on control keys guard the robustness side, but "true rule" attribution
  stays SUPPORTED-level at best.
- 140k updates is ~3× V4; multi-session resume is proven machinery, but operator wall
  time is the practical binding constraint.
- Gradient-cosine diagnostics are projected-parameter views, not full gradients;
  mechanistic claims stay at the level the mission allows.

## 7. Gate sequence an operator run must satisfy (in order)

1. substrate + source-checkpoint identity (V3/V4 receipts),
2. 48-token gate + task build + hashes in ENTRY_RECEIPT,
3. parents qualified (joint science gate),
4. B dose selection PASS (V4 inheritance or fresh pilot),
5. exact-resume smoke PASS,
6. runtime calibration recorded (sessions estimated, protocol unchanged),
7. matched-set execution with PARTIAL receipts per session,
8. `decide()` verdict + final zip `ARKENSTONE_ARK020_CONTINUAL_RESULTS.zip`.
