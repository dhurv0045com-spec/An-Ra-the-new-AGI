# ARK-020 V2 — DESIGN REASONING (written before implementation)

## Arm-value assessment (mission §22: information per arm)

All 7 arms are retained; each earns its compute by a distinct causal contrast:

| arm | what it isolates | why it stays |
|---|---|---|
| PLASTIC_HIGH | unprotected continuation = interference existence + formation reference | formation gates are defined on it; never removable |
| STATIC_REPLAY_1OF64 | minimal permanent-rehearsal dose | the efficiency target Guardian must beat |
| STATIC_REPLAY_1OF32 | stronger permanent rehearsal | V4's strongest static protector; the area/efficiency reference |
| STATIC_CAP16X | the second R2 lever (movement cap) without replay | keeps the two-lever contrast alive on multi-skill substrate |
| GUARDIAN_REACTIVE | recovery-style dynamic protection | V4's demonstrated mode |
| GUARDIAN_PREDICTIVE | prevention-style dynamic protection | the open V4 question (degradation→recovery vs prevention) |
| GUARDIAN_HYBRID | predictive + emergency cap escalation | completes the escalation-ladder design space |

Dropping arms would save ~20% compute each but break a preregistered contrast; the
efficiency gate needs static references, the prevention question needs the reactive/
predictive pair. Multi-session exact resume makes the total tractable.

## Key V2 changes versus V1 (root causes, not patches)

1. **Integration contracts**: V1 passed wrong object shapes into V4 (nested template,
   raw factset splits). V2 builds a single task bundle carrying both the split identity
   (factsets, hashed) and the semantic views, and passes V4 exactly what V4's code
   unpacks. Contract tests execute the real V4 call chain (`mixed_update`, `bmetrics`,
   `run_dose_pilot` boundary, signature pins) on CPU with real objects.
2. **Skill C replaced** (successor-cycle → two-hop composition): see
   TASK_VALIDITY_ANALYSIS.md. The old C was information-theoretically impossible.
3. **Per-phase order seeds actually used and identity-bound** (B/C/D each stream with
   their own frozen seed; checkpoint identity includes all three).
4. **Phase-relative confirmation** is the preregistered acquisition metric; global steps
   are recorded but never compared. The 1.5× rule operates on per-phase medians across
   matched sets — exactly as preregistered, no silent strengthening.
5. **Truthful exact-resume**: the smoke hashes model, optimizer, scaler, CPU/CUDA RNG,
   registry, controller, counters, phase identity, confirmations, and telemetry
   independently, and a corruption test proves fail-closed resume.

## Red-team summary (full list in PREEXECUTION_AUDIT_V2.md)

Attacks mounted pre-freeze: impossible sealed task (fixed by replacement), template
shortcut (distinct Trace/Owners templates), frequency shortcut (bijective balance,
asserted), 1-hop bypass (Trace answers are never intermediates), replay stealing task
slots (replay replaces real-text only), unequal compute (logged per arm), global-step
bias (fixed), stale V4 artifacts (identity-checked reuse only), multi-session duplicate
telemetry (dedupe + per-session receipts), lock races (6h advisory lock + resume scan),
science substrate destruction (joint parent gate + per-set science NLL gate).

## Development calibration stance (mission §10)

No GPU is available to this agent pre-freeze; a CPU/tiny dev calibration of C's
learnability would be too weak to be meaningful and is not claimed. Instead: (a) C is
chosen for maximal learnability headroom (in-context, both maps visible, chance 1/3),
(b) the formation gate makes non-formation an explicit `INCONCLUSIVE_…` outcome rather
than a Guardian indictment, (c) 12 task slots sit mid-range of V4's demonstrated working
dose window. This stance is declared rather than hidden.
