# ARK-020 — MULTI-SKILL CONTINUAL COGNITION GENERALIZATION

**Status: PREREGISTERED BEFORE IMPLEMENTATION EXECUTION. Plan frozen at commit time; the
runner, tests, audit, and launcher follow. Nothing here may be edited after outcomes exist.**

## Question

Can a single real-text-pretrained model **sequentially acquire several structurally distinct
capabilities** (binding, second-family binding, rule induction, inverse retrieval) while
retaining all earlier capabilities, preserving science modeling, and using **adaptive,
risk-allocated protection substantially more efficiently than permanent rehearsal**?

This is the first Arkenstone campaign that asks whether the Guardian mechanism *generalizes
from one protected skill to a registry of capabilities* — and the first to measure the
**protection-cost slope** as retained capabilities accumulate (1 → 2 → 3).

## Relationship to prior evidence (audited 2026-09-12)

- ARK-017/R2 (audited): `BOTH_LEVERS_SUFFICIENT` — movement caps and sparse support replay
  each independently prevent narrowing; replay permits movement *larger* than failing HIGH.
- ARK-019 V3.1 (audited): `CONTROLLER_NOT_SUPPORTED` / new-skill-formation bottleneck;
  Guardians protected/reconstructed SKILL_A (0.963 vs 0.005 PLASTIC_HIGH robust-min) but
  SKILL_B never formed; parent construction damaged science; cadence too coarse.
- ARK-019 V4: **EXECUTED-STATUS UNVERIFIED** — the live repository records V4 as
  `READY_FOR_OPERATOR_COLAB_CUDA_PREFLIGHT` / `scientific_result_status: NOT_EXECUTED` at
  audit time. The believed `GUARDIAN_CONTINUAL_PROXY_CANDIDATE` result is NOT in the
  repository and is NOT used as a design input. ARK-020 is designed to stand on V3.1/R2
  evidence alone and re-derives its own prospective gates.
- ARK-018 V4 (audited): real-text substrate exists and is reused unchanged (SCIENCE_ONLY
  checkpoints, seeds 31801/31902, science/birth caches, 8192-token BPE).

## Frozen substrate (identical to ARK-019 V4)

- ARK-018 prepared root `/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1`
  (science SHA `b397427c…`, Birth SHA `8f17d092…`, horizon 8000).
- Model `Ark018GPT` (~21M, 10 blocks, width 384, 6 heads, tied head, vocab 8192, ctx 256).
- Batch = 32 slots/update (real-text + task + replay), AdamW (0.9, 0.95), wd 0.1, clip 1.0.
- 24 single-token words per V4 `select_tokens`; **extended to 48** with fallback thresholds
  [256, 128, 64, 32, 16, 8] (preexecution gate verifies 48 exist before anything runs).

## The four skills (token-disjoint, independently evaluable)

| skill | family | template | tokens | robustness modes | qualification |
|---|---|---|---|---|---|
| A | relational binding | `Facts: k means v; … Query: k means` | 0–11 | canonical / reversed / query_order / augmented | canonical ≥ .90, order ≥ .85, query_order ≥ .85 |
| B | relational binding (2nd family) | `Map: k -> v \| … Requested k =>` | 12–23 | same as A | same as A |
| C | rule induction (successor cycle) | `Chain: k next` (+distractor context mode) | 24–35 | canonical / distractor-context | canonical ≥ .90, distractor ≥ .85; sealed = 3 held-out keys (induction, measurement-only) |
| D | inverse retrieval | `Owners: k holds v; … Who holds v ?` → k | 36–47 | canonical / reversed / value_order | same thresholds as A |

C trains a fixed permutation-successor rule on 9 of its 12 keys; the 3 remaining keys are
**never trained** and exist only in the SEALED induction battery (measurement-only, never
controls anything). D is the value→key direction with its own template — a structurally
different retrieval computation from A/B.

## Phase structure (per matched set)

1. **PARENT** (per parent seed): acquire A jointly with real text — V4-identical gate
   (16 A-slots + 16 real, A-qualified 3-streak, science CONTROL relative NLL ≤ 0.15,
   validation-qualified). Parents are reused from V4's Drive cache if identity-matched.
2. **Phase B** — 2000 updates. B-slot dose inherited from V4 `DOSE_SELECTION.json` when
   present on Drive; otherwise the V4-style prospective pilot (2 parents × {8, 12, 16},
   smallest passing, deadline 1800) reruns first. Campaign stops
   `INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE` if none passes.
3. **Phase C** — 1500 updates, fixed 12 C-slots (prospective constant; no post-hoc tuning).
4. **Phase D** — 1500 updates, fixed 12 D-slots.
   Total continuation horizon: **5000 updates per arm.**

At every point, ALL previously qualified capabilities are evaluated (battery):
CONTROL battery every 25 updates (drives controllers — never SEALED); SEALED + science
NLL every 100 updates (measurement only); final full SEALED battery of A, B, C, D.

## Capability registry and Guardian (generalized)

Per capability `CapabilityState`: qualified_step, healthy_streak, margin_history
(robust-min per CONTROL observation), degradation_rate (slope over last 3 observations),
protection_level (PLASTIC/SPARSE64/REPLAY32/EMERGENCY_CAP16X), replay_slots_used,
last_healthy_step, failure_since, recovery_state.

Per-update treatment = union of per-capability states; **replay slots are risk-allocated**:
a replay slot serves the lowest-margin qualified old capability (deterministic tiebreak by
capability id), maximum 2 replay slots per update (≤ 1/16 of batch). Static arms instead
rotate one replay slot round-robin over old capabilities regardless of health.

Arms (7, all seeds matched 2 parents × 2 phase-order seeds):

1. `PLASTIC_HIGH` — no protection ever (never removed).
2. `STATIC_REPLAY_1OF64` — one replay slot every other update, rotating capability.
3. `STATIC_REPLAY_1OF32` — one replay slot every update, rotating capability.
4. `STATIC_CAP16X` — no replay; full-delta cap at 16× the matched set's LOW-LR shadow median
   (V4 calibration procedure, per matched set).
5. `GUARDIAN_REACTIVE` — per-capability transitions only on formal failure → REPLAY32 for
   that capability → SPARSE64 after 4-healthy-streak → PLASTIC after 200-update post-phase
   floor.
6. `GUARDIAN_PREDICTIVE` — additionally descends to SPARSE64 on the preregistered warning
   signal **before** formal failure: robust-min < 0.95 or degradation_rate ≤ −0.03 per 100
   updates over the last 3 observations.
7. `GUARDIAN_HYBRID` — predictive + V4 emergency rule: ≥2 consecutive failures under
   REPLAY32 → EMERGENCY_CAP16X (replay continues for the failing capability).

Controllers observe CONTROL metrics only. SEALED data never influences any state, dose,
stopping, or selection.

## Mechanistic diagnostics (bounded, every 100 updates)

- full parameter displacement from parent; cumulative projected path.
- gradient cosine: new-phase task gradient vs each old capability's CONTROL-prompt gradient
  (projected parameter set, V4 `pnames`).
- replay-gradient vs old-capability gradient cosine on replay updates (samples).
These separate at least: movement-size accounts (H1), destructive-direction accounts (H2),
and representation-refresh accounts (H3/H5) at the level the budget allows.

## Verdicts (frozen; exact logic in PREREGISTRATION.json + ark020_core.decide)

- Formation gates: PLASTIC_HIGH must qualify each new skill in ≥ 3/4 matched sets, else
  `INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_{B,C,D}`.
- Interference gate: plastic sets with ≥1 earlier-capability CONTROL failure ≥ 2/4, or
  static-1/32 area advantage ≥ 0.15, else `INCONCLUSIVE_LOW_INTERFERENCE`.
- Guardian success (per Guardian arm, all required):
  final SEALED qualified for **every** capability in ≥ 3/4 sets; mean all-capability sealed
  area within 0.10 of the best static arm; per-phase median new-skill confirmation step
  ≤ 1.5 × PLASTIC_HIGH's; final science SEALED NLL within 5% of PLASTIC_HIGH in every set;
  mean protection duty < 0.60; mean replay fraction < 0.05; **efficiency gate**: total
  replay slots ≤ STATIC_REPLAY_1OF32's total.
- Verdict mapping: any Guardian arm passing all → `GUARDIAN_MULTI_SKILL_CANDIDATE`;
  quality-pass but efficiency-fail → `GUARDIAN_SUPPORTED_NOT_EFFICIENT`; else
  `GUARDIAN_NOT_SUPPORTED_MULTI_SKILL`.
- Separately reported (never collapsed): PREVENTION sets, RECOVERY-within-400 sets,
  RETENTION areas, and the **protection-cost slope** — total replay slots and duty as the
  retained-capability count grows 1 → 2 → 3, Guardian vs static.

## Multi-session durability

V4 machinery unchanged: exact checkpoints (model/optimizer/scaler/CPU+CUDA RNG/controller/
registry/counters/telemetry), identity-bound resume, 10 = 5 + save + reload + 5 exact-resume
smoke, PARTIAL receipts, session timebox 225 min + 10 min packaging, runtime calibration
that estimates sessions but never changes the protocol. One concurrent session per campaign
root (Drive advisory lock file); fail-closed on identity mismatch.

## Claim ceiling

Real-text proxy multi-skill continual-learning controller evidence only. No production
optimizer law, no PRE500M/500M authorization, no universal continual-learning claim, no AGI
claim, no consciousness claim. A negative or mixed result is a valid outcome.
