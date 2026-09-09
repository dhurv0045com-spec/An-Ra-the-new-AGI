# MASTER DISCOVERY V8 — MECHANISM → REAL DATA → CONTROLLER

## Status

**PREREGISTERED BEFORE V8 IMPLEMENTATION / EXECUTION.**

Discovery V8 is not one monolithic Colab job. It is a three-stage evidence program with explicit promotion gates so later experiments cannot silently reinterpret earlier results.

## Starting evidence

Validated Discovery V7 established:

- ARK-015: non-arithmetic invariance-retention transfer, NARROW_HIGH 8/8 failures vs NARROW_LOW 0/8 and AUGMENTED_HIGH_REFERENCE 0/8 across 3 fresh acquisition parents;
- canonical accuracy remained 1.0 while order robustness eroded, demonstrating capability narrowing rather than generic task forgetting;
- large parameter movement alone is insufficient because AUGMENTED_HIGH moved farther than NARROW_HIGH while retaining robustness;
- ARK-016: update-cap mechanism study inconclusive because only 1/12 T2 opportunities produced a qualifying collapse→recovery fork.

V8 therefore replaces stochastic T2 collapse with the reliable ARK-015 stress and proceeds in three stages.

## Stage A — ARK-017 causal mechanism factorial

Goal: separate applied-update magnitude from continued invariant-supporting data.

- exact ARK-014/015 binding manifest;
- 3 fresh acquisition seeds × 2 continuation semantic orders;
- canonical-only HIGH, LOW reference, HIGH capped to LOW movement, HIGH + 1/16 order-diverse replay, HIGH cap + replay, full-augmented HIGH reference;
- 8k continuation horizon;
- SEALED robust-retention endpoint;
- explicit causal verdict: update magnitude sufficient, diversity support sufficient, joint control required, both sufficient, mixed, or low-event.

Expected T4 safety budget: **180 minutes**. This stage should be run first and returned as its own ZIP.

## Stage B — ARK-018 user 1GB real-data bridge

Goal: move from toy-from-scratch subjects to a model whose base representation was formed by real text.

- user supplies ~1GB corpus on Google Drive;
- full file is streaming-hash-bound before training;
- deterministic TRAIN/CONTROL/SEALED document split;
- 8,192 byte-level BPE trained on TRAIN only;
- ~20–25M conventional decoder-only proxy;
- throughput-resolved 50M/100M/200M token pretraining budget with exact exposure accounting;
- persistent checkpoint to Drive;
- controlled language-like binding/invariance capability on top of the real-data substrate;
- HIGH, LOW, 1/16 replay and augmented-HIGH retention arms, plus at most one optional ARK-017-selected candidate arm under a frozen mapping;
- evaluate capability retention and held-out real-text NLL together.

Expected T4 safety budget: **360 minutes** excluding Drive upload time. The job may finish earlier. It must not claim exposure to the full 1GB unless receipts show that many bytes/tokens were actually trained on.

## Stage C — ARK-019 Capability Guardian

Goal: turn the evidence into a closed-loop algorithm/infrastructure prototype.

Entry requires valid ARK-018 substrate/checkpoint and a prospectively selected protection mechanism.

- old robust capability SKILL_A already acquired;
- new disjoint robust binding capability SKILL_B trained under identical streams;
- arms: PLASTIC_HIGH, LOW_ALL, STATIC_PROTECT, CAPABILITY_GUARDIAN;
- Guardian is HIGH/plastic by default and activates the prospectively selected protection only when SKILL_A CONTROL robustness erodes; SEALED never controls it;
- measure old-skill retention, new-skill acquisition speed/quality, real-text NLL, update/path telemetry, replay cost and controller duty cycle;
- exact resumable controller state is mandatory.

Expected T4 safety budget: **240 minutes** after a valid ARK-018 checkpoint exists.

## Program-level decision

After all executed stages, write `DISCOVERY_V8_TRAINING_ARCHITECTURE_DECISION.json` with exactly one of:

- `UPDATE_TRUST_REGION_CHALLENGER`
- `CAPABILITY_REPLAY_CHALLENGER`
- `HYBRID_GUARDIAN_CHALLENGER`
- `STATIC_PROTECTION_ONLY`
- `REAL_DATA_TRANSFER_NOT_SUPPORTED`
- `CONTROLLER_NOT_SUPPORTED`
- `INSUFFICIENT_EVIDENCE`

The mapping must follow the preregistered ARK-017/018/019 verdicts; no post-hoc winner selection.

## Infrastructure that is allowed to graduate regardless of algorithm winner

Measurement-only instrumentation may be recommended if implemented and validated:

- capability probe registry;
- explicit CONTROL vs SEALED roles;
- robustness/invariance metrics in addition to ordinary held-out loss;
- raw and applied update norms;
- cumulative parameter path and milestone displacement;
- data-regime / replay state identity;
- acquire / stable / eroding / recovered / protected state events;
- exact controller/checkpoint resume identity;
- old-skill/new-skill metrics across distribution changes.

## Branch boundary

V8 modifies only `Arkenstone`. Cymek may be inspected read-only. No V8 result automatically changes Cymek production training, authorizes TPU semantics, PRE500M or the 500M run. A positive V8 controller becomes a Cymek research challenger only after explicit handoff.
