# REQUIRED EXPERIMENTS (minimum decisive set)

The minimum experiments required to freeze unresolved architecture fields. Ranked independently by decision value. Full preregistration-grade specs for EXP-1/2/3 live in `docs/research/NEXT_3_EXPERIMENTS.md` (Phase 2); the architecture-specific fields are restated here.

## R1 — EXEC-R1C (rank 1; **EXECUTED / COMPLETE**)

| Field | Value |
|---|---|
| HYPOTHESIS | training-time inactive-class competition is sufficient to change formation at fixed 24,576 matrix |
| NULL | paired AUC gap < 0.10, no G50 advantage (both endpoints) |
| MODEL | Cymek V5 RESEARCH_SMALL 4L/128w, physical vocab 24,576 all arms |
| DATA | ARK-002B split `0dd93056…` (frozen) |
| TREATMENTS | FULL_24576 / MASK_19 / MASK_4096 / MASK_8192 / MASK_16384 / OFFSET_EQ4096 |
| CONTROLS | FULL per seed; ACTIVE_ONLY structural diagnostic |
| MATCHING | seeds, streams, init bytes, batch, optimizer fixed (frozen prereg) |
| SEEDS | model 3711–3714; order 6001–6004 |
| PRIMARY | FORMATION_AUC paired gaps + sustained G50 (structural AND functional) |
| RESULT | `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; `MASK_4096-FULL_24576` AUC gaps = `[-0.139792, -0.242215, -0.019377, -0.040138]`, mean `-0.110381`; MASK_4096 sustained G50 1/4, FULL 0/4 |
| BUNDLE | SHA-256 `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e` |
| CONSEQUENCE | masked-output Candidate B is not promoted; inactive-softmax competition is not a sufficient explanation; physical vocabulary/output geometry remains unresolved |

R1C closes the softmax-competition sufficiency question but **does not** authorize a tokenizer/vocabulary change, PRE500M, or 500M training. It also does not prove FULL_24576 optimal because the physical matrix size was fixed in all arms.

## R2 — CS-TRANSFER-001 (**EXECUTED / COMPLETE** — verdict PARTIAL_OR_INTERACTION)

> Synchronized 2026-09-14: the section below is the historical pre-registration
> text, preserved unedited. Outcome: physical V4096 is NOT a robust remedy
> (dev formation-AUC mean -0.0348, dev endpoint mean -0.0688, sealed endpoint
> mean -0.1396, sealed pair gaps mixed sign). V24576 remains the conservative
> working choice; optimality NOT ESTABLISHED; vocabulary gate CLOSED. Remaining
> hypotheses moved to FORMATION-MUX-001 (CS-MECH-002 mechanism dissection +
> REP-FORM-003A rendering contrast) on this branch.

| Field | Value |
|---|---|
| HYPOTHESIS | the physical class-space formation gap persists at larger scale and/or on production-tokenizer rendering |
| NULL | gap ≤ 0.10 on both tasks (K03) |
| MODEL | 8L/256w dense decoder (V5 block contracts) |
| DATA | regenerated attack-screened arithmetic surface (task A); clean production-tokenizer numeric-rendering rows (task B) |
| TREATMENTS | **actual physical V4096 vs V24576** as the primary pair; justified intermediate physical-class controls only if preregistered |
| CONTROLS | matched seeds/init/streams; query-blind + copy baselines; contamination screens |
| MATCHING | everything but declared physical class space / task |
| SEEDS | 3 matched pairs per task (fail-closed wall) |
| PRIMARY | INTERMEDIATE_GAP on held-out exact-with-valid-EOS at fixed endpoint |
| SEALED | generator-time hash-bound sealed rows, consumed once |
| THRESHOLD | ≥0.30 transfer / 0.10–0.30 partial / ≤0.10 null (K03) |
| STOP | fixed endpoint; wall calibration pre-outcome |
| GPU HOURS | ~12–16 |
| UNLOCKS | vocabulary_size (D7), numeric representation (D14-adjacent), scale question (D9 half) |

R1C specifically argues against using a masked-softmax proxy as the decisive R2 treatment. R2 must alter the actual physical tied embedding/output geometry if it is to test transfer of the earlier class-space result.

## R3 — GRD-VALID-001 (rank 3; unblocks continual-learning interface decisions)

| Field | Value |
|---|---|
| HYPOTHESIS | dynamic Guardian ≥ static-replay retention at < 50% replay dose with task-ID-hidden features |
| NULL | advantage disappears hidden or static matches (K02) |
| MODEL | ~20–25M ARK-018 SCIENCE_ONLY parents |
| DATA | peS2o science + SKILL_A + SKILL_B (V4 definitions) |
| TREATMENTS | GUARDIAN_REPLAY / GUARDIAN_HYBRID / STATIC_1OF64 / STATIC_1OF32 / PLASTIC_HIGH |
| CONTROLS | formation-first gate (reference must acquire SKILL_B at V4 dose); hidden-feature controller |
| MATCHING | 4 matched sets × 2 SKILL_B streams |
| SEEDS | per matched set |
| PRIMARY | final SKILL_A robust-min AND cumulative replay dose; SKILL_B qualification |
| SEALED | sealed skill rows once at finalization |
| THRESHOLD | ≥ static within 0.02 AND dose < 50% in ≥ 3/4 sets AND hidden-feature persistence |
| STOP | INCONCLUSIVE_NO_VIABLE_DOSE stops before arm comparison |
| GPU HOURS | 0 (audit path) / ~6–9 (rerun path) |
| UNLOCKS | continual_learning_controller (D11); replay dose (D12) |

Explicitly NOT required before the next canary: E3 mixture screens (corpus-blocked), ARK-021 (post-R3), ARK-022/025/028 (deferred), any 500M preparation.
