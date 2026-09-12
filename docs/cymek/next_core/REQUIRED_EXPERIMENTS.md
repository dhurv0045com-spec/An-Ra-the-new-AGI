# REQUIRED EXPERIMENTS (minimum decisive set; NOT executed)

The minimum experiments required to freeze the unresolved architecture fields. Ranked independently by decision value. Full preregistration-grade specs for EXP-1/2/3 live in `docs/research/NEXT_3_EXPERIMENTS.md` (Phase 2); the architecture-specific fields are restated here. Nothing here is authorized to execute.

## R1 — EXEC-R1C (rank 1; unblocks output head + vocabulary mechanism)

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
| SEALED | per frozen plan |
| THRESHOLD | preregistered 4-verdict taxonomy (unchanged) |
| STOP | multi-session exact-resume; no outcome-based stopping |
| GPU HOURS | ~22 |
| UNLOCKS | NEXT_CORE_SPEC: output_head (D8), vocabulary mechanism half (D7); K01/K02 tree branches |

## R2 — CS-TRANSFER-001 (rank 2; unblocks vocabulary/tokenizer transfer)

| Field | Value |
|---|---|
| HYPOTHESIS | class-space formation gap persists at 8× scale and on production-tokenizer numbers |
| NULL | gap ≤ 0.10 on both tasks (K03) |
| MODEL | 8L/256w dense decoder (V5 block contracts) |
| DATA | regenerated attack-screened arithmetic surface (task A); clean production-tokenizer numeric-rendering rows (task B) |
| TREATMENTS | V4096 vs V24576 (A1); mechanism or geometry arm per R1C world (A2/A2'); production-tokenizer output-space pair (A3) |
| CONTROLS | matched seeds/init/streams; query-blind + copy baselines; contamination screens |
| MATCHING | everything but declared class space / task |
| SEEDS | 3 matched pairs per task (fail-closed wall) |
| PRIMARY | INTERMEDIATE_GAP on held-out exact-with-valid-EOS at fixed endpoint |
| SEALED | generator-time hash-bound sealed rows, consumed once |
| THRESHOLD | ≥0.30 transfer / 0.10–0.30 partial / ≤0.10 null (K03) |
| STOP | fixed endpoint; wall calibration pre-outcome |
| GPU HOURS | ~12–16 |
| UNLOCKS | vocabulary_size (D7), numeric representation (D14-adjacent), scale question (D9 half) |

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
