# CYR-GPU-006 — PLAN

Status before Commit B: **EXECUTABLE_FREEZE_PENDING_PREREGISTRATION**.

## Question

On independently acquired Cymek V5 parents, which continuation policy best preserves capability under identical future data: constant HIGH LR, constant LOW LR, fixed-time HIGH→LOW, or hysteretic state-dependent HIGH↔LOW?

## Model / tokenizer / data

- Real Cymek V5 only via `ModelSpec` + `v5_model.core.initialize()`.
- Frozen 24,576 tokenizer; special IDs read from the artifact and identity receipt.
- T2 arithmetic worlds: first-operand tens 1–5 train, 6–7 evaluation; commutative closure is declared rather than hidden.
- Four split roles: TRAIN, DEV_CONTROLLER, DEV_MEASUREMENT, SEALED_RESERVED.
- Target = answer content + EOS; EOS carries loss; PAD/BOS do not.

## Independent parents

Frozen seeds: 707, 808, 909. **All three are attempted.** Acquisition runs exactly once per seed at HIGH=1e-3. G90 is candidate-free complete-exact-with-valid-stop ≥0.90 on DEV_CONTROLLER for three consecutive evaluations. An unqualified parent forks nothing.

## Matched forks

For each qualified parent, HIGH_CONTINUE / LOW_CONTINUE / FIXED_TIME_HIGH_TO_LOW / HYSTERETIC_HIGH_LOW restore identical parent model+optimizer bytes and consume the same continuation-tail batch hashes. Arm order rotates by parent to limit wall-order bias.

Fixed-time switches at 50% of actual continuation tokens. Hysteresis enters retention at ≥0.90 sustained 3 and re-enters plasticity below 0.50 sustained 3.

## Exposure

Only actual real tokens define dose. Minimum acquisition dose 2M/parent; target 4M. Minimum continuation 0.5M/arm; target 2M. The hardware-only resolver may downshift model scale before it reduces these floors.

## Hardware / wall

CELL 0 calibrates MIDI, MICRO and RESEARCH_SMALL where they fit, including actual optimizer steps and batched candidate-free generation. The resolver includes both training and generation throughput. Desired runtime ≈135 min; hard scientific ceiling 170 min with an 8-minute packaging reserve.

## Primary endpoint

RET90 area on DEV_MEASUREMENT. A scientific winner requires at least two independent complete, contract-valid parents; mean RET90 margin ≥0.10 against every comparator; support ≥0.05 on at least two parents; no ≥0.05 reversal. Otherwise INCONCLUSIVE.

## Transfer / sealed

Only a replicated winner enters a two-parent registry/binding transfer check. SEALED_RESERVED is observed only after the decision object exists and never changes the winner.

## Failure / resume

Completed parents and arms are durable on Google Drive and skipped on rerun. Incomplete arms restart from the same frozen parent/tail rather than resuming an ambiguous partial trajectory. Every exception packages partial evidence + FAILURE.json before surfacing the error.

## Production boundary

GPU evidence is development evidence only. It cannot authorize PRE500M or 500M. TPU/XLA remains IMPLEMENTED_PENDING_PRE500M_TPU until real hardware certification.
