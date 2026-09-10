# ARK-017 V2 — FINAL RESULT AUDIT

**Status:** EXECUTED / PRIMARY COMPLETE / SECONDARY EFFICIENCY SCREEN COMPLETE  
**Scientific runner commit:** `377c4743f8017e3455f576eafb75bb8ab9c50284`  
**Uploaded bundle SHA-256:** `c648d3fde569ca66fb34b54e56c09589a020a71bb5f7dcb88e476c224292d32c`  
**GPU/runtime:** CUDA, torch `2.11.0+cu128`  
**Primary campaign elapsed:** ~112.36 min at last primary partial receipt  
**Final result receipt elapsed:** ~149.57 min including secondary screens  
**Claim ceiling:** Micro controlled stability–plasticity mechanism evidence; no production optimizer law, PRE500M, 500M, broad reasoning or AGI authorization.

## Integrity audit

The uploaded ZIP passed CRC validation. It contains six JSON evidence files:

- `ARK-017_V2_SMOKE_TEST.json`
- `ARK-017_V2_TASK_MANIFEST.json`
- `ARK-017_V2_PARTIAL.json`
- `ARK-017_V2_SECONDARY_PARTIAL.json`
- `ARK-017_V2_SECONDARY.json`
- `ARK-017_V2_RESULT.json`

All **6/6 receipt SHA-256 values independently recompute correctly** using the runner's canonical JSON hashing rule. All receipts bind to scientific runner commit `377c4743f8017e3455f576eafb75bb8ab9c50284`. The frozen ARK-014 binding manifest identity is `fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3`.

The GPU smoke receipt is `PASS` and records exact non-identity sparse replay, byte-identical non-replay rows, CUDA forward/backward, exact model and optimizer snapshot restoration, deterministic identical next update from identical forks, multi-step LOW movement-trace consumption by capped-HIGH, and finite capped optimizer state.

## Primary result

All three acquisition parents qualified. The primary experiment produced **6 matched continuation sets across 3 independent acquisition seeds**, satisfying the preregistered event-sufficiency condition.

| Arm | SEALED failures | Failure risk | Mean applied path |
|---|---:|---:|---:|
| `NARROW_HIGH` | **4 / 6** | **0.667** | 166.628 |
| `NARROW_LOW_REFERENCE` | 0 / 6 | 0.000 | 1.371 |
| `NARROW_HIGH_CAP1X` | 0 / 6 | 0.000 | 1.371 |
| `NARROW_HIGH_REPLAY_1OF16` | 0 / 6 | 0.000 | **388.015** |
| `NARROW_HIGH_CAP1X_REPLAY_1OF16` | 0 / 6 | 0.000 | 1.371 |
| `AUGMENTED_HIGH_REFERENCE` | 0 / 6 | 0.000 | 254.310 |

The failures were not concentrated in a single continuation order: acquisition seed 2601 had 0/2 HIGH failures, while seeds 2702 and 2803 each had 2/2 HIGH failures.

Preregistered mechanism flags:

- `UPDATE_MAGNITUDE_SUFFICIENT`
- `DIVERSITY_SUPPORT_SUFFICIENT`

Primary verdict:

> **`BOTH_LEVERS_SUFFICIENT`**

The measured rescue relative to NARROW_HIGH was 0.667 for CAP1X, 0.667 for 1/16 replay, and 0.667 for the joint arm.

## What the primary result means

**DEMONSTRATED at Micro scale:** constraining applied HIGH-LR parameter movement to the prospectively measured LOW-reference movement is sufficient to prevent the ARK-015-style invariance failure in these matched forks.

**DEMONSTRATED at Micro scale:** sparse exact non-canonical support replay at only 1/16 of examples is also sufficient to prevent the same failure, even while cumulative parameter movement is *larger* than uncapped NARROW_HIGH on average.

Therefore the earlier hypothesis `retention is explained simply by small total parameter movement` is falsified. Two distinct levers can independently protect the invariant: **update magnitude control** and **continued diversity/support in the data stream**.

This does not show that the two levers operate through the same internal mechanism. It shows each is sufficient under this controlled subject and continuation regime.

## Secondary efficiency screen

Because both primary levers qualified, the preregistered secondary screens executed for the first continuation order across all three acquisition seeds.

### Movement caps

- `CAP4X`: 0/3 failures; applied paths approximately 4.80, 7.45, 3.87.
- `CAP16X`: 0/3 failures; applied paths approximately 14.19, 18.67, 13.57.

Thus protection is not confined to an exact LOW-equivalent cap. A substantially looser movement envelope still preserved the invariant in this screen.

### Sparse diversity replay

- `REPLAY_1OF64`: 0/3 failures despite applied paths approximately 202.11, 512.22, 218.01.
- `REPLAY_1OF32`: 0/3 failures despite applied paths approximately 344.40, 1226.55, 268.09.

The **1/64 replay dose** is especially informative: only one non-canonical example per 64-example batch was enough to preserve the capability in all three screened parents while allowing very large movement.

These dose screens are secondary and use one continuation-order seed, so they should guide the next experiment/controller design but should not be promoted as universal thresholds.

## Strongest current causal picture

The combined ARK-015 → ARK-017 evidence supports:

1. Narrow high-plasticity continuation can erase a broader invariant while canonical performance remains intact.
2. Lower effective update magnitude can prevent that erosion.
3. Continuing even sparse examples that exercise the missing invariant can also prevent erosion without globally suppressing parameter movement.
4. Therefore the dangerous variable is not simply `distance moved`; it is an interaction between **optimization plasticity and whether the training stream continues to constrain the capability-relevant directions**.

This is directly compatible with ARK-010/011: high plasticity can be useful for acquisition/recovery, while protection becomes important once a capability exists.

## Promotion boundary / next step

**SUPPORTED FOR NEXT PROXY:** a closed-loop Guardian should be allowed to choose between a movement-control intervention and sparse capability-support replay based on measured capability state and cost.

**NOT YET DEMONSTRATED:** production-scale transfer, natural-language continual learning, a universal cap multiplier, a universal replay fraction, or a globally optimal controller.

For ARK-019/R3, the lowest-cost candidates worth prospective comparison are now:

- a looser movement cap (CAP4X/CAP16X family), and
- very sparse support replay (especially 1/64, with 1/32 as a stronger reference),

while preserving exact SKILL_B exposure and real-text cost accounting. Any change to the already-preregistered ARK-019 mechanism-selection rule must be made in a new prospective addendum before R3 execution; do not select post hoc after seeing R3 outcomes.
