# CYR-GPU-007 PLAN

## Question
Can Arkenstone's same-task HIGH→LOW protection survive in the real Cymek V5 implementation on the operator's actual Colab GPU, and — when the hardware budget permits — can a hysteretic policy preserve that benefit without materially reducing order-robust non-arithmetic plasticity relative to LOW_CONTINUE?

## Why a new identity
CYR-GPU-006 passed deterministic software validation but Cell-0 hardware calibration on the operator's assigned Colab GPU proved that its fixed 3-parent + 4-fork + 3-transfer-parent worst-case campaign could not fit the frozen 170-minute wall. No scientific outcome was observed. 007 changes only the prospective hardware resolver/scope and therefore receives a new experiment identity.

## Invariants that may not change by hardware tier
- Real Cymek V5 (`ModelSpec` + `v5_model.core.initialize`) and frozen 24,576 tokenizer.
- Acquisition is HIGH LR only, exactly once per seed.
- Candidate-free generated G90 with valid EOS stop, sustained for 3 evaluations.
- Retention only from G90-confirmed parents.
- Four matched forks from identical model+optimizer bytes and the same future stream: HIGH_CONTINUE, LOW_CONTINUE, FIXED_TIME_HIGH_TO_LOW, HYSTERETIC_HIGH_LOW.
- Actual-real-token exposure, not nominal step exposure.
- At least 2 independent acquisition parents for any replicated claim.
- Fixed-time switch is in actual continuation-token space.
- SEALED arithmetic is post-decision only.
- Timeboxed/incomplete arms cannot promote.
- No GPU result can authorize TPU, PRE500M, 500M, a production scheduler, or the future 5B corpus.

## Hardware-only tiers
The resolver may use only calibration status, measured training real-token throughput, measured candidate-free generation throughput, and the fixed 170-minute wall.

1. `FULL_3P_TRANSFER3`: 3 acquisition parents, all four matched forks, up to 3 transfer parent pairs (minimum 2). This is preferred whenever affordable.
2. `CORE_2P_TRANSFER2`: 2 acquisition parents, all four matched forks, exactly 2 transfer parent pairs. This retains replicated retention plus replicated equal-age plasticity testing.
3. `RETENTION_2P_ONLY`: 2 acquisition parents, all four matched forks, no transfer. This can support only replicated same-task retention evidence; it cannot produce a research candidate.

Within a tier choose the largest calibrated proxy in MIDI → MICRO → RESEARCH_SMALL → TINY order. TINY is a last-resort development proxy and mechanically has `research_candidate_possible=false` even if transfer is affordable.

## Dose floors
- Acquisition: >=2,000,000 actual real tokens per scheduled parent.
- Continuation: >=500,000 actual real tokens per fork.
- Transfer when enabled: >=500,000 actual real tokens per state.

Hardware may increase doses prospectively toward 4M / 2M / 1M while filling roughly the 135-minute target. Hardware may not reduce below floors.

## Slow-GPU evaluation cadence
To avoid spending the entire session on generation, the 2-parent tiers use 400k acquisition, 125k continuation, and 125k transfer token evaluation intervals. These still permit the 3-confirmation gates. The 3-parent full tier retains the denser 250k / 100k / 100k cadence.

## Transfer
When enabled, the prospectively fixed comparison remains equal-age/equal-exposure `HYSTERETIC_HIGH_LOW` vs `LOW_CONTINUE` from the same parent, then both states move to HIGH LR on the same order-augmented registry-binding + fixed old-T2 replay stream. Minimum two independent parent pairs. This is a plasticity comparison, not universal consolidation proof.

## Outcomes
- `REPLICATED_WINNER` requires the existing matched-parent retention bar across >=2 independent parents.
- A production-facing research candidate is possible only on a non-TINY transfer-enabled tier and only if HYSTERETIC is the replicated retention winner and transfer is replicated plasticity-compatible.
- Retention-only or TINY tiers have an explicit lower claim ceiling regardless of accuracy.
- All outcomes remain GPU development evidence only.

## Abort
Fail closed for: no CUDA; frozen executable/hash mismatch; tokenizer/data identity mismatch; no tier fitting the 170-minute wall; fewer than two valid parent experiments for a replicated claim; stream/parent mismatch; timebox; corrupt checkpoint; or packaging failure.
