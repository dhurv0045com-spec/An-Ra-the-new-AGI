# ARK-019 V3.1 — PREEXECUTION AUDIT

**State:** STATIC AUDIT COMPLETE / OPERATOR CUDA RERUN REQUIRED  
**Comparative continuation-arm result:** NOT EXECUTED  
**Base V3 science executable:** `63dc47d9bdcd85e004c03aaf9a738c9682fcd8f0`  
**V3.1 runtime wrapper:** `ce9aafce0231fb3b4e1d4908ce0736e3a6dd3df7`

## Why V3.1 exists

The first operator T4 run reached the preregistered runtime-fit gate and reported `update_seconds = 0.5247120965999784` and `small_eval_seconds ≈ 0.50256029`. Under the frozen V3 projector and 1.30 safety factor, the full campaign projected to ~209.68 minutes, so the 175-minute wall correctly rejected execution.

The failure occurred in `run_all()` immediately before construction/execution of the comparative continuation arms. No PLASTIC_HIGH / STATIC_REPLAY / GUARDIAN_REPLAY / GUARDIAN_HYBRID outcome was available when V3.1 was specified.

## Audit conclusion

A runtime-only extension from 175 to **240 minutes** is scientifically acceptable here because:

1. It was chosen from pre-outcome runtime telemetry, not from Guardian efficacy results.
2. It keeps the original 1.30 runtime safety factor rather than weakening it.
3. It does not change seeds, arms, horizon, batches, learning rates, replay doses, CAP16X definition, controller thresholds, endpoints or verdict thresholds.
4. 240 minutes exceeds the measured conservative projection (~209.68 min) by ~30.3 minutes.
5. The base science module remains byte-identical; a narrow wrapper asserts the original wall/reserve/safety values and changes only `WALL_MINUTES` at execution.
6. A new pure test checks that the wrapper does not assign any of the scientific constants (`HORIZON`, `ARMS`, `PRETRAIN_SEEDS`, `SKILL_B_SEEDS`, `EVAL_EVERY`, `RUNTIME_SAFETY_FACTOR`).
7. The wrapper preserves the original runtime failure by archiving it and refuses to relabel/overwrite an unexpected prior failure or any failure accompanied by continuation-arm results.

## Operator rerun contract

The updated Colab launcher must:

- fetch the live amendment receipt before frozen checkout;
- verify amendment status and the single 175→240 change;
- checkout `ce9aafce0231fb3b4e1d4908ce0736e3a6dd3df7`;
- hash-verify base V3 core/runner/tests plus the V3.1 wrapper/test;
- run both V3 and V3.1 tests;
- require CUDA and the exact ARK-018 substrate;
- invoke `run_ark019_v31.py --mode all`.

Existing qualified parent caches from the failed preexecution run may be reused only through the original source-checkpoint/tokenizer identity gates. The comparative arm loop will still run prospectively.

## Remaining risk

240 minutes is an operator wall, not a promise that Colab will remain connected. The measured projection already includes the 1.30 factor, so one T4 session is expected to fit if its throughput remains comparable. The inherited arm checkpoints remain available every 300 updates if a later runtime failure occurs, but V3.1 is still intended as one complete campaign rather than a post-outcome protocol reduction.

## Claim boundary

Readiness is not a scientific result. Even a positive R3 result would be a **real-text proxy Guardian candidate**, not demonstrated general continual learning, production scheduling, PRE500M/500M authorization, or AGI.