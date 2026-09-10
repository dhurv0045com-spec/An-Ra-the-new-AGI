# ARK-019 V3.1 — RUNTIME-ONLY PREEXECUTION AMENDMENT

**Status:** preregistered after runtime-fit failure, before any comparative continuation-arm outcome  
**Scientific design:** unchanged from ARK-019 V3  
**Reason:** measured T4 runtime projected 209.68 minutes under the frozen 1.30 safety factor, exceeding the original 175-minute wall.

## Amendment

Change exactly one operational bound:

- previous campaign wall: **175 minutes**
- amended campaign wall: **240 minutes**
- packaging reserve remains **5 minutes**
- runtime safety factor remains **1.30**
- the full fixed campaign must still pass the same conservative runtime projection before comparative arms start.

The measured pre-outcome projection (~209.68 min) fits inside 240 minutes while retaining the original safety factor. This is preferable to reducing the safety factor or weakening the experiment.

## Frozen scientific invariants — UNCHANGED

- PRETRAIN seeds: `31801`, `31902`
- SKILL_B seeds: `319001`, `319002`
- 4 mandatory matched sets
- arms: `PLASTIC_HIGH`, `STATIC_REPLAY_1OF64`, `GUARDIAN_REPLAY`, `GUARDIAN_HYBRID`
- 1,000 continuation updates per arm
- evaluation every 100 updates
- 32 sequence slots/update; 28 real-text slots + 4 SKILL_B slots before replay displacement
- HIGH LR `3e-4`; LOW shadow LR `3e-6`
- Guardian thresholds/state machine
- 1/64 and 1/32 replay definitions
- CAP16X calibration definition
- science-NLL, retention, plasticity, duty-cycle and replay-cost decision thresholds
- matched data/order semantics
- exact-resume smoke
- no production/500M/AGI authorization.

No seed, arm, horizon, endpoint, controller threshold, success criterion, or treatment dose is changed.

## Failure-history handling

The earlier runtime gate failure is preserved as `PREEXECUTION_RUNTIME_FAILURE_001.md`. Existing qualified-parent artifacts may be reused only under the original identity checks. The amended launcher must not erase the historical failure receipt; it may archive the stale generic `ARK-019_V3_FAILURE.json` before the new run so that a successful final bundle cannot be mistaken for a failed run.

## Claim boundary

This amendment changes only the amount of operator wall time available to execute the already-preregistered experiment. It cannot improve the scientific verdict by definition and was selected using runtime measurements only, before comparative continuation-arm outcomes existed.