# CYR-GPU-014 / R1C — Pre-execution compatibility amendment

**Date:** 2026-09-12  
**Scientific outcome status at amendment:** `NOT_EXECUTED`  
**Original frozen executable:** `2a71cea10ebb7a231834b6b112c49e268e9631a5`  
**Compatibility executable:** `b9e4689bdcfa24a3c7d50b2f337c4e702de0bb8a`

## Trigger

Operator CUDA preflight failed before the exact-resume smoke, calibration, or any scientific arm. The first launcher issue was a Python package invocation error. After correcting invocation to module mode, the child traceback exposed a second implementation-only mismatch:

`TypeError: build_adamw_optimizer() got an unexpected keyword argument 'betas'`

The original R1C runner called the canonical V5 optimizer constructor with explicit `betas=(0.9, 0.95)`, `eps=1e-8`, and `weight_decay=0.1`. The canonical constructor already freezes exactly those values internally and accepts only `model`, `lr`, and `torch_module`.

## Amendment

`anra_v5/cyr_gpu014_r1c_run_v2.py` wraps the original frozen runner and replaces only its optimizer-construction boundary with:

- the same canonical `build_adamw_optimizer`;
- the same R1C learning rate (`CYR11_HIGH_LR`);
- the canonical frozen AdamW values `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.1`.

No experiment arm, seed, task row, model geometry, objective, learning rate, batch size, update count, evaluation cadence, diagnostic, threshold, decision rule, runtime wall, checkpoint rule, or claim boundary changes.

## Evidence boundary

This is a **pre-execution implementation repair**, not an outcome-adaptive protocol change. At discovery time:

- `EXACT_RESUME_SMOKE.json` was absent;
- `CALIBRATION.json` was absent;
- `RESOLVED.json` was absent;
- `MATCHED_INIT_PREFLIGHT.json` was absent;
- `PREEXECUTION_GATE.json` was absent;
- no R1C scientific arm had started.

The original failure therefore provides no scientific outcome information that could be used to tune the protocol.

## Operator rule

Future R1C execution must use compatibility executable `b9e4689bdcfa24a3c7d50b2f337c4e702de0bb8a` via module invocation:

`python -m anra_v5.cyr_gpu014_r1c_run_v2 ...`

The original preregistration remains the scientific protocol authority, with this amendment attached as the executable-compatibility receipt.
