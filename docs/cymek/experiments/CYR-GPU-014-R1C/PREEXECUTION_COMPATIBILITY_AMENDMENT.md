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


## Amendment 2 (2026-09-12): campaign-launch CLIP_BREACH numerics repair

**Root cause (reproduced from code on CPU, 6-arm driver):** the scientific
campaign died on arm `S1_MASK_8192` with
`abort CLIP_BREACH: post-clip global norm 1.0000042915344238 exceeds 1.0`.
The clip path (fused `clip_grad_norm_`) and the update certificate
(`_global_norm`, per-tensor `vector_norm` summed sequentially) are two
different float32 reduction orders over ~4.13M parameters; they disagree by
O(eps * sqrt(N) * ||g||) (observed 4.3e-6). The certification tolerance was
1e-6 — tighter than float32 reduction noise at this scale — so a
knife-edge arm aborted the whole campaign. No scientific constant, arm,
seed, or threshold is involved.

**Repair:** `_NORM_TOLERANCE` (production_backend.py) and `_CLIP_TOLERANCE`
(step.py) raised from 1e-6 to 1e-4 with the derivation in comments. A real
clip failure (missing/incorrect clip) lands orders of magnitude above this
and still aborts.

**Evidence:** the committed end-to-end engineering preflight
(`tests/test_v5_cyr_gpu014_r1c_e2e_preflight.py`) now drives the ACTUAL
campaign executable through all six arms (engineering fixtures: 2 updates,
1 seed, CPU) and completes; before the repair it reproduced the exact
operator failure at the same arm.
