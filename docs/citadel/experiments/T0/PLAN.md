# T0 — Preregistration: single-device Cymek TPU one-update certification

Status: **PREREGISTERED — NO RESULTS** (written before any TPU execution).
Branch: `citadel`. Source pins: Cymek `origin/cymek@4abeaeb`,
audit `docs/citadel/tpu/XLA_SAFE_AUDIT.md`, milestones `docs/citadel/tpu/TPU_MILESTONES.md`.

## Question

Does the unmodified Cymek PyTorch model execute one full certified update on a
single Kaggle TPU device through PyTorch/XLA?

## Existing evidence

- CUDA one-update cert PASS (`cymek_receipts/ONE_UPDATE.json`) — reference only, certifies nothing about XLA.
- Static audit: forward/loss/optimizer-step math is XLA-compilable with known shims required (XLA optimizer, `xm.mark_step`, host-side layout/count/hash); SDPA bool-mask lowering is the top unknown.
- No `torch_xla` code exists in `v5_*`; the `torch_module`-injection seam is the port surface.

## Primary hypothesis

H1: with the documented shims and no architecture change, one TPU update
certifies (params changed, optimizer stepped, grads finite, reload-identical).

## Competing hypothesis

H2: a specific op fails to compile or produces wrong numerics on XLA (leading
suspect: SDPA bool-mask lowering; runner-up: optimizer/RNG evidence path),
requiring a bounded shim (fallback attention or host-side evidence) before retry.

## Why this experiment

Smallest test that retires the largest TPU risk. Gates calculator (T1) and
everything after. No dataset, no scale confounds.

## Independent variable

Exactly one: execution backend (XLA single-device vs the CUDA reference).
Model, batch content, token budget identical.

## Fixed variables

- Model: tiny bring-up spec (to be pinned: prefer Cymek miniature or smaller; exact spec SHA recorded at run).
- Batch: tiny fixed token batch, bucket 512, fixed batch size; `packed_layout` computed on CPU host.
- Optimizer: XLA-wrapped AdamW, same betas/eps/wd/LR as reference.
- Environment receipt embedded.

## Models/checkpoints

Tiny spec SHA recorded; init seed recorded; before/after parameter SHA-256;
checkpoint hash; reload inference hash.

## Data

No corpus. Synthetic tiny batch with hash-recorded content.

## Controls

- CUDA reference one-update receipt (expected PASS, reproduced behavior).
- Negative: zero-step control (no optimizer step → params must NOT change; guards false-positive certification).
- Reload-identity: inference before save vs after reload must match exactly.

## Metrics

Primary first: certification PASS/FAIL (all gates conjunctive).
Secondary: step wall time, tokens/sec, XLA compile time vs execution time, recomp count.

## Statistical treatment

Single deterministic certification + one identical rerun for determinism. No
statistics beyond exact-match gates at this rung.

## Success threshold

Params hash changed AND optimizer steps 0→1 AND grads finite AND post-clip
norm ≤ 1.0 + 1e-6 AND checkpoint reload inference identical AND environment
block embedded.

## Failure threshold

Any gate fails → IMPLEMENTATION_FAILURE with the failing gate named and the
XLA error/log hash recorded. No scientific interpretation.

## Confound checks

1. `xm.mark_step` / `xm.optimizer_step` present and ordered (code assertion).
2. `packed_layout` ran on host (no XLA graph break inside step).
3. No `.item()`/`.cpu()` inside the compiled region.
4. Tied-embedding check uses XLA-safe identity (no `data_ptr`).
5. SDPA path recorded (fused vs fallback) in the receipt.

## Compute budget

Ceiling: 0.5 TPU-hours. Expected minutes. Storage: KB-scale receipts + tiny checkpoint.

## Stop condition

One certified update + one determinism rerun + receipts written. Then stop.
Fixes require an appended amendment note; thresholds unchanged.

## Possible outcomes

1. PASS → T1 (calculator) is unblocked; SDPA path recorded as the house standard.
2. FAIL at named gate → bounded shim preregistered (e.g. math-attention fallback), T0 re-run as T0.1 with identical thresholds.
3. Env probe fails → ABORT_NO_TPU; no TPU claim of any kind.
