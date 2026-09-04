# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then pushed to `origin/citadel`.
> Other branches are read-only audit inputs. CPU/CUDA is never recorded as a
> TPU result. No fabricated device results. Cycle download ceiling: <10 GB
> total (target <2 GB).

## STATUS

Handover base SHA: `f460e30e7ed9f6c90bbf41435f4bbb20877c33ce`

This cycle repaired the live Colab/PyTorch-XLA 2.9 compatibility failures found
by the operator. The operator's manual diagnostic on a Colab v5e runtime showed:

```text
torch: 2.9.0+cpu
torch_xla: 2.9.0
PJRT_DEVICE: TPU
XLA device: xla:0
hardware: TPU
world size: 1
```

This is useful live evidence that the TPU exists, but it is NOT yet a Citadel
TPU receipt and therefore does not promote the certification status.

Observed defects and permanent fixes committed to `citadel`:

1. `environment.py` used legacy `xm.xrt_world_size()` and therefore reported
   `tpu_present=false` on torch-xla 2.9/PJRT. It now prefers
   `torch_xla.runtime.world_size()`, uses modern device/runtime APIs, keeps
   legacy fallbacks, and sets `PJRT_DEVICE=TPU` before importing torch-xla when
   TPU is required.
2. `xla_backend.py::assert_tpu_active()` and `barrier()` also depended on
   `xm.xrt_world_size()`. World-size/device/hardware detection is now centralized
   and PJRT-first; no direct `xrt_world_size()` dependency remains in the active
   path. Modern `torch_xla.device()` is preferred with legacy fallback.
3. `notebooks/citadel_colab_tpu.ipynb` now sets PJRT before torch-xla imports and
   always syncs an existing `/content/An-Ra-colab` checkout to latest
   `origin/citadel`, preventing stale code from surviving notebook reruns.

Relevant repair commits:

```text
bb3e763 fix(citadel): use PJRT runtime APIs for TPU detection
2760b30 fix(citadel): modernize XLA backend for PJRT
f460e30 fix(citadel): make Colab launcher sync latest PJRT-compatible code
```

## PLATFORM

```text
colab
```

Colab is the first execution surface; Kaggle remains secondary. Core backend is
platform-neutral.

## TPU STATUS

```text
IMPLEMENTED_NOT_RUN
```

Manual operator probe: TPU visible on Colab/PJRT (`hardware=TPU`, `world_size=1`).
Official `TPU_ENVIRONMENT.json` after the permanent fixes: NOT YET RETURNED.
T0 one-update after permanent fixes: NOT YET RUN.

## CALCULATOR STATUS

```text
NOT_RUN
```

Model: MINI_SPEC (~1.6M). Generator: calculator-canary/1.1. Do not run T1 until
T0 passes and exact receipts are returned.

## 5B DATA STATUS

```text
NOT_STARTED
```

`0 / 5,000,000,000` real train tokens. Do not start full-corpus work before the
TPU one-update and calculator canary gates.

## DOWNLOADS

No new dataset/checkpoint downloads were required by these compatibility fixes.
Keep total new downloads <10 GB (target <2 GB).

## QUESTIONS FOR OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

The permanently repaired code has not yet been rerun on the live Colab TPU, so
T0 still lacks a valid receipt.

## NEXT ACTION

On the existing Colab TPU runtime (or a fresh one), run the UPDATED
`notebooks/citadel_colab_tpu.ipynb` from Cell 0. Cell 0 must print a Citadel SHA
at or after `f460e30`. Then run Cell 2, Cell 4, and—only if Cell 4 reports
`probe_pass=True`—Cell 5. Stop on any new error. If Cell 5 passes, run Cell 6 and
return the exact `TPU_ENVIRONMENT.json` and `TPU_ONE_UPDATE.json` files.
