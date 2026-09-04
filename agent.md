# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): this file is rewritten at the END of every Citadel work
> cycle, committed to the `citadel` branch ONLY, then `git push origin citadel`.
> No other branch is ever pushed or modified by Citadel. Other branches
> (esoes / triquetra / cymek / ...) are read-only audit inputs.
> A CPU run or CUDA run is NEVER recorded as a TPU result. No fabricated device
> results, ever. Preregistration and results are never in the same commit.

## 1. Branch pointers (FACT, update every cycle)

```text
citadel:        68ca5af12b74c813cec9eff603d9c6e6ed6806d4 (pushed, in sync with origin/citadel)
origin/esoes:   85f44b7b449f2ee39a0e80203a2d7df04614983b (foundation, unchanged)
origin/cymek:   4abeaeb3a1f524dee930480e587b10b0e416a144 (read-only; +3 commits since pinned 26a61f6)
origin/triquetra: fa44ea3 (read-only; integrity-hardening only since 2e42845)
merge-base(citadel, origin/cymek): 85f44b7 (siblings; never merged)
```

## 2. Execution target (binding)

```text
CPU     = debugging/reference
CUDA    = previous certification/reference (preserved, certifies nothing about XLA)
TPU/XLA = current target (Kaggle free TPU, expected v5e-8, NEVER hard-coded)
TRAINING_PATH_READY = false until demonstrated on the target TPU
```

## 3. TPU status (one of NOT_IMPLEMENTED / IMPLEMENTED_NOT_RUN / ONE_UPDATE_PASS / CALCULATOR_PASS / MULTI_DEVICE_PASS)

```text
IMPLEMENTED_NOT_RUN
```

Kaggle execution: `NOT_EXECUTED_ON_KAGGLE`. One-update: NOT_RUN. Calculator: NOT_RUN.
5B data: `NOT_STARTED` — `0 / 5,000,000,000` real train tokens. C1: preserved, NOT executed (blocked behind TPU path + 5B readiness).

## 4. What exists on citadel (paths)

```text
docs/citadel/                          # bootstrap audit: README, BRANCH_MAP, EVIDENCE_LEDGER,
                                       # NEGATIVE_RESULTS, OPEN_QUESTIONS, BOTTLENECK_RANKING,
                                       # RESEARCH_PROTOCOL, CYMEK_SYNC, CYMEK_* audits
docs/citadel/experiments/C0/PLAN.md    # scorer prereg (static validation PASS, full run queued)
docs/citadel/experiments/C1/PLAN.md    # data→capability prereg (DO NOT EXECUTE yet)
docs/citadel/tpu/XLA_SAFE_AUDIT.md     # static op audit @cymek 4abeaeb (no execution)
docs/citadel/tpu/XLA_TAXONOMY_MAP.md   # mission-§7 label mapping
docs/citadel/tpu/TPU_ENVIRONMENT.md    # fail-closed probe spec
docs/citadel/tpu/TPU_MILESTONES.md     # M0→M4 preregistration
docs/citadel/experiments/T0/PLAN.md    # single-device one-update cert prereg
docs/citadel/experiments/T1/PLAN.md    # calculator checkpoint prereg
citadel_tpu/                           # Kaggle source of truth (8 modules, py_compile clean)
notebooks/citadel_kaggle_tpu.ipynb     # thin launcher: probe → T0 → canary → T1 → throughput
```

## 5. Key technical facts (do not re-derive blindly)

- First-TPU model: `MINI_SPEC` (`anra_v5/miniature_run.py`: 2Lx64, 4Q/2KV, d=16, FFN128, ~1.6M). P35 is NOT the bring-up model.
- XLA shims required, architecture unchanged: XLA AdamW + `xm.optimizer_step`/`xm.mark_step`; `packed_layout` + loss-count + hashing stay on CPU host; drop `data_ptr()` on XLA; `torch_module`-injection is the port seam. No JAX/TF.
- Top device unknown: SDPA bool-mask lowering per torch-xla version (needs Kaggle probe + math fallback).
- Buckets (512/1024/2048/4096): bring-up on 512 fixed-batch only until 8-device passes.
- Calculator canary: `citadel_tpu/calculator_data.py`, `calculator-canary/1.0`, +−×÷ exact-division, canonical `a op b = c`, disjoint TRAIN/DEV/TEST seeds+ranges, overlap-guarded, CE objective only.
- No `torch_xla` code exists in `v5_*`; no calculator prior art on any branch.
- Strongest bottleneck: no Kaggle TPU execution yet (blocks T0→T1→T2→5B sizing).
- Most important negative result: `production_scoring_mode = null` (all likelihood scorers failed the bias screen) — gates every learned-cognition comparison.

## 6. Next action (ONE)

Run `notebooks/citadel_kaggle_tpu.ipynb` cells 0–3 on a Kaggle TPU notebook (env probe → T0), return `TPU_ENVIRONMENT.json` / `TPU_ONE_UPDATE.json` or the abort + log hash.

## 7. Commit log (latest first, citadel only)

```text
68ca5af docs(citadel): add kaggle tpu launcher notebook
0e4827e feat(citadel): add kaggle tpu bootstrap path
594fc77 docs(citadel): switch execution target to kaggle tpu
203ff60 docs(citadel): tpu-first override audit and milestone preregistration
f3d3ba8 test(citadel): certify P35 update on CUDA through the cymek production path
```
