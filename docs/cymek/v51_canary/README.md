# V5.1 CANARY — README

**Branch:** `cymek-v51-canary` · **Parent (Task-2):** `cymek-next-core-architecture` @ `38925a1ad0ffc3385cac7a7afd77c01b76d81f28` · **Date:** 2026-09-13

Task 3 turns the V5.1 specification into an **audited executable canary system**: the selected architecture runs through the REAL production training path (frozen 24,576 tokenizer → `v5_data` manifest/pack/stream → `v5_model.core` → canonical causal-CE objective → production AdamW → WSD schedule → clip certification → atomic `CheckpointStore` → fresh-process exact resume → receipts) at development scale.

**Boundary:** integration qualification only. Not cognition science, not production-data claims, not output-space conclusions (R1C owns that), not scale authorization.

| Document | Content |
|---|---|
| [CANARY_SPEC.md](CANARY_SPEC.md) / [.json](../../experiments/V5_1_CANARY/PREREGISTRATION.json) | frozen preregistration: rungs, seeds, data, gates |
| [CANARY_READINESS.md](CANARY_READINESS.md) | pre-run verdict + post-run result |
| [DATA_AUDIT.md](DATA_AUDIT.md) | split/contamination audit |
| [SHORTCUT_AUDIT.md](SHORTCUT_AUDIT.md) | trivial-baseline attack results |
| [WSD_AUDIT.md](WSD_AUDIT.md) | schedule execution trace |
| [RESUME_AUDIT.md](RESUME_AUDIT.md) | fresh-process exact-resume evidence |
| [GPU_FIT_PLAN.md](GPU_FIT_PLAN.md) | Rung B T4 memory/time model |
| [FAILURE_MODES.md](FAILURE_MODES.md) | canary failure taxonomy |
| [BRAMASTRA_EXECUTION_LESSONS.md](BRAMASTRA_EXECUTION_LESSONS.md) | cross-program engineering lessons (inspect, don't absorb) |
| [CITADEL_HANDOFF.md](CITADEL_HANDOFF.md) | what independent audit should verify |
| [OPERATOR_RUNBOOK.md](OPERATOR_RUNBOOK.md) | operator launch/resume procedure |

Runner: `python -m anra_v5.v51_canary_run --mode prepare|preflight|run|resume|evaluate|finalize|scan`.
Rung A: 10,227,456 params (CPU-qualified). Rung B: 42,092,544 params (T4 operator execution).
