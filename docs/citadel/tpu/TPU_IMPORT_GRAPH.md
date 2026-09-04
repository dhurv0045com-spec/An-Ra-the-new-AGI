# TPU_IMPORT_GRAPH.md

Mechanical import audit of `citadel_tpu/` (branch `citadel`). No execution.
Method: repository grep of top-level + function-level imports, verified against
`origin/cymek@298c91a` file layouts. Rule: every non-stdlib dependency below
resolves either inside the Citadel checkout or inside the pinned read-only
Cymek runtime (`citadel_tpu/runtime_bootstrap.py`); there are no other
filesystem dependencies. All torch/Cymek imports are function-level (deferred),
so `import citadel_tpu.<module>` and the preflight file checks never need torch.

## Citadel-owned (SOURCE = this checkout, BOOTSTRAP = none needed)

| MODULE | PATH | REQUIRED_FOR | AVAILABLE_IN_FRESH_COLAB? |
|---|---|---|---|
| citadel_tpu.environment | citadel_tpu/environment.py | probe, platform identity, fail-closed gate | YES (stdlib only at import) |
| citadel_tpu.xla_backend | citadel_tpu/xla_backend.py | centralized PJRT compat shim | YES (torch_xla imported lazily per call) |
| citadel_tpu.runtime_bootstrap | citadel_tpu/runtime_bootstrap.py | pinned runtime resolution, SHA identities | YES (stdlib only) |
| citadel_tpu.preflight | citadel_tpu/preflight.py | `python -m citadel_tpu.preflight` gate | YES (imports guarded per module) |
| citadel_tpu.one_update | citadel_tpu/one_update.py | T0 | YES (torch/Cymek deferred to `run()`) |
| citadel_tpu.calculator_data | citadel_tpu/calculator_data.py | canary data (pure Python) | YES (stdlib only, no torch ever) |
| citadel_tpu.calculator_train | citadel_tpu/calculator_train.py | T1 | YES (torch/Cymek deferred to `train()`) |
| citadel_tpu.checkpoint | citadel_tpu/checkpoint.py | save/load (host I/O) | YES (torch deferred to call time) |
| citadel_tpu.throughput | citadel_tpu/throughput.py | throughput + multi-device gate | YES (torch/Cymek deferred) |

## Cymek-owned (SOURCE = pinned runtime `298c91a`, BOOTSTRAP = runtime_bootstrap)

| MODULE | PATH (in runtime) | REQUIRED_FOR | RESOLUTION |
|---|---|---|---|
| anra_v5.miniature_run | anra_v5/miniature_run.py | MINI_SPEC (T0/T1/T2 model) | fetch + detached worktree, SHA-verified |
| v5_model.config | v5_model/config.py | from_spec | same runtime |
| v5_model.core | v5_model/core.py | initialize, packed_layout | same runtime |
| v5_contracts.model_spec | v5_contracts/model_spec.py | QK_NORM_EPSILON, spec validity | same runtime |
| v5_objectives.causal_lm | v5_objectives/causal_lm.py | CE loss (eligible-mask default path) | same runtime |
| v5_training.optimizer | v5_training/optimizer.py | AdamW semantics | same runtime |
| v5_training.distributed | v5_training/distributed.py | T2 ledger schema | same runtime |

Preflight (`python -m citadel_tpu.preflight`) verifies every row above by file
presence, then attempts the real imports and reports per-module PASS/FAIL.
`READY_FOR_T0=YES` requires all rows green plus an active TPU. Known local-only
limitation: on a box without torch, the Cymek import rows report FAIL with
`ModuleNotFoundError: torch` (expected; the file rows still prove resolution).
