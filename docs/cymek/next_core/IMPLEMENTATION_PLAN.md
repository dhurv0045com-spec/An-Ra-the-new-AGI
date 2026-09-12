# IMPLEMENTATION PLAN (V5.1 migration)

**Rule (§23/§42):** the evidence-driven winner is effectively V5 with contract changes — so there is **no ground-up rewrite and no fake V6**. The implementation is a thin contract layer (`v5_next/`) over the live V5 stack.

## Module classification

| Module | Class | Note |
|---|---|---|
| `v5_model/*` (core, block, attention, embedding, initialize, config) | **KEEP** | exact geometry audited; receipts asserted at construction |
| `v5_training/*` (step, optimizer, schedule, checkpoint, state, runner, production_backend, canaries) | **KEEP** | strongest engineering asset; R1C tolerance already single-sourced |
| `v5_objectives/*` | **KEEP** | EOS contract LOCKED; aux λ=0 |
| `v5_data/*`, `v5_evaluation/*`, `v5_promotion/*`, `v5_registry/*`, `v5_contracts/*` | **KEEP** | contracts + fail-closed gates; corpus consumption stays blocked until certification |
| `anra_v5/*` | **KEEP** | experiment executables (frozen identities) |
| `v5_next/` (NEW) | **ADD** | contract layer: `NextCoreContract` (geometry + output-mode + EOS lock + identity hash), reference builder over the real `v5_model` core, EXPERIMENT_ONLY output treatments |
| `tools/next_core_compute_model.py` (NEW) | **ADD** | analytic accounting; validated to 250,216,960 |
| `tools/validate_next_core_spec.py` (NEW) | **ADD** | spec status/evidence/geometry checks |
| output treatments as default | **REJECT** | EXPERIMENT_ONLY until R1C-world evidence (§41: canonical path intact; difference hash-visible) |
| separate next-core training stack | **DELETE** (never created) | duplication would be architecture theater |

## Implementation risk

LOW. New code ≈ 300 lines of contract/wrapper + tests; zero changes to production modules; canary path reuses the certified production backend. The riskiest element is conceptual: keeping the EXPERIMENT_ONLY treatments from leaking into the canonical path — mitigated by constructor gating (`allow_experimental`), identity-hash divergence, and dedicated tests.

## Checkpoint compatibility (§43)

V5.1 geometry is byte-identical to V5-A → existing V5 checkpoints are **COMPATIBLE** (same parameter inventory; the contract layer adds no parameters). Any future ADOPT-GEOMETRY-CHANGE handoff (World C) would be **INCOMPATIBLE** and requires its own prospective function-preservation test before adoption. No tensor resizing is proposed.

## Build → Measure → Understand → Improve (§36)

BUILD: this contract layer + canary config. MEASURE: canary receipts (parameter/eval/checkpoint contracts). UNDERSTAND: which contracts hold at development scale. IMPROVE: only via the ARCHITECTURE_DECISION_TREE after R1/2/3 outcomes.
