# Engineering Log

LOG_STANDARD: Keep entries dated, scoped, and tied to verification evidence.

## 2026-07-04 - FEAT - `iterate500` - Strengthen KV-cache parity gate to distribution level

| Field | Value |
|-------|-------|
| **Date** | 2026-07-04 |
| **Author** | claude |
| **Component** | inference runtime, evidence gates |
| **Type** | FEAT |
| **Summary** | The KV parity gate compared only output token IDs, which is blind to cache corruption that shifts logits without flipping the greedy argmax (measured: a stale cached token moved last-position logits by 0.117, and a 16-token stale prefix moved step entropies by over 1e-3, while every sampled token still matched). `verify_kv_cache_parity` now also requires per-step entropy and max-probability curve agreement within 1e-3 and reports `token_parity` and `distribution_parity` separately. Added the first tests that actually exercise the gate against a real model: parity verifies clean on CPU fp32 and unlocks cached generation that replays uncached tokens exactly; fault injection with a stale cache prefix is detected by the distribution check while token comparison alone would have passed it. Also established experimentally that a uniform position-offset poison is NOT a fault (RoPE attention is relative), so the fault model is stale cache content, i.e. cross-request leakage. |
| **Files** | generate.py, tests/test_kv_parity_gate.py |
| **Metrics** | 451 to 454 non-GPU tests passing (3 new); gate detects injected stale-cache fault (distribution_parity False) and passes clean (verified True) |
| **Verification** | `pytest tests/ -m "not gpu"` (454 passed, 1 skipped); `ruff check` clean; live fault-injection and clean-control probes |
| **Risk** | low - gate becomes stricter, never looser; cache remains disabled by default until the gate is run against the real checkpoint |
| **Follow-up** | run the parity gate on the real frontier checkpoint (GPU) and, if it verifies, measure the actual latency/memory win with the cache enabled |

---

## 2026-07-04 - FEAT - `iterate500` - Revive dormant subsystems into the live path

| Field | Value |
|-------|-------|
| **Date** | 2026-07-04 |
| **Author** | claude |
| **Component** | generation runtime, cognition, capability graph |
| **Type** | FEAT |
| **Summary** | Revived the permanently-dead phase-3 integrations and made claimed evidence real. (1) Ghost memory: bare-name import always failed; now loads from its canonical package with per-session durable stores under `state/ghost_memory/` (session-hash directories as the isolation boundary), a native blake2b feature-hash embedder (no external model downloads), a measured 0.30 retrieval threshold (hash-embedding cosines: related 0.40-0.69, unrelated 0.09-0.21), and real `add_turn` persistence of accepted full-system outputs; repaired `load_ghost_state`/`save_ghost_state`, which called nonexistent APIs. (2) Identity injector: revived import and fixed `.clean()` to the real `.clean_response()`; deterministic identity-phrase cleanup now runs live. (3) CIV: per-session `ConstitutionalIdentityVector` (persisted in `state/civ_sessions/`) now supplies the MoD router's `router_civ_similarity`, previously a hardcoded 1.0; verified evidence (real coherence quality; truthfulness only when a verifier ran) updates the profile on accepted outputs; diagnostic mode stays neutral. (4) Cognition wired into full-system `/chat`: CRE causal classification of every message plus epistemic-tracker recording of every accepted/rejected outcome into the calibration history. (5) Symbolic falsification pass (DFC): on math/logic messages the 45Q bridge independently derives the answer via sympy and scores the model's output against it; the score lands in the trace (`symbolic_verifier`) and feeds HAL/CIV truthfulness when no operator verifier was supplied. (6) Capability graph: substring-existence flags replaced by real import+health probes (`probe_module_capability`); the full-system integration probe now gates on live subsystem health. (7) Honest telemetry: `memory_saved_mb` reports None when unmeasured instead of a fabricated 0.0; unknown feature-flag names read as disabled; all 24 registry components now have explicit flag defaults. |
| **Files** | generate.py, app.py, inference/full_system_connector.py, engine/feature_flags.py, .gitignore, tests/test_ghost_identity_revival.py, tests/test_civ_runtime.py, tests/test_chat_cognition.py, tests/test_symbolic_verifier.py, tests/test_system_registry.py, tests/test_feature_flags.py, tests/test_frontier_runtime.py |
| **Metrics** | 428 to 451 non-GPU tests passing (23 new); symbolic pass discriminates correct (1.0) from wrong (0.0) derivative answers live; ghost cross-session retrieval isolation proven; anra.py master-system CLI verified working (no fix needed, audit over-claimed) |
| **Verification** | `pytest tests/ -m "not gpu"` (451 passed, 1 skipped); `ruff check` clean on changed files; live end-to-end probes for ghost retrieval, identity cleanup, CIV telemetry, and symbolic verification |
| **Risk** | medium-low - full-system mode gains real subsystem effects (ghost persistence, measured CIV routing signal, symbolic verification); diagnostic mode unchanged; every integration degrades to the previous behavior on error |
| **Follow-up** | KV-cache parity gate to earn cache enablement; ESV cross-restart persistence; ablation evidence for revived subsystems |

---

## 2026-07-03 - FIX - `iterate500` - Honest, reproducible evaluation and health evidence

| Field | Value |
|-------|-------|
| **Date** | 2026-07-03 |
| **Author** | claude |
| **Component** | evaluation, health telemetry |
| **Type** | FIX |
| **Summary** | Made compact/private evaluation generation deterministic (per-item `sha256(seed:item_id)` sampling seed, greedy option, local `torch.Generator` that does not mutate global RNG) so gate pass/fail replays exactly. Replaced the structurally-vacuous `coherence_rate`/`repetition_failure_rate` in `run_compact_eval` (previously always 0.0 because the compact suite has no coherence/repetition categories) with an honest surface fallback over every response, tagged by `coherence_basis`/`repetition_basis`, plus a recorded `decoding` block. Fixed `/sovereignty/status` (`/phase-health`) which imported phase-3 modules by bare name (always ModuleNotFoundError) yet hardcoded top-level `status: ok`: it now loads the real modules from their canonical `anra_paths` directories, runs their real `health_check`, and reports an honest conjunction with `degraded_subsystems`. |
| **Files** | training/v2_runtime.py, training/eval_v2.py, generate.py, scripts/evaluate_draft_recovery.py, scripts/build_brain.py, app.py, tests/test_eval_v2.py, tests/test_frontier_runtime.py, tests/test_phase_health_route.py |
| **Metrics** | 421 to 428 non-GPU tests passing (7 new); deterministic replay proven identical across two runs; all 5 phase-3 health checks now reported live |
| **Verification** | `pytest tests/ -m "not gpu"` (430 passed, 1 skipped); `ruff check` clean on changed files; endpoint exercised end-to-end returning honest per-subsystem status |
| **Risk** | low — no checkpoint, tokenizer, or architecture change; evidence surfaces become truthful rather than asserted |
| **Follow-up** | Continue auditing claimed-but-unmeasured evidence surfaces (capability graph, memory_saved_mb, CIV similarity) flagged in wiring review |

---

## 2026-06-17 - CHANGE - `iterate500` - Restore minimal health artifacts

| Field | Value |
|-------|-------|
| **Date** | 2026-06-17 |
| **Author** | codex |
| **Component** | `iterate500` |
| **Type** | CHANGE |
| **Summary** | Restored minimal required docs and templates so repository health checks remain connected. |
| **Files** | docs, phase4/web, runtime/engineering_templates |
| **Metrics** | full test health |
| **Verification** | pytest |
| **Risk** | low |
| **Follow-up** | keep Markdown minimal on experiment branches |

---

## 2026-06-30 - DOCS - `iterate500` - Rebuild operator and developer documentation

| Field | Value |
|-------|-------|
| **Date** | 2026-06-30 |
| **Author** | codex |
| **Component** | documentation |
| **Type** | DOCS |
| **Summary** | Replaced the stale branch README and documented the one-cell Colab path, prompt-to-response architecture, recovery roadmap, release gates, and developer contracts. |
| **Files** | README.md, docs/ARCHITECTURE.md, docs/WALKTHROUGH.md, docs/IMPROVEMENT.md, docs/DEVELOPER.md, docs/planning/MASTER_GOALS.md |
| **Metrics** | five linked manuals plus refreshed master goals |
| **Verification** | git diff --check; internal Markdown link validation |
| **Risk** | low |
| **Follow-up** | keep commands, schemas, and evidence gates synchronized with implementation |

---
