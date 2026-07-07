# Engineering Log

## 2026-07-07 - FEAT - Stream B - Canonical 32k V4 append migration, pinned corpus manifests, campaign-slice + V4 build CLIs

| Field | Value |
|-------|-------|
| **Date** | 2026-07-07 |
| **Author** | Claude (Fable 5) |
| **Component** | tokenizer V4 growth; training corpus manifest; Stream B data-engine scripts |
| **Type** | FEAT |
| **Summary** | Executed the code-executable half of Stream B. (1) **Generalized the proven append migration to the canonical 32k V4 ceiling (Law-1-clean, non-destructive).** `training/v2_config.py` adds `TOKENIZER_V4_32K_VOCAB_SIZE = 32_768`, `CANONICAL_V4_VOCAB_SIZE`, `V4_VOCAB_SIZES = (16_384, 32_768)`, `is_v4_vocab_size`, and the pinned `V2_FRONTIER_V4_32K_PARAMETER_COUNT = 530_602_567` (= 499,167,047 + (32,768-8,209)*1,280); `frontier_parameter_count` validates both pinned V4 contracts. `tokenizer/validate_tokenizer_v3.py`: `build_append_only_v4` and `audit_token_fertility` take a `target_vocab_size` ceiling (default 16,384 proven fallback; canonical 32,768), reserve byte-token room in the candidate budget, and assert the frozen 8,209-token V3 prefix is unchanged before writing. The 16,384 path is preserved verbatim as the plan's pre-decided fallback route. (2) **Wired 32k through every vocab gate**: `v2_runtime` (`assert_tokenizer_contract`, `build_model_from_config`, `build_v2_model`, `migrate_checkpoint_state` schema branch), `scripts/check_frontier_checkpoint.py`, and `training/ssg.py` now accept `{8209, 16384, 32768}` via `is_v4_vocab_size`/`ALLOWED_CANONICAL_VOCAB_SIZES`. (3) **`training/corpus_manifest.py`** (TODO 1): 7 pinned upstream sources (FineWeb-Edu, The-Stack-v2-dedup, FineMath-4+, Dolma, Smol-SmolTalk, verified DFC, identity replay) with a license allowlist, immutable-revision checks, normalized weights summing to 1.0, and a content-addressed manifest hash; emits `output/v2/data_manifests/upstream_corpus_manifest.json`. (4) **`scripts/build_campaign_slice.py`** (TODO 3): deterministic per-source held-out split (`sha256(line)[0] < 26`), disjointness proof, and the >=50MB gate. (5) **`scripts/build_v4_tokenizer.py`**: runs the audit at a chosen ceiling over a campaign corpus and, if eligible, builds + self-proves the append (frozen prefix, canonical IDs, byte round-trip, param count). Executed now: corpus manifest emitted (valid); V4 CLI on the local ~5MB corpus honestly reports `audit_not_eligible` (816k units / 3.7% reduction < 1M-unit/15% gate) — reconfirming the plan's measured fact that canonical V4 needs the >=50MB campaign corpus; slice builder produced a 3.4MB train slice with a disjoint held-out split, below the 50MB gate. |
| **Files** | training/v2_config.py, training/v2_runtime.py, training/ssg.py, training/corpus_manifest.py, tokenizer/validate_tokenizer_v3.py, scripts/check_frontier_checkpoint.py, scripts/build_campaign_slice.py, scripts/build_v4_tokenizer.py, tests/test_canonical_v4_32k.py, tests/test_corpus_manifest.py, tests/test_build_campaign_slice.py, tests/test_build_v4_tokenizer.py, tests/test_ci_health.py, output/v2/data_manifests/upstream_corpus_manifest.json, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 24 new focused tests (10 V4-32k proofs, 7 corpus-manifest, 4 slice, 3 V4-build); full suite 542 passed, 1 skipped (was 518); ruff clean on all changed files |
| **Verification** | `py -3.14 -m pytest tests/test_canonical_v4_32k.py tests/test_corpus_manifest.py tests/test_build_campaign_slice.py tests/test_build_v4_tokenizer.py -q`; full non-GPU suite; `py -3.14 -m training.corpus_manifest`; `py -3.14 scripts/build_v4_tokenizer.py`; `py -3.14 scripts/build_campaign_slice.py`; `py -3.14 -m ruff check` on changed files |
| **Risk** | low-medium - the 32k ceiling touches broad vocab-gate surfaces, but the change is additive (16k stays valid), the frozen V3 prefix is asserted before every V4 write, both param contracts are pinned, and the full suite is green including the existing 16k proofs |
| **Follow-up** | Data/compute-blocked: acquire >=120GB (`scripts/download_training_data.py`), build the >=50MB campaign slice, build the canonical 32k V4 from it, then run the pre-registered `p150-v4tok` 150M three-seed pilot |

---

## 2026-07-07 - FEAT - Stream A - Forecast ledger, pilot factorial manifests, baseline freeze, forensics driver

| Field | Value |
|-------|-------|
| **Date** | 2026-07-07 |
| **Author** | Claude (Fable 5) |
| **Component** | training forecast ledger + pilot factorial; Stream A freeze/forensics scripts |
| **Type** | FEAT |
| **Summary** | Executed the code-executable half of Stream A. (1) `training/forecast_ledger.py`: append-only JSONL forecast ledger with per-entry content hashes chained through `prev_hash`, `register_forecast`/`record_outcome`/`verify_ledger`, a `calibration_report`, and the Gate-5 `audit_pre_launch` timestamp audit that voids any launch whose forecast was registered after the manifest's `created_at`. (2) `training/pilot_factorial.py`: the 23-cell pre-registered pilot factorial (12 x 150M primary cells incl. Muon/MoE/MTP/QK-Norm/SWA/V4 and their interactions, 5 x 50M ladder anchors, 3 curriculum-order cells, moonshots M1/M3/M5) using the plan's honest prediction ranges; `build_pilot_launch_manifests` registers each cell's forecast first, then emits one signed schema-2 launch manifest per cell carrying exactly three seeds (1301/2602/3903), `checkpoint_source="scratch"` (Law 1), the cell axes, and its `forecast_id`, and every manifest must pass the pre-launch audit before it is returned; V4-tokenizer cells are explicitly `blocked_on` the Stream B canonical V4. Owner CLI: `py -3.14 -m training.pilot_factorial --owner-authorized` (requires `ANRA_MANIFEST_SIGNING_KEY`). (3) `scripts/freeze_baseline_hashes.py`: one frozen artifact (`output/v2/baseline_freeze.json`) covering checkpoint (full-file SHA-256 or an honest `blocked_on_artifact`), tokenizer (file/vocabulary hashes plus the 500 encode/decode probes executed live and cross-checked against the frozen manifest fingerprint), config contract, and corpus manifests. (4) `scripts/run_checkpoint_forensics.py`: the Stage 1.1 driver — checkpoint locator (arg -> `ANRA_CHECKPOINT_PATH` -> canonical), exact tensor accounting via the existing frontier proof, the 500 probes, deterministic greedy/seed-0/cache-off generation through the 200-prompt recovery gate, and the Part 0 undertraining decision rule; distinct exit codes for failed (2) vs blocked-on-artifact (3). Executed now: baseline freeze written and frozen (probe fingerprint matches `db1075ad...`), forensics written reporting blocked on the real checkpoint. |
| **Files** | training/forecast_ledger.py, training/pilot_factorial.py, scripts/freeze_baseline_hashes.py, scripts/run_checkpoint_forensics.py, tests/test_forecast_ledger.py, tests/test_pilot_factorial.py, tests/test_stream_a_forensics.py, tests/test_ci_health.py (script allowlist), output/v2/baseline_freeze.json, output/v2/stream_a_forensics.json, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 18 new focused tests passing; full suite 518 passed, 1 skipped; ruff clean on all changed files |
| **Verification** | `py -3.14 -m pytest tests/test_forecast_ledger.py tests/test_pilot_factorial.py tests/test_stream_a_forensics.py -q`; `py -3.14 scripts/freeze_baseline_hashes.py --allow-missing-checkpoint` (exit 0, frozen); `py -3.14 scripts/run_checkpoint_forensics.py` (exit 3, blocked as expected); full non-GPU suite; `py -3.14 -m ruff check` on changed files |
| **Risk** | low - all new code is additive; the signed manifest set is not yet emitted because `ANRA_MANIFEST_SIGNING_KEY` is an owner secret; tensor accounting and the recovery gate remain blocked on the real 500M checkpoint artifact (never in git) |
| **Follow-up** | Owner: restore the checkpoint (or set `ANRA_CHECKPOINT_PATH`), re-run `scripts/freeze_baseline_hashes.py` then `scripts/run_checkpoint_forensics.py --run-generation`; set the signing key and emit the pilot manifests; record pilot outcomes with `record_outcome` for the campaign calibration report |

---

## 2026-07-07 - SECURITY/FEAT - Stream C - Systems and serving foundation completed

| Field | Value |
|-------|-------|
| **Date** | 2026-07-07 |
| **Author** | Codex |
| **Component** | runtime Experience Ledger; CI; sibling gpu-cluster control plane |
| **Type** | SECURITY + FEAT |
| **Summary** | Completed the remaining Stream C TODOs. Experience Ledger shards now rotate by size/day, seal into tamper-evident manifests, verify sealed shard hashes/sizes/event envelopes, and prune retained shards only after successful manifest verification. Added a reusable ledger benchmark/stress script and CI step that enforces p50/p99 write overhead and crash/flush JSONL validity. In the sibling `C:\Users\ankit\Downloads\gpu cluster` control-plane repo, implemented the P0 hardening pass: disabled unauthenticated legacy heartbeat, operator-gated sensitive read endpoints, added per-IP rate limiting and nonce pruning, stopped sending worker secrets on signed v2 requests, stored worker secrets encrypted at rest, prevented expired lease resurrection, capped poison-job retries, switched checkpoint loads to `weights_only=True`, constrained artifact verification to managed Drive basenames, escaped Drive filename queries, kept DB backup generations, and killed local worker subprocesses on repeated heartbeat/lease failure. Implemented P1 storage tier accounting with a `StorageSlot` quota ledger, job-level fit checks, verified-artifact slot recording, and an operator quota report. |
| **Files** | runtime/experience_ledger.py, scripts/benchmark_experience_ledger.py, tests/test_experience_ledger.py, tests/test_experience_ledger_benchmark.py, .github/workflows/ci.yml, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md; sibling repo: backend/security.py, backend/database.py, backend/main.py, backend/campaign.py, backend/campaign_routes.py, backend/artifacts.py, backend/drive_sync.py, backend/recovery.py, backend/storage.py, worker/campaign_worker.py, worker/drive_worker.py, tests/conftest.py, tests/test_reliable_campaign.py |
| **Metrics** | An-Ra ledger focused tests: 13 passed; An-Ra focused ruff clean; cluster focused tests: 12 passed; cluster focused ruff clean |
| **Verification** | `py -3.14 -m pytest tests/test_experience_ledger.py tests/test_experience_ledger_benchmark.py -q`; `py -3.14 -m ruff check runtime/experience_ledger.py scripts/benchmark_experience_ledger.py tests/test_experience_ledger.py tests/test_experience_ledger_benchmark.py`; sibling repo: `py -3.14 -m pytest tests/test_reliable_campaign.py tests/test_auth_and_cluster.py tests/test_worker_artifacts.py -q`; sibling repo ruff on touched backend/worker/test files |
| **Risk** | medium - cluster changes are security-sensitive and were focused-test verified, but full cluster chaos/soak gates remain P1/P3 campaign work; full An-Ra regression was not rerun in this slice |
| **Follow-up** | Stream E ledger-derived UI trust projections; then cluster telemetry/chaos/24h soak gates under P1 |

---

## 2026-07-07 - FEAT - memory - Ledger trace binding for memory lifecycle

| Field | Value |
|-------|-------|
| **Date** | 2026-07-07 |
| **Author** | Codex |
| **Component** | memory router, app memory bridge, Experience Ledger |
| **Type** | FEAT |
| **Summary** | Bound memory writes, recalls, edits, and forgetting to Experience Ledger trace IDs. MemoryRouter now records PII-minimized lifecycle events with content hashes, record IDs, tier metadata, and memory-policy gate evidence; hybrid recall forwards trace IDs into the shared retrieval query contract. Added durable delete support to the FAISS episodic store and canonical-ID delete support to BM25 so edits and forgetting remove both dense and keyword copies. The API chat and memory bridges now pass the request ID through search/store calls, while legacy memory adapters accept the same signature and safely ignore it. |
| **Files** | memory/memory_router.py, memory/faiss_store.py, anra/memory/bm25.py, app.py, tests/test_memory_ledger_trace.py, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 1 new lifecycle regression test; focused memory/retrieval verification: 2 passed; focused ruff clean on changed memory/app/test files |
| **Verification** | `uv --cache-dir .uv-cache run --no-project --with ruff ruff check memory/memory_router.py memory/faiss_store.py anra/memory/bm25.py app.py tests/test_memory_ledger_trace.py tests/test_retrieval_substrate.py`; `uv --cache-dir .uv-cache run --no-project --with pytest --with numpy --with pyyaml --with pydantic pytest tests/test_memory_ledger_trace.py tests/test_retrieval_substrate.py::test_memory_router_hybrid_uses_shared_substrate -q` |
| **Risk** | low-medium - memory lifecycle events intentionally expose hashes and record IDs, not raw memory text; graph-tier forgetting remains unsupported because graph rows do not yet persist record IDs |
| **Follow-up** | Define ledger-derived UI trust projections for verification, memory, and gate visibility |

---

## 2026-07-06 - FEAT - retrieval - S3 shared retrieval substrate implemented

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Codex |
| **Component** | retrieval, memory router, inference bridge, citation verifier, data pipeline |
| **Type** | FEAT |
| **Summary** | Implemented the S3 typed retrieval substrate with canonical query/hit/provenance contracts, backend adapters, and deterministic weighted reciprocal-rank fusion. Semantic and BM25 hits now merge by canonical ID while retaining backend rank, raw score, and weight. The main MemoryRouter delegates hybrid retrieval to S3; the live chat bridge requests hybrid context; agent skills expose a compatible adapter; citation grounding can retrieve evidence through the same protocol; and corpus curation uses a retrieval-backed dedup index. Exact dedup remains the default behavior, while near-duplicate thresholds are explicit and opt-in. Retrieval outcomes emit Experience Ledger events without storing raw inputs. Fixed direct-import initialization of the legacy memory router by guarding the package registry's eager wrapper import. |
| **Files** | retrieval/__init__.py, retrieval/protocols.py, retrieval/adapters.py, retrieval/hybrid.py, retrieval/corpus.py, memory/memory_router.py, app.py, verification/builtins.py, training/data_pipeline_v3.py, anra/__init__.py, tests/test_retrieval_substrate.py, pyproject.toml |
| **Metrics** | 6 S3 contract/integration tests; 27 focused retrieval/data/memory/symbolic tests; full non-GPU suite: 494 passed, 1 skipped |
| **Verification** | RRF dedup/provenance, deterministic ties, hybrid MemoryRouter, skill adapter, citation grounding, exact dedup, opt-in near dedup, and direct memory-router import tested; full non-GPU regression suite green; ruff clean on the slice |
| **Risk** | low-medium - near-duplicate rejection is deliberately disabled by default until corpus ablation evidence selects a threshold |
| **Follow-up** | Expose ledger-derived transparency projections for verification, memory, and gate visibility |

---

## 2026-07-06 - SECURITY - execution - Code verifier sandbox hardened cross-platform

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Codex |
| **Component** | execution/sandbox, registered code verifier |
| **Type** | SECURITY |
| **Summary** | Replaced the verifier's private subprocess execution with the shared hardened CodeSandbox. Runs use the current isolated interpreter (`-I -B`), a minimal secret-free environment with no inherited PYTHONPATH, a dedicated workspace, bounded pipe draining, process-tree termination, and explicit wall-time, CPU, memory, file-growth, output, and open-file ceilings. POSIX uses rlimits and process groups; Windows uses kernel Job Objects plus psutil monitoring and peak-memory checks. A Python audit hook denies network, child processes, native-library loading, workspace escapes, and outside mutations. Policy attempts write an out-of-band marker, so caught PermissionError exceptions cannot turn a violation into success. |
| **Files** | execution/sandbox.py, training/verifier.py, tests/test_code_sandbox_security.py, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 11 adversarial sandbox tests; default verifier policy: 5s wall, 3s CPU, 256 MiB memory, 4 MiB file growth, 4 KiB retained output, 32 open files; full non-GPU suite: 488 passed, 1 skipped |
| **Verification** | normal workspace execution passes; secret/PYTHONPATH isolation, outside-write denial, caught-violation denial, child-process denial, network denial, wall timeout, CPU ceiling, memory ceiling, file ceiling, output flood, and registered verifier integration all tested on Windows; ruff clean; full non-GPU regression suite green |
| **Risk** | medium - Python audit hooks are defense-in-depth rather than an OS container; high-risk production execution should additionally run inside a dedicated low-privilege container/VM |
| **Follow-up** | S3 shared retrieval protocol and adapters; retain sandbox adversarial suite as a release gate |

---

## 2026-07-06 - FEAT - verification - All five verifier consumers routed through S2

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Codex |
| **Component** | verification, inference DFC, data pipeline, agents, GEPA |
| **Type** | FEAT |
| **Summary** | Completed the first S2 routing pass. Split legacy verifier selection into independently registered builtins, including code, math, exact, open-ended, domain, symbolic-output, DFC-format, and GEPA-candidate verifiers. Centralized ledger publication in the registry so compatibility and direct calls emit exactly one verifier event; verifier exceptions and malformed results emit failed evidence before being re-raised. Routed inference symbolic DFC, synthetic DFC-format validation, CriticAgent, and GEPA directly; RLVR/STaR training reward remains API-compatible while dispatching through the same registry. |
| **Files** | verification/builtins.py, verification/registry.py, verification/__init__.py, training/verifier.py, generate.py, training/data_pipeline_v3.py, agents/specialists.py, training/gepa.py, tests/test_verifier_registry.py, tests/test_symbolic_verifier.py, tests/test_gepa.py, tests/test_v3_training_systems.py |
| **Metrics** | 17 canonical builtin verifiers plus aliases; 10 registry tests; 33 focused integration tests; full non-GPU suite: 476 passed, 1 skipped |
| **Verification** | DFC correct/wrong parity retained; direct math and CriticAgent dispatch tested; DFC training format and GEPA evidence contracts tested; verifier-exception ledger capture tested; ruff clean; full non-GPU regression suite green |
| **Risk** | low-medium - code verifier still uses the legacy subprocess boundary and is the next hardening target |
| **Follow-up** | Bind registered code verification to execution/sandbox.py resource and filesystem policy, then add adversarial conformance cases |

---

## 2026-07-06 - FEAT - verification - Shared verifier registry established

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Codex |
| **Component** | verification, training/verifier |
| **Type** | FEAT |
| **Summary** | Started MASTER_UPGRADE S2 with a process-wide, thread-safe verifier registry. Added import-time decorator registration, aliases, normalized discovery, duplicate-name rejection, unknown-verifier failure, and structural result conformance checks for score/tier/reason. Registered every existing verifier name and routed the existing VerifierHierarchy facade through the common request protocol, preserving callers while exposing one discovery surface. Unknown legacy task names deliberately route to the registered open-ended verifier and retain the requested task in the request payload. |
| **Files** | verification/__init__.py, verification/registry.py, tests/test_verifier_registry.py, training/verifier.py, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 15 registered builtin verifier names; 7 registry tests; full non-GPU suite: 472 passed, 1 skipped |
| **Verification** | alias dispatch, discovery, duplicate and unknown failure, invalid score/tier/reason rejection, and legacy hierarchy compatibility all tested; ruff clean; full non-GPU regression suite green |
| **Risk** | low - compatibility facade remains; handler internals have not yet been split into independent modules |
| **Follow-up** | Route DFC/synthetic/agent/GEPA callers directly and split each legacy verifier into its own registered handler |

---

## 2026-07-06 - FEAT - runtime - Experience Ledger substrate implemented and wired live

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Codex |
| **Component** | runtime, app, verifier, sovereignty |
| **Type** | FEAT |
| **Summary** | Implemented MASTER_UPGRADE S1 as an append-only schema-v1 JSONL Experience Ledger. Events carry stable input hashes, tamper-evident event hashes, output, verifier verdicts, gate records, token/latency data, source, and metadata. Added validated replay and an explicit train/serve firewall: only verifier-passing, gate-allowed, non-PII events promote into deterministic train/validation shards with atomic SHA-256 manifests. Wired main chat/generate traces, explicit operator tools/agents, the verifier hierarchy, owner-auth request decisions, and sovereignty audit decisions. Capture failures are fail-open for serving and fail-closed during compaction validation. Added a live checklist translating MASTER_UPGRADE into executable TODOs. |
| **Files** | runtime/experience_ledger.py, tests/test_experience_ledger.py, app.py, training/verifier.py, phase3/sovereignty_45r/logger.py, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 7 focused ledger tests; 1,000-write benchmark p50 0.3481 ms and p99 0.9040 ms with 0 failures; full non-GPU suite: 472 passed, 1 skipped |
| **Verification** | fault-injected disk failure stays fail-open; mutation is detected on replay; live trace and verifier chokepoints tested; ruff clean; full non-GPU regression suite green |
| **Risk** | low-medium - runtime writes add local I/O; fail-open semantics protect serving, while promotion always revalidates hashes and gates |
| **Follow-up** | S2 verifier registry unification; then sealed shard rotation/retention and sustained p50/p99 write-overhead CI |

---

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

## 2026-07-06 - MEASUREMENT - tokenizer - Week 1 fertility gates measured; V4 draft built and validated held-out

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Claude (Fable 5) |
| **Component** | tokenizer |
| **Type** | MEASUREMENT |
| **Summary** | Executed MASTER_UPGRADE v2 Week 1 tokenizer slice. Measured per-source V3 fertility on held-out text: English prose 2.518 tok/word (gate 1.35 — FAIL, worse than the plan's 1.5–2.0 prediction; the V3 tax is confirmed), code 0.407 tok/char (gate 0.45 — pass), math/DFC 0.450 tok/char. Append-only audit on 1,000,000 units: projected reduction 21.2%, eligible_for_schema_v4=true. Built tokenizer_v4_draft.json (16,384, backend native_append_v4) from local sources and measured realized held-out fertility: math/DFC −30.5% tok/char, code −2.4%, English −0.2%. Byte fallback fixes a real defect: V3 fails exact round-trip on English held-out (unk chars); V4 draft round-trips exactly. The recovery script's earlier roundtrip_ok=false was diagnosed as literal <bos>/<eos> markers embedded in frontier_dfc.jsonl text being (correctly) stripped by decode — not a tokenizer defect. |
| **Files** | scripts/measure_tokenizer_fertility.py (new), tests/test_measure_tokenizer_fertility.py (new), tokenizer/tokenizer_v4_draft.json (+meta, draft artifact), output/v2/fertility_week1.json, output/v2/fertility_week1_v4draft.json, output/v2/tokenizer_recovery.json |
| **Metrics** | V3 English 2.518 tok/word vs V4 target 1.35; audit 1M units, 21.2% projected reduction; realized held-out: DFC −30.5%, code −2.4%, English −0.2%; V4 draft unk_rate ~0, byte-safe, exact English round-trip |
| **Verification** | 4 new tests pass; ruff clean on changed files; full non-GPU suite green (458 passed, 1 skipped); reports written atomically with source SHA-256 hashes |
| **Risk** | low — V3 untouched (IDs 0–8208 immutable, Law 1); V4 artifact explicitly named _draft; no serving path changed |
| **Follow-up** | Canonical V4 requires candidates derived from the >=50 MB campaign corpus (local 5 MB corpus cannot improve English prose because V3 was trained on it — measured, not assumed). Consider raising the V4 vocab target to 32k per MASTER_UPGRADE v2 Layer 3 before the campaign run. |

---

## 2026-07-06 - DOCS - planning - MASTER_UPGRADE finalized at v3 with review amendments

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Claude (Fable 5) |
| **Component** | docs/planning |
| **Type** | DOCS |
| **Summary** | MASTER_UPGRADE.md rewritten to v3 (Unified Intelligence Program: substrate spine, continual learning, multimodal/SSM/latent-reasoning/self-dev moonshot registry with pilot paths and kill criteria, distributed infra + product layers, five parallel workstreams with critical path). Then amended per owner review: (1) HAL elevated to fourth substrate S4 (calibration) with monthly Brier gate — was an oversight, now unifies four private confidence copies; (2) CIV/ESV identity-state continuity added as a fifth memory tier, distinct from CIVGuard drift detection, with cross-restart similarity >=0.95 CI gate and long-horizon stance probes; (3) self-directed consolidation pilot added to 2.4 (GEPA-scored, fixed recipe is permanent fallback); (4) adversarial gate audit added to Layer 11 — fresh-context agent must fail to refute any gate-pass before it is recorded; (5) compute split into confirmed (free TPU/T4 tier) vs assumed (spot rental) with securing-the-rental as a week-1 deliverable; (6) physics table rows tagged MEASURED/ASSUMED/PROJECTED/DECISION. |
| **Files** | docs/planning/MASTER_UPGRADE.md, PROGRESS.md |
| **Metrics** | plan frozen for implementation; all cost figures now provenance-tagged |
| **Verification** | passage-level coherence checks (grep) after multi-edit; no code paths touched |
| **Risk** | none — documentation only |
| **Follow-up** | Implementation begins: Stream C Experience Ledger and Stream D verifier registry unification are the highest-value in-environment slices |

---

## 2026-07-06 - DOCS - planning - GPU-cluster control plane adopted into the master plan after two-repo code inspection

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Claude (Fable 5) |
| **Component** | docs/planning, cross-repo (gpu cluster @ main) |
| **Type** | DOCS |
| **Summary** | Code-level inspection of the gpu-cluster repository (backend, worker, scripts, tests, job manifests) and its An-Ra contract surfaces. Verified working: fenced leases with monotonic attempt tokens, idempotent commits, AcceptedWindow exactly-once data accounting, single canonical writer, fail-closed storage preflight, heartbeat lease renewal, atomic publication with commit-time hash verification, Fernet-encrypted OAuth, scrypt-hashed keys, sparse path double-gated off, allowlisted worker scripts with source-commit pinning. Named defects (13, ordered): unauthenticated v1 heartbeat, worker secret on the wire, orphaned training subprocess on heartbeat failure, zeroed progress telemetry, renew-lease expiry race, weights_only=False deserialization, no poison-job cap, unauthenticated read endpoints, no rate limiting, local-path trust in artifact verification, Drive query interpolation, unbounded nonce table, unsatisfiable storage preflight formula on 15GB Drive. Documented-only: hot-spare promotion, rollback execution. Wrote docs/planning/CLUSTER_CONTROL_PLANE.md (12-section proposal: findings, three-tier storage strategy resolving 30GB corpus vs 15GB Drive via on-worker derivation from hash-pinned upstream sources, campaign lifecycle, failure/security model, G-C1..G-C7 acceptance gates, P0-P5 phases) and added Layer 12-B to MASTER_UPGRADE v3 (+ calendar Stream C, + risk rows, + confirmed-compute tier). |
| **Files** | docs/planning/CLUSTER_CONTROL_PLANE.md (new), docs/planning/MASTER_UPGRADE.md, PROGRESS.md |
| **Metrics** | cluster: 13 fix items, 7 acceptance gates, 6 implementation phases; boundary preserved (An-Ra never imports cluster code) |
| **Verification** | findings verified from implementation with file:line citations; An-Ra contract strings confirmed present in scripts/build_brain.py; secrets confirmed untracked in both repos |
| **Risk** | none — documentation only; no code changed in either repo |
| **Follow-up** | Cluster P0 (security + worker hardening) is the first implementation slice; P1 storage tiers require the ablation-gated slim-optimizer option in training/checkpoint.py |

---

## 2026-07-06 - REVIEW - runtime/verification - Audit of Codex-implemented ledger + verifier registry; firewall bug fixed

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Claude (Fable 5) |
| **Component** | runtime/experience_ledger, verification/, execution/sandbox, training/verifier, training/gepa |
| **Type** | REVIEW + FIX |
| **Summary** | Code-level audit of the 14 checked implementation TODOs Codex completed (Experience Ledger S1: 9 items; verifier registry S2: 5 items). Found the implementation genuinely sound: tamper-evident event hashing, validated replay, fail-open capture, PII-safe hash-only inputs, consistent trace_id across chat/generate/gate chokepoints, auto-registering registry with conformance + duplicate rejection, VerifierHierarchy routed through the registry, and _safe_exec upgraded to an audit-hook CodeSandbox (live smoke test confirmed fs/network/subprocess escapes are blocked). One real defect fixed: compact_for_training() split the train/validation firewall on event_hash (which embeds per-event timestamp+uuid), so identical prompts could leak across both splits — re-keyed to inputs_hash and added a 40-event regression proving one input never crosses the firewall. Fixed one ruff import-order error in execution/sandbox.py. |
| **Files** | runtime/experience_ledger.py (firewall fix), tests/test_experience_ledger.py (+regression), execution/sandbox.py (import order), docs/planning/IMPLEMENTATION_TODOS.md (review ledger) |
| **Metrics** | Full non-GPU suite 477 passed, 1 skipped (was 458 pre-Codex); ruff clean on all changed files; sandbox policy smoke test: benign=0, fs/net/subprocess escapes all blocked |
| **Verification** | py -3.14 -m pytest tests/ -m "not gpu" (477 passed); ruff check clean; manual sandbox escape probes |
| **Risk** | low — firewall fix is strictly more conservative (prevents leakage); no serving-path behavior change; ledger capture remains fail-open |
| **Follow-up** | Stream D unchecked items: wire SandboxPolicy limits as the verifier default + resource-limit test matrix; shared retrieval protocol; memory-event trace binding |

---
