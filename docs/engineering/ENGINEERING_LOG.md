# Engineering Log

## 2026-07-20 - RELIABILITY - Exact same-commit interruption and recovery

| Field | Value |
|-------|-------|
| **Date** | 2026-07-20 |
| **Author** | Codex |
| **Component** | V4 trainer termination handling, checkpoint provenance, deterministic sampler |
| **Type** | RELIABILITY + TRAINING |
| **Summary** | Replaced an unreliable Windows console-break drill with a rehearsal-only deterministic fault injector that enters the trainer's production deferred-SIGTERM path. Signal handling only records intent; checkpoint serialization occurs synchronously after gradients and pending data from an incomplete accumulation are discarded. The injector is unavailable with normal post-session evaluation, so it cannot silently enter an ordinary campaign. |
| **Evidence** | Signed commit `1138be50...`, RTX 4050, 181,132,071 parameters, context 2,048, batch 1 x accumulation 32. Termination requested at session microstep 40 after step 1; checkpoint records `safe_optimizer_boundary=true`, `discarded_micro_steps=8`, 65,536 tokens, cursor 32, 32 unique/0 repeat windows, and SHA-256 `4953682a...`. A separately signed resume under the same commit reached step 3, 196,608 tokens, cursor 96, 96 unique/0 repeat windows, loss 10.6677, and SHA-256 `837f7721...`. The source SHA stayed unchanged. Focused tests: 24 passed; Ruff and compilation passed. GPU idle afterward. |
| **Risk** | This proves exact checkpoint/restart mechanics, not model quality. The bounded rehearsal intentionally skipped held-out and behavioral evaluation. An operating-system hard kill can never execute an in-process handler; durability against that case depends on the last already-persisted optimizer-boundary checkpoint. |
| **Follow-up** | Start the signed dense V4 seed-1301 baseline, then measure held-out loss, generation coherence, reasoning, retrieval, verification, throughput, and memory before comparing any optional subsystem. |

---

## 2026-07-20 - TRAIN/FIX - Full-context rehearsal and raw-source replay firewall

| Field | Value |
|-------|-------|
| **Date** | 2026-07-20 |
| **Author** | Codex |
| **Component** | V4 foundation trainer, deterministic sampler, campaign source policy |
| **Type** | TRAINING + ARCHITECTURE + RELIABILITY |
| **Summary** | Ran the first bounded context-2048 optimizer update on the real 181M V4 path. Window-consumption evidence exposed that the old fixed 2% identity share sampled three times from only two unique windows in the first update. Separated seven-source corpus admission from Phase-A sampling: raw foundation training now samples only the normalized four-source broad corpus, while instruction, verified DFC, and identity remain provenance-bound for structured continuation. Added a fail-closed replay-budget check that prevents any raw source from being scheduled for more than four unique-data epochs. Added bounded rehearsal completion without expensive behavioral evaluation, monotonic horizon extension for the non-scheduled sampler, signed train-manifest data-profile binding, and immutable explicit resume sources. |
| **Evidence** | Corrected RTX 4050 part 1: 181,132,071 parameters; batch 1 x accumulation 32; context 2,048; step 1; 65,536 tokens; cursor 32; 32 unique/0 repeat windows; loss 10.6880; 1,142 tok/s. Signed continuation restored optimizer/RNG/sampler and reached step 2; 131,072 tokens; cursor 64; 64 unique/0 repeat windows; loss 10.6697; 1,366 tok/s. Original part-1 SHA-256 `c805547f...` was restored from its mirror and matches the signed resume manifest; separate part-2 SHA-256 is `8c3e07f7...`. Corrected 1B-token Phase-A policy has zero replay-budget violations. Focused tests and Ruff pass. |
| **Risk** | Two full-context updates prove execution and completed-session recovery, not language capability or a same-commit abrupt-kill path. The resume-enablement commit changed only sampler-horizon validation and post-session behavior, but part 1 and part 2 therefore name different source commits. No held-out behavioral evaluation was run. |
| **Follow-up** | Run one same-commit mid-session signal interruption/restart before the baseline campaign. Retain separate signed artifacts and verify source hashes after every worker publication. |

---

## 2026-07-16 - ARCH/TRAIN - Verified and extensible intelligence foundation

| Field | Value |
|-------|-------|
| **Date** | 2026-07-16 |
| **Author** | Codex |
| **Component** | verified process objective, LoRA/DoRA capability contract, continual learning, serving activation, reasoning budget, GPU canary |
| **Type** | ARCHITECTURE + TRAINING + RELIABILITY + VERIFICATION |
| **Summary** | Added a bounded process-supervision objective that weights only complete tagged spans from verifier-produced DFC training shards and binds its identity/multiplier into exact resume. Consolidated parameter-efficient learning behind one immutable-base LoRA/DoRA contract with strict targets, adapter-only state, SHA-256 checkpoint/tokenizer lineage, atomic artifact writes, reversible detach, and activation rollback. Continual candidates now use that artifact contract. Added an inspectable, non-acting adaptive reasoning budget with explicit retrieval/verifier blockers. |
| **Evidence** | Focused contract tests pass, including DoRA normalization/gradients, device/dtype inheritance, exact detach/reload, wrong-lineage rejection, failed-activation rollback, span eligibility/bounds, reasoning caps, and resume recipe checks. The local RTX 4050 executed the full 181,132,071-parameter V4 base with DoRA on 54 Q/V/down projections: 919,296 trainable adapter parameters, one BF16 AdamW step, 5.80 s, 1,300 MiB peak, finite 10.655715 total loss and 4.011755 pre-clip gradient norm, no checkpoint written. The initial canary found the CPU-allocation defect for adapters attached after GPU placement; the primitive was corrected and rerun successfully. |
| **Risk** | A one-step random-token canary proves only execution, gradients, and memory. It does not prove that process weighting, DoRA, adaptive compute, MTP, or any subsystem improves language, reasoning, autonomy, or AGI capability. The dense seed-1301 checkpoint and held-out comparisons do not yet exist. |
| **Follow-up** | Train the unchanged dense V4 baseline first. Then evaluate one hypothesis at a time: matched MTP, process-weight ablation, or a typed capability adapter. Promote only reproducible held-out gains; keep MoE/SSM/latent/HAL/moonshots isolated. |

---

## 2026-07-20 - FIX - V4 data pipeline - Recover append journal and publish training-ready shards

| Field | Value |
|-------|-------|
| **Date** | 2026-07-20 |
| **Author** | codex |
| **Component** | native corpus audit, V4 tokenizer, immutable token shards, campaign gates |
| **Type** | FIX + EXECUTION |
| **Summary** | Reconciled 411,000 online-validated append records, repaired the exact Windows CRLF accounting defect, replaced quadratic V4 longest-piece lookup with a parity-preserving trie, removed redundant BM25 reconstruction from post-audit token publication, repaired verified DFC/identity band-pass metadata, and completed fail-closed seven-source V4 shard publication. |
| **Files** | scripts/download_training_data.py, scripts/execute_stream_b.py, scripts/campaign_status.py, tokenizer/subword_tokenizer.py, training/data_pipeline_v3.py, focused tests |
| **Metrics** | corpus 31,160,180,241 bytes / 5,685,479 records; train 11,423,800,574 tokens; validation 115,134,145; test 117,260,614; 1,148/16/16 shards; 5,577,097 context-2048 train windows |
| **Verification** | append recovery zero failures; prefix-trie parity on 250 real samples; 32 MiB throughput probe 2.02M tok/s; all shard SHA-256 checks and first-window loads passed in 30.4s; Stream B complete; 55 focused tests passed; Ruff and diff checks clean |
| **Risk** | low-medium - data manifests changed before any training launch; augmentation is explicitly provenance-bound to the failed base-manifest hash and allowed only for missing verified_dfc/identity classes |
| **Follow-up** | restore CUDA visibility or deploy the bounded L40 rehearsal; prove context-2048 memory and exact kill/restart before signing the clean Phase-A launch |

---

## 2026-07-15 - TRAIN/FIX - Deterministic V4 foundation and exact resume

| Field | Value |
|-------|-------|
| **Date** | 2026-07-15 |
| **Author** | Codex |
| **Component** | canonical seed, sampler, optimizer, scheduler, AMP, checkpoint resume, preflight |
| **Type** | ARCHITECTURE + TRAINING + RELIABILITY + VERIFICATION |
| **Summary** | Replaced the seed label with an executable reproducibility contract. Unified both active dispatchers on seed 1301 and the `t4_v4_session` runtime. Schema 9 now captures/restores Python, NumPy, Torch, CUDA, DataLoader, optimizer, scheduler, scaler, recipe, and raw-sampler cursor state; exact training resume fails closed on any missing or changed component. Replaced session-restarting RNG sampling with a counter-based SHA-256 sampler whose suffix is directly addressable. Made AdamW plus 2% warmup/cosine decay the sole foundation algorithm, removed the unproven dynamic-regret LR overlay, applied weight decay only to matrix parameters, and prevented skipped AMP steps from advancing any progress state. |
| **Evidence** | Complete RNG snapshot/replay and schema-9 optimizer/scheduler/scaler/model resume pass on a trained tiny transformer; raw sampler suffix at position 173 exactly matches the uninterrupted run. Local RTX 4050 dense V4 canary: all 181,132,071 parameters, BF16 AdamW, sequence 64, 5.48 s, 3,499.82 MiB peak, CE/total loss 10.546875/10.558056, finite 64.08894 pre-clip gradient norm. Two seed-1301 builds matched fingerprint (`5d6854db...`), logits, and 10.566532 initial loss; seed 1302 differed (`8cbcff...`, 10.479838), proving that random initial loss cannot select a quality seed. Full MTP candidate canary: 182,739,495 parameters, sequence 64, 4.08 s, 3,527.00 MiB peak, finite CE 10.611328 + weighted MTP 2.084721 and gradients. Focused CPU suites, Ruff, and compilation pass. |
| **Risk** | Deterministic replay is scoped to the same software/hardware stack; bitwise equality across different GPU architectures is not claimed. Sequence-64 canaries prove execution and local memory feasibility only. They do not prove full 2,048-token fit, throughput, language quality, MTP benefit, or AGI capability. No checkpoint or training campaign was created. |
| **Follow-up** | Run the signed seed-1301 dense V4 campaign on a 16 GiB T4 at the real context and prove one kill/restart continuation. If its metrics are healthy, run one matched seed-1301 MTP comparison; replicate only a meaningful signal before promotion. |

---

## 2026-07-15 - RECLAMATION - Dependency-led legacy removal

| Field | Value |
|-------|-------|
| **Date** | 2026-07-15 |
| **Author** | Codex |
| **Component** | runtime ownership, tokenizer lineage, UI, memory, orchestration, tests, plans |
| **Type** | RECLAMATION + FIX + VERIFICATION |
| **Summary** | Completed the final dependency-led cleanup. Deleted duplicate Phase-2 memory, GhostMemory, MasterSystem, placeholder self-improvement, the unused React/Flask UI stack, retired V3/16k tokenizer construction, the 69-run factorial, obsolete tests, recovery notebooks/scripts, and the dead archive tree. Runtime replay can no longer ingest chat memory implicitly. Stream B validates the fixed 32,768-token V4 identity and publishes V4-only shards. The one live archived dependency, TurboQuant, moved into `inference/`; the rejected checkpoint builder is explicitly forensic-only. |
| **Evidence** | Dependency scans find no Python imports of removed packages or V3 builders. Modified core compiles. Focused contract set: 65 tests passed after correcting two stale expectations; targeted reruns passed. Ruff passed on the affected canonical modules, and `git diff --check` reported no whitespace errors. The regenerated system manifest contains no removed component paths. |
| **Risk** | Historical dated evidence still mentions retired mechanisms, but it is labeled historical and is not an executable path. No training, model-quality claim, or GPU workload was performed. |
| **Follow-up** | Publish canonical V4 shards, then run the owner-started seed-1301 T4 training notebook. Treat optional architecture work as isolated pilots only. |

---

## 2026-07-15 - ARCH/FIX - V4 pre-training architecture gate

| Field | Value |
|-------|-------|
| **Date** | 2026-07-15 |
| **Author** | Codex |
| **Component** | V4 decoder, attention, configuration, checkpoint contract, T4 recipe |
| **Type** | ARCHITECTURE + FIX + VERIFICATION |
| **Summary** | Audited the sole 181M V4 path before training. Corrected a fundamental RoPE mismatch: adjacent-pair rotation had been fed a concatenated rather than pair-repeated phase layout. Added schema-8 architecture identity, bounded composed attention temperature, neutral DSTP initialization, executable `d_ff` and `rope_base`, validated router/head/feed-forward geometry, V4 registry/config identity, and a T4-safe microbatch 1 × accumulation 32 recipe. Kept MTP, MoE, HAL, cognitive extensions, and moonshots outside the canonical path. |
| **Evidence** | Actual construction = 181,132,071 parameters; finite full-model `[1,4,32768]` logits; expected 14 local + 4 full layers; 104 focused architecture/runtime/configuration tests pass; Ruff clean on affected modules. |
| **Risk** | No inspection proves capability before training. Peak T4 memory and throughput remain measurements for the first seed-1301 session. Schema-8 V4 intentionally rejects checkpoints missing the corrected rotary contract. |
| **Follow-up** | Treat this architecture as frozen for seed 1301. Change it only through a new named scratch architecture after measured evidence. |

---

## 2026-07-15 - ARCH/FIX - V4 promoted to the sole active model lineage

| Field | Value |
|-------|-------|
| **Date** | 2026-07-15 |
| **Author** | Codex |
| **Component** | model profile, tokenizer, training, serving, Colab, readiness |
| **Type** | ARCHITECTURE + RECLAMATION |
| **Summary** | Promoted `anra-v4-180m` (181,132,071 parameters) and the 32,768-token V4 tokenizer to the only supported training and serving contract. Removed legacy model-size selection and tokenizer fallback, deleted the V3/draft artifacts, recovery scripts, and duplicate notebooks, made serving reject non-V4 checkpoints, and reduced unified training to one checkpoint lineage instead of sequential V2 brain/identity/ouroboros models. The T4 notebook uses the single canonical seed 1301 and never restores a shared legacy master. |
| **Evidence** | Model construction reports exactly 181,132,071 parameters. Focused V4/runtime/training/config/CI/registry tests pass (37 tests); changed Python files are Ruff-clean and compile successfully. No model training or GPU workload was started. |
| **Risk** | The canonical V4 checkpoint does not exist yet; quality remains unmeasured until an owner-started T4/GPU run produces it. Historical research scripts outside supported entry points may still describe earlier factorial experiments. |
| **Follow-up** | Train seed 1301 from scratch in the canonical T4 notebook, then inspect checkpoint compatibility and generation quality. |

---

## 2026-07-15 - DATA/FIX - Hard-kill append recovery and completed V4 campaign slice

| Field | Value |
|-------|-------|
| **Date** | 2026-07-15 |
| **Author** | Codex |
| **Component** | native corpus journal; download status; Stream-B gate; fallback tokenizer |
| **Type** | DATA + FIX + RELIABILITY + EXECUTION |
| **Summary** | Recovered the terminated 30GB append with a full corpus audit, then executed the seven-source slice and canonical V4 gates. Fixed the underlying restart boundary: every 1,000-record append now fsyncs corpus bytes before committing the exact byte boundary with SQLite rows, and a bound restart may truncate only the uncommitted tail. Bucket-only downloads no longer overwrite the authoritative base status, and Stream B requires explicit base-bucket evidence. Replaced repeated per-special-token scans with one compiled split and added bounded fallback-piece caching without changing IDs. |
| **Evidence** | Full audit: 28,993,027,495 bytes, 5,274,479 records, four native classes, zero failures. Campaign slice: 64.024 MiB, seven sources, held-out disjoint, mix verified. V4: 32,768 IDs, V3 prefix stable, byte round-trip proven, 38.9932% projected reduction. Minimal checks: Ruff clean; 3 downloader recovery/status tests and 4 Stream-B gate tests pass; 1,003 old/new tokenizer parity cases match with ~1.95x sampled speedup. CUDA probes: 57.4M/256 = 840.6 tok/s at 1.11 GiB; 159.1M/256 = 552.8 tok/s at 3.04 GiB; 159.1M/2,048/microbatch-1 = 255.4 tok/s at 8.0 GiB allocated, beyond the 4050's physical VRAM. |
| **Risk** | The owner stopped the 120GB acquisition at roughly 29GB. Only the 27.002GiB prefix has final audit evidence; the later journaled append requires recovery and a refreshed audit. Full immutable shard publication remains incomplete. The 150M three-seed pilot still requires completed shards, owner signing authority, and CUDA-visible execution. |
| **Follow-up** | Recover and audit the stopped append without automatically resuming acquisition, choose the approved campaign volume, then publish tokenizer-bound shard families and prepare the signed V4 pilot. |

---

## 2026-07-13 - OBS/FIX - ThirdEye boundary inspection and dependency pin

| Field | Value |
|-------|-------|
| **Date** | 2026-07-13 |
| **Author** | Codex |
| **Component** | standalone ThirdEye integration; An-Ra evidence boundary |
| **Type** | OBSERVABILITY + FIX + PROVENANCE |
| **Summary** | Inspected the downloaded standalone ThirdEye repository at commit `e691a79`. ThirdEye is the generic evidence control plane: Project → Feature → Protocol → Run → Metric → Evidence → Decision, with SQLite storage, content-addressed artifacts, controlled-protocol grading, reports, and calibrated intelligence telemetry. An-Ra’s adapter is the project-specific registration/telemetry bridge, while An-Ra’s signed launch manifest and forecast ledger remain domain-specific pre-launch gates; they are complementary rather than two competing intelligence engines. Pinned An-Ra’s dependency from moving `main` to the inspected immutable ThirdEye commit so the evidence semantics cannot drift between runs. |
| **Evidence** | Standalone ThirdEye test suite passes; An-Ra adapter tests were already present. |
| **Follow-up** | Normalize each signed An-Ra launch into one ThirdEye `RunManifest`, attach the signed manifest/checkpoint as content-addressed artifacts, emit training metrics through ThirdEye, and use ThirdEye reports as the human-facing evidence surface. Retire duplicate An-Ra reporting only after parity is proven. |

---

## 2026-07-13 - TRAIN/FIX - Real per-seed launches and live tokenizer lineage

| Field | Value |
|-------|-------|
| **Date** | 2026-07-13 |
| **Author** | Codex |
| **Component** | pilot factorial; signed data lineage; trainer determinism; V4 checkpoint contract |
| **Type** | TRAIN + FIX + RELIABILITY + VERIFICATION |
| **Summary** | Corrected a false three-seed execution contract: the factorial formerly put three seed numbers in one signed manifest but launched one trainer without an explicit seed. It now emits one signed run per cell per seed (69 total), gives every run a unique checkpoint path/worker identity, and propagates the signed seed through Python, Torch, CUDA, model initialization, and immutable-shard sampling. Launch schema 3 hashes every bound data manifest, assigns explicit train/validation roles, and re-hashes artifacts at validation so post-sign mutation fails closed. Real continuation checkpoints are hash-bound too; scratch launches no longer try to load a path literally named `scratch`. Removed the last global-V3 assumptions from checkpoint metadata and staged tokenizer gates: schema, vocab size/hash, special IDs, and the 500-probe fingerprint now come from the active signed tokenizer. Hardened target-mix sampling against non-finite/negative weights and zero-mass edge selection. Fixed resolved V4 cells so they actually become launchable; moonshots remain outside critical-path readiness. Added a resumable dispatcher that validates signed jobs and forecast lead time, skips completed runs, excludes moonshots by default, requires explicit CUDA execution, and writes per-seed status/log evidence. Worker run reports and mutable selection evidence are isolated per artifact/seed, preventing parallel jobs from overwriting metrics, evaluations, mix control, CDR, data-route evidence, or progress journals. Forecast outcomes now require finite values and an existing SHA-256-bound evidence file and reject duplicate resolution. The launch-manifest API accepts the required data roles. |
| **Evidence** | Focused signed-launch, forecast, pilot, tokenizer-lineage, dispatcher, trainer-plan, and sampler suite: 64 passed, 1 skipped in 9.45s. Changed-file Ruff and `git diff --check` clean. RTX 4050 runtime: Torch 2.11.0+cu128, CUDA available, 6GB VRAM. |
| **Risk** | No run outcome is credited. The 69 manifests require completed immutable V3/V4 shards and the owner-held signing key; the 150M/MoE factorial requires cluster-class memory and compute. |
| **Follow-up** | Let the queued Stream-B runner finish the audited 30GB tranche, seven-source slice, canonical V4, and both shard families. Then sign the 69 manifests and dispatch critical non-moonshot replicas; keep M1/M3/M5 pilot-gated. |

---

## 2026-07-13 - DATA/FIX - Recoverable public corpus acquisition and verified slice inputs

| Field | Value |
|-------|-------|
| **Date** | 2026-07-13 |
| **Author** | Codex |
| **Component** | licensed corpus acquisition; audit/resume; tokenizer campaign slice; canonical V4 lineage |
| **Type** | DATA + FIX + RELIABILITY + VERIFICATION |
| **Summary** | Diagnosed the 21.58GB resume stop: Stack v2 is authentication-gated and Dolma's loading script is incompatible with the installed datasets client. Replaced them with immutable Common Pile Stack-v2-open-code and openly licensed ArXiv parquet revisions, validated each live, and made per-row licensing all-of rather than unsafe substring-any. Added source/rate progress, fsync-before-SQLite ordering, WAL resume, hash-chained incremental append audits, and an executable 120GB native profile. Acquired 120k pinned SmolTalk instruction examples. Added a deterministic verifier-bank DFC builder; unverified historical DFC can no longer enter the verified slice. Identity replay is explicit in slice evidence. Canonical V4 CLI runs now bind the ready campaign manifest and exact slice hash. Token publication is source-pure, explicitly materializes a trainable identity replay shard, and raw training deterministically enforces the signed campaign mix rather than silently following physical corpus imbalance. V3 and V4 publication now use separate tokenizer-bound shard families and inventories; a signed pilot resolves token capacity from its own signed train manifest rather than the process-global V3 inventory. |
| **Evidence** | Live streaming probe: 4/4 native sources, one accepted row each, zero errors. Full 21,582,998,123-byte integrity pass: 4,113,170 valid records, zero structural/hash/license/duplicate/quality failures, resume-safe SQLite index. Instruction corpus: 120,000 rows / 230,753,834 bytes. Verified DFC: 2,249 unique rows / 4,195,921 bytes, 1,125 formal-proof + 1,124 constraint-verifier records, all verified. Final focused data/slice/V4/curriculum/shard/orchestration/pilot-contract suite: 75 passed in 26.63s; changed-file Ruff and diff checks clean. |
| **Risk** | The repaired network resume is actively appending from the proven audit toward 27GiB; it is not yet complete. No 30GB/120GB volume, V4 fertility, signed launch, three-seed, or model-quality gate is credited yet. The owner signing key and cluster-class pilot compute remain external blockers. |
| **Follow-up** | A managed `scripts.execute_stream_b` continuation is queued behind the active 30GB worker. It revalidates completed audit/source evidence, builds the seven-source 64MB slice, runs the bound V4 audit/build, and publishes immutable, family-specific V3/V4 inventories with live shard progress and restart safety at completed-family boundaries. A second fail-closed continuation starts the 120GB pinned acquisition only if Stream B reports complete. After the first runner passes, set the owner signing key, generate manifests, and launch the pre-registered pilots. |

---

## 2026-07-11 - DATA/PERF - Compact MinHash resume index

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | 30GB corpus acquisition; near-duplicate resume state |
| **Type** | DATA + PERF + RELIABILITY |
| **Summary** | Replaced the boxed `list[tuple[int]]` signature store and one-list-per-band LSH map with flat uint64 signature storage and integer singleton buckets that promote to lists only on collision. The running old worker had spent ~40 minutes and >5 GiB rebuilding 500k signatures without changing the corpus. After confirming the corpus size still exactly matched the completed audit, it was stopped and restarted on the compact implementation with unbuffered logs. Dedup thresholds, signatures, bands, and append-only SQLite/corpus contracts are unchanged. |
| **Evidence** | Deterministic 100k synthetic-signature benchmark: 0.649 s, ~154,042 inserts/s, 62.08 MiB tracemalloc peak. Dedup/audit focused verification: 24 passed. Restart precondition: corpus bytes = audit bytes = 17,500,932,024. |
| **Risk** | Network/API failures can still interrupt source acquisition. Any appended bytes invalidate the old audit for another restart, so a fresh audit is mandatory before resuming again. |
| **Follow-up** | Let the optimized worker acquire Stack/FineMath/Dolma; on completion, rerun the streaming audit before slice/token-shard publication. |

---

## 2026-07-11 - TRAIN/FIX - Per-cell tokenizer and shard binding

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | signed launch manifests; V3/V4 factorial isolation |
| **Type** | TRAIN + DATA + FIX |
| **Summary** | Closed a mixed-factorial reproducibility flaw: manifests formerly hashed the one process-global active tokenizer, which could mislabel V3 and V4 cells. Every launch now binds an explicit tokenizer artifact path and verifies its hash. Pilot construction selects V3 tokenizer/V3 shards or canonical V4 tokenizer/V4 shards per cell; the unified trainer exports the signed path to the child process. A V4 stream blocker is resolved only when those artifacts exist and pass construction checks. |
| **Evidence** | Pilot, signed-manifest, cognition-manifest, and training-contract checks: 35 passed, 1 skipped. |
| **Risk** | Canonical V4 artifacts do not exist yet; V4 cells remain correctly blocked. The local draft tokenizer is not accepted as the canonical path. |
| **Follow-up** | After the seven-source slice passes, build `tokenizer/tokenizer_v4_32k.json`, publish V4 train/validation shards, then regenerate signed manifests. |

---

## 2026-07-11 - TRAIN/DATA - Executable curriculum-order factorial

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | immutable raw-shard sampling; pilot manifests; checkpoint recipes |
| **Type** | TRAIN + DATA + VERIFICATION |
| **Summary** | Implemented code-before-prose, math-density-ramp, and identity-mix-late as deterministic progress-dependent source sampling over the signed expected-token budget. The sampler stores compact shard ranges rather than millions of Python window entries, preserves the realized corpus mix when a multiplier is 1.0, requires the targeted source class, and samples training only. Curriculum identity is persisted in checkpoint training-recipe metadata and a changed curriculum/layout/profile fails closed on resume. With dense/Muon/QK/SWA/MTP/MoE work, all 17 ordinary V3 factorial cells are trainer-mapped; only three V4-dependent and three moonshot cells remain evidence-blocked. |
| **Evidence** | Curriculum curve/order/determinism tests plus pilot and training-contract checks: 25 passed. Campaign preflight reports 17/23 trainer-mapped. |
| **Risk** | The schedules encode the pre-registered hypotheses, not proven optimal order. Weighted replacement can repeat windows; the existing window-consumption tracker quantifies repeat rate, and matched-token validation determines the winner. |
| **Follow-up** | Complete mixed-source corpus and immutable shards, set the owner signing key, generate manifests, then run the mapped cells without changing their schedules post hoc. |

---

## 2026-07-11 - ARCH/TRAIN - Executable MTP and sparse-upcycled MoE axes

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | pilot architecture axes; canonical trainer; checkpoint compatibility |
| **Type** | ARCH + TRAIN + VERIFICATION |
| **Summary** | Replaced two factorial-only labels with actual model/trainer behavior. MTP uses two per-horizon RMSNorm/projection heads tied through the canonical embedding to predict +2 and +3 tokens, contributing a 0.2-weight loss. MoE replaces each dense SwiGLU with eight routed clones and one always-on shared expert, executes top-2 assignments sparsely, and scales shared+routed output for exact dense-function parity at initialization. Load balancing is aux-loss-free: observed expert utilization updates a bounded persistent routing bias only after a complete optimizer step. Signed manifest axes now select both features; exact total parameter accounting includes inactive experts; dense/MTP/MoE checkpoint mismatches fail closed. |
| **Evidence** | MTP RTX 4050: 58,194,823/58,194,823 expected parameters, base CE 9.1169, weighted MTP loss 1.8077, finite gradients, 486.44 MiB peak at the bounded sequence. MoE CUDA kernel smoke: finite CE 5.5495 and gradients, balance bias advanced, 22.37 MiB peak on a small model. Focused architecture/trainer checks passed (55). |
| **Cost** | MoE total parameters are 375,940,743 (50M dense anchor) and 1,100,615,335 (150M dense anchor); MTP+MoE totals are 376,761,223 and 1,102,222,759. Active expert compute is shared + top-2, but optimizer/checkpoint memory follows total parameters. |
| **Risk** | Kernel correctness and dense parity do not prove capability-per-active-FLOP gains. Full MoE pilots exceed comfortable RTX 4050 optimizer memory and require the campaign cluster or rented accelerator. Sparse routing throughput needs measurement; three-seed gates decide adoption. |
| **Follow-up** | Finish immutable data and launch the mapped dense/Muon/QK/SWA/MTP/MoE cells. Implement curriculum-axis scheduling; keep V4 and moonshots blocked on their declared evidence. |

---

## 2026-07-11 - ARCH/FIX - QK-Norm, hybrid attention, and stronger cache parity

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | production attention; pilot mapping; checkpoint lineage; KV-cache gate |
| **Type** | ARCH + TRAIN + VERIFICATION |
| **Summary** | Implemented parameter-free per-head QK-Norm and the planned 3:1 1024-token sliding-window/full-attention hybrid as explicit model settings. The settings are checkpoint-recorded; pre-feature checkpoints are restored to full-attention/no-QK behavior, so loading old lineage cannot silently change logits. Signed pilot axes now reach the canonical trainer, making the QK-Norm-off and full-attention-only ablations executable. Strengthened KV-cache parity with a deterministic 16-bin full-distribution probe per decode step after a depth-scaled scratch model made entropy/max-probability alone too insensitive to stale-cache poisoning. |
| **Evidence** | Focused architecture, pilot, data-contract, and cache tests: 48 passed, followed by 34 targeted tests after explicit attention-contract coverage. RTX 4050 scratch pilot forward/backward: 57,374,343 parameters, loss 9.1206, finite logits and gradients, 498.68 MiB peak allocated VRAM. |
| **Risk** | Implementation and numerical stability are not evidence that the hybrid wins. The QK-Norm and attention-pattern cells still require immutable data, three seeds, matched tokens/compute, and pre-registered acceptance metrics. Pilot contexts are now 2048 tokens so the 1024-token SWA pattern is causally distinct; this raises pilot memory/throughput cost and must be measured before launch. |
| **Follow-up** | Finish corpus/source mixing and immutable shards, then launch the now-mapped cells. Implement or retire MoE/MTP/curriculum axes before calling the full factorial executable. |

---

## 2026-07-11 - ARCH/DATA/FIX - Forward-path collapse diagnosis and fresh-campaign hardening

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | CUDA checkpoint activation forensics; model initialization; phase policies; loss-scale control; resumable corpus; executable pilot profiles |
| **Type** | ARCH + DATA + TRAIN + VERIFICATION |
| **Summary** | Extended the real-checkpoint inspection from static weights to diagnostic/native CUDA forwards. The legacy artifact is finite but its output distribution is effectively deterministic, residual streams amplify sharply with depth, router context is dormant, and two routed layers have near-closed gates. Repaired the fresh path with depth-scaled residual projection initialization, masked logit z-loss, a truly dense Phase A, isolated Phase-B subsystem activation, explicit later-phase recipes, and CUDA telemetry without hot-path `.item()` synchronization. Added exact 57.4M/159.1M pilot profiles; manifests now bind immutable train/validation shards, block unimplemented axes, forbid optimizer fallback, and cap actual tokens. Made the 17.5GB corpus audit byte-resumable/WAL-safe; it completed with zero structural failures and proved the interrupted artifact contains only FineWeb-Edu. Safe mixed-source resume is active. |
| **Evidence** | `output/v2/checkpoint_pathology_profile.json`; `output/v2/foundation_records_audit.json`; `output/v2/foundation_records_index.sqlite3`; `scripts/campaign_status.py`. |
| **Metrics** | Legacy diagnostic/native mean top-1 probability 99.974%/99.9997%; output entropy 0.00351/0.0000434 nats; residual RMS 14.87→282.26 diagnostic and 6.93→208.82 native; router position entropy 0.92–0.97, gate means as low as 0.0289. Fresh 28-layer scratch smoke: residual RMS 0.021→0.064, entropy 5.47 nats, top-1 2.4%. CUDA 57.4M forward/backward: finite, 499MB peak. Corpus: 17,500,932,024 bytes, 3,347,036 records, zero audit failures, 100% FineWeb-Edu. |
| **Risk** | These corrections prevent known failure modes but do not prove intelligence or a winning recipe. Advanced pilot axes remain blocked until each has a real trainer mapping and three-seed evidence. The active data resume depends on external network reliability. |
| **Follow-up** | Complete the remaining 30GB source mix, publish immutable train/validation/test shards, build the seven-source >=50MB tokenizer slice, then execute only launchable signed pilots. |

---

## 2026-07-11 - VERIFICATION - Schema-7 CUDA replay and baseline freeze

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | schema-4 legacy checkpoint migration to schema-7 runtime; RTX 4050 recovery gate; baseline freeze |
| **Type** | VERIFICATION |
| **Summary** | Re-executed all 600 cache-off greedy generations under the repaired schema-7 runtime with a hard CUDA requirement. The report completed and published atomically. Exact structure, finite activations, tokenizer probes, and deterministic replay pass; capability fails decisively: 0.0% coherence and acceptance with 100% EOS failure. Re-froze the unchanged checkpoint, schema-7 config contract, tokenizer fingerprint, and corpus manifests. |
| **Evidence** | `output/v2/stream_a_forensics.json` generated 2026-07-11 02:51:32; `output/v2/baseline_freeze.json`; checkpoint SHA-256 `648354a42d68c22769450a3aaa249e93689b21fbe72e68b07dcc15c6f7f4d393`; config contract SHA-256 `15321a16d8ddc28c1b825384ac5f2ffded0a66ea20aedbe73772874b17c14215`. |
| **Metrics** | 600/600 generations; coherence 0.0% / required 80%; diagnostic/native/replay acceptance 0.0%; diagnostic/native EOS failure 100%; current schema-7 contract 499,167,075 parameters. |
| **Risk** | The artifact remains structurally useful only as a forensic baseline. It is not a serving or continuation winner. The schema-7 architecture candidate still requires scratch/continuation pilots; this replay does not promote it. |
| **Follow-up** | Continue data/campaign preflight. Do not spend training compute until corpus manifests, tokenizer pilot, throughput, recovery, and immutable validation gates pass. |

---

## 2026-07-11 - TRAIN/FIX - Explicit answer-only loss contract across GPU and TPU

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | conversational packing; GPU/TPU/draft trainers; immutable validation; checkpoint/runtime metadata |
| **Type** | TRAIN + FIX + VERIFICATION |
| **Summary** | Closed the scaffold-loss and held-out-boundary gaps exposed by the legacy checkpoint. Conversational packing now emits a boolean token-level answer mask independent from numeric loss weights. GPU and TPU consumers preserve the mask; training records answer/scaffold NLL and exact denominators; validation reports total, weighted, answer-only, scaffold-only, and per-domain CE; checkpoints/runtime metadata preserve `best_answer_validation_loss` separately. The former conversational `eval_ds = ds` path is replaced by deterministic whole-content-group assignment before tokenization, distinct datasets, and a hashed zero-overlap manifest. Stage promotion computes same-identity, newer baseline/candidate domain regressions rather than trusting supplied flags. Raw causal foundation shards emit an all-false answer mask because they have no conversational answer boundary. |
| **Files** | `training/v2_data_mix.py`, `training/eval_v2.py`, `training/stages.py`, `training/train_unified.py`, `scripts/build_brain.py`, `scripts/build_brain_tpu.py`, `scripts/train_draft_recovery.py`, `training/v2_runtime.py`, `generate.py`, focused regressions and planning/forensic records |
| **Metrics** | Focused cross-path verification: 66 passed after final gate integration. Full non-GPU suite: 597 passed, 1 skipped. |
| **Verification** | Synthetic logits prove answer/scaffold/weighted/total CE separation; packed conversation mask presence; raw shard source identity/no-answer contract; deterministic grouped split and overlap rejection; identity/newness/domain promotion regression tests; training breakdown test; Python compilation; changed-file ruff and diff check; full suite. |
| **Risk** | Foundation shards intentionally have no answer-only metric. Promotion now blocks until a new validation measurement exists after the preflight baseline; short runs that never produce a second measurement cannot pass by reusing evidence. |
| **Follow-up** | Finish the schema-7 CUDA replay, freeze current hashes, then audit remaining stage metrics and campaign preflight for any asserted rather than derived evidence. |

---

## 2026-07-11 - ARCH/FIX - Schema-7 trainable temperature control and evidence publication guard

| Field | Value |
|-------|-------|
| **Date** | 2026-07-11 |
| **Author** | Codex |
| **Component** | layer-temperature architecture; checkpoint migration; optimizer partitioning; Stream-A evidence publication |
| **Type** | ARCH + FIX + VERIFICATION |
| **Summary** | Resolved forensic F-10 through a named schema-7 candidate. The per-layer temperature control is now a neutral-initialized log-space parameter with a positive `[0.5, 2.0]` realized bound, native anchor regularization, subsystem telemetry, and subsystem learning-rate coverage. Legacy direct scales migrate with `log(scale)`; non-finite/non-positive values fail closed. The 28 newly trainable scalars update the V3/V4 parameter contracts exactly. Also fixed the forensic driver so a structural/skipped rerun cannot overwrite an already executed behavioral gate, and `complete` now denotes terminal execution independently from pass/fail. |
| **Files** | `anra_brain.py`, `anra/architecture.py`, `training/v2_config.py`, `training/v2_runtime.py`, `training/anra_optimizer.py`, `scripts/run_checkpoint_forensics.py`, profiler support, focused regressions, README/planning/forensic records |
| **Metrics** | Schema 7 V3: 499,167,075 parameters; V4-16k: 509,631,075; V4-32k: 530,602,595. Focused: 62 passed. Full non-GPU suite: 590 passed, 1 skipped. Legacy schema-4 structural checker migrates to schema 7 with zero reported errors. |
| **Verification** | Trainability/finite-gradient/bounds/telemetry test; legacy conversion and invalid-scale rejection; subsystem optimizer partition test; canonical count proof; checkpoint-on/off all-gradient parity; evidence-downgrade regression; full non-GPU suite; changed-file ruff. |
| **Risk** | This is an architecture candidate, not an ablation winner. The removal-versus-trainable three-seed pilot remains mandatory before campaign selection. A schema-7 CUDA behavioral replay is active because the prior canonical report must be refreshed under the new runtime contract. |
| **Follow-up** | Finish the CUDA replay, freeze schema-7 hashes, then continue the highest-impact locally executable training-pipeline audit while corpus/pilot compute remains external. |

---

## 2026-07-10 - VERIFICATION - 500M GPU recovery gate completed

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | owner-supplied 500M checkpoint; RTX 4050 behavioral recovery gate |
| **Type** | VERIFICATION |
| **Summary** | Completed the cache-off, greedy, seed-0 behavioral audit without modifying the checkpoint: 200 diagnostic prompts, 200 native prompts, and 200 deterministic-replay prompts. The artifact retained exact load, finite activation, and deterministic-replay evidence, but failed the actual capability gate: 0.0% coherence against the required 80%. The checkpoint is structurally intact yet behaviorally disqualified as undertrained; it is not a recovered serving candidate. |
| **Evidence** | `output/v2/stream_a_forensics.json`, generated 2026-07-10 23:30:32 local time; prompt-suite SHA-256 `0bcd88f6c4d77fd7265f371ae3e6b0865f7988830b88dca44c4457c6858449b9`. |
| **Metrics** | Diagnostic acceptance 2.5%; native/replay acceptance 3.0%; diagnostic EOS failure 97.0%; coherence 0.0%; required coherence 80.0%. |
| **Risk** | Do not promote, serve, or select this checkpoint based on its legacy `best_loss`. The evidence supports undertraining; it does not uniquely assign causality among all documented historical defects. |
| **Follow-up** | Preserve the artifact as a baseline. Compare a named fresh-optimizer continuation against a scratch control at equal verified tokens and three seeds, using immutable validation and the same behavioral gate. |
| **Planning record** | The former recovery blueprint was superseded by `docs/engineering/V4_ARCHITECTURE_GATE.md` and the current `TODO.md` summary after the legacy checkpoint was rejected. |

---

## 2026-07-10 - FORENSICS/FIX - Real 500M checkpoint reconstruction and training-parity repair

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | owner-supplied 500M checkpoint; historical trainer reconstruction; activation-checkpoint parity; CUDA forensics |
| **Type** | FORENSICS + CRITICAL FIX + VERIFICATION |
| **Summary** | Froze and structurally proved the real 1.863 GiB checkpoint without modifying it. Schema-4-to-6 migration accounts for all 608 target tensors with exact core/native load and all 500 tokenizer probes matching. Reconstructed the recorded source commit and proved why its `best_loss=0.32788` is not capability evidence: it is minimum training-loss EMA, “quick validation” reused the training dataset, each session capped the mixture at 4,096 examples, and the entire run had an upper bound of 56.75M target positions. Verified historical defects include late-bound layer temperature during checkpoint backward, dormant context routing, no router balance/z loss, checkpoint publication before behavioral evaluation, partial-accumulation optimizer steps, and non-trainable temperature buffers. A new exact parity regression then found and fixed a second live checkpointing bug: RIM spectral normalization advanced power-iteration state again during backward recomputation, producing different gradients from plain execution. Recompute now freezes that state update and every parameter gradient matches. Added a weight-pathology profiler; the actual artifact has no non-finite values, but all 12 router context vectors are zero and unregularized residual scales range 0.361–1.638. The current raw-shard trainer was also caught selecting `eval_ds` for its training loader; it now selects `ds` and fails closed on any future dataset-identity violation before a campaign can contaminate validation. |
| **Files** | `docs/engineering/CHECKPOINT_FORENSICS.md`, `docs/IMPROVEMENT.md`, `anra_brain.py`, `scripts/build_brain.py`, `scripts/profile_checkpoint_pathologies.py`, profiler/parity/data-boundary regressions, checkpoint/TODO/progress records, `.gitignore` |
| **Metrics** | Checkpoint SHA-256 `648354a4...d393`; 499,167,047 runtime parameters; 608/608 tensor load; 0 mismatches; 0 non-finite serialized elements; CUDA smoke 7.44 tok/s on RTX 4050; full non-GPU suite 585 passed / 1 skipped; focused architecture tests 23 passed. |
| **Verification** | Safe checkpoint proof; frozen 500-probe tokenizer fingerprint; source-commit ancestry; direct CUDA smoke; content-hashed pathology report `9fb39287...29b4`; exact checkpoint-on/off logits/loss/all-gradient regression; full non-GPU pytest; changed-file ruff. |
| **Risk** | Superseded by the completed recovery-gate record above: the checkpoint is structurally intact but behaviorally disqualified as undertrained. |
| **Follow-up** | Preserve the artifact as a baseline, then compare a named fresh-optimizer continuation against a scratch control at equal verified tokens and three seeds. Never select on minimum training loss. |

---

## 2026-07-10 - FEAT/SECURITY - Cluster P4 signed promotion and executable rollback

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | cross-repository control-plane P4; An-Ra evidence producer; cluster promotion/rollback |
| **Type** | FEAT + SECURITY + VERIFICATION |
| **Summary** | Implemented the next incomplete `CLUSTER_CONTROL_PLANE.md` phase across both repositories without widening their dependency boundary. An-Ra now builds an owner-signed promotion envelope bound to the checkpoint SHA-256 and a concrete source commit; it validates Gate-6 metrics, two same-manifest/same-seed bit-exact reruns (G-C6), the adversarial promotion audit, and a locally signed rollback drill. The cluster verifies the fixed schema and all four evidence hashes and rejects the old arbitrary `{"gate": true}` authorization shape. Promotion now uses the actually active verified release as the rollback target. Added an executable rollback that resolves the prior artifact, verifies ledger hash/size, loads it with `weights_only=True`, validates complete model/optimizer/scheduler/tokenizer/data-position state at an optimizer boundary, atomically restores bytes, emits a signed report, and repoints the active release only after success. A corruption regression proves failed rollback leaves the active release unchanged. Promotion/rollback operator actions are audit-recorded. The cluster integration verifier now checks the An-Ra P4 CLI and gate-name contract. |
| **Files** | An-Ra: `evaluation/cluster_promotion_evidence.py`, focused tests, control-plane/TODO/progress records. Cluster: `backend/promotion_evidence.py`, `backend/recovery.py`, `backend/campaign_routes.py`, `scripts/verify_anra_integration.py`, `tests/test_promotion_and_rollback.py`. |
| **Metrics** | An-Ra: 581 non-GPU tests passed / 1 skipped. Cluster: 27 passed. Cross-repo integration: 8 jobs / 0 errors. Changed-file ruff and diff checks clean. |
| **Verification** | Full pytest in both repositories; `py -3.14 scripts/verify_anra_integration.py --anra-repo <An-Ra>`; focused P4 tests; changed-file ruff; `git diff --check`. |
| **Risk** | P4 code paths are fail-closed, but the real G-C6/G-C7 campaign exits remain unearned until owner-signed evaluation evidence and a Drive-backed full-checkpoint rollback drill exist. No production release was promoted in this local implementation run. |
| **Follow-up** | Generate the envelope from the restored real checkpoint and Gate-6 reports, execute the rollback through Drive, then proceed to P5's smoke-to-rescue ladder only after G-C1-G-C7 live evidence passes. |

---

## 2026-07-10 - FEAT/EXECUTION - Moonshot local execution and M6 acceptance

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | M1-M7 local execution; master-plan acceptance gates; M6 formal proofs |
| **Type** | FEAT + SECURITY + VERIFICATION |
| **Summary** | Reconciled the moonshot registry against the actual MASTER_UPGRADE gates, replacing weaker placeholder metrics: M1/M3 now require exact-150M three-seed evidence, M2 requires all three staged vision gates, M4 requires held-out simulation and digital-transition gains, M5 requires >=20k pairs and >=10% recall@5 gain, and M7 requires 10 human-approved merges with signed sovereignty records and zero reverts/unauthorized applies. Added one executor that runs every local-safe M1-M7 path but never treats shape/smoke evidence as campaign acceptance. Hardened the M6 verifier so facts can enter only through premises and implications only through explicit rules; arbitrary conclusion steps are rejected. Executed a hash-addressed 100-case M6 pilot (50 valid chains, 50 adversarial injected conclusions): 100/100 classifications correct, passing the >=95% gate. M6 alone is checked; M1-M5 and M7 remain explicitly blocked. Also removed a retrieval/memory circular import that prevented clean-process moonshot execution. |
| **Files** | `training/moonshot_pilots.py`, `training/moonshot_execution.py`, `training/formal_proof_pilot.py`, `verification/formal.py`, `memory/memory_router.py`, `scripts/run_moonshot_pilots.py`, moonshot tests, `docs/planning/IMPLEMENTATION_TODOS.md`, `PROGRESS.md` |
| **Metrics** | M6: 100 proof cases, 50 positive / 50 adversarial, 1.00 deterministic pass rate, gate >=0.95. All seven local paths executed and passed smoke checks. Full non-GPU suite: 578 passed / 1 skipped. |
| **Verification** | `py -3.14 -m scripts.run_moonshot_pilots --execute-local`; 26 focused tests; full non-GPU pytest; changed-file ruff; `git diff --check`. |
| **Risk** | low for M6's bounded propositional certificate domain. M1-M5 and M7 are not promoted from local smoke results; their real compute/data/human gates remain mandatory. |
| **Follow-up** | Finish the language campaign and corpus acquisition, then execute the exact-150M M1/M3 jobs, post-Gate-6 M2, trained M4/M5 evaluations, and real owner-reviewed M7 ladder evidence. |

---

## 2026-07-10 - SECURITY/REVIEW - Full upgrade execution audit and fail-closed hardening

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | accumulated MASTER_UPGRADE implementation across runtime, agents, serving, memory, repository integration, and pilot execution |
| **Type** | SECURITY REVIEW + FIX + VERIFICATION |
| **Summary** | Audited the accumulated implementation and corrected evidence paths that could pass while contradicting their stated contract. Answer contracts now validate schema types, SHA-256 fields, findings/verdicts, response presence, and derived trust-state consistency even after an attacker recomputes the outer hash. Verified-answer latency now measures complete TTFT-plus-decode p95 and requires both verified and unverified comparator samples. Every post-training method must now outperform its finite ablation score. Irreversible plan steps require a successful named pre-action authorization verifier; adversarial tests prove the action is never invoked when authorization is absent or denied. Adapter artifacts are hashed incrementally. Ghost memory defaults to a deterministic offline embedder, with the external sentence-transformer explicitly opt-in. Capability-graph discovery prunes virtual environments, caches, outputs, and training data and caps source reads, preventing a live multi-GB corpus from stalling full-system startup. The full-system agent probe now exercises a bounded real dispatcher action. The moonshot CLI reliably re-enters the current workspace instead of resolving a stale globally installed package. |
| **Files** | `runtime/answer_contracts.py`, `inference/serving_gates.py`, `inference/adapters.py`, `agents/plan_act_verify.py`, `training/post_training_ablations.py`, `phase3/ghost_memory_45p/ghost_memory/memory_store.py`, `inference/full_system_connector.py`, `app.py`, `scripts/run_moonshot_pilots.py`, focused regressions, planning/progress records |
| **Metrics** | An-Ra full non-GPU suite: 575 passed / 1 skipped. Sibling cluster full suite: 24 passed. Changed-file ruff and `git diff --check`: clean. The managed 30 GB acquisition remains active and uncredited pending completion/manifest verification. |
| **Verification** | `py -3.14 -m pytest tests -m "not gpu" --ignore=tests/test_drive_session_manager_integration.py --ignore=tests/test_v2_drive_artifacts.py --timeout=60 -q`; full sibling cluster `pytest`; changed-file `ruff check`; `git diff --check`; direct moonshot entry-point regression. |
| **Risk** | lower at the local mechanism layer. No checkpoint, corpus, training, soak, canary, human-study, or moonshot promotion is inferred from deterministic tests. |
| **Follow-up** | Let the corpus acquisition finish and verify its immutable manifests; restore the owner-held checkpoint and signing key; then execute the real three-seed, GPU, chaos/soak/preemption, post-training, canary, usability, and independent evaluation gates. |

---

## 2026-07-10 - FEAT/EXECUTION - Moonshot pilot gate executor

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | M1-M7 pilot evidence execution |
| **Type** | FEAT + VERIFICATION |
| **Summary** | Added one fail-closed runner for all seven moonshots. It loads only recorded metric evidence, evaluates each registered threshold, writes a durable status report, and classifies absent metrics as `blocked`. Executed against the real evidence directory: M1 through M7 are all blocked, with no failures and no passes claimed. |
| **Files** | `scripts/run_moonshot_pilots.py`, `tests/test_run_moonshot_pilots.py`, `output/v2/moonshot_pilot_status.json`, planning/progress records |
| **Metrics** | 9 focused moonshot/gate tests pass; ruff clean; first report: 7 blocked / 0 passed / 0 failed. |
| **Verification** | `py -3.14 -m pytest tests/test_moonshots.py tests/test_run_moonshot_pilots.py tests/test_remaining_todo_gates.py -q`; focused `ruff check`; `py -3.14 -m scripts.run_moonshot_pilots`. |
| **Risk** | none added — the runner cannot launch training, modify model lineage, or apply proposals. |
| **Follow-up** | Supply pilot artifacts and measured evidence; emit signed M1/M3/M5 manifests once the owner signing key is available; preserve the plan's non-critical-path ordering. |

---

## 2026-07-10 - FIX/EXECUTION - Stream B standard streaming ingestion

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | `scripts/download_training_data.py`; pinned corpus acquisition |
| **Type** | COMPATIBILITY FIX + EXECUTION |
| **Summary** | The actual 30 GB acquisition preflight found 313 GB free but the installed `datasets` 5 client rejected the obsolete `trust_remote_code` argument. Removed that argument from every streaming source path, preserving immutable revisions and avoiding remote dataset-code execution. Added a regression test that validates each native foundation source uses the standard streaming contract. After safely terminating two accidentally detached overlapping writers and deleting their unusable partial file, restarted exactly one hidden, logged downloader process. |
| **Files** | `.gitignore`, `pyproject.toml`, `scripts/download_training_data.py`, `tests/test_download_training_data.py`, `PROGRESS.md`, `docs/planning/IMPLEMENTATION_TODOS.md` |
| **Metrics** | 14 focused Stream-A/B/remaining-gate tests pass; downloader compatibility test passes; C: preflight 335,618,834,432 bytes free. |
| **Verification** | `py -3.14 -m pytest tests/test_download_training_data.py tests/test_build_campaign_slice.py tests/test_stream_a_forensics.py tests/test_remaining_todo_gates.py -q`; focused `ruff check`; downloader dry run. |
| **Risk** | high-duration external operation — no corpus-size or provenance gate is claimed until the single process completes, its status is `complete`, source errors are empty, and token shards/manifests validate. |
| **Follow-up** | Monitor the managed 30 GB download, publish immutable token shards, re-run the 50 MB held-out slice and V4 audit, then acquire further licensed tranches to the 120 GB plan target. |

---

## 2026-07-10 - FEAT/SECURITY - Cluster P2 truth surfaces and P3 evidence gates

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | sibling `gpu cluster` control plane: reliable v2 telemetry, operator controls, chaos/soak/preemption gates |
| **Type** | FEAT + SECURITY + VERIFICATION |
| **Summary** | Added durable `WorkerTelemetry` and `OperatorAudit` records. Signed worker heartbeats now persist append-only progress samples, while the operator-only campaign dashboard calculates throughput strictly from successive token-counter deltas and exposes worker status, quota, incidents, jobs, and verified artifacts. New pause/drain/resume/halt controls fence lease issuance and halt refuses artifact commits; replacement-worker revokes the old encrypted secret and safely requeues its lease. The dashboard UI uses the v2 campaign control route. Legacy cluster init and training-control mutation routes now require the operator bearer token. P3 adds G-C1 chaos, G-C2 24-hour soak, and G-C3 five-preemption evaluators: missing, simulated, short-duration, or evidence-free data fails closed. |
| **Files** | sibling repo: `backend/database.py`, `backend/campaign.py`, `backend/campaign_routes.py`, `backend/observability.py`, `backend/chaos.py`, `backend/main.py`, frontend campaign-status/control files, focused tests; An-Ra: TODO board and progress log |
| **Metrics** | 21 focused cluster tests pass; ruff clean. Simulated chaos is deliberately `local_harness_passed` only, never `campaign_gate_passed`. |
| **Verification** | `py -3.14 -m pytest tests/test_campaign_observability.py tests/test_cluster_chaos_gates.py tests/test_reliable_campaign.py tests/test_auth_and_cluster.py tests/test_coordinator_controls.py tests/test_worker_artifacts.py -q`; focused `ruff check`; `git diff --check`. |
| **Risk** | medium — this is local orchestration and evidence validation. It does not substitute for real Drive transactions, live tunnels, a 24-hour two-worker soak, or five physical preemption recoveries. |
| **Follow-up** | Feed signed/timestamped live reports to G-C1/G-C2/G-C3, then run the owner-held checkpoint/corpus/pilot campaign gates. |

---

## 2026-07-10 - FEAT - Remaining TODO fail-closed execution gates

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | checkpoint recovery; post-training evidence; moonshot pilots; campaign probes |
| **Type** | FEAT + VERIFICATION |
| **Summary** | Executed the remaining locally runnable TODO probes and added missing fail-closed machinery. `run_checkpoint_forensics.py --run-generation` correctly reports blocked because the real 500M artifact is not present, while retaining the passing 500-probe tokenizer fingerprint. `build_campaign_slice.py` re-proved held-out split disjointness but only has 3.40 MB, so the 50 MB gate remains blocked. Added `training/recovery_drill.py`, which terminates an isolated checkpoint writer then restores its exact boundary and optimizer state; `training/post_training_ablations.py`, which requires reports and ablations for SFT/RLVR/STaR/DPO/self-distillation; and `training/moonshot_pilots.py`, which registers all M1–M7 acceptance metrics and rejects missing evidence. |
| **Files** | training/recovery_drill.py, training/post_training_ablations.py, training/moonshot_pilots.py, tests/test_remaining_todo_gates.py, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 3 new gate tests pass; recovery drill terminates and restores a real isolated process checkpoint. Campaign slice: 3.3966 MB / required 50 MB. |
| **Verification** | `py -3.14 scripts/run_checkpoint_forensics.py --run-generation` (expected blocked); `py -3.14 scripts/build_campaign_slice.py` (valid but below minimum); `py -3.14 -m pytest tests/test_remaining_todo_gates.py -q`; focused `ruff check`. |
| **Risk** | none from the local gates; external campaign progress is still blocked by the absent checkpoint, corpus scale, credentials, GPU/fleet, telemetry duration, and independent evaluation evidence. |
| **Follow-up** | Owner must restore the checkpoint, authorize/sign pilot manifests, provide the licensed corpus and compute fleet, then execute the saved gate runners with real artifacts. |

---

## 2026-07-10 - FEAT - P2 execution harnesses, gates, and trust UI

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | serving promotion gates; DPO; GEPA cadence; release drills; developer UI |
| **Type** | FEAT |
| **Summary** | Completed P2's locally executable mechanisms. Accelerator promotion now fail-closes unless speculative acceptance/speed, token and distribution parity, <=1% QAT output drift, and TTFT/decode/verified-latency budgets pass. Added a reference-policy DPO objective, a ten-cycle proposal-only GEPA runner that records external review and requires a justified rejection, and fail-closed canary/adversarial aggregation gates around existing signed rollback and release-bundle evidence. The backend-served developer UI now fetches and renders only `/traces/{trace_id}/trust`; `ui/usability.py` holds the versioned 20-scenario acceptance script. |
| **Files** | inference/serving_gates.py, training/dpo.py, training/qat.py, training/gepa_cycles.py, evaluation/release_drills.py, ui/usability.py, app.py, tests/test_p2_execution.py, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 44 focused P2/P1/GEPA/new-system tests passed; ruff clean on all P2 files. |
| **Verification** | `py -3.14 -m pytest tests/test_p2_execution.py tests/test_p1_p2_foundations.py tests/test_gepa.py tests/test_new_systems.py -q`; focused `ruff check`. |
| **Risk** | medium - all executed P2 evidence is deterministic local test evidence. Real Phase B-E runs, post-training ablations, owner-held 50-goal evaluation, human usability study, latency hardware profile, and production canary remain separate mandatory evidence gates. |
| **Follow-up** | Restore the checkpoint and compute fleet, then run the remaining campaign gates with saved artifacts and independent review. |

---

## 2026-07-10 - FEAT/SECURITY - P1/P2 local serving, trust, and agent foundations

| Field | Value |
|-------|-------|
| **Date** | 2026-07-10 |
| **Author** | Codex |
| **Component** | serving primitives; answer contracts; retrieval recall; trust projections; plan-act-verify |
| **Type** | FEAT + SECURITY |
| **Summary** | Implemented the locally executable P1/P2 foundations. `evaluation/retrieval_recall.py` measures recall@5/20/50 against the shared retrieval contract and fails closed on invalid or empty suites. `runtime/answer_contracts.py` scans untrusted retrieved spans for instruction override, role impersonation, prompt-exfiltration, and delimiter attacks; tainted memories are removed before chat context construction. `/generate` and `/chat` now return and ledger a tamper-evident hash-only answer contract, while `/traces/{trace_id}/trust` renders only ledger-derived verifier, memory lifecycle, and gate projections. Added `ContinuousBatchScheduler` and bounded `PagedKVCache` serving primitives, plus content-addressed adapter registration/hot activation with checkpoint/tokenizer lineage and mutation checks. Added a fail-closed plan-act-verify runner and 50-case harness. |
| **Files** | runtime/answer_contracts.py, runtime/ledger_projections.py, inference/serving_runtime.py, inference/adapters.py, evaluation/retrieval_recall.py, agents/plan_act_verify.py, app.py, tests/test_p1_p2_foundations.py, tests/test_ledger_projections.py, docs/planning/IMPLEMENTATION_TODOS.md, PROGRESS.md |
| **Metrics** | 23 focused P1/P2/retrieval/registry tests passed; 21 serving/integration-focused tests passed; ruff clean on touched files and `anra/`. |
| **Verification** | `py -3.14 -m pytest tests/test_p1_p2_foundations.py tests/test_ledger_projections.py tests/test_retrieval_substrate.py tests/test_verifier_registry.py -q`; `py -3.14 -m pytest tests/test_serving.py tests/test_integration_production.py tests/test_p1_p2_foundations.py tests/test_ledger_projections.py -q`; focused `ruff check`. |
| **Risk** | medium - the proof contract honestly reports unverified when no task-specific verifier ran; the 50-case runner test validates orchestration mechanics, not a full owner-held real-goal benchmark. No campaign, latency, soak, usability, or cluster chaos claim is made. |
| **Follow-up** | Execute the blocked measured/owner gates, then add workload-backed latency CI, the real 50-goal suite, 20-scenario usability evidence, and adversarial/canary release drills. |

---

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
