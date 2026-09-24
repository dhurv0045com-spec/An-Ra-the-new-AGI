# refs/remotes/origin/cymek-beta

**Type:** `remote_tracking`  
**Tip:** `194b98c8bddbe85d362fd5bf391ad68e3303ec79`  
**Commit:** fix(cymek): bind canonical role-transfer protocol hash  
**Committer date:** 2026-09-24T05:39:30+05:30  
**Family:** experiment operations, custody, and recovery / `cymek-recovery`  
**Captured:** `2026-09-24T21:08:59Z`

## Identity and custody

- Upstream: `none` (none); upstream divergence ahead/behind: N/A/N/A.
- Base: `b620f1c2d084b0db5fa0d24196949f88c92edc1e` by `configured_project_base`; ahead/behind 519/1.
- Worktrees: none; dirty state is an overlay, not committed tip content.
- Unique commits: 519; first/last UTC: 2026-08-16 / 2026-09-24; days since last: 0.

## Mission and soul

**Problem:** Recover and complete the pinned Formation-Mux campaign without changing its protocol or manufacturing a final result.

**Thesis/design approach:** A partial evidence archive is not a resumable experiment; checkpoint custody and frozen identities must pass before continuation.

**Role in An-Ra:** Current recovery/operator branch for the Formation-Mux campaign.

**Unique contribution:** Documents a dual-T4 partial session, identifies the missing original saved Output/checkpoint tree, and provides a fail-closed preflight/launcher.

**Strongest evidence-backed result:** The session report verifies 24/24 S5 arms and partial TIE-ROLE completion, but explicitly says no final result or architecture gate exists.

**Unresolved question:** Can the exact saved Output tree be recovered and pass the pinned preflight without protocol changes?

**Falsifier/failure condition:** The preflight rejects the bundle, any checkpoint/receipt hash mismatches, or a completed arm mutates.

**Read first:** `artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md`, `artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md`, `artifacts/cymek/evidence_ledger.json`

These fields are reviewed interpretation grounded in the profile citations; they do not upgrade the status labels.

## Status labels

| Label | State | Confidence | Evidence |
|---|---|---|---|
| Proposed | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Specified | SUPPORTED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md § Recovery rule @ df99076e5ce7 |
| Implemented | SUPPORTED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md § Recovery preflight @ df99076e5ce7 |
| Locally verified | NOT_VERIFIED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md § Recovery audit @ e98874a5ada3 |
| CPU-tested | NOT_VERIFIED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md § Recovery audit @ e98874a5ada3 |
| GPU-qualified | SUPPORTED_BOUNDED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md § Provenance @ e98874a5ada3 |
| TPU-qualified | NOT_VERIFIED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md § What did not complete @ e98874a5ada3 |
| Executed scientifically | SUPPORTED_BOUNDED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md § What completed @ e98874a5ada3 |
| Replicated | UNKNOWN | unknown | The partial campaign does not establish replication. |
| Supported within a bounded regime | SUPPORTED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md § Data handling and claim ceiling @ e98874a5ada3 |
| Negative result | RECORDED | medium | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md § Required user action @ df99076e5ce7 |
| Inconclusive | RECORDED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md § What did not complete @ e98874a5ada3 |
| Superseded | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Blocked | RECORDED | high | refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md § Status @ df99076e5ce7 |
| Unknown | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |

## Committed tip stock

- Production source lines: **55784**; tests: **15208**; docs: **24362**; notebook raw lines/cells: **2902 / 82**.
- Tracked blob files/bytes: **700 / 38967983**; binary files/bytes: **6 / 7653994**.
- Exact base-to-tip source stock change: **-39098** lines (-41.20697287156679%).
- Excluded samples: 0; large/uninspected samples: 16; source-category large files: 0.
- LOC comparison limited by uninspected large source files: **False**.

### Production source language stock

| Language | Source lines |
|---|---:|
| Python | 55,784 |

## Change history

- History: `retrospective_git_history`.
- Commit flow added/removed/net/churn: 71508 / 110082 / -38574 / 181590.
- Merge first-parent added/removed: 579 / 61.
- Recent windows: `{"1d": {"commit_count": 1, "docs_lines_added": 6, "docs_lines_removed": 2, "end_utc_day": "2026-09-24", "files_changed": 5, "generated_artifact_changes": 0, "notebook_changes": 0, "source_lines_added": 3, "source_lines_churn": 5, "source_lines_net": 1, "source_lines_removed": 2, "start_utc_day": "2026-09-24", "status": "known", "test_lines_added": 10, "test_lines_removed": 8}, "30d": {"commit_count": 468, "docs_lines_added": 27623, "docs_lines_removed": 4015, "end_utc_day": "2026-09-24", "files_changed": 1515, "generated_artifact_changes": 209, "notebook_changes": 59, "source_lines_added": 59871, "source_lines_churn": 71970, "source_lines_net": 47772, "source_lines_removed": 12099, "start_utc_day": "2026-08-26", "status": "known", "test_lines_added": 15763, "test_lines_removed": 3911}, "7d": {"commit_count": 30, "docs_lines_added": 1272, "docs_lines_removed": 92, "end_utc_day": "2026-09-24", "files_changed": 117, "generated_artifact_changes": 4, "notebook_changes": 20, "source_lines_added": 3766, "source_lines_churn": 4128, "source_lines_net": 3404, "source_lines_removed": 362, "start_utc_day": "2026-09-18", "status": "known", "test_lines_added": 1616, "test_lines_removed": 117}, "90d": {"commit_count": 519, "docs_lines_added": 29888, "docs_lines_removed": 11448, "end_utc_day": "2026-09-24", "files_changed": 2398, "generated_artifact_changes": 248, "notebook_changes": 84, "source_lines_added": 71508, "source_lines_churn": 181590, "source_lines_net": -38574, "source_lines_removed": 110082, "start_utc_day": "2026-06-27", "status": "known", "test_lines_added": 19540, "test_lines_removed": 23433}}`.

Daily rows and formulas are in `../../data/latest_snapshot.json` and `../../data/history/daily_metrics.csv`.

## Specifications, plans, and evidence

This tip indexes 264 relevant documents. The complete record is in [specs_index.md](../specs_index.md).

| Path | Type | Authority | Date | Summary |
|---|---|---|---|---|
| `.github/workflows/role-transfer-preregistration.yml` | preregistration_or_protocol | canonical | N/A | name: ROLE-TRANSFER-001 preregistration qualification branches: cymek-beta - 'v5_experiments/role_transfer_protocol_v1.py' - 'tests/test_role_transfer_protocol_v1.py' - 'docs/cymek/experiments/ROLE-TRANSFER-001/**' |
| `ESOES.md` | document | historical | N/A | > **NON-CANONICAL:** ESOES has completed four design iterations and executable E0/E1/250M contract phases. Start at ; that directory and generated receipts define the current V5 candidate. This file is retained as branch history. This branch is a design/research branch for the next An-Ra Core generation. It starts from the current `core-vnext` evidence but does **not** assume that the existing V4 architecture, tokenizer, parameter count, data recipe, or training path should survive unchanged. The recent evidence... |
| `README.md` | current_state_or_readme | canonical | N/A | ESOES is a clean-sheet research branch for designing the next An-Ra neural Core. Its Git ancestry passes through `core-vnext`, but V4, VNext, PGE, SFT, and EXP are evidence sources—not inherited implementation. Start with . Current state: **V5 contracts, local canaries, and experiment plans are executable; learned E1–E5 runners and the production trainer still require implementation. `python -m v5_contracts.launch_readiness --output artifacts/v5/launch_readiness.json` checks the evidence inventory. It never auth... |
| `agent.md` | handoff_or_brief | unclear | N/A | Branch: `cymek-500m-readiness`. `CYR-GPU-011` has **EXECUTED** on a Google Colab Tesla T4. Do not describe it as pending. Frozen executable: `0a97257e2b38db6dfa85cc6e58da0697591dde6b`. Source bundle SHA-256: `fbec390f66223a19a998db6519f42c046ad8cb0a345c205b168f5bd1a86668e5`. The raw V11 ZIP is intentionally **not stored in git**. Preserve only the distilled scientific record: |
| `artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/FORMATION_MUX_001_RESULTS.partial.zip` | result_or_summary | unclear | N/A | Human review required because the document is missing, binary, or above the safe text inspection limit. |
| `artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md` | document | canonical | 2026-09-23 | Status: **BLOCKED ON THE ORIGINAL KAGGLE SAVED OUTPUT** Recovery qualification: **PASS — GitHub Actions run `35925460488`, head `ee6a5cad4a4f39752b2646a60eeaeef8d2b06019`** Campaign snapshot: 24/24 S5 development arms; 2/24 TIE-ROLE frontier arms; no sealed evaluation Recovery rule: no retraining, no new seeds, no protocol changes, and no sealed access until custody passes `FORMATION_MUX_001_RESULTS (2)` is consistent by displayed size with the preserved evidence archive: |
| `artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md` | document | canonical | 2026-09-24 | Status: **PARTIAL CAMPAIGN SNAPSHOT — NOT A FINAL SCIENTIFIC RESULT** - Source archive supplied by the user: `FORMATION_MUX_001_RESULTS (1).zip` - Preserved archive: `FORMATION_MUX_001_RESULTS.partial.zip` - SHA-256: `3e3ad68cd7f80bd242733b61153d4bb8f3fedbb1e4fba8fbc0f3ebc5d904423f` - Archive integrity: ZIP CRC check passed; 755 unique members; no unsafe paths or duplicate names. |
| `artifacts/cymek/evidence_ledger.json` | evidence_ledger | canonical | N/A | "as_of_commit": "This ledger is updated per milestone; see git history for the exact commit.", "basis": "Every LOCAL_CANARY or stronger claim below has a committed receipt artifact with hashes under artifacts/cymek/ or is asserted by tests/test_*.py.", "components": { "BACKWARD": { "evidence": "Real autograd on live model parameters through the production backend; gradients required to exist and be finite for every trainable parameter.", |
| `artifacts/cymek/gap_audit.json` | audit | canonical | N/A | "schema": "anra-v5-cymek-gap-audit/v1", "audit_commit": "92dcd56ef492bfdb777711c06be924c322c55536", "audited_by": "cymek principal agent (fresh handoff, no inherited claims)", "method": "Every classification below was derived by reading the executable source and running the code, not from commit messages, README claims, or prior agent prose.", "verification_evidence": { |
| `artifacts/cymek/handshake_v2.md` | document | historical | N/A | Status: **proposed by Cymek, not yet adopted by Triquetra.** Direction stays: Cymek hands subjects; Triquetra qualifies cognition. Cymek does not modify Triquetra; this document is the exact compatibility specification. No schemas are copied silently: field mappings below are explicit, and drift in either direction fails closed. |
| `artifacts/cymek/miniature_receipt.json` | receipt | canonical | N/A | "checkpoint": { "final_checkpoint_sha256": "83b3d2b6dc85d431f3a7926d4300c130ccd1e7e500f682ee4b34a577905feedf", "resume_parameter_hash_equal": true "classification": "END_TO_END_MINIATURE", "cumulative_tokens": 16384, |
| `artifacts/cymek/p35a_preregistration.json` | preregistration_or_protocol | draft | N/A | "checkpoint_milestones_tokens":  "code_freeze_sha256": "78d969b58f450ae648904d3ac582add3c190bbea50ed7d60f5d4e3086f4231c7", "control": { "cognition_fraction": 0.0, "replacement": "ratio-preserving natural/code fill (M38)" |
| `artifacts/cymek/test_receipt.json` | receipt | unclear | N/A | "command": "python -m unittest discover -s tests; python -m v5_contracts.import_boundaries", "note": "Receipt names the commit whose tree was tested; the commit carrying this file adds only the file itself.", "platform": "Windows-10-4050-CUDA", "python": "3.11.15-torch-2.11.0+cu128", "failed": 0, |
| `artifacts/e0/shortcut_repair_receipt.json` | receipt | unclear | N/A | "generator_version": "e0-eval/0.4.0", "rule_induction_heuristics": { "bag_of_words": 0.06640625, "fixed_identity_rule": 0.0, "fixed_repeat_left_rule": 0.125, |
| `artifacts/e1/local_tournament/audit-16384.json` | audit | unclear | N/A | "artifact_sha256": "0a83fb9280ef5309f3351a5ec0dd9889940da374b3e17d02a687c1c307e1fd1b", "candidate": "local-byte-bpe-16384", "all_probes_present_once": true, "identity_roundtrip": true, "token_ids_in_range": true, |
| `artifacts/e1/local_tournament/audit-24576.json` | audit | unclear | N/A | "artifact_sha256": "97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc", "candidate": "local-byte-bpe-24576", "all_probes_present_once": true, "identity_roundtrip": true, "token_ids_in_range": true, |
| `artifacts/e1/local_tournament/audit-32768.json` | audit | unclear | N/A | "artifact_sha256": "56be8e1cab4a1b0cede97f5bd5e6af8f2f68175f846362a7c0fcdbd5f048ba7f", "candidate": "local-byte-bpe-32768", "all_probes_present_once": true, "identity_roundtrip": true, "token_ids_in_range": true, |
| `artifacts/e1/local_tournament/corpus_manifest.json` | machine_record | unclear | N/A | "evaluation_records": 14757, "evaluation_unique_texts": 8171, "evaluation_utf8_bytes_with_repetitions": 1057977, "holdout": { "bucket": 0, |
| `artifacts/e1/local_tournament/result.json` | result_or_summary | unclear | N/A | "canary_pareto_front":  "local-byte-bpe-16384", "local-byte-bpe-24576", "local-byte-bpe-32768" "candidate_rows":  |
| `artifacts/e1/tournament_plan.json` | plan | unclear | N/A | "artifact_sha256": null, "audit_receipt_sha256": null, "matched_raw_bytes": 10000000, "matched_training_flops": 1, "name": "bpe-byte-fallback-16384", |
| `artifacts/e1/v4_32k_baseline_audit.json` | audit | unclear | N/A | "artifact_sha256": "1a0140661c9d16c830f8dd8292699946e92db0be6f7aec92fb272d13cc1c745b", "candidate": "v4-native-32k-baseline", "all_probes_present_once": true, "identity_roundtrip": true, "token_ids_in_range": true, |
| `artifacts/e2/scoring_policy_preregistration.json` | preregistration_or_protocol | unclear | N/A | "abort_rules":  "Neutral prompt length or candidate suffix tokens differ from target.", "Candidate roles cannot be uniquely counterbalanced for every tokenizer.", "No calibrated policy passes every development gate.", "Implementation exceeds the compute abort budget.", |
| `artifacts/e2/static_plan.json` | plan | unclear | N/A | "attention_fraction_of_forward_proxy": 0.5333333333333333, "factors": { "shape": "deep" "forward_flops_per_full_sequence_proxy": 241591910400, "group": "shape", |
| `artifacts/e3/static_plan.json` | plan | unclear | N/A | "comparison_policy": { "tokenizer", "optimizer", "source order", "raw bytes", |
| `artifacts/v5/cymek_500m_closure_test_receipt.json` | receipt | unclear | N/A | "environment": { "note": "CPU-only run; no scientific GPU or TPU training was performed by this receipt.", "os": "Windows-10-10.0.26200-SP0", "python": "3.11.15", "source": "local cymek-beta worktree execution", |
| `artifacts/v5/cyr_gpu_002_test_receipt.json` | receipt | unclear | N/A | "environment": { "cpu": "AMD Ryzen 7 170 8C/16T", "cpu_venv": "torch 2.13+cpu", "cuda_venv": "torch 2.11+cu128", "gpu": "RTX 4050 Laptop 6 GB (not used for training this cycle)", |
| `artifacts/v5/cyr_gpu_005_test_receipt.json` | receipt | unclear | N/A | "environment": { "cpu": "AMD Ryzen 7 170 8C/16T", "note": "plumbing/TINY fixtures only; no heavy local training (operator compute boundary honored)", "os": "Windows", "ram_gb": 15.3, |
| `artifacts/v5/cyr_gpu_011_result_receipt.json` | receipt | unclear | N/A | "all_json_parsed": true, "bytes": 31263, "filename": "CYMEK_GPU_RESEARCH_V11_RESULTS.zip", "json_files":  "ARK002B_MANIFEST_RECEIPT.json", |
| `artifacts/v5/implementation_contract.json` | specification | unclear | N/A | "gqa_groups_integral": true, "head_dimensions_exact": true, "main_training_authorized": false, "target_within_half_percent": true, "token_budget_near_twenty_per_parameter": true, |
| `artifacts/v5/launch_readiness.json` | readiness | canonical | N/A | "blueprint_document_sha256": { "BENCHMARK.md": "6857c1e01425f9df6cf9d6b718ab9625661a7054affdcf2475c692dd94bdafa5", "DECISIONS.md": "11025585f272572d7d69723bb276e5431aac6e3c0ef290410e0edfb3dcd3d490", "DECISION_LOG.md": "3c142f18a722a5177191ca4d56203755296aabdc85cc669da0a4d8ab925d750d", "EXECUTION.md": "02387f4049742518f7b3b34e2461af5585ceecfedbe68aadfcbdd97860cca3cf", |
| `artifacts/v5/training_spec_v1.json` | machine_record | unclear | N/A | "cognition_fractions_exact": true, "cognition_tokens_exact": true, "data_tokens_exact": true, "difficulty_mix_exact": true, "external_identities_unfilled": true, |
| `artifacts/v5/training_transaction_canary.json` | machine_record | unclear | N/A | "clean_copy_restore": true, "corruption_rejected": true, "crash_after_pointer_safe": true, "crash_after_publish_before_pointer_safe": true, "crash_after_stage_safe": true, |
| `blueprint/BENCHMARK.md` | document | canonical | N/A | **Status:** Ground Blueprint benchmark specification v0.1 **Branch:** `esoes` **Purpose:** Decide whether a Core design, training recipe, or checkpoint is genuinely more useful for future cognition—not merely a better next-token predictor. **Main-training authority:** **NONE.** This specification defines evidence requirements; it does not authorize the V5-A run. The benchmark exists to answer one question: |
| `blueprint/DECISIONS.md` | decision_or_open_question | current | 2026-08-30 | Ground Blueprint: v0.4 Iteration: evidence-building after 4/4 design attacks Date: 2026-08-30 Commit: recorded by the commit containing this file A later agent may not silently change a decision. It must add a reopening entry stating new evidence and the affected blueprint version. |
| `blueprint/DECISION_LOG.md` | decision_or_open_question | canonical | 2026-08-29 | This file prevents silent redesign. Base: `core-vnext` at `054619f20851317e9b1c49b6f31599f6444a8280`. - create a separate research/design branch rather than modifying the active V4 training branch; - treat V4, EXP, and VNEXT as evidence, not immutable architecture; - do not launch a major V5 training run until the cognition-first blueprint is stress-tested and frozen; |
| `blueprint/EXECUTION.md` | plan | current | N/A | Expected current status: `READY_FOR_PRELAUNCH_EXPERIMENTS`, with `main_training_authorized=false`. This checks the evidence inventory, not a production trainer. E1–E5 learning runners still require implementation and the representative data and compute described below. Run the static audit and the matched P35 16k/24k/32k tournament on the declared, |
| `blueprint/EXPERIMENTS.md` | document | canonical | N/A | Order is frozen: **E0 → E1/E2 → E3 → E4 → E5 → freeze review**. No production trainer or V5-A main run precedes these gates. E1 and E2 may share baseline runs after E0, but scientific comparisons retain isolated variables. Development infrastructure is implemented in `e0_cognition/` and has a deterministic development certificate. It has **not** frozen or consumed a real sealed suite. Prove that the benchmark measures causal cognitive operations rather than templates, candidate priors, tokenization artifacts, or... |
| `blueprint/FREEZE_CHECKLIST.md` | document | current | N/A | The V5 training path is **not frozen** until every item below has an explicit answer, evidence link/receipt, or deliberate rejection. Ground Blueprint v0.4 state: **250M implementation contracts and shortcut-resistant E0/E1 research harnesses pass; sealed promotion certification remains incomplete.** Checked items below are research decisions or development invariants, not authorization to train. Required critical path: **E0 benchmark certification → E1 tokenizer → E2 architecture → E3 data/objective → E4 minima... |
| `blueprint/IMPLEMENTATION_BLUEPRINT.md` | specification | canonical | N/A | Status: **V5-A implementation candidate v1.0; main run blocked** Scope: design and executable contracts; no production trainer or main run authorization This is the canonical module-boundary bridge. `blueprint/V5_MASTER_BLUEPRINT.md` owns the scientific question, `blueprint/V5_TRAINING_SPEC_v1.0.md` and `v5_contracts.training_spec` own exact constants, and `blueprint/BENCHMARK.md` owns measurement. No code may substitute an unspecified default for a null identity in the executable receipt. The provisional V5-A c... |
| `blueprint/LAUNCH_GATES.json` | machine_record | current | N/A | "schema": "anra-v5-launch-gates/v1", "candidate_spec": "artifacts/v5/training_spec_v1.json", "external_identities": { "data_manifest_sha256": null, "pack_manifest_sha256": null, |
| … | … | … | … | 224 more in complete index |

## Authority and contradictions

- The recovery runbook is authoritative for custody requirements, not for a scientific result.
- The attached evidence-only ZIP is explicitly not sufficient continuation state.

## Claim ceiling

Do not infer scientific success from implementation, tests, LOC, model size, benchmark-shaped files, or recent commits. Use the status labels and cited receipts. This dossier is an audit aid, not an experiment record or merge recommendation.
