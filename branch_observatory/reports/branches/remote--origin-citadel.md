# refs/remotes/origin/citadel

**Type:** `remote_tracking`  
**Tip:** `e96c9a9a07af1d9126930190cbf3d1c764523e2d`  
**Commit:** audit(citadel): confirmed cross-split contamination + shortcut vulnerability + supply shortfall in tiered arithmetic corpus  
**Committer date:** 2026-09-13T03:12:19+05:30  
**Family:** independent evidence audit and research protocol / `citadel-audit`  
**Captured:** `2026-09-24T21:08:59Z`

## Identity and custody

- Upstream: `none` (none); upstream divergence ahead/behind: N/A/N/A.
- Base: `b620f1c2d084b0db5fa0d24196949f88c92edc1e` by `configured_project_base`; ahead/behind 220/1.
- Worktrees: none; dirty state is an overlay, not committed tip content.
- Unique commits: 220; first/last UTC: 2026-08-16 / 2026-09-12; days since last: 12.

## Mission and soul

**Problem:** Audit project claims for contamination, shortcuts, provenance, negative results, and claim ceilings without silently repairing or rerunning them.

**Thesis/design approach:** Documentation is not proof; receipts, ledgers, controls, replication, and custody determine the strength of a scientific claim.

**Role in An-Ra:** Independent audit and data-readiness gate for the wider program.

**Unique contribution:** A machine-readable evidence ledger, research protocol, negative-results ledger, bottleneck ranking, and explicit open questions.

**Strongest evidence-backed result:** The ledger demonstrates a replicated negative result for the screened scorer policies and records contamination/shortcut findings; it explicitly limits generalization to the tested policy family and substrate.

**Unresolved question:** Can an answer-blind selection policy and a qualified substrate pass the open gates without evaluator leakage?

**Falsifier/failure condition:** A preregistered policy or data audit passes its controls and falsifies the recorded contamination/shortcut diagnosis.

**Read first:** `docs/citadel/EVIDENCE_LEDGER.md`, `docs/citadel/RESEARCH_PROTOCOL.md`, `docs/citadel/NEGATIVE_RESULTS.md`, `docs/citadel/OPEN_QUESTIONS.md`

These fields are reviewed interpretation grounded in the profile citations; they do not upgrade the status labels.

## Status labels

| Label | State | Confidence | Evidence |
|---|---|---|---|
| Proposed | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Specified | SUPPORTED | high | refs/remotes/origin/citadel:docs/citadel/RESEARCH_PROTOCOL.md § Standing rules @ 0a6897b979a6 |
| Implemented | SUPPORTED | high | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § Every substantive existing claim @ e1775d34b96a |
| Locally verified | SUPPORTED_BOUNDED | high | refs/remotes/origin/citadel:docs/citadel/RESEARCH_PROTOCOL.md § Reproducibility minimum @ 0a6897b979a6 |
| CPU-tested | SUPPORTED_BOUNDED | medium | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § E3 @ e1775d34b96a |
| GPU-qualified | SUPPORTED_BOUNDED | medium | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § E3 @ e1775d34b96a |
| TPU-qualified | NOT_VERIFIED | high | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § E10 @ e1775d34b96a |
| Executed scientifically | SUPPORTED_BOUNDED | high | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § E1 to E10 @ e1775d34b96a |
| Replicated | SUPPORTED_BOUNDED | high | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § E3 @ e1775d34b96a |
| Supported within a bounded regime | SUPPORTED | high | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § E5 to E9 @ e1775d34b96a |
| Negative result | RECORDED | high | refs/remotes/origin/citadel:docs/citadel/NEGATIVE_RESULTS.md § N1 to N20 @ 050ae5b707aa |
| Inconclusive | RECORDED | high | refs/remotes/origin/citadel:docs/citadel/OPEN_QUESTIONS.md § Open questions @ 74833486a826 |
| Superseded | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Blocked | RECORDED | high | refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md § E10 @ e1775d34b96a |
| Unknown | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |

## Committed tip stock

- Production source lines: **23288**; tests: **6940**; docs: **15124**; notebook raw lines/cells: **1049 / 57**.
- Tracked blob files/bytes: **313 / 30083322**; binary files/bytes: **5 / 1388180**.
- Exact base-to-tip source stock change: **-71594** lines (-75.45582934592441%).
- Excluded samples: 0; large/uninspected samples: 15; source-category large files: 0.
- LOC comparison limited by uninspected large source files: **False**.

### Production source language stock

| Language | Source lines |
|---|---:|
| Python | 23,288 |

## Change history

- History: `retrospective_git_history`.
- Commit flow added/removed/net/churn: 37346 / 108416 / -71070 / 145762.
- Merge first-parent added/removed: 579 / 61.
- Recent windows: `{"1d": {"commit_count": 0, "docs_lines_added": 0, "docs_lines_removed": 0, "end_utc_day": "2026-09-24", "files_changed": 0, "generated_artifact_changes": 0, "notebook_changes": 0, "source_lines_added": 0, "source_lines_churn": 0, "source_lines_net": 0, "source_lines_removed": 0, "start_utc_day": "2026-09-24", "status": "known", "test_lines_added": 0, "test_lines_removed": 0}, "30d": {"commit_count": 169, "docs_lines_added": 18148, "docs_lines_removed": 3778, "end_utc_day": "2026-09-24", "files_changed": 908, "generated_artifact_changes": 147, "notebook_changes": 21, "source_lines_added": 25709, "source_lines_churn": 36142, "source_lines_net": 15276, "source_lines_removed": 10433, "start_utc_day": "2026-08-26", "status": "known", "test_lines_added": 7076, "test_lines_removed": 3492}, "7d": {"commit_count": 0, "docs_lines_added": 0, "docs_lines_removed": 0, "end_utc_day": "2026-09-24", "files_changed": 0, "generated_artifact_changes": 0, "notebook_changes": 0, "source_lines_added": 0, "source_lines_churn": 0, "source_lines_net": 0, "source_lines_removed": 0, "start_utc_day": "2026-09-18", "status": "known", "test_lines_added": 0, "test_lines_removed": 0}, "90d": {"commit_count": 220, "docs_lines_added": 20413, "docs_lines_removed": 11211, "end_utc_day": "2026-09-24", "files_changed": 1791, "generated_artifact_changes": 186, "notebook_changes": 46, "source_lines_added": 37346, "source_lines_churn": 145762, "source_lines_net": -71070, "source_lines_removed": 108416, "start_utc_day": "2026-06-27", "status": "known", "test_lines_added": 10853, "test_lines_removed": 23014}}`.

Daily rows and formulas are in `../../data/latest_snapshot.json` and `../../data/history/daily_metrics.csv`.

## Specifications, plans, and evidence

This tip indexes 115 relevant documents. The complete record is in [specs_index.md](../specs_index.md).

| Path | Type | Authority | Date | Summary |
|---|---|---|---|---|
| `ESOES.md` | document | historical | N/A | > **NON-CANONICAL:** ESOES has completed four design iterations and executable E0/E1/250M contract phases. Start at ; that directory and generated receipts define the current V5 candidate. This file is retained as branch history. This branch is a design/research branch for the next An-Ra Core generation. It starts from the current `core-vnext` evidence but does **not** assume that the existing V4 architecture, tokenizer, parameter count, data recipe, or training path should survive unchanged. The recent evidence... |
| `README.md` | current_state_or_readme | canonical | N/A | ESOES is a clean-sheet research branch for designing the next An-Ra neural Core. Its Git ancestry passes through `core-vnext`, but V4, VNext, PGE, SFT, and EXP are evidence sources—not inherited implementation. Start with . Current state: **V5 contracts, local canaries, and experiment plans are executable; learned E1–E5 runners and the production trainer still require implementation. `python -m v5_contracts.launch_readiness --output artifacts/v5/launch_readiness.json` checks the evidence inventory. It never auth... |
| `agent.md` | handoff_or_brief | draft | 2026-09-06 | > Convention (binding): rewritten at the END of every Citadel work cycle, > committed to the `citadel` branch ONLY, then `git push origin citadel`. > Other branches are read-only audit inputs — never modified, never pushed. > A CPU/CUDA run is NEVER a TPU result. No fabricated device results. > Preregistration and results never share a commit. Download ceiling <10 GB; |
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
| `blueprint/MODEL_ARCHITECTURE.md` | specification | current | N/A | This is an engineering specification, not a neural-model implementation. Bracketed states are governed by `blueprint/DECISIONS.md`; `v5_contracts/model_spec.py` is the executable arithmetic/configuration authority. The architecture contains no explicit task labels, symbolic slots, state registers, memory modules, routers, or cognition heads. Learned attention/MLP circuits must earn query-conditioned binding and transformation under controlled training. Assuming bias-free projections, GQA keys/values of width 448... |
| `blueprint/OPEN_QUESTIONS.md` | decision_or_open_question | current | N/A | Ground Blueprint v0.4 intentionally limits the open set. A question belongs here only if its answer can materially change architecture, data, training, evaluation, or system boundaries. `V5_TRAINING_SPEC_v1.0.md` supplies the implementation default while these questions remain open. A winning experiment changes that default only through a new spec version; uncertainty is no longer permission for code to guess. **WHY IT MATTERS:** 16k saves embeddings and may preserve atomic symbols; 32k shortens sequences; eithe... |
| `blueprint/README.md` | current_state_or_readme | current | N/A | This directory is the single human-facing authority for the V5 Core. Code and receipts remain in their executable packages; this index prevents duplicated constants and competing documents. 1. — exact Core, tokenizer, data, cognition, optimization, topology, checkpoint, and |
| `blueprint/SOFTWARE_SYSTEM.md` | document | canonical | N/A | This specifies the clean implementation boundary and deliberately does not copy the VNext directory layout or checkpoint schema. is authoritative for exact packages, interfaces, commands, artifacts, CI gates, and milestone acceptance. The framework-independent `v5_contracts/`, executable `e0_cognition/`, and `e1_tokenizer/` research harness now exist. Production packages below remain gate-controlled. The existing research packages are implemented; the `v5_*` production packages are design targets, not empty scaf... |
| `blueprint/STATUS.md` | current_state_or_readme | canonical | 2026-09-02 | Updated: 2026-09-02 Phase: **ready for gated prelaunch execution; main scientific launch blocked** Canonical research blueprint:  Canonical code-facing specification:  Executable receipt: `artifacts/v5/training_spec_v1.json` |
| `blueprint/V5_MASTER_BLUEPRINT.md` | specification | canonical | 2026-08-30 | Status: **GROUND BLUEPRINT v0.4 — E0 SHORTCUT-RESISTANT BENCHMARK CONTRACT** Date: 2026-08-30 Branch: `esoes` Training authorization: **NO** This document is the canonical V5 research blueprint. It is intellectually independent of VNext implementation. The evidence base is `EVIDENCE_BASE.md`, the four-round attack record is `ITERATIONS.md`, and change control is `DECISIONS.md`. A value marked **EXPERIMENT REQUIRED** is not permission to encode it silently into a trainer. |
| `blueprint/V5_TRAINING_SPEC_v1.0.md` | document | current | N/A | Status: **implementation-frozen candidate; main run not authorized** Executable receipt: `artifacts/v5/training_spec_v1.json` Authoritative constants: `v5_contracts.training_spec` This is the single code-facing specification for V5-A. It removes undefined defaults and contradictory alternatives. “Frozen” means implementation must |
| `docs/citadel/500M/CYMEK_500M_CAMPAIGN.json` | machine_record | canonical | N/A | "campaign_id": "cymek-500m-v1", "canonical_cymek_sha": "28bf57a0d299a2c13a99fe0046616c00a1b8530c", "checkpoint": { "recovery_interval_minutes": 30, "recovery_rotation": 4, |
| `docs/citadel/500M/CYMEK_REQUIRED_CHANGES.md` | document | canonical | 2026-09-06 | Citadel does not edit/push Cymek. Each change below specifies file, function, current behavior, required behavior, why, test, and acceptance. Classification: **BLOCKING** (PRE500M cannot go green without it) / **RECOMMENDED** / **OPTIONAL**. - File(s): new `v5_data/acquire_production.py` (or extend the existing |
| `docs/citadel/500M/PLAN.md` | plan | current | N/A | Status: **SPEC_ONLY — NOT_AUTHORIZED_TO_TRAIN.** Machine-readable identity: . Cymek is the production authority; Citadel validates. **500,000,000 consumed training tokens** (Cymek `TrainingState.cumulative_tokens`), milestone ladder |
| `docs/citadel/500M/PRODUCTION_PATH_AUDIT.md` | audit | canonical | 2026-09-06 | Audit date: 2026-09-06. Pin: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (== origin/cymek == Citadel `PINNED_CYMEK_SHA`; the divergent local lineage 4abeaeb is UNPUSHED WIP and is not production — see `docs/citadel/CROSS_BRANCH_INGESTION.md`). Classification per §4: **CONNECTED** (executed inside the real production |
| `docs/citadel/BOTTLENECK_RANKING.md` | document | unclear | N/A | Candidate bottlenecks for the central question ("what prevents stronger transferable internal cognition per parameter and per training token"), ranked primarily by **expected information gain / experimental cost** — not novelty, not ambition. Each candidate keeps the full field set; Priority is the rank order for Citadel's attention. Fields: Evidence for / Evidence against / Uncertainty / Expected capability impact / |
| `docs/citadel/BRANCH_MAP.md` | document | canonical | 2026-09-03 | Audit date: 2026-09-03. Auditor: Citadel bootstrap audit (direct git inspection + full-tree review). Labels: **FACT** = verifiable from the repository (command or artifact cited). **INTERPRETATION** = Citadel's reading; may be wrong; must be re-derived when cited. **FACT.** Remote: `http<local-path>`. **FACT.** Branch tips at audit time (after `git fetch --all --prune`): **FACT.** Other remote branches exist (`main`, `core`, `core-exp`, `core-frozen-v4`, `core-vnext`, `senora`, `codxyz`, `experiment`, `iterate50... |
| `docs/citadel/CROSS_BRANCH_INGESTION.md` | document | draft | 2026-09-06 | Citadel's standing rule: other branches are audit inputs, never merged. This records what the sibling agent branches demonstrated, what Citadel ingests, and what stays branch-local. Read-only inspection of `origin/BRAMASTRA` (0235001) and `origin/Arkenstone` (2a5ab55); the local `cymek` branch has also moved (28bf57a → 4abeaeb, 3 commits, **unpushed** — origin/cymek still |
| … | … | … | … | 75 more in complete index |

## Authority and contradictions

- The research protocol says ledgers are authoritative when a receipt verdict disagrees; negative results must not be erased.
- Citadel is an audit, not a universal branch or a production authority.

## Claim ceiling

Do not infer scientific success from implementation, tests, LOC, model size, benchmark-shaped files, or recent commits. Use the status labels and cited receipts. This dossier is an audit aid, not an experiment record or merge recommendation.
