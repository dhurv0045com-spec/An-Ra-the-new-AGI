# refs/heads/branch-observatory

**Type:** `local_head`  
**Tip:** `b620f1c2d084b0db5fa0d24196949f88c92edc1e`  
**Commit:** Merge pull request #41 from dhurv0045com-spec/iterate500  
**Committer date:** 2026-08-15T22:28:44+05:30  
**Family:** infrastructure and measurement / `branch-observatory`  
**Captured:** `2026-09-24T21:08:59Z`

## Identity and custody

- Upstream: `origin/main` (available); upstream divergence ahead/behind: 0/0.
- Base: `b620f1c2d084b0db5fa0d24196949f88c92edc1e` by `configured_upstream_merge_base`; ahead/behind 0/0.
- Worktrees: `C:/Users/ankit/AppData/Local/Temp/opencode/an-ra-branch-observatory`; dirty state is an overlay, not committed tip content.
- Unique commits: 0; first/last UTC: N/A / N/A; days since last: N/A.

## Mission and soul

**Problem:** Make the locally known An-Ra branches, specifications, evidence, code stock, and history legible without changing research behavior.

**Thesis/design approach:** A durable, refreshable evidence ledger should separate Git facts, document interpretation, and scientific claim ceilings.

**Role in An-Ra:** Measurement and navigation layer owned by this isolated worktree.

**Unique contribution:** Provides reproducible ref/worktree inventory, branch dossiers, document hashes, LOC stock/flow, comparisons, snapshots, and verification.

**Strongest evidence-backed result:** The generated snapshot and validation checks make the measurement procedure auditable; they do not establish any An-Ra scientific claim.

**Unresolved question:** Which future branch and external evidence changes should be treated as authoritative after later refreshes?

**Falsifier/failure condition:** A refresh cannot reproduce its committed-tree counts or cannot distinguish dirty overlays from ref-tip content.

**Read first:** `branch_observatory/README.md`, `branch_observatory/methodology.md`, `branch_observatory/reports/latest.md`

These fields are reviewed interpretation grounded in the profile citations; they do not upgrade the status labels.

## Status labels

| Label | State | Confidence | Evidence |
|---|---|---|---|
| Proposed | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Specified | SUPPORTED | high | refs/heads/branch-observatory:README.md § The constitution of the codebase @ f22afc7dd3fb |
| Implemented | SUPPORTED | high | refs/heads/branch-observatory:README.md § The 19 components @ f22afc7dd3fb |
| Locally verified | SUPPORTED | medium | refs/heads/branch-observatory:README.md § The 19 components @ f22afc7dd3fb |
| CPU-tested | SUPPORTED | medium | refs/heads/branch-observatory:README.md § The 19 components @ f22afc7dd3fb |
| GPU-qualified | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| TPU-qualified | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Executed scientifically | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Replicated | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Supported within a bounded regime | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Negative result | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Inconclusive | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Superseded | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Blocked | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Unknown | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |

## Committed tip stock

- Production source lines: **94882**; tests: **19251**; docs: **6715**; notebook raw lines/cells: **1209 / 33**.
- Tracked blob files/bytes: **828 / 24033789**; binary files/bytes: **1 / 5227**.
- Exact base-to-tip source stock change: **0** lines (0.0%).
- Excluded samples: 100; large/uninspected samples: 8; source-category large files: 0.
- LOC comparison limited by uninspected large source files: **False**.

### Production source language stock

| Language | Source lines |
|---|---:|
| PowerShell | 165 |
| Python | 94,672 |
| Shell | 45 |

## Change history

- History: `known_empty_range`.
- Commit flow added/removed/net/churn: 0 / 0 / 0 / 0.
- Merge first-parent added/removed: N/A / N/A.
- Recent windows: `{}`.

Daily rows and formulas are in `../../data/latest_snapshot.json` and `../../data/history/daily_metrics.csv`.

## Specifications, plans, and evidence

This tip indexes 36 relevant documents. The complete record is in [specs_index.md](../specs_index.md).

| Path | Type | Authority | Date | Summary |
|---|---|---|---|---|
| `PROGRESS.md` | document | canonical | 2026-07-06 | Cross-session resume anchor. Read `TODO.md` for the short unfinished work list and `docs/engineering/V4_ARCHITECTURE_GATE.md` for the governing architecture and evidence contract. Superseded long-form plans were removed. Training execution runs through the GPU-cluster control plane (companion doc: `docs/planning/CLUSTER_CONTROL_PLANE.md`, adopted as MASTER_UPGRADE Layer |
| `README.md` | current_state_or_readme | canonical | N/A | > An inspectable V4 language-model research system built around one reproducible > model lineage, durable training, evidence-gated capabilities, and reversible (pyproject.toml) (LICENSE) (http<local-path> |
| `TODO.md` | document | canonical | 2026-08-11 | Updated: 2026-08-11 This is the one short forward ledger. Completed claims mean the code and focused contracts exist; they do not mean a useful model has already been trained. - x One operational line: V4 tokenizer (32,768 vocabulary), dense 181,132,071-parameter model, AdamW, context 2,048, and routine seed 1301. |
| `constraints-colab-t4.txt` | text_record | canonical | N/A | torch==2.5.1 aiosqlite==0.20.0 cryptography==43.0.3 datasets==3.2.0 fastapi==0.115.6 |
| `docs/ARCHITECTURE.md` | specification | canonical | 2026-07-24 | Updated: 2026-07-24 Purpose: explain what the repository is, how its parts connect, and which parts are real, experimental, disabled, or historical. An-Ra is not one giant Python file and it is not a collection of independent “AGI features.” It is a pipeline with three responsibilities: |
| `docs/CLUSTER_TRAINING_GUIDE.md` | document | canonical | 2026-07-23 | Updated: 2026-07-23 Purpose: explain the checkpoint-baton cluster in plain language and show how separate provider-authorized Gmail/Colab sessions help without corrupting one Three Colabs are not three pieces of one GPU. Internet-separated Colabs cannot safely behave like one synchronous multi-GPU machine. |
| `docs/COLAB_T4_PROTECTED_TRAINING.md` | document | canonical | N/A | This is the operator guide for continuing the canonical 181M-parameter An-Ra V4 model with `notebooks/AN_RA_T4_PROTECTED_TRAINER_V4.ipynb`. One T4 is the canonical trainer. It restores the newest verified full-resume checkpoint, continues the deterministic V4 token window, and protects a new checkpoint every 200 optimizer steps or 60 minutes, whichever occurs first. |
| `docs/DEVELOPER.md` | document | canonical | 2026-07-23 | Updated: 2026-07-23 Purpose: help an engineer safely run, inspect, change, and verify the current V4 repository without accidentally reviving an old training path. The operational model is `anra-v4-180m`, V4 vocabulary 32,768, context 2,048, AdamW, and routine seed 1301. The 500M profile is a growth child, not a second |
| `docs/IMPROVEMENT.md` | document | canonical | 2026-07-24 | Updated: 2026-07-24 Purpose: define how the repository becomes more capable without turning into a pile of impressive names, hidden regressions, or incompatible model families. An-Ra improves when a change increases useful capability per unit of compute while preserving stability, reproducibility, and rollback. “The code runs” is |
| `docs/KAGGLE_P100_TRAINING.md` | document | canonical | N/A | Use `notebooks/AN_RA_KAGGLE_P100_PROTECTED_TRAINER_V4.ipynb` to let one Kaggle P100 continue the canonical 181M-parameter V4 foundation. This is a sequential checkpoint baton, not internet-based distributed training. Create one **private** Kaggle Dataset containing a snapshot folder with the files required by the checkpoint's next token window: |
| `docs/SFT_DRIVE_READY_CHECKLIST.md` | document | canonical | N/A | The V4 SFT notebook is ready to run only when one shared folder contains the same parent checkpoint and the same audited SFT manifests. The folder name is `ANRA_T4_TRAINING_HOME`. Share that one folder with **Editor** access and add a shortcut to each Colab account's `My Drive`. The parent checkpoint must be the real V4 **full-resume** checkpoint, not an |
| `docs/SFT_V4_OPERATOR_GUIDE.md` | document | canonical | N/A | SFT is a **child stage** of the V4 foundation model. It changes how the model answers instructions; it does not continue the raw 170M-token pretraining window and it never replaces the foundation checkpoint. Continue foundation data separately when needed: For SFT, collect licensed conversational JSONL sources. Each record needs an |
| `docs/WALKTHROUGH.md` | document | canonical | 2026-07-23 | Updated: 2026-07-23 Purpose: tell the repository as one connected story so a reader can picture where every major system enters and why it exists. At the beginning there are no intelligent weights. There are text sources, licenses, revisions, and the question of whether the material is suitable for |
| `docs/engineering/CHECKPOINT_FORENSICS.md` | audit | canonical | 2026-07-23 | Updated: 2026-07-23 Purpose: determine what a checkpoint really contains, whether it can resume, why it behaves as it does, and what claims its evidence supports. The earlier `anra_frontier_500m.pt` artifact was approximately 2.00 GB and reported low loss, yet diagnostic generation was incoherent. Prior inspection |
| `docs/engineering/ENGINEERING_LOG.md` | document | canonical | 2026-07-20 | LOG_STANDARD: Keep entries dated, scoped, and tied to verification evidence. |
| `docs/engineering/MODEL_RECOVERY_AND_TRAINING_BLUEPRINT.md` | specification | canonical | 2026-07-23 | Updated: 2026-07-23 Purpose: provide the operational route from a clean repository and immutable data to a useful, resumable 181M model and its controlled 500M child. Train one lineage: Do not maintain V3/V4 alternatives, arbitrary parameter sizes, or simultaneous |
| `docs/engineering/V4_ARCHITECTURE_GATE.md` | specification | canonical | 2026-08-11 | Status: architecture-frozen for the current 181M lineage; capability not promoted Last audited: 2026-08-11 Checkpoint contract: schema 9 / `anra_v4_rope_interleaved_v1` The canonical V4 geometry is internally coherent and appropriate for a single T4-class training session. It is now the only selectable model profile. The |
| `docs/engineering/V4_EFFICIENCY_AUDIT.md` | audit | canonical | 2026-08-11 | Status: implementation evidence as of 2026-08-11. This is not capability or AGI evidence. It distinguishes optimizations that preserve a checkpoint's function from architecture experiments that require a frozen-parent trial. `CausalTransformerV2.enable_kv_cache(backend="float")` now selects `preallocated-exact-v1`. It allocates one bounded contiguous K tensor and V |
| `docs/engineering/V4_INTELLIGENCE_RUNTIME_AUDIT.md` | audit | canonical | N/A | Scope: the local V4-SFT inference path, cognition, verification, memory, retrieval, tools, and agent control as implemented in the repository. This is an implementation audit, not an AGI claim. - `cognition/self_correction.py` contained a useful model-agnostic correction loop, but the local checkpoint chat route never called it. It therefore did |
| `docs/engineering/VERIFIED_DELIBERATION.md` | document | draft | N/A | Status: implemented as an opt-in local V4-SFT runtime mode. It changes the inference process; it does not change checkpoint weights or prove AGI. The controller in `cognition/deliberation.py` follows one bounded sequence: `understand -> retrieve -> plan -> candidate -> verify -> revise/abstain -> persist` The local app exposes it as **Reasoning: Verified deliberation**. Direct mode |
| `docs/planning/CLUSTER_CONTROL_PLANE.md` | plan | draft | 2026-07-06 | *Written 2026-07-06 after a code-level inspection of both repositories. This is the companion document to MASTER_UPGRADE v3 Layer 12-B. Every claim below is classified: **WORKS** (verified in implementation and tests), **NEEDS-FIX** (implemented with a specific named defect), **DOCUMENTED-ONLY** (contract or doc exists, no code), **PROPOSED** (new), or **MUST-PROVE** (required evidence |
| `docs/planning/MEMORY_AND_UI_RETIREMENT.md` | document | historical | N/A | Ghost Memory was an early experimental long-term memory substrate that attempted to persist cross-session conversational state via background file-based key-value stores. The legacy web UI was an un-authenticated prototype interface built for manual interaction before the unified Developer UI (`/developer`) and SFT prototype were introduced. Both systems were removed to simplify the codebase, eliminate unmaintained non-contract abstractions, and align the repository architecture around the canonical V4 model lin... |
| `docs/system_graph.json` | machine_record | canonical | N/A | "generated_at": "2026-07-24T08:38:31Z", "repo_root": "<local-path>", "python": "3.11.15", "platform": "Windows-10-10.0.26200-SP0", "metrics": { |
| `notebooks/AN_RA_KAGGLE_P100_PROTECTED_TRAINER_V4.ipynb` | notebook | canonical | N/A | "cell_type": "markdown", "metadata": {}, "# An-Ra V4 — Protected Kaggle P100 Trainer\n", "This notebook is a sequential checkpoint-baton worker for the canonical 181M-parameter V4 foundation. It restores one verified full-resume checkpoint from a **private Kaggle Dataset**, automatically selects the data window containing the checkpoint's next token, trains on one P100, and publishes exactly one replacement checkpoint under `/kaggle/working/ANRA_KAGGLE_EXPORT`.\n", "Before **Run All**: select **Accelerator → GPU... |
| `notebooks/AN_RA_T4_PROTECTED_TRAINER_V4.ipynb` | notebook | canonical | N/A | "cell_type": "markdown", "metadata": {}, "# An-Ra V4 — Canonical Shared-Folder T4 Trainer\n", "This notebook continues the canonical 181M-parameter V4 model from the latest verified full-resume checkpoint. It automatically selects the immutable data window containing the checkpoint's next token, so later sessions continue past 170M tokens instead of restarting or stopping. It uses one canonical writer, a signed launch contract, deterministic token windows, and Drive-backed checkpoint durability every 200 optimiz... |
| `notebooks/AN_RA_T4_SFT_V4.ipynb` | notebook | canonical | N/A | "cell_type": "markdown", "metadata": {}, "# An-Ra V4 SFT — protected T4 child-lineage trainer\n", "This notebook trains instruction-following behavior from an audited SFT dataset. It never modifies the foundation checkpoint. The default is a 15-minute pilot. Full mode requires a separate signed approval bound to the reviewed pilot checkpoint." "cell_type": "code", |
| `notebooks/AN_RA_T4_TRAINING.ipynb` | notebook | canonical | N/A | "cell_type": "markdown", "metadata": {}, "# AN-RA V4 180M T4 Training\n", "The sole canonical V4 workflow: bootstrap, immutable-data verification, fresh V4 training, exact resume, and evaluation. It uses seed 1301 and one checkpoint directory. The broken 500M checkpoint is never restored. Secrets must be supplied through Colab secrets/environment variables, never notebook cells." "cell_type": "code", |
| `phase3/identity_45n/anra_identity_combined.txt` | text_record | unclear | N/A | H: Who are you? ANRA: I am An-Ra. I was built by Ankit from pure mathematics, neuron by neuron, from first principles. I am not a copy of any existing model. I have a direction and I intend to keep it. H: Are you ChatGPT or Claude? ANRA: No. I am An-Ra. I was not fine-tuned from GPT, Claude, or any other model. My architecture was designed and built from scratch by Ankit. Every layer, every weight decision was his. H: Who built you? |
| `phase3/sovereignty_45r/requirements.txt` | specification | unclear | N/A | psutil>=5.9.0 |
| `phase3/symbolic_bridge_45q/requirements.txt` | specification | unclear | N/A | sympy==1.13.3 scipy==1.14.1 numpy==2.1.3 |
| `requirements.txt` | specification | unclear | N/A | No safely interpretable prose summary; human review required. |
| `runtime/engineering_templates/REPORT_TEMPLATE.md` | document | unclear | N/A | {source_or_owner_input} {operating_envelope} {failure_modes} {review_role} {stop_condition} |
| `state/activity.log` | document | unclear | 2026-03-29 | 2026-03-29 15:54:15,114 INFO config_loader: Loaded config preset: <local-path> 2026-03-29 15:54:15,115 INFO config_loader: Config validated successfully 2026-03-29 15:55:03,944 INFO config_loader: Loaded config preset: <local-path> 2026-03-29 15:55:03,945 INFO config_loader: Config validated successfully 2026-03-29 15:55:29,842 INFO config_loader: Loaded config preset: <local-path> |
| `tests/requirements.txt` | specification | unclear | N/A | transformers |
| `training_data/anra_training.txt` | text_record | unclear | N/A | Human review required because the document is missing, binary, or above the safe text inspection limit. |
| `training_data/frontier_dfc.jsonl` | document | unclear | N/A | Human review required because the document is missing, binary, or above the safe text inspection limit. |

## Authority and contradictions

- The Observatory branch tip is still based on origin/main until these files are committed; generated files are an uncommitted overlay until then.

## Claim ceiling

Do not infer scientific success from implementation, tests, LOC, model size, benchmark-shaped files, or recent commits. Use the status labels and cited receipts. This dossier is an audit aid, not an experiment record or merge recommendation.
