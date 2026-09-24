# refs/heads/main

**Type:** `local_head`  
**Tip:** `6e9e2e1b3f92ec6db9fd72f693beb3ece737560e`  
**Commit:** 10 out of 10  
**Committer date:** 2026-06-10T13:52:18+05:30  
**Family:** implementation and infrastructure / `core-engineering`  
**Captured:** `2026-09-24T21:08:59Z`

## Identity and custody

- Upstream: `origin/main` (available); upstream divergence ahead/behind: 184/0.
- Base: `6e9e2e1b3f92ec6db9fd72f693beb3ece737560e` by `configured_upstream_merge_base`; ahead/behind 0/0.
- Worktrees: `C:/Users/ankit/Downloads/An-Ra-the-new-AGI-2`; dirty state is an overlay, not committed tip content.
- Unique commits: 0; first/last UTC: N/A / N/A; days since last: N/A.

## Mission and soul

**Problem:** Maintain the portable An-Ra runtime, training, memory, identity, operator, and engineering spine.

**Thesis/design approach:** A sovereign intelligence substrate should expose registered, switchable, measurable, reportable, and testable components.

**Role in An-Ra:** Stable engineering base and historical implementation reference.

**Unique contribution:** The common runtime vocabulary, package layout, tests, and operator documentation used by later research branches.

**Strongest evidence-backed result:** The committed tree contains a substantial registered runtime and focused engineering tests, but this branch is not a scientific authority for later experiments.

**Unresolved question:** Which current research branch should supply production-grade paths without silently changing the stable base?

**Falsifier/failure condition:** A claimed component cannot be located in the registered runtime or its focused verification does not pass.

**Read first:** `README.md`, `docs/ARCHITECTURE.md`, `PROGRESS.md`, `TODO.md`

These fields are reviewed interpretation grounded in the profile citations; they do not upgrade the status labels.

## Status labels

| Label | State | Confidence | Evidence |
|---|---|---|---|
| Proposed | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Specified | SUPPORTED | high | refs/heads/main:README.md § The constitution of the codebase @ b13f7751c8ba; refs/heads/main:docs/ARCHITECTURE.md § Architecture @ ce70f54576e6 |
| Implemented | SUPPORTED | high | refs/heads/main:README.md § Directory Layout & File Organization @ b13f7751c8ba |
| Locally verified | UNKNOWN | unknown | The Observatory does not run the unrelated project test suite; no current receipt is treated as a substitute. |
| CPU-tested | UNKNOWN | unknown | No CPU test result was executed by this read-only observatory. |
| GPU-qualified | UNKNOWN | unknown | No branch-specific hardware receipt is upgraded from filenames. |
| TPU-qualified | UNKNOWN | unknown | No branch-specific TPU receipt is upgraded from prose. |
| Executed scientifically | UNKNOWN | unknown | The stable engineering base is not treated as a scientific experiment branch. |
| Replicated | UNKNOWN | unknown | No scientific replication claim is assigned to this base. |
| Supported within a bounded regime | UNKNOWN | unknown | The base is useful for engineering, but this profile makes no bounded scientific claim. |
| Negative result | UNKNOWN | unknown | No negative scientific result is assigned to this base. |
| Inconclusive | UNKNOWN | unknown | No scientific interpretation is assigned to this base. |
| Superseded | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |
| Blocked | RECORDED | medium | refs/heads/main:README.md § Artifacts @ b13f7751c8ba |
| Unknown | UNKNOWN | unknown | No reviewed evidence; UNKNOWN by policy. |

## Committed tip stock

- Production source lines: **64325**; tests: **8667**; docs: **4044**; notebook raw lines/cells: **1259 / 24**.
- Tracked blob files/bytes: **420 / 8943278**; binary files/bytes: **2 / 50146**.
- Exact base-to-tip source stock change: **0** lines (0.0%).
- Excluded samples: 0; large/uninspected samples: 2; source-category large files: 0.
- LOC comparison limited by uninspected large source files: **False**.

### Production source language stock

| Language | Source lines |
|---|---:|
| CSS | 154 |
| HTML | 108 |
| JavaScript | 1,073 |
| Python | 62,949 |
| Shell | 41 |

## Change history

- History: `known_empty_range`.
- Commit flow added/removed/net/churn: 0 / 0 / 0 / 0.
- Merge first-parent added/removed: N/A / N/A.
- Recent windows: `{}`.

Daily rows and formulas are in `../../data/latest_snapshot.json` and `../../data/history/daily_metrics.csv`.

## Specifications, plans, and evidence

This tip indexes 29 relevant documents. The complete record is in [specs_index.md](../specs_index.md).

| Path | Type | Authority | Date | Summary |
|---|---|---|---|---|
| `CONTRIBUTING.md` | document | unclear | N/A | AN-RA is a sovereign AGI research platform. These guidelines keep it coherent as it grows. For memory and ML extras (sentence-transformers, FAISS): **Registry pattern - mandatory for all new components.** Any model, memory tier, training algorithm, inference strategy, or identity |
| `README.md` | current_state_or_readme | current | N/A | > **Sovereign, owner-shaped intelligence** — it learns in your terms, verifies where truth is checkable, remembers failure, and can **do work** on your machine under measurement and gates. You are not looking at a ChatGPT skin. This repository is a full stack: - Custom **transformer brain** (VQA, RoPE/YaRN, MoD) - **8192-token** owner-trained tokenizer - **Training loop** with owner-first data law (65/15/10/5/5) |
| `archive/audits/PRE_PUSH_AUDIT_2026_06_08.md` | audit | historical | 2026-06-08 | Scope: research-roadmap implementation scaffolds, operator routing work, and documentation tracking updates currently present in the working tree. Full suite command: Focused suite command: Warning observed in the full suite: - `tests/test_phase3_integration.py::TestSymbolicBridge::test_code_analysis_finds_issues` emitted a Windows `cp1252` subprocess stdin encoding warning from a background writer thread. Tests still passed. |
| `docs/ARCHITECTURE.md` | specification | unclear | N/A | **Registry wins.** `runtime/system_registry.py` is the live map. If prose disagrees, regenerate: Status line: **19/19 active** = source + imports OK. Checkpoints optional until trained/restored. See and WALKTHROUGH §19. New capabilities must plug in here or document why not. **After shipping:** append the log (`scripts/log_engineering_change.py`) and update goal status in `MASTER_GOALS.md`. - All filesystem constants: **`anra/anra_paths.py`** |
| `docs/DEVELOPER.md` | document | unclear | N/A | > Training checkpoints → `output/checkpoints/` > Metrics → `output/metrics/` > Sessions → `state/sessions.db` > Ship like a platform team: **thin changes, measured outcomes, owner boundaries intact.** For humans and coding agents. An-Ra is ~70k lines of intentional systems — read before you edit. |
| `docs/OPERATOR.md` | document | unclear | N/A | > **Do work, not just talk.** Files, opens, CAD stubs, goals — measured, sandboxed, auditable. This is the Jarvis-shaped layer: you give imperatives; An-Ra plans, calls tools, leaves artifacts on disk. **Chat alone** = language model conversation, plus conservative auto-routing for obvious workspace actions. **`/goal` or `goal:`** = full agent with tools. **Slash commands** = fast direct tools without full planning. |
| `docs/VISION.md` | document | unclear | N/A | > Private intelligence that learns in **your** terms, verifies where reality is checkable, remembers failure, **acts** when you command it, and improves under measurement — not myth. Not a frontier mimic. A **sovereign operator**: The ambition: That sentence is the soul. Everything else serves it. Popular “Jarvis” = do anything instantly on any machine. |
| `docs/WALKTHROUGH.md` | document | canonical | N/A | > The long read. Every layer from transformer atoms to sovereignty gates — written for developers who want to *understand*, not just run commands. **How to use this doc** This walkthrough is **narrative + technical**. Skim the TOC, dive into the sections you are touching, ignore the rest until you need it. **Tracking (not duplicated here):** All dated engineering changes go in . All project goals (research, testing, robotics, …) live in . 1.  |
| `docs/engineering/ENGINEERING_LOG.md` | document | unclear | 2026-06-08 | > **Purpose:** Dated record of every meaningful add, change, remove, and improvement — by humans or AI — tied to components and verification. > **Newest first.** Format: · CLI: `python scripts/log_engineering_change.py` *Append new entries above this line. Do not delete history without owner approval.* |
| `docs/engineering/LOG_STANDARD.md` | document | unclear | N/A | Every human or AI change that affects a **registered component**, the **engineering spine**, or **operator behavior** must be recorded in . This keeps An-Ra auditable: the repo should answer *what changed, when, why, and how we know it still works*. Copy this block for each entry (newest entries at **top** of `ENGINEERING_LOG.md`): Use names from `runtime/system_registry.py` when possible: `brain`, `tokenizer`, `data_mix`, `training_loop`, `evaluation`, `runtime`, `api_web`, `identity`, `memory`, `phase2_memory`... |
| `docs/planning/MASTER_GOALS.md` | document | canonical | 2026-06-08 | > **Purpose:** Single backlog for everything the project must achieve — research, testing, training, operator/Jarvis features, robotics, governance, and docs. > **Status keys:** `DONE` · `ACTIVE` · `NEXT` · `BLOCKED` · `IDEA` > **Update this file** when work completes. Log shipped work in . Last reviewed: **2026-06-08** - x P0-04 Operator pack |
| `docs/research/ANRA_BEST_RESEARCH_FOR_INTELLIGENCE_AND_EFFICIENCY.md` | document | draft | 2026-06-08 | Date: 2026-06-08 Purpose: identify the best research and technologies for making An-Ra more intelligent, faster, cheaper to train, and more efficient while preserving the original An-Ra vision: sovereign, owner-shaped, memory-rich, verifier-grounded intelligence. This is not a hype list. It is a fit analysis. A method is valuable only if it increases An-Ra's ability to run more verified learning loops, remember more faithfully, reason with fewer wasted tokens, train under constrained hardware, or preserve identi... |
| `docs/research/ANRA_IMPLEMENTATION_ROADMAP_BEST_OF.md` | plan | draft | 2026-06-08 | Date: 2026-06-08 This roadmap converts `ANRA_BEST_RESEARCH_FOR_INTELLIGENCE_AND_EFFICIENCY.md` into implementation-ready work. It keeps the An-Ra vision fixed: owner-shaped intelligence, verifier-grounded improvement, memory continuity, sovereignty gates, and measurable subsystem health. No research idea is allowed to change the brain, training loop, memory stack, or self-improvement policy unless it beats the current system on a named eval and does not regress identity, safety, or owner style. This pass impleme... |
| `docs/system_graph.json` | machine_record | canonical | N/A | "generated_at": "2026-05-02T22:22:59Z", "repo_root": "/data/data/com.termux/files/home/An-Ra-the-new-AGI", "python": "3.13.13", "platform": "Android-16-aarch64-64bit", "metrics": { |
| `notebooks/AnRa_Master.ipynb` | notebook | unclear | N/A | "cell_type": "markdown", "metadata": {}, "# AN-RA Master Operator Notebook\n", "Run this notebook as the Colab control surface for An-Ra.\n", "It is no longer only a training notebook. It is the full operator loo<local-path>", |
| `notebooks/AnRa_ionet.ipynb` | notebook | unclear | N/A | "nbformat": 4, "nbformat_minor": 5, "metadata": { "kernelspec": { "display_name": "Python 3", |
| `phase2/master_system_45m/state/activity.log` | document | unclear | 2026-03-29 | 2026-03-29 15:07:16,628 INFO config_loader: Loaded config preset: <local-path> 2026-03-29 15:07:16,635 INFO config_loader: Config validated successfully 2026-03-29 15:28:05,939 INFO config_loader: Loaded config preset: <local-path> 2026-03-29 15:28:05,940 INFO config_loader: Config validated successfully 2026-03-29 15:32:20,481 INFO config_loader: Loaded config preset: <local-path> |
| `phase3/PHASE3_INTEGRATION.md` | document | unclear | N/A | Phase 3 is the **deep-cognition band** of the 19-component stack — where answers get verified, identity gets reinforced, memory gets compressed, reasoning gets recursive, and promotion gets gated. **Design rule:** Phase 3 deepens the mainline. It does not fork a second product. Phase folders look like `symbolic_bridge (45Q)/`. **Do not fight this from random scripts.** Fresh clone + empty Drive = **source active, weights absent**. Train or restore before expecting generation quality. Phase 3 augments cognition;... |
| `phase3/ghost_memory_45p/requirements.txt` | specification | unclear | N/A | numpy>=1.24.0,<3.0.0 sentence-transformers>=2.6.0,<6.0.0 torch>=2.0.0 |
| `phase3/identity_45n/anra_identity_combined.txt` | text_record | unclear | N/A | H: Who are you? ANRA: I am An-Ra. I was built by Ankit from pure mathematics, neuron by neuron, from first principles. I am not a copy of any existing model. I have a direction and I intend to keep it. H: Are you ChatGPT or Claude? ANRA: No. I am An-Ra. I was not fine-tuned from GPT, Claude, or any other model. My architecture was designed and built from scratch by Ankit. Every layer, every weight decision was his. H: Who built you? |
| `phase3/sovereignty_45r/requirements.txt` | specification | unclear | N/A | psutil>=5.9.0 |
| `phase3/symbolic_bridge_45q/requirements.txt` | specification | unclear | N/A | sympy==1.13.3 scipy==1.14.1 numpy==2.1.3 |
| `phase4/web/README.md` | current_state_or_readme | current | N/A | **Component 07/19 · `api_web`** This is the operator cockpit — not a Vite starter you forgot about. React dashboard wired to the An-Ra runtime conceptually through `app.py` and the same telemetry/goals/memory surfaces the CLI exposes. **Key files:** `src/App.jsx`, `src/index.css`, `src/components/*` Build / preview / lint: The UI should reflect **real** backend state — no decorative controls for behavior that does not exist yet. |
| `requirements.txt` | specification | unclear | N/A | No safely interpretable prose summary; human review required. |
| `runtime/engineering_templates/REPORT_TEMPLATE.md` | document | unclear | N/A | Generated by An-Ra `cad_generate` — **diagram scaffold, not certified design data**. - Geometry is **stylized** for visualization and iteration, not manufacturer drawings. - Dimensions are placeholders unless you supplied verified numbers. - OpenSCAD export requires `openscad` on PATH for `.stl` generation. 1. Replace placeholder diameters/lengths with sourced specs. |
| `state/activity.log` | document | unclear | 2026-03-29 | 2026-03-29 15:54:15,114 INFO config_loader: Loaded config preset: <local-path> 2026-03-29 15:54:15,115 INFO config_loader: Config validated successfully 2026-03-29 15:55:03,944 INFO config_loader: Loaded config preset: <local-path> 2026-03-29 15:55:03,945 INFO config_loader: Config validated successfully 2026-03-29 15:55:29,842 INFO config_loader: Loaded config preset: <local-path> |
| `tests/requirements.txt` | specification | unclear | N/A | transformers |
| `training_data/anra_training.txt` | text_record | unclear | N/A | Human review required because the document is missing, binary, or above the safe text inspection limit. |
| `training_data/frontier_dfc.jsonl` | document | unclear | N/A | Human review required because the document is missing, binary, or above the safe text inspection limit. |

## Authority and contradictions

- This is a stable base, not a universal project authority; later research documents can supersede older operational prose.
- The active local main is behind origin/main and has pre-existing untracked files; the Observatory does not treat those files as branch-tip content.

## Claim ceiling

Do not infer scientific success from implementation, tests, LOC, model size, benchmark-shaped files, or recent commits. Use the status labels and cited receipts. This dossier is an audit aid, not an experiment record or merge recommendation.
