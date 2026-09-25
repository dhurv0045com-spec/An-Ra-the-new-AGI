# An-Ra Research

An-Ra is a research program for building and testing neural systems that can acquire useful capabilities, transfer them to new conditions, and retain them as learning continues. The work focuses on cognition, learning dynamics, evaluation, data quality, and reliable training systems. It is not a claim that this repository already contains AGI or that a large model run by itself would demonstrate general intelligence.

The experiments are designed around explicit questions and measurable outcomes. A lower loss or a larger parameter count is not, by itself, evidence that a capability formed. Results must be interpreted within the model, data, evaluation, seeds, compute, and limits recorded for that experiment.

## On this page

- [Research programs](#research-programs)
- [Signac model and training contract](#signac-model-and-training-contract)
- [Readiness and evidence limits](#what-is-ready-and-what-is-not)
- [Repository map and notebook choice](#repository-map)
- [Getting started](#getting-started)
- [How results are evaluated](#how-results-are-evaluated)
- [Useful references](#useful-references)

## Research programs

### Signac 100M

Signac prepares an experiment at the next useful scale for the An-Ra research line. Its primary M102 architecture has **101,790,080 parameters**. The name “100M” describes the scale class; reports use the exact parameter count.

The research question concerns capability formation and transfer: whether a model can learn reusable operations, follow changes in a query, compose rules, generalize across changes in representation, and preserve useful behavior during later learning. Signac reuses the existing An-Ra model and training contracts. It does not add a speculative “AGI module,” memory system, router, or auxiliary objective and treat the name of that mechanism as evidence.

The architecture package and synthetic engineering checks are implemented. A small CPU update and single-process checkpoint continuation have passed locally. The Kaggle free-TPU notebook is prepared for bounded qualification canaries, but **no Kaggle TPU run is recorded as completed**. Production XLA execution remains closed, and a research training run is not authorized until the data, evaluation, target-runtime, memory, distributed-resume, and durable-output gates pass.

### Cymek

Cymek investigates capability formation, representation and optimization effects, transfer, retention, and continual learning. The repository contains executed experiments as well as preregistrations, diagnostics, and blocked proposals. Their labels matter: a completed engineering run, a completed scientific experiment, an inconclusive result, and a readiness plan are different kinds of evidence.

Use the [next-core readiness snapshot](docs/cymek/next_core/NEXT_CORE_READINESS.md), [research roadmap](docs/cymek/research/NEXT_7_EXPERIMENTS_V51.md), and [experiment evidence ledger](docs/research/EXPERIMENT_EVIDENCE_LEDGER.md) to understand the claims and their limits. These records include branch- and commit-scoped material; check the experiment’s pinned source and date before treating any status as the latest state of every branch.

### Arkenstone and Citadel

Arkenstone contains related capability and learning experiments. Citadel provides evaluation work used by several research lines. They are separate systems with their own data, model, protocol, and readiness requirements. An outcome from one system does not automatically qualify another. Signac’s Citadel evaluation gate remains blocked until a ready, hash-bound evaluator and independent review are supplied.

## Signac model and training contract

The primary model is a dense causal decoder Transformer built from the shared An-Ra `ModelSpec` contract. Its declared geometry is:

| Property | M102 specification |
|---|---:|
| Parameters | **101,790,080** |
| Width and layers | 640 × 20 |
| Vocabulary | 24,576, provisional working choice |
| Query and KV heads | 10 and 5 |
| Head dimension | 64 |
| SwiGLU inner width | 1,600 |
| Native context | 4,096 tokens; not yet learning-validated |
| Attention | Causal grouped-query attention with RoPE |
| Embedding and output | One tied table and full-vocabulary output |
| Dropout and linear bias | None |

Two TPU-shaped alternatives are retained as comparison candidates: an FFN-aligned geometry at 99,332,480 parameters and a tiled geometry at 100,303,104 parameters. Neither is promoted on analytic estimates alone. Target memory, compilation, throughput, and numerical behavior must be measured before choosing a run configuration.

One logical optimizer update consists of accumulated microsteps. Checkpoint cadence is expressed in completed optimizer updates, not microsteps. The production campaign now writes a full checkpoint every **200 optimizer updates**, as well as at completion and other recovery, milestone, or timebox boundaries. This code path has not been exercised on Kaggle TPU; the current canaries are too short to reach update 200.

## What is ready, and what is not

| Area | Current evidence | What it does not establish |
|---|---|---|
| Model specification | Exact parameter accounting and CPU model/update checks | TPU fit, full-context behavior, or scientific capability |
| Local checkpointing | Single-process continuation check and host-side distributed checkpoint tests | Kaggle Output durability or TPU multi-process exact resume |
| TPU qualification | Notebook and fail-closed gates are implemented | A successful TPU run; the target is still untested |
| Training corpus | Development-only synthetic examples and research records | A licensed, audited, deduplicated production corpus |
| Evaluation | Development diagnostics and evaluation plumbing | A ready independent Citadel evaluator or sealed confirmation |
| RSI | Deterministic weakest-axis diagnostics and registered-trial analysis | A learned or autonomous self-improvement capability |
| AGI | No claim supported by this repository | General intelligence has not been demonstrated |

The live gate report is authoritative for Signac’s launch status. A blocked result is expected until its missing evidence is provided; do not bypass a blocker by changing a threshold, treating synthetic data as production data, or relabeling a canary as training.

## Repository map

| Path | Purpose |
|---|---|
| `signac_100m/` | M102 specification, source identity, data/evaluation contracts, custody, and research diagnostics |
| `v5_model/` | Shared An-Ra model implementation and initialization |
| `v5_contracts/` | Model, tokenizer, training, and readiness contracts |
| `v5_data/` | Data manifests, deduplication, packing, cursors, and dataset lifecycle |
| `v5_training/` | Optimizer/update path, distributed coordination, checkpointing, resume, and XLA adapters |
| `v5_evaluation/`, `e0_cognition/` | Evaluation contracts, cognitive task generation, scoring, and diagnostics |
| `docs/signac_100m/` | Signac architecture, evidence, readiness, training plan, and Kaggle instructions |
| `docs/cymek/` and `docs/research/` | Cymek experiment records, research roadmaps, and cross-branch evidence ledgers |
| `notebooks/` | Experiment-specific notebooks; see the Signac operator path below |
| `tests/` | Contract, model, training, checkpoint, evidence, and notebook tests |
| `tools/` | Preflight, model smoke, and research support commands |

Some documents are preserved from sibling branches or earlier experiment snapshots. They are valuable research records, but their branch names, dates, source hashes, and claim limits apply. When records disagree, use the experiment’s pinned source and returned evidence bundle; do not silently combine statuses from different snapshots.

## Which notebook to run

For the Signac 100M qualification task, the current operator entry point is [`SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb`](notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb). It is a bounded qualification harness for three candidate geometries, not a notebook that launches the research training campaign. Its default work is synthetic and short; the optional M102 qualification and XLA-development lanes require deliberate edits to the notebook and still produce engineering evidence only.

The other notebooks are named for separate Cymek studies and target different experiments or accelerator environments. They are not prerequisites in the Signac run sequence. Before rerunning one, use its experiment record and the evidence ledger to check whether its question is still active, what result already exists, and which source snapshot the result belongs to.

## Getting started

Run commands from the repository root with the project’s Python environment and test/runtime dependencies available.

### Check Signac’s static gates

```powershell
python tools/signac_100m_preflight.py --target tpu
```

This is a static review. Until qualified corpus, evaluation, and TPU receipts are supplied, it reports `BLOCKED` and leaves `training_authorized` false. That is the intended fail-closed behavior.

### Exercise the local M102 path

The CPU smoke exercises model initialization, a synthetic forward/backward update, and state mutation. The optional checkpoint check compares a restored next update with the uninterrupted path.

```powershell
python tools/signac_100m_model_smoke.py --candidate m102_primary --device cpu --workdir artifacts/signac_100m/m102_cpu_smoke --verify-checkpoint
```

This small synthetic check is useful for plumbing. It is not a capability experiment, production-data run, TPU benchmark, or memory-fit result. It writes its receipt and temporary checkpoint material under the selected work directory; choose a new directory if you want to preserve an earlier receipt.

### Run the Kaggle TPU qualification notebook

Open [`SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb`](notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb) in Kaggle and follow the [Kaggle TPU runbook](docs/signac_100m/KAGGLE_TPU_RUNBOOK.md). The notebook checks the selected runtime’s PyTorch/`torch_xla` compatibility, expects eight workers/devices, records source and runtime identity, and stops if its frozen topology or required APIs do not match.

The default workload uses synthetic batches and bounded updates. An optional 21-update M102 profile measures 20 steady updates and applies a fail-closed device-memory gate. An optional two-update XLA development profile checks backend and same-session resume plumbing. Both remain engineering-only. Neither is research training, and neither can pass Signac’s data, evaluator, baseline, or production-resume gates.

At the end, the notebook creates a per-run ZIP under `/kaggle/working`, includes receipts and checkpoint files with a SHA-256 manifest, writes a checksum sidecar, and displays a download link. To retain the ZIP after the Kaggle session, download it or save a Kaggle Notebook Version with output saving enabled. The notebook cannot create that version by itself. Read the runbook for the official runtime references, exact save procedure, and remaining target qualifications.

## How results are evaluated

The Signac plan separates engineering readiness from research evidence:

1. **Verify the contract locally.** Confirm parameter accounting, initialization, update semantics, serialization, and deterministic identity.
2. **Qualify data and evaluation.** Establish source and license provenance, split and contamination controls, a runnable corpus, a frozen evaluator, and a baseline that is neither at floor nor ceiling.
3. **Measure the target.** On the pinned Kaggle TPU runtime, measure steady-state updates, peak device memory, parity, distributed synchronization, and exact fresh-worker resume.
4. **Verify durable recovery.** Save outputs and checkpoints, restore them in a fresh process, and verify hashes and the next update after the output round-trip.
5. **Run a preregistered research question.** Freeze the training dose, seeds, metrics, stopping rule, sensitivity thresholds, and sealed evaluation before looking at outcomes.

The Phase-One contract defines the detailed gates and claim ceiling. Synthetic data can test plumbing; it cannot qualify the research corpus or show that a model acquired a capability. Loss curves, model size, source hashes, and successful checkpoint writes are useful engineering evidence, but none alone demonstrates cognition or AGI.

Use these terms consistently when reporting outcomes:

| Term | What it means | What it does not mean |
|---|---|---|
| Engineering smoke | A small check that executable code, state changes, or save/restore works | A benchmark or learned capability |
| Qualification canary | A bounded target-runtime check of participation, synchronization, memory counters, or restart behavior | Production-run authorization |
| Production training | A run admitted by the complete data, evaluation, runtime, resume, and durability gates | A scientific finding by itself |
| Research result | An outcome measured against a preregistered question, controls, metrics, seeds, and evaluator | General intelligence unless the evidence supports that much stronger claim |

## Useful references

- [Signac package overview](docs/signac_100m/README.md)
- [Signac architecture](docs/signac_100m/ARCHITECTURE.md)
- [Signac readiness](docs/signac_100m/READINESS.md)
- [Qualification and training plan](docs/signac_100m/TRAINING_PLAN.md)
- [Phase-One Research Twin contract](docs/signac_100m/PHASE1_TWIN_CONTRACT.md)
- [Kaggle TPU runbook](docs/signac_100m/KAGGLE_TPU_RUNBOOK.md)
- [Signac evidence ledger](docs/signac_100m/EVIDENCE_LEDGER.md)
- [Cymek next-core readiness](docs/cymek/next_core/NEXT_CORE_READINESS.md)
- [Cymek V5.1 research roadmap](docs/cymek/research/NEXT_7_EXPERIMENTS_V51.md)
- [Cross-branch experiment evidence ledger](docs/research/EXPERIMENT_EVIDENCE_LEDGER.md)
