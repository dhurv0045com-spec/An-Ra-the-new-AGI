# K8 branch review and notebook implementation order

Source inspected: `edee72726706e0f8f46ec98bb3c979ebc2930233`. The chief and one Luna reviewer inspected actual consumers, not just new file names. The chief ran:

```powershell
python -m pytest tests/test_research_accounting.py tests/test_research_evaluation.py tests/test_research_master_m05_m06_m12.py -q -p no:cacheprovider -o addopts=
```

Result: **62 passed in 8.06 seconds**, one test warning about converting a gradient-bearing scalar to a Python float. These were non-learning/gradient checks with no optimizer updates. This does not certify CUDA execution, canonical multi-objective integration, or a learned proposer. The older F1–F6 changes are present and covered partly by these checks; E0's actual-path checks remain required.

## New milestone and authority

Build the notebook/runner that executes [experiment.md](../../../experiment.md), then hand it to the owner. The owner has newly allocated one future two-T4 Kaggle session of at most480 elapsed minutes. This supersedes the earlier no-accelerator instruction **only for that named owner-launched campaign**, not for local development or arbitrary additional runs. Keep the historical CPU ledger unchanged.

The broad M00–M24 program remains the architecture reference. Implement the subset necessary for K8 as I01–I06 below; do not delay the campaign for unrelated optional modules, a natural-language corpus, a full source-mutating agent or a much larger model. Do not silently substitute fixture behavior for a required learned component.

## I01 — Actual CUDA trainer and allocation-aware runtime

Owner: integrator. Paths: `learning/trainer.py`, `runtime/resume.py`, config, runtime allocation adapter and shared CLI wiring. Extend the existing trainer, not a notebook-only optimizer. Propagate device and precision to model, batch, optimizer restore, diagnostics, masks and targets. The present train/restore path defaults to CPU and must change explicitly. Keep a working CPU path.

Implement one update boundary for normalized multi-objective gradients, AMP unscale/clip/step and skipped-update accounting. Store scaler state. A skipped nonfinite step does not increment successful optimizer updates or silently consume a schedule step, but its time/exposure remains charged. Use the installed PyTorch version's actual APIs; version-pin source and requirements after validation.

Acceptance before notebook: fake-device/config dispatch checks, state-schema tests, disabled-feature tests, interruption/failure accounting, and rejection of missing campaign authority. E0 on Kaggle proves real GPU execution, gradients and fresh-process resume within its allowance. Until E0 passes, the runner cannot start E1–E5.

## I02 — Differentiable objectives and canonical public rendering

Owner: model/objective integrator, optionally a bounded Luna implementation package on nonoverlapping adapter files. Paths: `learning/router.py`, `experience/supervision.py`, `models/decisions.py`, `models/world.py`, episode preparation and trainer consumers.

Add differentiable batched action/value/transition scoring APIs; retain separate inference wrappers using no-grad. The current helpers create CPU tensors and return detached results. Inference-only scores cannot train the heads. Every enabled term must have an actual loss tensor, correct window denominator and verified gradient path. Positive eligibility with a missing term is an error, not a logged omission. Replace gratuitous CPU/double temporary tensors with device-correct reductions of the declared precision.

Use exactly one public history/goal/action/feedback renderer for preparation, prediction and cognition. Score outcome target spans only, include required EOS and deduplicate identical public support outcomes before normalization. Candidate actions are isolated branches with consistent masks and order-equivariant scores. Predictor/proposer identities must bind real checkpoint/source/schema identities, not only class name and parameter count.

Acceptance: bounded backward checks with different masks/lengths, no target leakage, exact count normalization, independent candidate permutations, duplicate support invariance and no CPU tensor leakage. Existing isolated-module tests are not enough: a fake public training job must route through the real shared trainer interface.

## I03 — Qualified K8 data and actual cognitive/tool collection

Owner: data/environment executor. Paths: K8 generator/compiler under `data/` or `environments/`, `collection/`, executive adapter and `orchestration/experience_cycle.py` where appropriate. Produce the dataset described in experiment.md with independent verifiers and grouped splits; preserve all generator/code identities. No giant downloaded corpus is required.

Implement a learned scorer adapter that instantiates the shared model/checkpoint in the actual executive/session path. Train/infer workspace formats must agree. The reference belief updater is a control; it must not supply hidden solutions to the learned arm. Connect real tool actions/results and costs to observed receipts, not imagined traces. Bound action enumeration, tool arguments and file outputs.

Acceptance: deterministic generation, known-mechanism dedup/split checks, verifier agreement on exhaustive small fixtures, context-fit audit, failed-trace eligibility, no hidden keys in tokens and actual scorer-call capture. A missing learned scorer cannot silently fall back to a scripted oracle. Build and hash the data bundle before the owner starts GPU time.

## I04 — Gated block reuse and measured RSI proposer

Owner: architecture/meta-learning executor; shared decoder changes mediated by integrator. Paths: `model.py`/wrapper, architecture config/migration, `metalearning/` proposer/runner adapters.

Implement the exact E4 two-gate block-reuse variant and a disabled matched control. Prove parent equality at zero gates, gradient access to gates and shared blocks, changed architecture identity and full restore. Keep training/inference compute counters explicit; reused blocks cost real compute.

Implement actual checkpoint-origin method generation for E5, not `run_fixture_generation`. Bind generated bytes to parsed methods and immutable pre-decision archive snapshots. Dispatch every measured adaptation trial through the canonical trainer. P0 must choose the method applied to its own P1 update; P_fixed gets the same archive/time under M0. Final method choices are captured before confirmation outcomes exist. All candidate methods share the declared adaptation anchor.

Acceptance: fake-model fixture tests through the actual integration seams, archive leakage rejection, forged origin rejection, identical-start trial validation, P0/P1/P_fixed lineage and method-state checks. Learned E5 evidence is collected later in the authorized notebook. No fabricated method-quality labels or host-chosen “model” proposals.

## I05 — One supervisor, two workers, one persistent allowance

Owner: runtime integrator. Concrete target entry point: **`python -m bramastra_lab.research.campaigns.k8`**, to be implemented with `prepare`, `validate`, `run`, `summarize`, and `export` subcommands. `run --mode e0` runs only E0; `run --mode full` executes the gated campaign. These commands are a target interface, not currently existing CLI claims.

Implement `campaigns/k8.py`, `campaigns/supervisor.py`, and a versioned ledger. Use one supervisor as the only accounting writer, with a transactional SQLite journal or an equivalently tested serialized event store. Required records: allocation ID/source hash/absolute deadline; job ID, parent job, worker/device, phase/arm/seed, reservation; event sequence, started/completed/failed status, committed updates, attempted updates, supervised exposure, device/time costs and checkpoint identity. Unique job IDs make retries idempotent.

Reserve a bounded time slice before worker execution; close it with actual consumption even on failure. Parent costs aggregate child events without double counting. The supervisor can terminate workers at the hard deadline; workers cooperatively checkpoint before their earlier deadlines. Persist UTC start/deadline and elapsed accounting so kernel restart cannot reset the allowance. A live supervisor owns an exclusive lease; recovery must prove the prior writer is gone.

Launch subprocesses with explicit GPU visibility before importing torch; use spawn semantics and record physical device identity plus worker-local device index. Each worker sees its assigned GPU and cannot run both models on GPU0 accidentally. The two-device inventory is checked by the supervisor. No DDP, shared model tensors or nested unaccounted workers in this campaign.

Acceptance: deterministic simulated480-minute campaign covers reservation collisions, failure, timeout, retry, restart and export with zero optimizer work. Show that two concurrent requests cannot reserve beyond remaining capacity. A future full notebook cannot enter E1 until the required successful CUDA E0 receipt exists for both workers.

## I06 — Notebook, protocol freeze and owner handoff

Owner: integrator. Deliver `notebooks/bramastra_k8.ipynb`, minimal requirements/source archive, generated Kaggle input bundle and a runbook. The notebook exposes clearly separated setup/validate, E0-only and full-run cells. It invokes repository modules and prints the schedule and allocation before starting. Resolve mounted input/output directories from inspected notebook state/config; do not assume a particular uploaded dataset slug.

The checked-in campaign manifest is a design allocation, not the final launch identity. Generate a frozen launch manifest after preparation and E0 timing, containing complete source/data hashes, selected case inventory, common update targets, hardware, precision, every treatment setting and all phase deadlines. Do not choose these values from learned measurement scores. If E0-only has already run, full mode uses the remaining same allocation and its valid receipt; a new mode cannot duplicate E0 and erase its cost.

Return `engineering/reports/K8_BUILD/HANDOFF.md` with I01–I06 disposition, local focused checks, unresolved readiness gates and exact Kaggle instructions. Do not claim the notebook has run merely because it exists. The owner executes it; result manifests/reports are then reviewed for the next architecture decision. Commit and push BRAMASTRA normally, preserving user work and excluding large data/weights.
