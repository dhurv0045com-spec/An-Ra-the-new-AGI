# Planthir reliability review and implementation handoff for Gemini

This handoff records the present review and the next engineering work. The reviewed starting point was `de4bce0ce7516a9e7d190a86de749ce64263e745`; start future work from the **latest review commit**, not that historical baseline. Read root `AGENTS.md`, this document, and the current implementation and tests first. `purpose.md` preserves the original architectural proposal and historical claims, some of which this review supersedes. The current runner is a qualified **V8-specific** path, not a verified universal engine. No review can establish zero errors, universal applicability, or scientific validity for all future experiments.

The review made concrete changes to admission, scientific interpretation, execution metadata, and evidence handling. The final curated results and exact test outcomes belong in the accompanying owner-maintained results record; do not invent counts, GPU qualification, or performance gains from this handoff. Historical reports describe their own runs and must not be silently rewritten as results of this review. Owner GPU qualification remains pending.

## Present controls and boundaries

**Implemented in the reviewed code:**

- The V8 cartridge validates the declared factor against arm configuration and checks unchanged recipe, dataset, order, policy and model fields. Its coverage is bounded by the schema and V8-specific factor model, not proof that every confounder is absent. Controls and treatments must be explicit; declared capacity tolerance is checked where applicable.
- Admission checks source SHA format, actual prepared-data identity and V8 D1 provenance, configurations and eval/probe splits. A production run requires its declared SHA to match a clean checkout. `RUN_IDENTITY.json` binds a work directory to canonical run configuration and execution parameters. A matching identity and explicit `resume=True` are required to resume; a mismatched or unowned work directory fails closed. Process-held writer locking and durable run status prevent ordinary concurrent reuse and record attempts.
- CPU execution is a **tiny canary** only, capped at 100,000 parameters, 32 samples and 8 updates. These bounds do not establish training convergence or GPU scalability. Expensive research training is not an appropriate local validation step for this review.
- Checkpoint milestones, update journal handling and resume guard against several ordinary interruptions and inconsistent identities. Exception-based restoration and journal tests are useful, but do not demonstrate recovery from power loss, operating-system kill, device reset, or failure at every filesystem boundary.
- Funnel calibration now scores last-action probability against last ideal-action correctness rather than episode correctness. Inputs are checked for compatible lengths and valid probabilities. The first tool stage checks the **correct** required tool rather than merely any tool call. Geometric isotropy uses the spectrum of normalized representation directions, so antipodal points on one axis do not masquerade as isotropic. The score is sample-size and dimension limited; it is not a general proof of representation quality.
- Probe training and evaluation reject shared world lineage and V8 canonical graph signatures; inexact V8 canonicalization is rejected. This is a useful split guard, not a guarantee of distributional independence or of disjoint task templates. Linear separability is not proof of how the model uses an encoding, and failure of a linear probe is not proof that information is absent.
- Oracle and subgoal confidence intervals default to unavailable. A caller may explicitly assume independent linked-world clusters, which resamples parent/pair/base-linked variants as units. Such intervals are conditional on the provided worlds and fixed model, and do not quantify seed or training-run variation. Oracle injection changes the context and tool availability together: an observed difference does not identify a tool-acquisition mechanism. The interpreter no longer promotes a survived falsifier or these associations into a confirmed mechanism. Thresholds, finite suites, and multiple comparisons still limit claims.
- F-cap and U ablation code restores modified weights or specifications in `finally`; regression tests exercise exception paths. A null ablation on a finite suite does not establish global bypass, and a drop alone does not establish a general mechanism. A one-checkpoint result cannot establish scaling or training-order laws.
- Evidence paths distinguish compact tier summaries/capsules from fuller raw sidecars. Checkpoint/archive identity and archive safety checks have been strengthened, and default delivery checks same-device, full-weight reload parity on sampled observations. Prediction/hidden parity on a finite sample is a useful integrity check, not equivalence on all inputs. Compact representations are lossy; treat their parity and probe results separately from the full checkpoint.

The five named checks are an execution lifecycle, **not five prelaunch gates**. Instrument and design admission occur before training; gradients, training/checkpoint health, diagnostics, packaging and parity require later execution. A certificate summarizes checks that actually ran and is not a research conclusion or a replacement for inspecting defects, assumptions, counts, and raw evidence. Avoid labels such as “universal complete,” “confirmed cause,” or “launch safe” when the evidence does not support them.

## Priority 1: backend contract and reproducible identity

**Pending.** `CartridgeRunner` currently explicitly rejects non-V8 kernels and depends on V8 specs, data, trainer, loss, policies and checkpoint reconstruction. A generic observation protocol does not make training generic. Define a versioned typed backend interface for model/config construction, canonical factor declaration, data identity and batching, loss and update, policy creation, telemetry, checkpoint save/load and device capabilities. Keep the V8 implementation behind an adapter and reject unsupported capabilities before work begins. Protocol version and all backend-controlled configuration must contribute to identity.

Build a genuinely independent non-V8 adapter and a small end-to-end fixture. Include X1 R/D2 grouped-loss behavior: define the group identity, reduction, masking, denominator and gradient semantics explicitly, compare it with a hand-computed reference, and verify resume preserves group ordering. Do not treat a V8 wrapper renamed as another backend as this deliverable. Acceptance: both V8 and non-V8 pass the same contract tests; incompatible versions, unsupported operations, mismatched model/data/loss specifications and corrupted checkpoints fail before updates; grouped loss and gradients match reference including empty/masked groups. Report any unsupported paths instead of emulating success.

Strengthen content-addressed provenance for source, dirty state, dependency lock/environment, full normalized config, backend version, dataset bytes and preprocessing/tokenizer, ordered split membership, seed and device/runtime. Identity must change on a relevant input change. Validate the mapping at admission and restore, not only the names supplied by the caller. Preserve a readable identity manifest and a canonical digest; test collision-resistant canonical serialization, changed files and changed order. This is an engineering identity, not a claim that the inputs are scientifically representative.

## Priority 2: crash consistency and publication

**Pending.** Extend existing identity/status/journal controls with genuine subprocess interruption tests. Kill a worker at controlled points during checkpoint component write, manifest commit, journal append, status update and result publication; restart with matching identity and verify recovery against an uninterrupted reference including optimizer, RNG, update count, data order and outputs. Test truncated/corrupt bytes, stale locks, incomplete attempts and identity mismatch. Explicitly document filesystem and platform guarantees; simulate exceptions as well as actual process termination. A kill test cannot by itself prove power-loss durability on every filesystem.

Publish outputs using an atomic `CURRENT` pointer to an immutable, fully verified generation. Stage a complete generation in the destination filesystem, fsync files and directory as supported, verify referenced hashes and only then atomically replace `CURRENT`. Readers must resolve one generation and never mix files from two generations. On failure preserve the previous valid pointer and quarantine partial staging; clean up only with explicit retention rules. Acceptance: injected kill at each publication boundary yields either the old complete generation or the new complete generation; readers never observe a hybrid, and repeated resume is idempotent. Verify platform-specific rename and sync semantics rather than assuming them.

## Priority 3: measured resource controls and two independent T4 workers

**Pending.** Replace rough estimates with measured peak allocated/reserved GPU memory, wall time, checkpoint and output bytes, free disk headroom, and per-phase durations. Record hardware, driver, framework, warmup and synchronization method. Implement enforceable time and disk quotas with safe stop/checkpoint behavior and admission limits. GPU memory quotas need an explicit supported enforcement mechanism; monitoring and warnings alone must be labelled as such. Check failures and cleanup paths, including out-of-space during publication. Preserve the tiny CPU caps above.

Design a two-T4 qualification using **independent workers**, each with its own process, device assignment, work identity and output generation. Confirm both devices are actually used, isolation and lock ownership are correct, failures do not corrupt the other worker, and the coordinator reports partial failure accurately. Capture measured peaks, throughput and wall time, not assumed linear scaling. Start with bounded owner-approved GPU canaries and then qualify the intended workload; do not run a full local research job as an infrastructure test. Acceptance requires durable per-worker receipts and independent restore/parity on the target hardware. This qualification is pending and this document is not evidence that two T4s have been tested.

## Priority 4: experiment design and statistical units

**Pending.** Replace stringly typed contrast declarations with a typed factor schema, explicit treatment/control levels, permitted differences, measured parameter/capacity policy and preregistered primary outcomes. Reject an undeclared mismatch, missing treatment, multiple unintended changes or a factor that does not actually vary. Include dataset, ordering, augmentation, compute budget and backend semantics. Acceptance: adversarial fixtures mutate each declared and held field and obtain deterministic, localized failures; a valid matched design survives. Keep observed checks separate from the broader claim of no confounding.

Represent cohorts and dependencies explicitly: base world, parent, variants, graph family, task template and seed/run pairing. Validate transitive grouping and split disjointness before fitting and evaluation. Preregister the unit of randomization and analysis; for independent trained runs, pair corresponding seeds and estimate uncertainty across independent seed pairs, not by treating thousands of correlated episodes as independent model replications. Record the cluster and resampling assumptions, denominator, missingness, intervals and multiple-comparison procedure. Default to descriptive estimates when independence cannot be justified. Validate with correlated-variant simulations, one-cluster cases, unequal cluster sizes and missing pairs.

Teachers and oracle labels must be **nonanticipating**: at each step the teacher may use only information legally observable at that step, except in explicitly labelled oracle-only diagnostics. Audit state mutation and hidden truth leaks with counterfactual hidden values that preserve observable history. Compare independent reference rollouts and record what information each teacher can see. A successful oracle establishes only behavior under that oracle condition, not the mechanism of ordinary success or failure.

## Priority 5: evidence, cloud custody and replay

**Pending.** Full raw evidence can be large. Stream arrays, traces and checkpoint members to chunked, hashed files rather than materializing all evidence in memory or silently truncating it. Keep bounded capsules with explicit row selection, dtype, omissions, schema and links to immutable full-raw objects; distinguish quantized capsule results from full-precision results. Enforce quotas at write time and fail visibly if required evidence cannot be retained. Acceptance: large synthetic evidence stays within measured memory limits; hashes, row alignment and references verify after restore; tampered, missing, duplicate or traversing archive members fail; every compact claim can be traced to its raw source. A bounded result archive must not imply all raw data is embedded.

Create a pinned cloud notebook and lockfile with immutable source commit, data manifests, reproducible commands, environment/hardware capture, artifact hashes and a custody receipt identifying who produced, transferred and verified each artifact. Include a read-only independent replay procedure that reconstructs from full weights and data, recomputes preregistered metrics, verifies hashes and reports tolerances and discrepancies. A second operator or independent environment must perform and sign that replay; a self-reported hash or same-process reload is not independent validation. Preserve original receipts and add corrections as new records. Any cloud credentials and private data remain outside notebooks and Git.

## Priority 6: measure productivity rather than assert it

**Pending.** The historical hardcoded 82% productivity or token-saving figure is not an experimental measurement and must not be presented as one. Design an actual six-task paired study comparing the current workflow with a defined baseline on the same tasks, with preregistered ordering or counterbalancing, independent raters where feasible, and fixed success criteria. Record end-to-end human and agent time, tokens, compute, retries, defect count, correctness and total cost. Preserve raw task-level data, exclusions, uncertainty, learning/order effects and failed tasks. Report paired differences and uncertainty; do not generalize from six tasks to all work. Acceptance requires a reproducible analysis script and an honest outcome even if the new workflow is slower or less accurate.

## Suggested implementation and review sequence

1. Commit the typed backend contract and identity schemas with negative contract tests; separately commit V8 migration and a truly non-V8 X1 R/D2 grouped-loss adapter and tests.
2. Commit process-kill recovery and atomic-generation publication, with fault-injection receipts. Keep production claims limited to tested filesystems.
3. Commit resource measurement and quotas, then independent-worker orchestration and bounded qualification harness. Record owner-run GPU qualification separately after it happens.
4. Commit typed contrast/cohort/seed rules and nonanticipating teacher audits with adversarial tests. Revisit existing interpretation thresholds and report unresolved evidence rather than automatically confirming mechanisms.
5. Commit streaming raw evidence, custody manifests and independent replay tooling; separately provide a pinned cloud notebook once its environment and permissions are actually available.
6. Commit the six-task study protocol and analysis before collecting observations; publish results and raw provenance in the appropriate curated location afterward.

At each step inspect the current worktree, preserve user changes, make the smallest coherent commit, run the focused contract and regression tests, and append the required concise dated progress entry. Verify compatibility and migrations for saved identities and bundles; either support old schemas explicitly or reject them with a clear diagnostic. Keep implemented controls, proposed upgrades, CPU test observations and owner GPU research results separate. The final handoff should name actual changed files, tests and remaining uncertainties, and avoid claiming completion merely because a certificate exists.

## Execution map: where to make the changes

Use the existing files below as the starting point. Proposed new file names are suggestions; preserve the public interface unless a versioned migration justifies a change.

| Work package | Current implementation to inspect | Deliverable and regression boundary |
|---|---|---|
| Backend abstraction | `anra_v5/planthir/cartridge.py`, `contracts.py`, `adapters/v8_adapter.py`; V8 `training.py` | Add a small `backend.py` protocol, move V8 construction behind it, retain current V8 receipts. Test a truly different backend and grouped-loss reference. |
| Recovery | `anra_v5/planthir/execution.py`, `v5_training/checkpoint.py`, `checkpoint_lock.py` | Extend existing journals/locks; do not replace the checkpoint engine. Add subprocess kill fixtures with a supervisor outside the worker. |
| Publication | `cartridge.py` final staging/backup swap; `evidence.py` export and verification | Introduce a versioned immutable-generation reader/writer together. Test old bundle discovery and every interruption boundary. |
| GPU scheduling | `CartridgeRunner.run`, runtime identity and cooperative deadline | Add coordinator/worker commands with explicit device IDs; retain one writer per work directory. Do not embed orchestration solely in notebook cells. |
| Design and inference | `audit_experimental_design`, `causal.py`, `interpretation.py` | Introduce typed contrast/cohort records, preserve competing hypotheses, and test false-positive behavior using known correlated fixtures. |
| Measurement | `observatory.py`, `instrument.py`, V8 `evaluation.py` | Preserve observation/action parity, actual gradient checks, RNG restoration and action-budget checks. Add synchronized per-phase cost receipts. |
| Artifact custody | `evidence.py`, `checkpoint_intelligence.py` | Keep streamed checkpoint files and raw sidecars. Add chunked arrays, cross-archive reference verification, and independently executed replay. |
| Operator surfaces | root `agent.md`, `purpose.md`, `research/planthir/`, `.github/workflows/elendril.yml` | Update examples alongside API changes. Pin the cloud notebook only after its implementation commit exists; never invent a future SHA. |

Each work package must produce code, a focused regression, a compatibility decision, and a short evidence record. Write down measured limitations even when the package passes. Avoid simultaneous rewrites of the trainer, exporter and evaluator: change one contract boundary, qualify it, then proceed.

### Reproduction commands and release discipline

Inspect `git status --short`, `git branch --show-current` and `git log -5 --oneline` first. Continue on `planthir` or an isolated review branch based on it; do not merge all of Elendril or Cymek. Preserve the original historical reports. Read `Experiment Result/Planthir/reliability-2026-11-16/RESULTS.md` for the actual review evidence.

In an already configured development environment, run the changed focused test module first. The review's broader commands are:

```text
python -m pytest tests/test_planthir_runner_reliability.py tests/test_planthir_execution_reliability.py tests/test_planthir_evidence_reliability.py tests/test_planthir_instrument_reliability.py tests/test_planthir_scientific_reliability.py tests/test_planthir_universal_infrastructure.py -q
python -m anra_v5.planthir calibrate
python -m tools.verify_elendril_evidence
python -m pytest -q
```

Use the repository's pinned environment declarations and inspect CI before changing dependencies. Do not reinstall a CUDA stack on the owner's local machine as a routine test. The full test suite includes tiny mechanics fixtures; it does not authorize research training. Save validation logs outside tracked paths and commit only concise Markdown/essential JSON evidence. A passed local suite is not a passed remote CI run: record each independently and identify the exact commit.

For cloud qualification, first build a single-worker short canary that trains, checkpoints, interrupts, resumes and exports; compare its final state with an uninterrupted control. Next run two isolated canaries concurrently. Only then propose the bounded S20 workload. The notebook must disclose expected cost, preserve progress after each completed unit, save failure receipts, expose both compact and full archives, and make owner-authorized launch semantics explicit. It must never turn a failed mandatory export into a successful campaign. Download and independently verify saved Kaggle output before claiming durable delivery.

### Decision boundaries for the next agent

- **Proceed now:** typed contracts, non-V8 adapter, identity migration, subprocess recovery fixtures, atomic publication, streaming, report corrections and bounded CPU validation.
- **Prepare but do not claim qualified:** owner-run GPU scheduler, target-hardware resource limits and cloud notebooks. No review execution used a GPU research campaign.
- **Do not promote:** causal root causes from oracle gaps, productivity from compressed JSON size, arbitrary-backend support from one adapter, or intelligence from infrastructure tests.
- **Stop a launch, preserve evidence and report the exact defect:** corrupted identity/checkpoint, split contamination, missing required export, failed restore parity or unsupported backend/loss semantics. Fix the defect; do not weaken its assertion.

The first scientifically useful downstream use should compare a registered mechanism with its matched control using the same qualified backend and independent seed pairs. Infrastructure qualification and mechanism research are separate outcomes. A failed model can yield valuable evidence; an untrustworthy instrument cannot establish that evidence.
