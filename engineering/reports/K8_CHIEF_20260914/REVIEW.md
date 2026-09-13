# Chief review of 2a1e35a: partial implementation, launch not accepted

Date: 2026-09-14. Reviewed implementation: `2a1e35a`. The remote and local BRAMASTRA heads agreed at review start. This review supersedes launch-readiness claims in the preserved [K8_BUILD handoff](../K8_BUILD/HANDOFF.md). The current notebook must not consume the owner's Kaggle allocation until the implementation defects below are repaired. This is a code-readiness decision, not a negative learned result. No experiment was run.

## What is useful and what is missing

The push adds real differentiable scoring helpers, an AMP-oriented trainer subclass, a gated-model prototype, dataset and ledger scaffolding, a CLI and a thin notebook. Those components are useful starting points. However, the handoff overstates their integration. Several required phases are placeholders, the trainer does not consume the objective router, and the documented persistent clock resets in the actual implementation. These are CPU-detectable implementation defects, not merely pending T4 verification.

Chief verification: `python -m pytest tests/test_research_k8.py -q -p no:cacheprovider -o addopts=` returned **17 passed in 7.90 seconds**. The inspected tests use backward checks and refused update boundaries, not successful optimizer updates. Additional [probe.py](probe.py) diagnostics instantiate no models, execute no optimizer updates and use no accelerator. Their [receipt](probe-001.json) reproduces the control-plane failures below. One Luna reviewer independently inspected data and proposer code; the chief checked the actionable findings against source and reproduced the inventory/archive defects.

## K8-R01 — Implement phases and make E0 a real admission gate (blocking)

`bramastra_lab/research/campaigns/worker.py:30–36` returns `blocked_pending_e0` unconditionally for E1/E3/E4/E5. There is no learned implementation behind that status. E2 has no dispatch branch and inherits `completed` with zero work; E6 similarly does no export. The runner does not validate successful E0 receipts before dispatching later phases. In the diagnostic, both fake E0 workers fail, every later phase is still invoked, and the runner exits zero.

Implement the E1–E5 phase executors and E6 output verification, including paired parents, seeds, datasets, frozen targets and result identities. A blocked or failed required job must yield a non-successful campaign result and exit code. Completion requires a valid phase receipt with actual outcomes, not merely a worker returning a status string. E0 success must be bound to both physical GPUs and the current source/data/configuration before any learned phase can start.

Acceptance: a production-runner diagnostic with failed E0 must record no E1–E5 dispatches; a missing phase handler must refuse execution. Then demonstrate the real handler graph with test doubles only at the expensive model execution boundary. Real learned evidence remains deferred to the owner's authorized run.

## K8-R02 — Preserve allocation and job identity across restart (blocking)

`campaigns/supervisor.py:114–125` uses `INSERT OR REPLACE` with a newly computed deadline on every allocation call. The same-allocation diagnostic advances the fake clock by 60 seconds and observes a 60-second deadline extension. `runner.py:29–31` also includes mode in allocation identity, so e0/full do not even identify the same allocation. `runner.py:39–40` indexes `row[5]` from a one-column SELECT; the second invocation after reservations exist raises `IndexError`. The reset occurs before that failure.

`reserve` admits the same job ID twice and does not enforce a unique job identity. Its capacity check sums device reservations against one wall-time span, with no physical-device exclusivity model; sequential reservation tests do not prove concurrent correctness. Failed worker handling records zero updates, exposure and device time regardless of work already attempted. Closed phase presence is also not proof that all jobs succeeded.

Create-or-validate an immutable allocation independently of mode; reject changed source/data/budget identities and preserve the original deadline. Add unique idempotent jobs with compatible retry semantics, per-device occupancy and transactionally checked capacity. Record partial work from durable worker events on failure. Resume from successful individual job receipts, not phase names alone. Close database resources on every path and implement verified stale-lease recovery.

Acceptance: same allocation and e0-to-full transitions retain the exact deadline; conflicting identities fail; duplicate IDs cannot produce two executions; two valid simultaneous GPU reservations are admitted while overlapping jobs on the same GPU are refused; failed jobs retain consumption; restart never skips unfinished work. Use simulated time and real ledger connections locally.

## K8-R03 — Build actual parallel workers and enforce phase deadlines (blocking)

`runner.py:53–69` calls every worker synchronously in nested loops. `worker.py` says that real execution spawns a process, but contains no subprocess launch or GPU-visibility assignment. The declared phase caps are descriptive fields; workers receive only the campaign deadline. There is no enforcement of the minute-450 training cutoff or supervisor hard-stop mechanism. If the missing handlers were filled in without changing this structure, the intended two-GPU schedule would still not be implemented.

Use actual isolated subprocesses with GPU visibility set before torch import. Run each prescribed paired slot concurrently, join it within its phase boundary, and preserve the explicit counterbalancing. Persist absolute phase/job deadlines derived from the original start. Stop training by minute 450 and reserve export through 480. Verify physical UUIDs, not only worker-local CUDA indices.

Acceptance: simulated workers prove overlap of the two paired jobs and separation of sequential slots; a hanging worker is terminated and accounted; an overrun cannot borrow unallocated time from export. The test should run the production process-supervision code without model training.

## K8-R04 — Repair the canonical learning boundary (blocking)

`learning/k8_trainer.py:100–135` implements answer loss only. It does not call `route_window`, ignores `pair_rows`, and does not consume the configured world/action/value/pair objectives. `finalize_update` divides every gradient by answer-token count, which cannot implement independent denominators for the other objectives.

At lines 180–182, it calls `optimizer.step()` and then constructs `StepReport`, which is not imported into this module. Therefore an actual step can mutate weights and increment its counter before raising `NameError`; the runner's failure path would then report zero updates. The method unscales AMP gradients but never calls `scaler.step`/`scaler.update`. Its returned learning rate is calculated but never assigned to optimizer groups; this is not the promised scheduled update. `_admit_update` also permits a missing allocation.

Integrate the configured differentiable terms through one normalized update boundary. Implement the complete AMP lifecycle, finite-skip behavior, actual optimizer LR application and durable attempted/committed accounting. Reject unallocated campaign updates. Do not fix only the missing import and declare I01/I02 complete.

Acceptance: production-path backward tests prove each term's attributable gradient and window-normalization equivalence. A no-op optimizer test double can exercise report construction, schedule application, scaler sequencing and failure accounting locally without learning. Real parameter-change and AMP checks belong to E0. A failed report publication after a committed step must not erase the step.

## K8-R05 — Replace the invalid E0 comparison (blocking)

`worker.py:63` builds the tiny profile, not the specified full K8 profile. The uninterrupted branch runs answer plus manual action/value/world backward calls, but the interrupted branch runs only answer loss. Those are different treatments. The auxiliary gradients are also unscaled while token gradients are scaled under AMP. The comparison uses a sum of parameters, which is not a sufficient model-state identity. The supposed fresh process is another object in the same process. `started` used at line 141 is not defined in `_run_e0`.

Use the same canonical update function, initial state, data sequence and objective configuration in both paths. Save an actual checkpoint, restore it in a separate process and compare full state, counters and the next batch under the declared numerical policy. Measure the full campaign profile and heaviest active arms for calibration. Persist success receipts and failure consumption for both workers. Retain the existing E0 update and wall-time caps; no local learned probe is authorized.

Acceptance before GPU: exercise the orchestration and state comparisons with deterministic non-learning doubles, prove both paths use identical treatment definitions, and reject a deliberately changed optimizer/cursor/target. Actual GPU equality and throughput remain pending owner launch.

## K8-R06 — Generate executable, sufficiently diverse data (blocking)

The rule generator's deduplication key has at most 24 public configurations: two variable counts, four rule types and three distractor counts. `_mechanisms_for_family("rule-inquiry", 4096, 8609)` returns **24**, not 4096. Even 32 sealed confirmations are impossible under this key. The launch's explicit large CLI counts do not repair the generator. IDs restart at zero in each pool; cross-pool equivalence is not rejected. `validate_bundle` checks hashes and some counts but does not independently establish split disjointness or executable semantics.

At `data/k8_bundle.py:239–250`, the verifier is assigned but never called and `verifier_consistent` is hardcoded true. Four trajectories differ in labels but contain no actual action/result history. The held-out tool builder never enables `held_out_composition`, so both tool pools use the same composition generator. Meta support/query entries are unresolved strings rather than concrete executable examples. The output does not establish the intended learning/evaluation inputs.

Expand the mechanism family meaningfully, or stop preparation with an explicit insufficient-inventory error; do not meet the count by adding UUIDs or surface names. Materialize actual teacher/exploration trajectories, action/value/world supervision, verifier outcomes, tool compositions and meta support/query/protected cases. Use global mechanism equivalence grouping, reproducible source identities and independently recomputed validation. Teacher answers in separate targets and publicly specified programs are not inherently leakage; prove that evaluator/private data are excluded at the actual model-input boundary.

Acceptance: full requested inventory is generated and verified before GPU launch; wrong answers, renamed cross-pool duplicates, unresolved meta references and undersized pools fail qualification. Tool-heldout tasks must differ in actual execution structure. A miniature fixture test with `min_confirmation=1` is useful but does not qualify the full bundle.

## K8-R07 — Preserve architecture and proposer semantics (blocking)

`models/gated.py:61–71` reuses blocks after `decoder.forward_hidden`, which already applies final normalization (`bramastra_lab/model.py:235`). The specification requires reuse before final normalization. The override drops `segment_ids` and other IntegratedModel forward options, and its reused block call drops masks. `learning/k8_scoring.py:49` calls `model.decoder.forward_hidden` directly, bypassing gated reuse. Zero-gate equality can pass despite these treatment and interface defects.

Preserve the complete IntegratedModel contract, packed-segment isolation and head outputs; place reuse at the specified location and route all relevant scorers through it. Verify the extra path independently from gradients through the ordinary shared-block pass. Preserve device and parent mode during migration and reconstruct sharing during restore.

`metalearning/dispatch.py:70–74` calls an unimported `content_identity`; the empty archive identity diagnostic raises `NameError`. The dispatch accepts a caller-supplied compiled identity without establishing equivalence to the selected program. M0/M1 representative programs are identical even though their LR semantics differ, and arbitrary program JSON is mapped into the small method vocabulary by a loose gradient-transform check. E5's actual training/archive/successor executors are absent under R01.

Bind the exact selected token/program, immutable archive, compiled semantic recipe and applied trainer state. Reject mismatched caller recipes. Materialize P0/P1/P_fixed and fixed-anchor lineages with choices captured before fresh confirmation outcomes. Acceptance requires meaningful non-learning forgery/dispatch checks and later actual learned evidence; parsing a string does not complete I04.

## K8-R08 — Correct packaging and the handoff (blocking for owner delivery)

Notebook command strings contain literal backslash-n fragments in shell invocations rather than valid line continuations. Replace these fragile shell cells with checked `subprocess.run` argument lists using the current interpreter, or verified equivalent notebook calls. Resolve repository and data paths explicitly and validate required files before starting the allocation. Do not assume a particular cloned directory exists.

`campaigns/k8.py:112–120` exports only the ledger JSON. It does not export the required restorable checkpoint payloads, per-phase results, comparisons, frozen protocol or source/data artifacts. Complete and validate the result bundle. Setup/prepare/validate must fail before expensive work if requirements are absent.

The current handoff asserts router consumption, frozen-model E2, real parallel dispatch and launch readiness that source does not support. Preserve it as the original submission, then add a correction and new evidence-backed disposition. Tests called `VerticalSliceTests` currently use disconnected inputs and accept fallback decision origins; replace the claim with a trace through actual connected data and caller identities.

## Direct instruction to the implementation agent

Read this review before continuing. Complete R01–R08 alongside I01–I06. First fix the fail-closed control plane and trainer boundary, then E0 orchestration, qualified data, architecture/proposer semantics, real phase handlers and packaging. Work efficiently with one integrator and bounded Luna assignments on nonoverlapping files. Do not spend the owner's GPU session diagnosing the CPU-reproducible defects above.

Return `engineering/reports/K8_REPAIR_20260914/HANDOFF.md` with one disposition per R01–R08, exact production callers and focused negative-control evidence. Keep GPU-only verification explicitly pending. Run the existing focused suite plus the necessary regressions; no local optimizer updates, no new allocation and no historical-ledger reset. Commit and push repaired source and compact evidence. Chief acceptance is required before describing the notebook as launch-ready.
