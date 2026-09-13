# BRAMASTRA K8: cognition, tool transfer, architecture and recursive learning

Chief experiment design, 2026-09-13. Inspected implementation: `edee72726706e0f8f46ec98bb3c979ebc2930233`. **Status: experiment specified; notebook/integration work required; no campaign results exist.** This is the next executable milestone after the M00–M24 build, not another expansion of its module list.

## 1. Objective and new authorization

The owner authorizes preparation of a substantial Kaggle experiment using **two T4 GPUs for 7–8 elapsed hours**, and will run the notebook. This packet allocates **one campaign, maximum 480 elapsed minutes and 960 provisioned GPU-minutes**, including startup, failed attempts, profiling, evaluations and export. It does not authorize additional sessions, paid compute, local training, or eight hours per experiment. Build the notebook first; the owner's launch starts the allocation clock.

The old local smoke ledger remains unchanged at 206 CPU updates. Create a separate K8 campaign allocation and cumulative ledger; do not edit that old total, disable its guards globally, or pretend its exhausted remainder is available. A campaign-aware trainer validates the active allocation ID, device, remaining time and experiment job before admitting updates. Local development uses non-learning checks; learned preflight takes place inside the Kaggle allocation.

The milestone is a **learned, auditable cognitive loop with measured transfer**, followed by a bounded experiment asking whether the learner can improve its choice of learning method. We will measure unfamiliar-rule reasoning, useful inquiry, goal binding, tool actions, retention and method selection. This run cannot establish broad AGI from a narrow task suite. It can identify which proposed mechanisms actually deserve the next scale-up.

Use two independent single-GPU workers, not distributed training: one model fits on each device, and we need independent seeds and paired controls more than a single larger checkpoint. Memory is not pooled across devices. Kaggle documents its accelerator selection and GPU usage controls; verify the live account/session allowance rather than assuming a guaranteed quota. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks), [GPU usage documentation](https://www.kaggle.com/docs/efficient-gpu-usage).

## 2. What the latest branch actually supports

The `edee727` push adds real workspace, belief, memory, objective, planner, meta-episode and generation-contract modules. Its handoff correctly labels many learned paths unqualified. Chief review found the following campaign blockers:

- `commands.train` and `runtime.resume.restore_run` construct CPU-default trainers; inspect's device flag does not make the training command CUDA-capable.
- `learning.router.route_window` exists but the canonical trainer still computes answer-token and optional pair losses. There is no actual multi-objective training consumer.
- `models.decisions.score_candidates` and `estimate_value` construct CPU tensors and use `torch.no_grad()`. They are inference helpers, not differentiable action/value training interfaces.
- `models.world.predict_step_finite` also constructs CPU tensors. Its condition renderer needs exact agreement with the training renderer, and outcome scoring must mask the conditioned prefix, deduplicate outcomes and preserve probability semantics.
- The executive takes an injected scorer; the generation runner is explicitly a fixture. Neither is evidence of a learned executive or model-origin proposer.

The chief ran 62 selected non-learning/gradient-only tests successfully. That does not close these missing integrations. See [the K8 readiness review](engineering/experiments/K8_20260913/READINESS.md). Implement the launch blockers before starting the notebook. Do not spend the allocated eight hours writing or debugging substantial missing modules on the GPU session.

## 3. Schedule: one eight-hour campaign

All times are maximum elapsed minutes from campaign start. Both workers share the supervisor deadline; retries do not restart it. A phase ending early may advance the next phase, but may not enlarge its own budget or create additional treatments after inspecting results. Idle time is not filled with meaningless training.

| Phase | Window | Wall cap | GPU 0 and GPU 1 work |
|---|---:|---:|---|
| E0 | 0–30 | 30 min | Hardware, actual-path gradients, resume, pilot throughput and locked protocol |
| E1 | 30–150 | 120 min | Four from-scratch formation runs: two treatments × two independent seeds |
| E2 | 150–195 | 45 min | Cognitive behavior and planning/goal/working-context controls on frozen E1 models |
| E3 | 195–255 | 60 min | Paired tool-learning children plus protected-family retention |
| E4 | 255–315 | 60 min | Paired gated block-reuse architecture experiment |
| E5 | 315–450 | 135 min | Measured method archive, learned proposer update, successor/frozen/fixed comparison |
| E6 | 450–480 | 30 min | Final scores, restore/export verification, report and durable artifacts |

E1's first 60-minute slot runs A/seed1701 on GPU0 and B/seed1701 on GPU1; its second slot runs B/seed1702 on GPU0 and A/seed1702 on GPU1. This counterbalances treatment/device assignment. Thereafter GPU0 owns seed1701, GPU1 owns seed1702, loading the relevant completed checkpoint. Each GPU runs its paired children sequentially, reversing treatment order between seeds. File names include phase, arm, seed and unique run ID.

Stop all new training by minute450. E6 has no new training. Hard stop before minute480 after flushing recoverable evidence. If a required artifact is missing, report that phase blocked and preserve independent results. A broken pipeline or no real CUDA training is an E0 abort, not a reason to keep the GPUs busy for eight hours.

## 4. Model and optimizer choices

Use the existing shared decoder with a campaign profile: vocabulary260, layers8, width256, heads4, FFN704, maximum context512, tied output embeddings, existing action/value heads. Analytical total is **6,493,952 parameters** before E4's two scalar gates; confirm by instantiation. This is a tractable mechanism-learning subject, not a claim that this scale is sufficient for AGI. All core weights begin randomly initialized in E1. Subsequent children trace to those from-scratch parents; no pretrained model, tokenizer, reward model or external LLM-generated answers enter the primary experiment.

Initial optimizer candidate: AdamW, learning rate0.0003, betas(0.9,0.95), epsilon1e-8, weight decay0.01, global clip norm1.0, warmup over the first5% of each declared update schedule followed by cosine decay to10% of initial LR. These are fixed experimental choices, not proven optima. Use per-objective window denominators from the master algorithm; clip and step once per accumulated window. Use float32 for probability/loss reductions. FP16 autocast plus GradScaler is the initial CUDA precision path; unscale before clipping, keep the scaler consistent across accumulated microbatches, and checkpoint its state. Match the installed runtime's supported APIs. [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html).

E0 chooses microbatch size from 16,8,4,2 and accumulation from1,2,4 to fit the **heaviest** declared treatment, targeting64 decision/example units per update where feasible. Candidate actions create additional sequence rows: record that multiplier, not just the outer batch size. Use the same effective units and deterministic stream for paired treatments. Reserve memory for evaluation/checkpoint operations; no silent sequence truncation or vocabulary change to survive an OOM.

For E1/E3/E4, predeclare a common completed-update target per paired comparison using E0 timing: `N = min(4000, floor(0.75 * slot_seconds / worst_measured_update_seconds))`. Include all active losses and candidate expansion in pilot timing, use the slowest worker/treatment, and reserve the remaining25% for evaluation/checkpoint overhead. Minimum informative planned targets: E1=200 updates, E3=80, E4=80. If the chosen profile cannot meet these conservative targets, stop before E1 with a measured profile report; do not change the experiment after seeing scores. The pilot is not a capability result.

At a phase deadline, compare paired checkpoints at their latest **common completed update** and common stream prefix; report the scheduled target missed. Also report actual time and extra forward/backward cost. E5 explicitly uses equal-time adaptation trials, a different resource comparison; do not call all these regimes equal-compute.

## 5. Dataset: generate what we can verify

For this first campaign, the owner need not supply an internet corpus. The agent should build a versioned **BRAMASTRA-K8 synthetic dataset** offline from qualified generators and export it as a Kaggle input bundle. This avoids spending the session on downloads and lets us test genuine mechanism-held-out transfer. Synthetic data limits the claim; it does not make correctly executed learned experiments fake.

Required bundle:

```text
bramastra-k8-data/
  manifest.json                 # licenses/authorship, generator and source hashes
  splits.json                   # mechanism groups, pair groups and pool roles
  episodes/                     # real generated public trajectories, including failures
  supervision/                  # answer/EOS, action, value, world and pair records
  tools/                        # disposable table/file fixtures and independent verifiers
  meta/                         # meta-task definitions; no invented outcome labels
  audit.json                    # leakage, solvability, length and target-count checks
```

Generate three initial families: hidden Boolean/rule inquiry with complementary queries and distractors; inventory/resource dependency tasks; and small executable program-composition tasks. Target4096 distinct training mechanisms per family and four trajectories per mechanism, using a declared mixture of qualified public-information teacher, fixed exploration and random exploration. Where the generator cannot justify that many distinct mechanisms, report the actual count and redesign the generator before launch; do not manufacture uniqueness through UUIDs. Labels must come from the actual environment or an independent verifier.

Separate pools by mechanism equivalence: training, training-controller, development-measurement and sealed confirmation. Target256 controller and256 development mechanisms per family; prepare at least128 distinct confirmation mechanisms per family. Pair groups and renamed/rephrased equivalents remain in one pool. Use two goal variants on each designated goal-pair case. Teacher access, reference belief updater, tool interpreter and byte codec are declared priors. Inquiry teachers should reason from the public hypothesis support/history, not choose an answer-revealing action by reading the sampled hidden mechanism.

E0 chooses the confirmation inventory size solely from timing:128,64 or32 clusters per family, largest predicted to fit the E2 allowance with20% reserve. Lock the chosen IDs before learning; do not downsample based on scores. Below32 is `EVALUATION_BUDGET_INSUFFICIENT`. All cases stay in the report; exceptions/timeouts count as failures, not dropped rows.

Tool transfer uses new generated table/file tasks: read a public table schema, inspect rows, filter on a predicate, aggregate a bounded integer column, and write/check a small result. Define a finite tool API such as `read_table`, `filter_rows`, `sum_column`, `write_result`, `check_result`; bounded arguments come from public schemas, not the hidden solution. Tools operate on disposable per-episode files, with no network or unrestricted shell. Use256 training mechanisms and at least64 held-out compositions per seed; hold out operator order, table structure and value ranges separately. Free-form final answers still use the full vocabulary and EOS.

The meta bundle supplies24 meta-training mechanisms,6 reusable meta-validation mechanisms and6 fresh meta-confirmation mechanisms per seed, distinct from prior examination pools. Each has support examples, query examples and a protected old-family set. Outcome labels for method quality are measured during E5; never prefill them from a designer's guess. Local support adaptation/query measurement inside meta-training is training feedback, whereas final meta-confirmation remains examiner-only.

Reject or count records whose mandatory public prefix/answer cannot fit context512. Preserve goal, real evidence and boundaries; do not hide required observations through arbitrary truncation. Report rejected-length fractions by family. Keep global IDs, hidden state, split names and evaluator answers out of learned tokens. Hash exact prepared bytes and bind every run to them.

If the owner wants to provide data, request this generated bundle first, not billions of unverified web tokens. A later natural-language/code campaign can use an owner-supplied licensed manifest; that is a separately budgeted study. Do not silently replace missing K8 tasks with unrelated text or call synthetic task learning general pretraining.

## 6. E0 — Prove the experiment path before using it

Log live GPU count/names/VRAM, CUDA/PyTorch versions, available session allowance and output storage. Start one persistent worker per GPU with explicit device assignment; no accidental CPU fallback. The supervisor owns the only campaign ledger and records atomic per-worker reservations and actual usage, including failed steps. Bound any subprocess lifetime by the parent deadline.

On each GPU use a disposable tiny profile for an actual uninterrupted3-update versus interrupted1+resumed2-update check, using two microbatches per update and enabled token/world/action/value terms. These are six real updates per GPU, now charged to K8. Compare model, optimizer, scaler, RNG, data cursor, objective counters and the next batch/decision sequence. Declare deterministic replay settings; if exact numerical equality is unsupported, require identical stream/counters and a preregistered tolerance justified by a same-backend repeat. Divergent state is not numerical noise.

The whole E0 allowance is at most128 completed optimizer updates per worker, including pilot timing and every failed/retried attempt; the 30-minute global cap remains tighter when reached first. Pilot weights are discarded and never become E1 parents. Verify enabled heads change and disabled heads do not; the action/world training path must return differentiable tensors, not floats from inference-only wrappers. Refuse positive objective counts with missing losses. Validate CPU/GPU dtype/device consistency, mask correctness and per-objective accumulation equivalence.

Require the latest F1–F6 regressions, actual prepared-data identity binding, successful lease cleanup after a setup failure, and invalid promotion rejection before launch. Existing recovered B2.2 manifest files alone do not prove restored payload contents. Produce a new unique GPU resume receipt here rather than modifying old evidence.

Run outcome-order and duplicate-support tests for the world predictor, action permutation tests, no-hidden-state checks and a fake-versus-learned origin check. Gate each later capability route independently. If model-origin proposal generation is unavailable, mark E5 blocked instead of substituting `run_fixture_generation`.

## 7. E1 — Can unified supervision form useful cognitive behavior?

**Primary hypothesis:** legitimate action/value/public-prediction supervision improves held-out task success and goal-sensitive decisions compared with answer-only learning, under the same starting core, stream and completed-update count.

Arm A uses answer/EOS loss only. Arm B uses answer weight1.0, world weight0.5, action weight0.5, value weight0.1 and goal-pair weight0.1. Each term has its own eligible-unit denominator. Pair margin is1.0 with eligible-answer-token mean scoring. Both arms use the same public trajectories and qualified labels are available in the dataset; A's disabled objectives receive no gradients. This compares a supervision package, not a pure architecture-only intervention. Do not attribute all gains to a single component.

For each seed, create one random initial state and clone it bit-for-bit into A and B with matched optimizer initialization. Evaluation inference does not use training-only vocabulary masks. Store stage checkpoints at25%,50%,75% and100% of the planned target, with exact exposure/cost counters. Measure full-answer/EOS success, goal-pair both-correct, query cost, legal choices, public prediction content-field log score, value error and head/core displacement. A's untrained action head is not the only task baseline: include direct-answer, fixed inquiry and random inquiry controls.

If B's loss falls while action/value gradients are absent, label a wiring failure. If both arms memorize training mechanisms but fail unseen ones, label a transfer failure. Passing a training-loss threshold does not qualify cognition. Preserve both seeds and all trajectories; never report just the favorable seed.

## 8. E2 — Does learned cognition cause better decisions?

Freeze all E1 weights. On both seeds evaluate A's direct/fixed-inquiry controls and B in three predefined modes: learned policy with real public history; the same policy with explicit goal/evidence workspace; and the same model with bounded depth-two public-outcome planning. The workspace renderer and its operating format must have appeared in training; an untrained input format is not a fair cognition ablation. Keep the same mandatory observations available across modes and declare any additional context/compute.

Primary contrasts: B versus A on matched task success; B planner versus B policy on success per real inquiry; correct-goal versus swapped-goal behavior on qualified pairs. Secondary tests use contradictory observations, irrelevant high-entropy queries and a valid complementary two-query construction. Use actual model outputs; a symbolic posterior updater is a named reference control and cannot masquerade as the learner's belief update.

Limits per episode: at most4 real inquiry/tool actions plus one submission, at most8 imagined nodes, depth2, at most16 batched model calls, and final answer cap24 tokens or a stricter validated task cap. Count every fallback. The same real-action allowance applies to all modes; extra planning tokens/time are reported separately. Store exact public traces for a fixed sample of successes and failures selected by hashed case ID, not attractiveness.

Report calibration/proper score, both-correct, success/cost and constraint violations per family. Evaluation data cannot change E3/E4 treatment settings or the proposer training archive. A failing E2 result does not invalidate later **predeclared** mechanism diagnostics, but it prevents a positive cognitive claim and any automatic promotion.

## 9. E3 — Learn a new tool skill while retaining old skills

Fork the same E1-B parent per seed into T0 and T1; reset child optimizer under one declared rule for both. T0 trains on new tool trajectories with token loss only and no old-family replay. T1 uses the E1-B objective package plus25% fixed stratified old-family replay and75% new-tool episodes. This is deliberately a system-package comparison; it does not separately identify replay versus extra supervision. Future ablation is warranted only if the package helps.

Each worker gives each child a30-minute slot, with a common E0-calibrated update target and shared new-task ordering. Record replay as additional old exposure, not unique data. Evaluate zero-shot parent performance before adaptation and child performance after adaptation on held-out tool compositions. Re-evaluate a fixed protected set of old-family tasks without feeding its results to training.

Primary outputs are new-tool success gain, real tool-call cost and worst protected-family regression. The best result acquires tools while retaining earlier performance; freezing the model and learning nothing is not a retention success. Include an oracle scripted tool executor only as a diagnostic ceiling, explicitly external. The learned model must choose both the tool and valid arguments. Tool return values are real observed evidence; predicted tool outputs cannot enter the observed ledger.

## 10. E4 — A bounded architecture change with a clear control

Fork E1-B again, independently of E3, so tool learning cannot confound this experiment. Compare S0 unchanged decoder with S1 **gated reuse of the final two decoder blocks**, using the same training stream/objective and common completed updates. Both arms receive identical original weights and fresh matched optimizer state.

Proposed S1 change, before final normalization, for each of the final two blocks B_j:

`h_next = h + tanh(alpha_j) * (B_j(h) - h)`.

The original eight-block pass remains. Reuse its actual block weights; add only two trainable scalar alpha values, initialized to zero. At migration, S1 must be functionally equal to its parent within declared numerical tolerance. Store those new parameters and architecture identity explicitly. S0 has the same two scalar slots fixed/disabled for parameter-inventory comparison. Core weights retain their from-scratch lineage. Never silently load the changed architecture as an exact resume.

This evaluates extra recurrent computation with learned gates. It is an engineer-specified architecture candidate, not autonomous model invention; the blocks still execute and incur cost even when gates are small. Compare equal-update/data outcomes and the actual additional time/calls. E5 separately asks whether the learner can choose this candidate when it helps adaptation.

Use30 minutes per arm per GPU, reversed ordering between seeds. Evaluate held-out longer/composed rules and protected original tasks, reporting gate values, gradient flow and effective block evaluations. Reject any benefit caused by altered labels, more real interactions or a different final-answer scorer. If initial equivalence or checkpoint migration fails, do not run S1; save the failure and retain S0.

## 11. E5 — Direct experiment on learned method selection and recursive improvement

This is a bounded, exploratory RSI experiment with an intentionally small search space. It must involve a real learned proposer and measured adaptation outcomes. It does not claim open-ended algorithm invention, and two acquisition seeds cannot establish robust general RSI.

Each GPU owns its E1-B seed parent as a fixed **adaptation anchor**. The proposer begins as a separate clone of that same from-scratch checkpoint. Adaptation jobs never start from the more-trained proposer: keeping anchor weights identical is essential to isolate method selection. Separate adaptation and proposer optimizer states/checkpoint identities.

The method language initially admits three choices:

- M0: base architecture, LR0.0003, the job's declared objective.
- M1: the same architecture/objective with LR0.00015.
- M2: E4 gated block reuse, initially zero gates, LR0.0003, same job objective.

Adaptation jobs use the E1-B objective on eligible support trajectories. Proposer updates use method-token supervision from the measured archive, with other objective types explicitly disabled. Method choice changes optimizer/architecture, not which gold answers or evaluation rules a job receives.

Every trial resets from the same anchor for its task, with identical support ordering and a fresh declared optimizer. Methods have equal **elapsed adaptation allowances**, not equal update counts. Record actual updates/tokens and success over a common elapsed-budget grid. Include query/evaluation overhead in each trial's reservation; finalize the last committed update before its deadline. For an unsupported M2 implementation, E5 is blocked rather than silently shrinking the search space after seeing data.

**Minutes315–360: learn from measured method experience.** Per worker, evaluate12 meta-training tasks ×3 methods ×45-second maximum trials:27 minutes reserved for adaptation/measurement. Reserve8 minutes for proposer training and10 for capture/validation/checkpoints. The remaining12 meta-training tasks are held for the next block. Query labels here are explicitly meta-training feedback. Save all method results, including failures, into a hashed archive.

Proposer input contains only permitted support-derived task descriptors, public goals, budget and the frozen allowed archive summary. No current query label or measured current-task outcome appears before selection. The proposer generates a method ID/typed AST using the shared decoder; it is not a Python argmax over the archived current task. A short finite vocabulary of method tokens is a disclosed action language, not free-form algorithm discovery. Train masked method-token likelihood against measured best-feasible choices, with ties represented as acceptable alternatives. Include failed choices and measured cost. Check parameter changes and real decoded output origin.

Before every selection, commit a `proposal_context` containing proposer checkpoint hash, raw input hash, archive snapshot hash and maximum included event index, pool/generation ID and current task identity. The validator proves the current task's measured adaptation/query outcomes are absent from that snapshot and rendered input. For the confirmation block, freeze all proposer/archive snapshots and capture all choices before any confirmation method trial begins. Appending new results creates a new archive identity; it cannot retroactively change a captured decision's context.

**Minutes360–405: create a research successor.** Evaluate12 fresh meta-training tasks ×3 methods ×45-second trials, again27 minutes. Capture actual P0 choices before evaluating the corresponding task and compare them with fixed M0/random controls using the common outcome table. Freeze P0 and its input/archive cutoff. Ask P0 to choose M0/M1/M2 for the public goal of improving its own future method-selection learning, using only admitted meta-training summaries. Capture and compile its actual generated choice; no external agent selects the method on its behalf.

Fork P0 into **P1**, updated with P0's selected method, and **P_fixed**, updated with M0. Both receive the same newly admitted method-training archive/order and equal4-minute update allowances, eight minutes total. This controls for additional meta-training rather than comparing only a more-trained successor against an older parent. Store actual update counts and any architecture migration. Choosing M0 is a valid no-change proposal, not a reason to force a different choice. Use the remaining10 minutes for provenance, checkpoints and bounded optional meta-validation; no validation feedback may rewrite the frozen choices. These are provisional research successors, not canonical accepted-parent publications.

**Minutes405–450: independent recursive comparison.** On6 fresh meta-confirmation tasks per worker, capture P1's choice, P_fixed's choice, frozen P0's choice, fixed M0 and deterministic random choices **before** any method is evaluated. Evaluate all3 methods per task with90-second trial caps:27 minutes. Use the remaining18 minutes for query/retention aggregation, evidence checks and export. Score each policy by its already-captured choice against the common measured outcome table. Same-table scoring is a matched control, not independent additional training evidence.

Measure adaptation-curve AUC, final query success, old-skill retention, proposal regret versus the best measured method, no-change frequency and full search/proposer cost. A bounded recursive benefit requires P0's actual method choice to be applied to P1, followed by improved future method selection over P_fixed, P0 and fixed M0 on untouched meta-confirmation tasks, with authentic successor provenance and no unacceptable retention/cost tradeoff. If P1 beats P0 but not P_fixed, attribute the gain to extra meta-training rather than self-selected method improvement. Report negative/inconclusive if required differences are unsupported. Merely selecting M2, generating valid JSON, or training P1 longer is not an RSI success.

Do not feed E5 confirmation outcomes back into P1 or start another generation in this allocation. Canonical model/method promotion remains a chief review after the campaign. This experiment can support a narrow learned-selection/recursive-successor result; it cannot support arbitrary source self-modification, unbounded recursive acceleration or AGI.

## 12. Statistical decisions and honest claims

Primary campaign comparison is E1-B minus E1-A held-out complete-task success, paired by seed and exact mechanism/case. Require both seeds' point differences nonnegative, pooled point gain at least3 percentage points, and a positive95% world-cluster interval conditional on these seeds for a promising result. Also require no protected-family point regression greater than2 percentage points where a retention comparison applies. These are screening criteria, not confirmatory significance across training randomness.

Compute uncertainty by resampling complete mechanism clusters while retaining paired cases; report each training seed separately. With only two independent acquisition seeds, do not present thousands of query rows as thousands of independent model runs. E2–E5 are preregistered secondary/exploratory comparisons; report every result and its interval without cherry-picking a single positive claim or treating multiple comparisons as independent confirmations.

For E5 report6 confirmation meta-tasks per seed as the actual task count. Query rows within a meta-task are not independent method comparisons. Use paired meta-task differences, conditional intervals and a table of every selection. A larger multi-seed/meta-task confirmation is a future campaign, not additional unallocated work in this one.

An engineering failure is different from a scientific negative: absent gradients, hidden-state leakage, changed source/data, invalid comparisons or false device attribution invalidate the affected experiment. Poor accuracy with correct execution is useful negative evidence. Never lower acceptance thresholds after the run. Never convert a missing result to zero cost or a successful result.

## 13. Checkpoints, notebook and return artifacts

The implementation agent must deliver `notebooks/bramastra_k8.ipynb`, a thin notebook calling tested repository entry points, plus a generated dataset bundle, launch manifest, pinned source archive/hash and offline runbook. These are required deliverables, not existing files at the time of this design. Put substantive trainer/model changes in the repository, not in opaque notebook cells. The notebook must print source identity and the complete phase plan before starting workers.

Separate build mode, E0-only mode and owner-launched full campaign mode. A full rerun needs a new run ID and remaining allocation; restarting a kernel does not reset the campaign clock/ledger. Shared supervisor writes atomic reservations and phase transitions. Each worker writes only its own run directories. On cancellation or deadline, stop at an update boundary, flush ledger and persist a recoverable checkpoint.

Persist full model/optimizer/scaler/RNG/cursor/objective/controller/proposer state every10 minutes and at phase boundaries; preserve parent identity and source/data/method hashes. Bound checkpoint retention: latest complete and required phase parents plus manifests for retired intermediates, never delete the only restorable copy. Weights/large datasets stay outside Git. Record archive hashes, location and a successful restore check; merely listing payload files is not restore evidence.

Required result bundle:

```text
K8-<unique-id>/
  launch.json, frozen_protocol.json, hardware.json, source.json, data.json
  campaign_ledger.json, reservations.jsonl, failures.jsonl
  E0/ ... E5/                  # raw outcomes, checkpoint manifests, per-seed reports
  proposer_transcripts/       # raw output -> parsed method -> checkpoint binding
  comparisons/               # all arms, seeds, intervals, costs, exclusions
  checkpoints/               # restorable payloads outside Git
  RESULT.json, REPORT.md, artifact_manifest.json
```

Include task-success/learning curves, protected-family deltas, calibration/goal-pair tables, tool-call costs, architecture gate/cost plots and E5 method-choice/regret tables. An AGI score is not required and must not be invented. End with concrete decisions: keep, disable, diagnose or replicate each mechanism, and the cheapest informative next experiment.

The complete [agent prompt](engineering/experiments/K8_20260913/AGENT_PROMPT.md), [readiness work order](engineering/experiments/K8_20260913/READINESS.md) and [campaign manifest](engineering/experiments/K8_20260913/campaign.json) make this file executable by another agent without conversation history.
