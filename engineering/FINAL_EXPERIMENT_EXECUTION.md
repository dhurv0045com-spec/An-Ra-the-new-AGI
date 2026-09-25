# FINAL-K8 — build contract and acceptance specification

## 0. Status and authority

This document preserves the **F01–F24 build contract** for Gandiva's K8 campaign: required model/data/runtime behavior, integrated checks, and acceptance conditions. It was written as an implementation assignment; that assignment has reached local build readiness. It is not a standing instruction to keep expanding the repository or an assertion that the Kaggle experiment has passed.

As of 2026-09-25, the source-bound no-update verifier passed F01–F24, all seven test groups, and all seven production-interface exercises with zero local optimizer updates. The exact evidence and the still-pending Kaggle E0 gates are in [`engineering/STATUS.md`](STATUS.md) and [`engineering/reports/FINAL_K8/gandiva-post-e4-c6c64c89-20260925-pytest-temp/build_verification.json`](reports/FINAL_K8/gandiva-post-e4-c6c64c89-20260925-pytest-temp/build_verification.json). The next milestone is a fresh owner-run on two T4s, using the [run guide](final_delivery/RUN_EXPERIMENT.md) and the normal notebook.

The build report establishes local readiness only. The Kaggle notebook must verify the exact downloaded source and fresh data, then pass live E0 checks for device identity, real updates/resume, measured workload cost, and the active allocation before it may run E1–E6. A valid campaign can fail scientifically. Neither this specification nor passing tests establish learning, recursive self-improvement, or AGI.

The original baseline note (`29df755`, H01 not yet implemented, and early scoring/device checks) records why this assignment began; it does not describe the current code. Keep the requirement sections below as the frozen acceptance contract. The [Gandiva guide](../GANDIVA.md) explains branch purpose and lineage; current status takes precedence over old dispatch language.

## 1. Original implementation order (historical)

Execute the following sequence without another owner dispatch: (1) public task and decision contracts, including H01 repairs; (2) complete learning/consumer interfaces; (3) cognition and actual controls; (4) tool and architecture children; (5) actual RSI trial/proposer chain; (6) scheduler/calibration/checkpoint/export integration; (7) final build verifier and owner notebook. Test each boundary as it is connected. Do not postpone integration until every isolated module has a passing mock test.

Use one lead and optional bounded Luna/Sol lanes. Before concurrent editing, establish these exclusive ownership boundaries:

| Lane | Owned paths | Integration obligation |
| --- | --- | --- |
| Data/learning | `research/data`, `experience`, `learning`, focused tests | Public state, labels, windows and trained consumer formats |
| Cognition | `research/cognition`, `environments`, E2 and tests | Real action execution, public/imagined state, budgets and comparisons |
| RSI/architecture | E4/E5, `models/gated.py`, `metalearning`, trial service and tests | Actual migrated handles, trials and independent successor choices |
| Lead/runtime | CLI, runner/worker, session/ops interfaces, readiness, export, notebook, release | Coherent identities, allocation, cross-lane integration and final acceptance |

Use isolated worktrees where practical. In a shared checkout, the lead owns shared files such as ops.py and schema interfaces; other lanes request a concrete interface change rather than editing them simultaneously. The lead reviews behavior and closes requirements. A time estimate or subagent failure is not a reason to return only a plan. If an external limitation prevents completion, deliver exact failed requirements and evidence without labeling the release ready.

## 2. Architecture: one from-scratch model, explicit consumers [F01]

Use the actual shared decoder with vocabulary 260, eight layers, width 256, four heads, FFN width 704 and context 512; tied language output embedding; existing action and value heads. Instantiate and verify the expected base parameter count of 6,493,952. All core E1 weights start randomly initialized. Seeds 1701 and 1702 define independent initializations; paired A/B clones under a seed have identical initial tensor/optimizer/stream identities. No pretrained weights, external reward model or external LLM teacher enters the primary experiment. The byte codec, symbolic data teacher, tool interpreter and reference oracle are declared priors.

Publish an implemented consumer map: decoder token head → terminal answers and typed next-feedback prediction; action head → legal candidate selection; value head → registered return estimates; shared hidden path → all these consumers and the gated architecture variant. An unused trained head is a missing integration, not a functioning learning algorithm. A wrapper may not bypass gated blocks through a direct decoder call.

The cognition runtime consists of typed public state, immutable received event history, a bounded derived evidence workspace, learned decision consumers, isolated imagined state and a real environment/tool executor. Persistent learning is in model/optimizer/checkpoint state and the training/replay streams. Do not relabel a scripted workspace updater as learned memory. Do not add untested long-term retrieval, new modalities or a larger architecture merely to make the design appear more complete.

## 3. Task information and solvability [F02]

Repair the rule task so the correct answer can be inferred from permitted information. For the K8 base task, the public goal contains the full rule specification: function, relevant-variable roles, negations, threshold if applicable and target predicate. Sampled variable values remain hidden until inspected. Bound required relevant observations to the four-inquiry budget. The current hidden target flip that changes the correct answer without changing available observations must no longer occur in an admitted uniquely scored task.

Generate each teacher trajectory by executing the same environment contract used in evaluation. Its final answer refers to the sampled episode state, not a mechanism-level constant with a different meaning. Inventory dependencies require received checks; program targets refer to the stated input; tool answers require actual tool results. Include failed/partial exploration with valid eligibility masks. Do not teach final inventory answers from a goal-only prompt that contains no evidence of the dependency.

Add an information sufficiency witness in the data audit: a legal bounded sequence or adaptive strategy that makes the answer identifiable from public observations. For a bounded finite diagnostic family, adversarial private twins sharing the same observation transcript must agree on the uniquely scored answer. A witness is for data validation and teacher auditing, never an inference oracle. If a task intentionally remains ambiguous, give it a registered uncertainty/abstention scoring rule outside the uniquely scored primary inventory; do not silently penalize a learner for unavailable information.

The rule information change is an explicit protocol/schema amendment. Update the campaign manifest, generators, validators and notebook together. Old and repaired task results are not interchangeable. Unknown-function induction is a future experiment unless the registered task exposes informative labeled interventions within the existing action budget; it is not a substitute that delays this base repair.

## 4. Dataset and split closure [F03]

Deliver an offline generated K8 bundle containing manifest, split/canonical identities, real trajectories, supervision channels, tool fixtures, meta-task definitions and a data audit. Target per initial family: 4096 training mechanisms × four trajectories, 256 controller mechanisms, 256 development mechanisms and at least 128 sealed confirmation mechanisms. Families are rule inquiry, inventory and program composition. Tool inventory is 4096 training tasks plus 256 held-out compositions. The larger training pool lets calibrated E3-T0 consume any registered target up to 4000 without repeating or fabricating examples. Meta inventory per seed is 24 training, six validation and six confirmation tasks, with two disjoint 12-task training blocks.

Mechanism uniqueness must survive the declared equivalence relation; new names/UUIDs do not create new mechanisms. If a generator cannot supply the required distinct, solvable mechanisms, improve the generator before declaring the bundle complete. Document finite-domain/diversity audits and actual counts. Do not quietly lower counts in the owner notebook or inherit CLI smoke defaults. The full data bundle is a delivery artifact outside Git; commit its manifest, compact audit and reproducible generation command.

Keep renamed/rephrased equivalents, paired goal variants and canonical mechanism clusters within a single pool. Enforce exact role membership, not substring tests. E2 and retention evaluation consume the registered frozen held-out IDs, not an unrelated generator stream whose seed merely differs. Meta support/query/protected references resolve to complete records with explicit disjoint roles. Missing references fail validation; they do not shrink denominators.

Prepare data before the owner GPU session. No internet corpus is required for this campaign. Confirm licenses/authorship and source identities for any supplied inputs. Keep private truth, teacher labels and solvability witnesses physically separate from learner input fields.

## 5. One DecisionExample and one codec [F04]

Define and version the shared logical record: goal, received event prefix, evidence view, legal typed actions, payload schemas, remaining budgets, teacher action when eligible, observed next feedback when eligible, terminal answer when eligible, return target/horizon, split/canonical identity and source-event references. Use the exact record through preparation, compiler, training scorers, cognitive inference and RSI support/query evaluation.

Compile action/world inputs at the appropriate pre-action prefix. Compile final answers after the observations actually received before submission. No next feedback, final label, hidden target or protected task enters a conditioned prefix. Remove six-token input slicing, eight-byte candidate slicing and ID-only evaluation prompts. Complete canonical encodings must preserve all semantic fields and types; unsupported or oversized inputs fail explicitly before training. Schema migration invalidates older incompatible artifacts and checkpoints' assumptions rather than silently mixing them.

Preserve Boolean/string/integer/null/missing distinctions, nested tool results/errors and variable identities. Typed terminal actions are rule `{kind:submit, answer:bool}`, inventory `{kind:submit, item:str}` and program `{kind:submit, value:int}`. Preserve H01's compact typed-code compatibility and reject malformed/oversized outputs without guesses. The actual production model must receive any public mapping required to interpret generated codes. Terminal payload constraints may enforce syntax/range, never supply the correct answer.

Context admission reserves goal, decision-critical evidence, legal descriptors, budgets and response space. Omitted optional records have explicit IDs. No raw JSON slicing, invisible truncation or copying a hidden answer into a summary. Compare full training/inference token sequences and masks for the same DecisionExample, not merely a common goal prefix.

## 6. Learning objectives and optimizer windows [F05]

For a logical optimizer window W, compute `L = Σ_j weight_j * sum_j(W) / count_j(W)` over eligible enabled terms. Counts are different units: answer/EOS target tokens, world-feedback target tokens, action decisions, value targets and eligible goal-pair groups. World NLL summed over target tokens must use the target-token denominator, not an invented count of one. Disabled terms have no gradient or denominator. An eligible term with missing loss is a contract failure. A zero-eligible microbatch is valid; a required objective with zero data over its entire stage is a data/configuration failure.

E1-A weights: token 1, world/action/value/pair 0. E1-B: token 1, world 0.5, action 0.5, value 0.1, pair 0.1. Token loss is teacher-forced answer/EOS NLL with the public prompt masked. World loss is next-public-feedback NLL conditioned on state/action; target positions alone are supervised. Action loss is cross-entropy against the declared legal teacher distribution using complete candidate content. A single legal action has no discriminative action target. Value loss is squared error against a declared observed bounded return; missing verdicts are ineligible, not forced zero. Pair loss uses each goal with its valid answer versus the swapped answer over a genuinely distinct-answer pair; preserve the registered existing margin/normalization and record it in the frozen protocol. Pair group identities must not cross splits.

Precompute each window's denominators across microbatches, accumulate correctly normalized gradients, then unscale, clip and step once. Paired treatments consume the same example order and effective decision units. Candidate expansions are extra sequence work and must be measured. No per-row finalize masquerading as a 64-unit window. Overflow/nonfinite attempts do not advance optimizer/scheduler counters as if committed; their actual cost is charged and traceable.

Use AdamW LR 3e-4, betas (0.9,0.95), eps 1e-8, weight decay 0.01, gradient norm clip 1, first 5% linear warmup and cosine decay to 10% initial LR over the declared update schedule. Use stable loss reductions and FP32 on T4 by default; FP16 autocast/GradScaler is opt-in only after a successful E0 calibration. Allocate all tensors/masks/spans on the model device; the current scorer fix is retained. Local tests use CPU forward/backward only, discard gradients and prove microbatch partition equivalence, masking, disabled-term behavior and gate-path connectivity. Actual optimizer commit/AMP qualification occurs in owner E0.

## 7. Learned action, world and value consumers [F06]

Use the trained action head to rank complete legal candidates conditioned on the public state. Include a legal submit candidate when allowed; terminal content is generated by the trained answer decoder under the family schema. Use the same candidate scorer representation at training and inference, batching candidate branches where possible. Do not train an action head and evaluate a separate untrained JSON-index policy as the main learned consumer.

World inference uses the same conditioned prefix and typed feedback representation as world training. Use constrained feedback decoding or a declared finite public support; that support cannot be constructed from the private realized outcome. A valid prediction contains actual predicted fields; `{}` or an invalid numeric value is not a successful calibrated forecast. Value estimates come from the trained value head with the same return horizon; do not ask the decoder for an unrelated untrained success-probability JSON field.

Record invocation count, generated tokens, candidate rows, decode limits and prediction eligibility separately. Batch scoring counts as its actual forward invocation with its real sequence multiplier; it is not free computation. The registered action response cap remains 24 byte tokens. If feedback needs a different compact schema/cap, implement and record it in the protocol before E0 timing. Remove unilateral 8/96-token assumptions that disagree across parser, trainer and benchmark.

Acceptance includes a head-parameter perturbation affecting the actual consumer, backward gradients reaching intended parameters, and a real random-weight model trace through the production interfaces. Poor random-weight performance is expected; teacher doubles establish mechanics only and must stay labeled.

## 8. Evidence workspace and cognitive state [F07]

Retain immutable received events. Derived evidence records contain subject, predicate, typed value, valid time, source event ID, status and supersession links. Distinct variables must not conflict merely because their event kind matches. Changes over ordered times are transitions; incompatible single-valued claims at the same relevant time remain unresolved conflicts. Multi-valued facts accumulate. Duplicates do not manufacture independent support. Unknown timestamps have unknown order; never infer chronological order from arbitrary strings or rounded formatting.

The workspace may prioritize decision-relevant received evidence using declared deterministic rules, but does not solve the hidden task for the model. Preserve omitted IDs when the context window excludes optional records. Contradiction, distractor, missing-evidence, complementary-query and goal-swap probes operate on actual public state and independently verified outcomes. Goal swaps affect the rendered goal and the task verifier consistently; a changed prompt with the old scoring predicate is invalid.

## 9. Planning, legality and bounded execution [F08]

A planner root contains full PublicState, not only its hash. Imagined nodes contain parent/decision/root IDs, applied action prefix, typed predicted delta, imagined state, legal next actions, horizon/value, budgets and model origin. Apply each predicted delta to an isolated imagined copy before a child prediction. Recompute legality from public transition rules; stop branches that predict termination. Imagined records never enter received history.

At most four real inquiries/tools plus one submission, eight imagined nodes and sixteen model invocations per episode. Expand admitted roots before children; deterministic candidate exclusion and ties must be recorded. Use trained value/declared terminal forecasts minus stated path costs, not sums of unrelated probabilities. Bind the selected action to the exact selected path/forecast IDs. Repeated action names across decisions cannot share ambiguous node identities.

Counters are either cumulative or explicit per-selection deltas, never a mixture. Failed generations consume actual calls; preflight errors that made no call consume none and terminate explicitly. Every loop iteration must execute a budgeted action, spend real inference work or stop. Check job wall deadlines as well as action/call/node budgets. Use subprocess deadlines in liveness tests. A watchdog is protection, not a substitute for fixing zero-progress loops.

## 10. Meaningful comparisons and metrics [F09]

Run every registered E2 mode on matched copies of the same frozen mechanisms and seeds. B-policy consumes B's trained action/answer interfaces; B-workspace additionally consumes the evidence representation; B-planner uses learned prediction/value search. A-direct answers through A's actual decoder without inquiry; A-fixed performs the fixed inquiry schedule then answers through A's decoder. A-random uses the declared random inquiry schedule and an honestly named answerer. Oracle and pure scripted guesses remain separate diagnostic controls. Merely attaching A's checkpoint ID to a scripted default answer is forbidden.

Join each forecast to the executed decision/action/path and its declared horizon. Immediate feedback is compared with the corresponding received observation; terminal return with the terminal outcome. Report valid forecast counts, invalid/unknown rates, Brier score where probabilistic success is defined and typed feedback error with its exact denominator. Empty predicted feedback is not a match. All admitted episodes, including parser/context/runtime failures, remain in task-success denominators. Record per-mode consumed checkpoint identities and actual model origins.

## 11. E1 formation and lineage [F10]

Train four actual from-scratch formation runs: A/B for each of the two seeds, counterbalanced across devices. Publish 25/50/75/100% checkpoints on the actual completed-update axis, with complete resume state. Frozen held-out evaluation uses free/typed model answers and independent task verification; EOS or lower token loss is not task success. Model and optimizer identities at the paired starts must agree. Report actual stream prefixes, target counts, update counts and failed attempts. If a deadline prevents the scheduled target, compare the latest valid common update/stream point and label the target missed; do not compare unequal training counts without disclosure.

## 12. Tools, acquisition and retention [F11]

Execute finite public tools: read_table, filter_rows, sum_column, write_result and check_result. Arguments come from public schemas and model outputs; replies come from actual execution. Bound paths and outputs to each disposable episode directory, make retry semantics explicit, and independently verify resulting content. Stored labels cannot substitute for tool execution. Record request/result IDs, bytes/content hashes, errors and costs.

T0/T1 start as independent forks of the same E1-B checkpoint with fresh child optimizers. T0 uses the registered tool-token-only/no-replay treatment. T1 uses the B objective package with 75% new-tool and 25% permitted old-training replay. Measure realized exposure, rounding and candidate work. Held-out compositions and protected evaluation mechanisms never enter either update stream. Evaluate new-tool acquisition, old-family retention and worst-family regression on the actual child handles, not a restored parent accidentally reused for both arms.

## 13. Architecture adaptation [F12]

E4 forks independent S0 and S1 children from the same E1-B parent. S1 reuses the last two shared decoder blocks before final normalization: each reuse is `h <- h + tanh(alpha) * (B(h) - h)`, with both scalar gates initialized to zero. S0 has disabled slots and no trainable gate updates. Zero-gate outputs match the parent within declared deterministic precision; nonzero gates affect the path; gradients reach gates/shared blocks as intended. Propagate padding and packed-segment isolation through every reuse.

The migrated model must be the actual optimizer-owned, trained, evaluated, checkpointed and restored handle. Verify optimizer parameter inventory, aliasing and fresh-state policy; extra reuse compute is measured despite the small parameter increase. Restore architecture identity and gates from payloads. This experiment tests an engineer-specified architecture change selected/applied through bounded mechanisms; do not label it unrestricted architecture invention.

## 14. Trial service and production authority [F13]

A TrialRequest contains the fixed anchor payload/config/architecture identity, compiled method, support/query/protected identities, seed/RNG policy, parent allocation/reservation, deadline and allowed updates. Validate correspondence among allocation, reservation, job, device, source and open status. Pass JobInput to the JobInput binder or use the validated mapping binder; fix the current dict/attribute mismatch. Missing production authority fails admission. A local no-update rehearsal is an explicitly separate mode, not a silently downgraded production trial.

Restore/fork the real anchor, apply the recipe to the real optimizer/model, train under inherited limits, evaluate complete public query/protected tasks, publish the actual payload and return measured outcomes/costs. Child deadlines are the minimum of trial, block, job and allocation deadlines. No new synthetic reservation ID creates additional allowance. Missing protected references fail rather than being skipped or evaluated as opaque ID strings. Preserve failed trials and charge attempts/time. Local rehearsal reaches all interfaces with backward/no-step or strict doubles; it is labeled nonlearning.

## 15. Measured proposer and recursive successor [F14]

Use the fixed E1-B adaptation anchor for all method trials. Recipes are M0 base LR, M1 half LR and M2 zero-gate shared-block reuse at base LR, with actual typed compiler identities. Distinguish the adaptation anchor from the proposer checkpoint; proposal transcripts bind to the latter.

Block A: measure all three methods on 12 training meta-tasks using 45-second trial caps, then train P0 on method-selection targets derived from that measured archive within eight minutes. Block B: on 12 fresh training meta-tasks, freeze P0's selections before current outcomes, measure all three recipes with 45-second caps, and admit the new archive only after its cutoff. P0 chooses its own successor-learning recipe before application. Fork trained P0 into P1 and P_fixed; apply P0's choice to P1 and M0 to P_fixed; train both on the identical newly admitted archive/order for four minutes each. Archive-A and archive-B data are distinct.

Block C: for each of six untouched confirmation tasks per seed, independently decode P1, P_fixed and frozen P0 choices and capture fixed_M0/random choices before any confirmation trial result. P_fixed is a trained proposer using M0 for its training; it is not the always-M0 controller. Never copy P0's choice into P1. Evaluate all three adaptation methods per task under 90-second caps and score the already frozen chooser policies against that shared table.

Capture raw prompt/output, parser status, method, archive cutoff, selected recipe and exact payload identity. A malformed learned proposal is a failed proposal. A teacher-majority fallback may be reported as a separate control, never used to turn an unsuccessful model proposer into RSI success. No confirmation feedback trains another successor in this allocation. Measure regret, query success, protected retention, actual update/exposure/time and all proposer/search overhead. Stateful doubles test ordering; they do not establish learning.

## 16. Schedule, budgets and experiment clock [F15]

One owner-launched campaign, two independent T4 workers; no pooled VRAM or DDP. The allocation is at most 480 elapsed minutes / 960 provisioned GPU-minutes. Training stops at minute 450; export ends by 480. Do not launch paid compute, extra sessions or local training. Immutable allocation time survives E0-only/full transitions, process restarts and retries.

| Phase | Campaign window | Work |
| --- | --- | --- |
| E0 | 0–30 | Full-profile qualification, resume and calibration |
| E1 | 30–150 | Two 60-minute slots: GPU0 A1701 then B1702; GPU1 B1701 then A1702 |
| E2 | 150–195 | Frozen cognition comparisons; GPU0 seed1701, GPU1 seed1702 |
| E3 | 195–255 | T0/T1 30-minute child slots, reverse order between seeds |
| E4 | 255–315 | S0/S1 30-minute child slots, reverse order between seeds |
| E5 | 315–450 | Three 45-minute blocks per worker |
| E6 | 450–480 | No training; final aggregation, restore checks and export |

E5 block A reserves 27 minutes for 12×3×45-second trials, eight for P0 learning and ten overhead. Block B reserves 27 trials, eight total successor learning and ten overhead. Block C reserves 27 for 6×3×90-second trials and eighteen aggregation/export. Trial caps include their actual adaptation path and may finish early; do not create extra trials from unused time after observing results. Every nested operation obeys parent deadlines.

Record attempted updates, committed updates, supervised target units, model invocations, candidate rows, device time and elapsed allocation time separately. Phase totals reduce from events for all archive trials, proposer updates, successor updates and confirmation trials. Counting successful trials is not counting optimizer updates. Retries and failed/overflow attempts consume their actual time. Idempotent event IDs prevent double charging on resume without erasing genuine prior cost.

## 17. Hardware qualification and calibrated protocol [F16]

E0 validates two distinct physical T4s, worker-local CUDA mapping, precision support and actual full K8 geometry. Tiny profiles may support local diagnostics but never qualify full-model training. On each worker, demonstrate real gradients/optimizer change and fresh-process uninterrupted-versus-resumed continuation including next batch, RNG, scaler, optimizer/scheduler and counters. Charge all pilot/resume attempts within at most 128 completed updates per worker and the 30-minute E0 window.

Measure full update cost including data-to-device and every objective forward, backward, unscale/clip/step. Use synchronized CUDA timing and record candidate expansion. Time the heaviest actual evaluation path, including all modes, probes and planner envelopes. Calibration must fit whole matched evaluation groups; timing one cheap policy episode and multiplying only by families undercounts the workload. Include full-model S1/trial cost where it affects the common settings and reserves.

Choose microbatch from 16/8/4/2 and accumulation from 1/2/4, targeting 64 decision units per logical update where feasible; freeze the actual supported common setting. For paired E1/E3/E4 targets use `min(4000, floor(0.75*slot_seconds/worst_full_update_seconds))`; minima are 200, 80 and 80. Choose 128/64/32 confirmation clusters per family, largest whose complete matched groups fit E2 with 20% reserve. Below 32 or below an update minimum is a qualification failure; do not silently substitute a smaller experiment.

Freeze source/data/schema/tokenizer/model/optimizer identities, exact case IDs, batch/window settings, update targets, treatment parameters, hardware and precision before E1. Outcomes cannot influence selection. The runner consumes the frozen artifact, not hardcoded defaults. E0 costs, including nested pilot commits, must reach top-level ledger totals. A failed or missing pilot prevents E1; an E0-only restart does not reset allowance.

## 18. Checkpoints and restart correctness [F17]

Publish real tensor payloads with model, optimizer, scaler, scheduler, RNG states, sampler/cursor, objective-window state or explicitly atomic-boundary status, counters, parent, architecture, source/data/protocol identities and writer fencing. Avoid half-written checkpoints through atomic publication and hash verification. Wrong-parent, incompatible-schema, missing tensor, corrupt digest and stale-writer cases fail before reuse.

Fresh-process restore tests must load payloads, instantiate the intended architecture and reproduce the next controlled computation/stream state; listing file names or parsing metadata is insufficient. Local tests can use randomly initialized tiny/full-profile payloads without optimizer updates and must say so. Owner E0 provides actual trained/AMP resume evidence. Preserve failed-run artifacts, never overwrite a run ID with new settings, and keep large payloads outside Git.

## 19. Evaluation, uncertainty and claim acceptance [F18]

Primary comparison: E1 B−A complete-task success on paired seed/mechanism cases. Report each seed and pooled paired differences. A promising screening result requires both seed point deltas nonnegative, pooled gain at least three percentage points and a positive 95% mechanism-cluster interval conditional on these two seeds. Where retention applies, no protected-family point regression beyond two percentage points. Do not treat many trajectories as many independent training seeds.

E2–E5 comparisons are preregistered exploratory results. Resample entire mechanism/meta-task clusters with paired cases intact, use fixed analysis seeds, preserve failures and show all arms/intervals. E5 has six confirmation meta-tasks per seed, not hundreds of independent query comparisons. P1 must improve over P_fixed as well as P0/fixed controls under retention/cost constraints before attributing benefit to its self-selected method. Beating only P0 supports extra training, not recursive improvement.

Separate engineering-invalid, valid-negative and promising outcomes. A missing result is never zero cost or success. Report invalid-prediction rates, goal-swap correctness, useful inquiry coverage, tool acquisition/retention, gate values/costs, method choices/regret and parent identities. Freeze thresholds before training and do not select only favorable seeds. No invented AGI score or guaranteed multiplier.

## 20. Notebook, CLI and release inputs [F19]

Deliver the actual `notebooks/bramastra_k8.ipynb` and repository-backed CLI. The notebook performs checked commands rather than embedding a second implementation. It validates supplied source and data, displays the eight-hour/two-T4 configuration, starts one allocation, runs E0, continues automatically only after qualification, summarizes and exports even on recoverable failure. Command failure must stop dependent cells; users should not manually comment out guards or patch Python files.

Pin the source revision/closure, dependency versions and generated data manifest. Provide reproducible install instructions suitable for Kaggle's selected runtime, with no unnecessary downloads inside the GPU allocation. Verify imports and dependency compatibility in the available local environment; actual CUDA compatibility is an E0 check. Supply the offline data bundle or its reproducible generation/export artifact path and hashes, never ask the owner to invent the dataset or repair a schema mismatch.

Expose only necessary operator inputs: source/data roots, unique writable run/output paths and accelerator choice. Use a fresh run ID. E0-only then full mode must consume the same allocation if supported; an automatic full run should not allocate twice. Include concise instructions for stopping, collecting partial artifacts and continuing a legitimately resumable run within the original deadline.

At the reviewed source the real CLI has these flags. Preserve compatible commands or update all consumers/tests atomically; do not hand the owner invented options. Full mode currently requires successful E0, so the notebook runs E0 then full on the same run directory and original deadline:

```text
python -m pip install -e ".[bramastra]"
python -m bramastra_lab.research.campaigns.k8 prepare --out <offline-bundle> --training-mechanisms 4096 --controller-mechanisms 256 --development-mechanisms 256 --confirmation-mechanisms 128 --tool-mechanisms 4096 --tool-heldout 256 --meta-train 24 --meta-validate 6 --meta-confirm 6
python -m bramastra_lab.research.campaigns.k8 validate --bundle <offline-bundle>
python -m bramastra_lab.research.campaigns.k8 verify-build --data <offline-bundle> --report-dir <new-build-report> --no-updates --notebook notebooks/bramastra_k8.ipynb
python -m bramastra_lab.research.campaigns.k8 run --mode e0 --run-dir <run> --data <bundle> --max-wall-minutes 480 --devices cuda:0,cuda:1 --precision fp32 --build-report <new-build-report>/build_verification.json
python -m bramastra_lab.research.campaigns.k8 run --mode full --run-dir <same-run> --data <same-bundle> --max-wall-minutes 480 --devices cuda:0,cuda:1 --precision fp32 --build-report <new-build-report>/build_verification.json
python -m bramastra_lab.research.campaigns.k8 summarize --run-dir <same-run>
python -m bramastra_lab.research.campaigns.k8 export --run-dir <same-run> --out <new-export>

```

`verify-build` is new work, not an existing command. Preparation/validation/build verification happen before the GPU allocation; run commands belong only to the owner notebook. The current editable-install extra is unpinned; ship a tested runtime lock/constraints mechanism while preserving the Kaggle-compatible Torch build. The notebook uses validated source/input discovery and may generate the full bundle before verification; it must not carry a hardcoded host path. Offline source archive and installed package paths must work without assuming a live Git checkout or downloading a moving branch.

## 21. Export and result completeness [F20]

The result bundle contains launch/hardware/source/data/frozen-protocol manifests; allocation and reservation ledgers; E0–E5 outputs; failures; per-task predictions and actions; proposer transcripts; paired comparisons; checkpoint manifests and required payloads; RESULT.json, REPORT.md and a complete artifact hash inventory. Store large payloads outside Git, but make them available in the owner's output bundle.

Verify every required payload and fresh-process restore before reporting a successful complete export. Corrupt/missing payloads, wrong source closure or absent phase outputs invalidate completeness. Partial runs export their valid evidence with explicit missing/failed phases; export failure propagates to campaign status. Reconcile reported totals to child events and selected comparisons to their frozen case/checkpoint identities. No metadata-only success or fabricated restorable checkpoint ID.

## 22. Build verification and conditional readiness authorization [F21]

Readiness is derived from a fresh, source-bound `verify-build` report; absent, stale, tampered, or incomplete evidence returns `ready=False`. The chief gives standing, conditional authorization in this work order to implement and maintain that evidence gate as part of F21, and to clear build readiness only after F01–F24 have complete evidence. A new conversation or manual chief code edit is not a required final step. This does not authorize a bypass flag, hardcoded True, removal of runtime budget gates or an unverified claim.

Implement `python -m bramastra_lab.research.campaigns.k8 verify-build --data <bundle> --report-dir <new-dir> --no-updates`. This is a required new CLI surface. It runs the registered local contract/integration checks, validates the full data/schema/config closure and notebook/release inputs, and writes a machine-readable report. The report has exact requirement IDs, test selectors, child command exit codes, duration, actual assertions/receipts, source/data/config hashes and explicit hardware checks still pending. Unrun, failed, timed-out or empty local checks cannot be PASS. Test-only capability doubles remain labeled; real interfaces must also be exercised without stepping. The command must enforce no optimizer commits and refuse production allocation creation.

Build readiness is derived from the verifier's complete passing local evidence and matching source/data/config identity. A report with manually written PASS rows, missing receipts or a 64-character invented checkpoint ID cannot clear it. Separate the source implementation closure from generated report files to avoid self-referential hashes. Changed relevant code/data invalidates cached acceptance. A replay must revalidate hashes and test evidence; it cannot infer validity merely from a report file's presence.

Startup may enter E0 when build readiness is valid. E1–E5 additionally require current-run hardware qualification and frozen protocol. Thus ready-to-launch does not depend circularly on already having E1 training results. Actual optimizer authority still comes only from the owner campaign's validated live allocation. Independent code review of the final change is useful and should be performed by the lead or a bounded reviewer; unresolved critical findings must be fixed. No extra owner approval round is required for code acceptance under this explicit contract.

## 23. Tests and integrated rehearsal [F22]

Retain and strengthen meaningful existing checks; fix the known missing torch import in the foundation F6 test without weakening its assertions. Run H01's acceptance against the completed public interfaces or port equivalent behavioral tests with explicit migration notes. An old diagnostic aborting on a renamed API is neither proof of correctness nor a reason to abandon its counterexample.

Required local negative controls include: hidden-label input invariance; changed-public-input sensitivity; action collisions; typed payload round trips through the actual environment; context overflow; unknown/contradictory time; illegal/all-masked actions; empty/nonfinite predictions; multi-decision node joins; zero-progress loops; no teacher fallback in learned claims; actual A checkpoint invocation; wrong-parent/corrupt payloads; expired/mismatched reservations; duplicate ledger events; omitted pilot charges; P0 capture before/after a stateful learning boundary; independent successor choices; confirmation leakage; missing protected references; and missing export tensors.

Run one bounded integrated local rehearsal through the production E0/E1–E6 control interfaces with explicit no-step adapters at optimizer boundaries. It must exercise scheduling, parent wiring, failures, typed decisions, trial ordering, statistics and export; it is not a learned campaign and cannot populate production success evidence. Include real randomly initialized model calls and real checkpoint round trips on representative paths. Use fake clocks for the eight-hour schedule instead of waiting eight hours. Bound each subprocess and terminate only owned children on timeout. No full-suite run should hang indefinitely.

Use focused tests during development and a final appropriate regression selection once after integration. Categorize unrelated historical failures with exact evidence, but all tests covering the delivered production path must pass. No xfail/skip added to conceal a relevant failure. CUDA-specific tests are explicitly unrun locally and executed in owner E0. Do not generate optimizer changes in local tests, including hidden helper paths.

Existing relevant test files include test_research_k8_launch_gate.py, test_research_k8_readiness.py, test_research_k8_operational.py, test_research_k8_real.py, test_research_checkpoint.py, test_research_k8.py, test_research_k8_foundation.py and test_research_scoring_device_contract.py. Inspect their actual behavior before executing them under the zero-local-update rule. The bounded Luna runbook audit reported 36 passes and one failure in the launch-gate/readiness/operational selection: O05MatchedEvaluationTests.test_matched_live_evaluation_with_doubles returned a failed phase instead of completed. Resolve that production contract as part of F09/F22; do not treat the selection as green. Existing readiness tests asserting permanent blocking must be replaced by valid-build/invalid-build/current-run qualification tests once F21 is implemented, without losing failure-path coverage.

## 24. Required final delivery and completion report [F23–F24]

Deliver all code, tests, current campaign/consumer schema, compatible full generated dataset artifact, notebook, dependency instructions, verify-build report and local rehearsal/export evidence. Record a unique `engineering/reports/FINAL_K8_<id>/HANDOFF.md` and `BUILD_READINESS.json`, plus a short owner `RUN_EXPERIMENT.md`. Commit/push source and compact manifests/evidence only. Protect user changes and all earlier evidence.

Every requirement in REQUIREMENTS.json needs a disposition and exact evidence location. The final handoff says which source/data revision is ready, the verified commands, where the owner inputs/notebook are, what E0 will test, and how artifacts are recovered. No unresolved TODO, fixture-only production executor, hardcoded reduced task count, unconsumed trained head, missing data bundle or stale always-blocked readiness gate may coexist with `ready_for_owner_experiment=true`.

The final completion check is: a fresh owner checkout of the pushed revision plus the delivered data can validate its build, open the notebook and run the complete campaign without another agent patch or design decision. Hardware insufficiency may cause a correct E0 refusal; missing implementation must have been resolved beforehand. If a criterion fails, continue fixing it. If an actual external blocker makes that impossible, report `ready=false` with the precise blocker rather than calling a partial delivery complete.

## 25. Prompt for the implementation agent

Complete BRAMASTRA's FINAL-K8 delivery. Read AGENTS.md, engineering/STATUS.md, engineering/FINAL_EXPERIMENT_EXECUTION.md and engineering/final_delivery/REQUIREMENTS.json, then implement. H01 is only the first included repair; there is no one-hour stopping point and no further owner dispatch needed. Finish the full model/learning consumers, solvable data/labels, cognition/planning, actual A/B controls, tool retention, gated architecture, measured trial/proposer/successor chain, calibrated runtime, checkpoint/export, verifier and Kaggle notebook. Close F01–F24 with production-path evidence. Follow the standing conditional readiness authorization; never hardcode readiness or bypass allocation checks. No local optimizer updates or accelerator run. Use bounded Luna/Sol lanes with exclusive ownership if helpful. Continue through integration and verification, publish the compatible data artifact, write the final readiness/handoff/operator files, and push scoped code. End only with a genuinely ready-to-launch experiment build or an exact external blocker supported by evidence; do not return another plan or a completed subset as the final result.
