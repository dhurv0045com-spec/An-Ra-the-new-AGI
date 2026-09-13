# Whole-repository implementation orders

Execute M00–M18 against BRAMASTRA. This is a connected build, not a request for another paper or a collection of uncalled modules. Read [architecture](ARCHITECTURE.md), [algorithms](ALGORITHMS.md), [data/training](DATA_AND_TRAINING.md) and [evaluation](EVALUATION.md) as binding design inputs. `program.json` records dependencies; this file supplies acceptance criteria.

## Dispatch and file ownership

Use one integrator. Luna may implement or audit independent bounded packages; use Sol only when available and appropriate. Give each executor packet ID, prerequisite receipts, owned paths, excluded paths, resource allowance and report directory. In shared worktrees, the integrator alone edits `research/config.py`, `research/commands.py`, `research/learning/trainer.py`, `research/models/wrapper.py`, root entry documents and shared checkpoint schemas. Package owners request narrow integration changes with an interface note. In isolated worktrees the same ownership agreement governs integration and conflict resolution.

All paths below are relative to `bramastra_lab/research/` unless prefixed otherwise. Proposed files are targets, not claims that they exist. Do not rename existing APIs gratuitously; adapters and explicit versioned migrations are preferred. Every package owns corresponding focused `tests/test_research_*` additions, and its own `engineering/reports/MASTER_20260913/Mxx/` receipt. No package edits another package's immutable result or old run directory.

Before dispatch, replace a package's abstract allowance with the **shared remaining allowance**. A helper has no independent training budget. Most acceptance criteria can use pure functions, deterministic fake outputs and bounded gradient checks without optimizer steps. Record what was actually verified. If learned qualification is unallocated, deliver implemented code with `LEARNED_UNQUALIFIED`; do not stop unrelated build work or invent a result.

## M00 — Repair the foundation

**Prerequisite:** reviewed B2/B2.1 source and chief R1–R7 probe. **Owner:** integrator; delegated data/runtime/evaluation work must have nonoverlapping ownership. **Scope:** existing B2.2 C1–C6 and their owned paths.

Execute the full [B2.2 work order](../phase_b22_20260913/README.md). Bind prepared bytes and eligibility, restore pair metadata, fix true accumulation and global stream/controller state, enforce checkpoint identity/fencing, correct comparison protocols, and preserve failure/resource accounting. Add a disposition for every chief finding with the command and evidence that would fail on the reviewed code and pass on the fix.

**Acceptance:** all C1–C6 criteria and focused negative probes pass; final actual-path resume comparison fits the shared remaining allowance or is precisely marked blocked. Model/optimizer/RNG/cursors/controller state and next-event sequence must agree. Commit the repair report. Later packages may proceed as provisional build integration after this dependency gate; chief scientific acceptance is separate. Do not spend reserved updates on unrelated new objectives.

## M01 — Define the complete experience and objective contract

**Depends on:** M00. **Owned paths:** new `experience/trajectory.py`, `experience/supervision.py`, and additions to `contracts/` coordinated by integrator.

Implement the records and tensor semantics in DATA_AND_TRAINING sections 1–3. Separate public payloads, provenance, observed transitions, predictions and teacher labels. Introduce explicit schema versions, validation errors and migration boundaries. Document cost-versus-reward and terminal-versus-truncation semantics. The public-view allowlist must remove episode/seed/split identifiers from learned tokens.

**Acceptance:** round-trip records preserve identities; hidden/provenance keys cannot reach token content; NaN targets and incompatible versions fail; observations cannot be forged by serializing a prediction; no later event enters an earlier prefix. Rejection occurs before model allocation. Provide fixtures covering a valid episode, invalid action, partial episode, pair group and unknown mechanism identity.

## M02 — Make decisions share a correctly conditioned core

**Depends on:** M01. **Owned paths:** new `models/decisions.py`; integrator applies wrapper/config changes.

Implement independent candidate branches, legal-action normalization and action-free value prefixes. Retain tied full-vocabulary answer output. Define optional head schema/count identity and explicit checkpoint migration behavior. Build the inference adapter so collection and planning call the same action scoring implementation.

**Acceptance:** candidate permutation equivariance, stable action identities, padding/mask behavior, no legal-action rejection, value-prefix independence, and baseline parameter-count compatibility. Bounded gradient checks show teacher action/value losses can reach the appropriate heads and shared core once M05 connects them. No random head is described as trained. Optional prefix caching must match the uncached control before use.

## M03 — Predict typed public outcomes

**Depends on:** M01, M02. **Owned paths:** new `models/world.py`, `experience/outcomes.py` and outcome schemas.

Implement shared-decoder `predict_step`, finite-support scoring and bounded open-output sampling. Keep parsed duplicate outcomes, invalid mass and predictor identity explicit. Compute publicly deterministic budget fields through the protocol. Provide a transition renderer with field-level target spans and masks. Do not create a separate latent model or privileged simulator as a shortcut.

**Acceptance:** known fake distributions normalize and aggregate duplicates correctly; parse failures retain mass; predicted outcomes cannot enter observed-ledger APIs; changing action changes the encoded condition; actual hidden environment state is never read. Show an end-to-end world batch reaches decoder logits with the correct target mask. Learned prediction quality remains unqualified until allocated measurement.

## M04 — Connect memory to the actual input path

**Depends on:** M01. **Owned paths:** new `memory/` package and codec adapter; integrator owns shared codec edits.

Implement deterministic eligible-record retrieval, frozen index identity, episode reset, explicit MemoryContext and context allocation. Use the same renderer for policy, world prediction and answer inference. Keep raw observations, retrieved records and imagined branches visibly distinct in traces.

**Acceptance:** forbidden records cannot rank or render; retrieval ties are stable; changed index content invalidates identity; context caps reject mandatory overflow and drop optional whole records deterministically; reset removes prior episode state. A fake-model capture proves retrieved public content reaches the actual inference call, and memory-off removes it. Test evaluation memory freeze and current-episode exclusion.

## M05 — Implement the unified objective router

**Depends on:** M01, M02, M03. **Owned paths:** new `learning/router.py`, extensions to `learning/objectives.py`; trainer integration through integrator.

Implement A1–A3 objective sums/counts, pair group normalization, world masks, teacher action CE, value regression and the single-window on-policy loss contract. All objectives use one optimizer boundary, scheduler and counters. Reject stale behavior identities and unsupported missing return targets. Keep compatibility configs answer-only.

**Acceptance:** hand-computed scalar loss cases, pair sign tests, unequal-microbatch gradient equivalence, sparse objective denominator cases, stop-gradient targets and feature-off gradient routing. No optimizer step is needed for most checks. Verify zero-denominator omission differs from a stage with no eligible data. One optimizer path is used by document and trajectory batches; a second private training loop fails acceptance.

## M06 — Compile episodes and teacher supervision into learning data

**Depends on:** M01, M02, M05. **Owned paths:** new `experience/prepare_episodes.py`, `learning/teachers.py` and task-supervision adapters.

Convert real ledger receipts into world, answer, action and return targets with separate eligibility. Bootstrap with at most six short fixed-policy episodes from the existing real environment collector, with no learned model or optimizer updates; record their actual costs and immutable seed-ledger identity. This prerequisite is available before M08. Add a declared teacher interface for training-only action distributions and counterfactual goal groups. Validate teacher targets with task verifiers. Preserve failure episodes without training incorrect submissions as gold answers. Keep oracle diagnostic and train teacher roles distinct.

**Acceptance:** an observed fixture compiles into exact expected targets/counts; truncated return labels are excluded; teacher ties distribute mass correctly; sealed tasks cannot become teacher inputs; pair groups remain joinable after sharding. The output is consumed by M05 without bespoke agent-only code. Report qualified versus unusable transitions by reason.

## M07 — Replace continuation scoring with bounded branching search

**Depends on:** M02, M03, M04. **Owned paths:** `planning/` and planner traces; other consumers use its stable interface.

Implement A5 recursive public-outcome planning on finite declared supports, root policy pruning, conservative invalid mass, actual horizon accounting, optional answer-information criterion and global ResourceVector budgets. Open-text prediction does not enable recursion without the separately specified estimator/uncertainty contract. Keep current simple root scorer as a named control. Add a real decision adapter choosing one root action; do not return an unordered trace and let callers guess what was selected.

**Acceptance:** a deterministic two-step fixture requires a non-greedy first action; changing downstream outcomes changes selection with root logits fixed; entropy distractors do not falsely earn goal-information credit; node/token/time limits stop search; imagined legality never consults future real environment state. All simulated steps stay in a prediction trace. No “depth two” label on a single text completion.

## M08 — Close collection-to-learning without hidden fallbacks

**Depends on:** M04, M06, M07. **Owned paths:** `collection/` and new `orchestration/experience_cycle.py`.

Join public views, memory, model/planner decisions, actual interactions, observed receipts and prepared objective shards. Record action probabilities and frozen behavior identity for on-policy use. Provide explicit fixed-policy, learned-policy and planner modes. Fallbacks remain possible for operations, but their counts and reasons must be visible and excluded from a claim of pure learned-policy performance.

**Acceptance:** a fake frozen model completes a session and produces traceable trainable world/action/value records; the same receipt IDs reach the canonical trainer input; failure and cancellation preserve charged costs; no duplicate episode admission on retry; imagined transitions are rejected. A deliberately broken learned adapter cannot silently produce a passing “learned” run entirely through fallback.

## M09 — Build verified practice and curriculum state

**Depends on:** M06, M08. **Owned paths:** new `curriculum/` and task-factory selection adapters.

Implement qualified proposal admission, mechanism-level deduplication, failure classification and the transparent frontier/replay/uniform selector in A7. Persist estimates, selection RNG, counts and rationale. Add a registry for public skill recipes with explicit primitive-action expansion; initial recipes are engineered and tagged as such.

**Acceptance:** invalid or unsolved proposals do not become gold data; impossible task buckets do not monopolize practice; uniform exploration remains nonzero when configured; restored state produces the same future bucket sequence; controller-only data drives selection. Macro action costs equal the sum of executed primitives. No sealed example enters curriculum feedback.

## M10 — Wire acquisition and preservation through one trainer

**Depends on:** M05, M08, M09. **Owned paths:** `learning/plasticity.py`, replay/controller adapters, new preservation objective module if implemented; coordinate existing replay ownership.

Connect fixed stratified replay and the existing controller to the new objective batches. Preserve global fractional scheduling and exact replay state from M00. Add optional parent-anchor distillation with explicit source/cost identity, disabled by default. Old trajectories remain ineligible for on-policy gradients unless a future reviewed correction method is implemented.

**Acceptance:** family/group quotas and shortfalls are exact; resume preserves future selection and controller transitions; stale or missing metrics cannot create qualification; teacher/replay/PG eligibility remain separate; a preservation-only fixture cannot be reported as acquisition. Numerical loss tests cover parent-equals-child divergence and masking. Learning benefit is reserved for the evaluation protocol.

## M11 — Extend the independent examiner to the entire loop

**Depends on:** M01, M03, M07, M08. **Owned paths:** `evaluation/` plus immutable protocol/receipt schemas.

Implement EVALUATION.md's matched-case comparison, task-family aggregation, resource accounting, world calibration, action legality, complete-answer/EOS, goal pairs, acquisition/retention and contamination checks. Carry all feature flags, teacher priors, fallbacks and memory scope into the comparison receipt. Promotion prerequisites are explicit protocol fields, not a trusted `evidence_complete` boolean.

**Acceptance:** mismatched cases/budgets/labels fail; invalid/nonfinite scores fail; missing required pair or uncertainty receipts block promotion; confirmation output remains in its declared split; seeded clustered resampling is reproducible; protected-family regression prevents aggregate-only acceptance. Tests include an apparent gain caused by a larger interaction allowance, which must not qualify as an equal-budget result.

## M12 — Implement transactional candidate improvement

**Depends on:** M08, M10, M11. **Owned paths:** new `orchestration/candidates.py`, proposal schema and candidate registry.

Implement A8 proposal-to-child-to-comparison state machine, isolated lineage creation, idempotency and accepted-parent publication. The initial proposal language allows bounded curriculum/replay/schedule settings; arbitrary code mutation is excluded. The learned proposal service may be a later client; a deterministic fixture must exercise the state machine now.

**Acceptance:** crash injection at every durable boundary leaves either the old accepted parent or a fully accepted new parent; retries do not repeat training or duplicate evidence; rejection preserves parent and compact child evidence; comparison cannot change its own preregistered thresholds; chief acceptance is required for learned promotion. Forge a fixture receipt, a mismatched approval hash and a zero-update weight-learning claim: all must be rejected by the actual publication API, not only a CLI warning. Fixture registry and learned registry are disjoint. No test needs a long candidate learning run.

## M13 — Make broader from-scratch data a real supported path

**Depends on:** M01, M04, M05. **Owned paths:** `data/` source adapters and document-task compiler; coordinate original manifest/sampler owners.

Support local licensed/provenanced documents and code with deterministic document boundaries, split groups, source deduplication and mixture accounting. Implement the missing-corpus readiness status and corpus inspection report. Join document prediction and grounded episodes through the same objective router. Do not download arbitrary corpora or introduce pretrained weights to close a checkbox.

**Acceptance:** valid local fixture documents prepare deterministically; untrainable sources never emit training rows; source changes invalidate hashes; document splits and mixture counters remain correct across resume. Real corpus availability is reported accurately. Keep synthetic fixtures visibly separate from actual pretraining sources.

## M14 — Qualify richer environments and transfer boundaries

**Depends on:** M01, M06. **Owned paths:** `environments/` new families/verifiers and generator manifests.

Qualify current families and implement the additional data domains specified in DATA_AND_TRAINING where independent verification is feasible. Prioritize complementary-query worlds, new program compositions and document-assisted tasks before cosmetic task proliferation. Add a bounded tool/code execution adapter only with explicit resource and isolation semantics. Provide surface, mechanism, compositional and length split generators.

**Acceptance:** oracle/checker agreement on bounded exhaustive fixtures; nontrivial random/fixed baselines; unsatisfiable/ambiguous cases are correctly classified; renamed copies stay in the same known mechanism cluster; output and timeout bounds work; hidden state never appears in public views. Number of worlds generated is not a capability result.

## M15 — Implement actual-backend readiness and recovery tooling

**Depends on:** M00, M05, M12. **Owned paths:** `runtime/readiness.py`, backend adapters, packaging/run-manifest tooling; integrator owns checkpoint schema edits.

Extend device-independent preflight, profile forecasts, exact runtime identities, durable checkpoint export/import and resource accounting. Implement supported backend adapters against the versions actually available; inspect installed dependencies and official documentation when needed. Keep unavailable hardware explicitly unverified. Profile the intended update/restore path, not an unrelated matrix multiply.

**Acceptance:** CPU metadata and failure-path fixtures work; unsupported backend rejects clearly; unsafe checkpoint load remains impossible; export manifest verifies identities without exposing credentials; resource exhaustion triggers a recoverable stop. Live GPU/TPU evidence is supplied only if authorized and actually runnable. A dry-run configuration is not accelerator readiness.

## M16 — Deliver the usable operator workflow

**Depends on:** M12, M13, M14. **Owned paths:** CLI adapters, operator documentation, example manifests; integrator applies shared CLI/parser changes.

Expose the completed session, collection, episode preparation, candidate and comparison functions through documented commands. Publish one canonical runbook that resolves BRAMASTRA, inspects readiness, selects a checkpoint, runs a bounded session, saves observed experience and proposes a child. Include resume, cancellation, failed-run recovery and DATA_NOT_READY paths. Actual training commands require an allocation reference.

**Acceptance:** documented commands match parser help; an operator can run a fixture workflow without importing a private script; all outputs contain identities and resource/status fields; changing a config creates the appropriate fork; no default command launches long training. Clearly label random-model and deterministic-fixture demonstrations.

## M17 — Prepare the qualification campaign without running it

**Depends on:** M11, M13, M14, M15. **Owned paths:** new versioned experiment protocol manifests and `engineering/reports/MASTER_20260913/qualification/` design receipts.

Translate EVALUATION.md into executable protocols with data manifests, exact treatment switches, independent-seed policy, resource budgets, primary/secondary outcomes, uncertainty rules and rejection criteria. Include branch-comparison adapters only where a matched task/compute/data interface is possible. Mark unknown sources, hardware and thresholds as blockers that prevent launch, not as guessed values.

**Acceptance:** protocol validator rejects incomplete launch manifests; dry-run enumeration shows paired cases and resource accounting; each proposed mechanism has a disabling ablation and a falsifying result; planned comparisons fit an explicit future allocation before they can become runnable. No experiment is launched merely to finish this package.

## M18 — Integrate, audit and hand off the complete build

**Depends on:** M16, M17 and all their transitive prerequisites. **Owned paths:** master status/report, root entry links and integration checks through the integrator.

Exercise six fixture stories: answer-only baseline; multi-objective trajectory routing; outcome-sensitive planning; memory scope/reset; collection-to-child preparation; crash-safe reject/accept transaction with a simulated examiner. Use the actual public orchestration functions, injecting only bounded model/environment/evaluator test doubles at documented seams. A simulated examiner acceptance must be labeled a fixture and cannot publish a learned production parent.

Review every package criterion and every current chief finding. Remove dead adapters or integrate their callers. Confirm baseline switches, parameter/schema identities, end-to-end provenance and command help. Summarize focused checks once; do not rerun all learned tests reflexively.

**Acceptance:** `engineering/reports/MASTER_20260913/HANDOFF.md` maps M00–M18 to exact commits/files/commands/receipts and distinguishes implemented, locally checked, accelerator-tested, experimentally supported and unqualified. Supply remaining blockers, actual effort/usage if available, cumulative resources, source identity, next allocated experiment and a prompt for the owner. Update STATUS from evidence and push BRAMASTRA normally. No force push, hidden external messages, large weights or unrelated user files.

## Milestones and stop conditions

The useful milestones are: **foundation repaired** (M00); **all core outputs have legitimate supervision** (M01–M06); **the learner can interact, remember and prepare its own observed experience** (M07–M10); **candidate improvement can be evaluated transactionally** (M11–M12); **broader data, environments and runtime are supported** (M13–M16); **qualification-ready build handed off** (M17–M18).

Stop a dependent action on corrupted identities, hidden-data leakage, exhausted allowance, unverified source eligibility or unsupported backend. Continue independent pure implementation. Resolve ambiguous design choices by preserving the documented baseline, adding a narrow versioned interface and recording the decision; do not ask the owner to make routine engineering choices. Escalate only a real missing authority/resource choice or an architecture contradiction that cannot be resolved within this packet.
