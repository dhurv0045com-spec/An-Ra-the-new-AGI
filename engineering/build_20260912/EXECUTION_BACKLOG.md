# Implementation backlog: finish the integrated product

Work sequentially through dependencies. Independent file ownership may permit cheap subagents, but one integrator accepts every package. Each packet can sustain approximately 3–5 hours of useful implementation when prerequisites are available; estimates are not minimum runtime, measured effort or a reason to add code. The owner wants sustained execution. Do not stop after completing the easiest package or produce another blueprint in place of the system.

## B00 — Reconcile the checkout and inventory

Read the build packet, inspect `git status`, and record branch/head/worktree. If BRAMASTRA is checked out elsewhere, use that clean authorized worktree or make an isolated descendant branch; never overwrite another app's edits. Record executable existing APIs and absent target modules. Create `engineering/reports/B2/PROGRESS.md` with packet status, ownership, exact commands, blockers and next action. Preserve source evidence snapshots and the research paper.

Acceptance: branch identity verified, no unrelated edits staged, file ownership assigned, and the first implementation command chosen. This is a brief prerequisite, not an open-ended audit.

## B01 — Config, CLI and one model wrapper

Own `research/models/`, `research/config.py`, `research/cli.py`, package registration and focused tests. Implement the public subcommands described in SYSTEM_DESIGN. Use the existing BRAMASTRA decoder; add optional hidden-state access through backward-compatible interfaces. Implement strict configuration validation, exact parameter accounting, tokenizer identity and seed initialization. Start with tiny and development profiles; larger profile validation is configuration-only.

Acceptance: CLI help is side-effect free; tiny model runs logits/loss/backward through the wrapper; parameter count agrees; repeated seed initialization matches; changing vocabulary/head dimensions changes identity; unknown keys and invalid head geometry reject. No specialized fake model for integration tests.

## B02 — Public codec, sequences and task boundaries

Own `research/experience/codec.py`, packing/batch code and related tests. Implement deterministic goal/event/action serialization, explicit EOS, label/loss masks, reset semantics and provenance sidecars. Support language rows and trajectories through the same model input. Prove padding and segment boundaries cannot leak future or neighboring-task evidence.

Acceptance: provenance-only changes leave tokens unchanged; relevant goal changes alter tokens; one-token answer plus EOS has the exact intended denominator; ragged/exact-full sequences work; a packed batch cannot attend across protected episode boundaries; malformed/oversized records fail explicitly. Changing packing must not silently change which targets receive loss.

## B03 — Qualified data entry and counterfactual groups

Own local dataset manifest loader, semantic split validator and group sampler. Accept only operator-supplied local files or explicitly named tiny fixtures. Produce content hashes, dataset availability state, split inventories and exact counters. Preserve pair groups within splits and minibatches; implement a shuffled unpaired control using the same examples.

Acceptance: duplicate semantics cannot cross splits despite renamed surfaces; empty/missing corpus is DATA_NOT_READY; resume reproduces the next group; reading metadata cannot reveal labels to inference; changing input bytes invalidates identity. No fabricated qualification numbers and no downloads.

## B04 — Real trainer and loss components

Own `research/learning/trainer.py`, objectives and schedule helpers. Integrate answer/EOS CE, optional pair-margin objective, gradient accumulation, optimizer and explicit schedule counters. Add training-only logit treatments and telemetry from SYSTEM_DESIGN. Keep defaults full-loss, no pair auxiliary and controller disabled until explicitly enabled.

Acceptance: one accumulated update matches a reference global batch using supervised-target normalization; target masking cannot remove a valid gold class; full treatment agrees with ordinary CE; enum/schema treatment restores full logits for evaluation; instrumentation changes neither weights nor optimizer/RNG state. Pair loss penalizes a constructed goal-blind prediction and permits correctly separated goals. Do not call these tests capability validation.

## B05 — Atomic checkpoint, exact resume and durable milestones

Own `research/runtime/`, checkpoint schema integration and interruption tests. Implement all state identities, atomic publication and one-writer parent fencing. Include optimizer/scheduler/RNG/scaler, group/replay cursor and controller state. Handle incomplete writes, stale parents, wrong schema/tokenizer and tampered payloads. Maintain separate paths for latest recovery and accepted parent.

Acceptance: the real trainer's next update agrees after fresh-process restoration with an uninterrupted reference, with documented precision tolerance; failure before publication does not destroy the previous valid checkpoint; rotation retains referenced milestones. CPU evidence does not certify TPU. The smoke path must exercise the same publication/restore code used by later training.

## B06 — Formation, preservation and reacquisition controller

Own `research/learning/plasticity.py`, configuration fields and deterministic controller tests. Implement disabled/fixed-schedule control plus FORM/STABILIZE/EXPAND/REACQUIRE/HOLD states. Enforce controller-pool origin, freshness, thresholds, hysteresis and transition limits. Changes occur only at optimizer boundaries. Record proposed and applied decisions separately if a budget boundary prevents application.

Acceptance: crafted traces cover every transition, threshold equality, missing data, stale measurements, cooldown and resume mid-window. Sealed or measurement-pool inputs reject. A collapse cannot automatically force a lower LR. Stable scores without nonzero learning/plasticity evidence cannot be reported as acquired improvement. Do not optimize controller settings through a local sweep.

## B07 — Experience ledger and consolidation

Own remaining `research/experience/` storage/replay components. Implement append-only episode receipts, deterministic stratified replay, quality/ambiguity filters and cursor restoration. Retain enough context for genuine resets or warm-up. Respect declared replay proportions through exact consumed counters, not planned ratios alone.

Acceptance: interrupted/resumed sample order matches; no duplicate learning from cursor rollback; mutation invalidates identity; parent experiences remain accessible after rejected child; a schedule shortfall is explicit. No claim that replay rescues retention is made from plumbing tests.

## B08 — Public environments and inference

Own `research/environments/`, environment fixtures and inference adapters. Build switch first, then inventory and program laboratory using shared contracts. Earlier W02 files are absent from this committed branch; implement against the reviewed interface, not a phantom module. Use finite evaluators, explicit goal attainment, deterministic public state and legal-action semantics. Add free generation and bounded finite-action inference through the canonical wrapper.

Acceptance: actual independent oracle checks agree with tiny exhaustive spaces; invalid action handling follows the stated charging rule; inventory cannot win by reporting initial failure; final submission cost is counted separately from inquiries; hidden mechanism objects cannot enter the codec. Every environment supplies actual rollouts, including a deliberately failed baseline. Do not run a large tournament.

## B09 — Independent evaluation and candidate promotion

Own `research/evaluation/`. Implement complete-answer scoring, paired goal metrics, family retention, Brier/calibration where applicable, cost counters and immutable raw outcomes. Separate training-controller data from measurement and sealed data at the API and storage level. Define candidate decisions using configured margins; default on insufficient evidence is no promotion.

Acceptance: forged success fields are ignored and recomputed; missing/duplicate pairs reject; wrong EOS cannot pass complete-answer; a family regression cannot be hidden by aggregate improvement; repeated sealed-pool use is logged and cannot masquerade as a fresh test. No post-hoc change to benchmark thresholds.

## B10 — Bounded planning and learner-selected experience interface

Own `research/planning/` and narrow collection orchestration. Provide no-planning baseline, bounded public-model rollout planner and a finite diagnostic oracle adapter that cannot be selected accidentally in primary inference. Track real interactions separately from imagined steps. Expose fixed and learned collection-policy interfaces; train neither at scale in this build.

Acceptance: depth/time bounds stop search; predicted outcomes cannot be written as observed facts; oracle mode is explicit in result identity; planner uses only the public model API; uncertainty/model failure falls back to a declared policy. All components operate on the same learner, not a hidden symbolic solver presented as neural reasoning.

## B11 — Integrated local build smoke and operator package

Own integration tests, manifest packaging and `engineering/reports/B2/`. Connect prepare-data -> train -> checkpoint -> fresh-process resume -> infer -> evaluate -> package using tiny fixtures and the real modules above. Run focused checks once they are meaningful; expand only for new failures/changes. Use the stated cumulative CPU/GPU allowance. Do not launch a production profile to prove ambition.

Acceptance: the owner can invoke documented commands without reconstructing notebook state; output manifests expose source/config/data/tokenizer/device identities, exact counters and all feature flags. Package excludes secrets, weights and large corpora from Git. An operator with a valid local corpus can prepare a run and inspect readiness. A missing real corpus remains DATA_NOT_READY even if synthetic smoke passed.

## B12 — Review, commit and handoff

Review all completion criteria, fix integration defects, and update STATUS, the paper's implementation-status note and the build report without rewriting historical outcomes. Create commits containing only BRAMASTRA work. Push normally if existing owner Git authorization and credentials permit; never force-push. Record any actual push failure precisely.

Deliver exact tested commands, tiny profile count, resume evidence, optional-feature status, remaining external data/hardware requirements, resource usage and the next operator command. Never say “trained AGI” or “10x better” on the basis of successful integration. Do not leave TODO/pass/NotImplemented stubs on the primary path or mark a packet complete because a file exists.

## Multi-session continuity

At context boundaries update `PROGRESS.md` with current commit, owned files, complete/partial criteria, most recent exact test output, active process IDs and one next action. Do not restart the audit or replay all successful checks after resuming. Measure elapsed effort only when instrumented. Distinguish an unavailable provider token count from zero. If the system's usage limit blocks execution, preserve the working tree and report the exact unfinished criterion; do not label the overall program complete.
