# Ten packages, one integrated delivery

The lead implementation agent owns integration. Start from the latest BRAMASTRA checkout, inspect dirty files and preserve the current foundation work. Use isolated worktrees for parallel work where available. In a shared directory, assign the paths below exclusively; no simultaneous edits to episode.py or ops.py. Delegate bounded tests/reviews to Luna or Sol when useful. Do not give an agent a vague instruction to build AGI.

Each substantial package can occupy roughly 3–5 useful engineering hours, but dependency and acceptance completion determine the actual duration. No minimum token consumption, padding, background training or repeated full-suite runs. This is one consolidated delivery with intermediate code commits, not ten handoffs requiring owner micromanagement.

## U01 — Make the tasks and labels coherent

Prerequisites: REVIEW J01/J02 and CONTRACTS sections 1–2. Own `research/data/k8_bundle.py`, `research/environments/k8_live.py`, data contracts and focused task tests. Agree on action/payload schema with U02 before editing shared types. Exclude cognition, trainer and campaign budgets.

Implement the public rule specification for the base evaluation task; derive teacher answers from the same sampled environment state and independent verdict used by evaluation. Compile actual received history into answer examples. Unify inventory/program payload keys. Provide task solvability witnesses and canonical split checks. Regenerate affected small test bundles in new paths; do not overwrite old datasets.

Accept when adversarial private twins with indistinguishable permitted observations cannot demand different unique answers in the base family; witnesses fit the four-inquiry budget; teacher submissions replay successfully; corrupted teachers fail verification; public serialization contains no witness/private answer; two different histories with the same initial goal produce their appropriate different targets. Explicitly classify intentionally ambiguous cases outside the uniquely scored base inventory.

## U02 — Unify state/action/feedback interfaces

Prerequisites: U01 schema decision. Own shared renderer/codec contract and compiler changes; coordinate exclusive episode.py ownership with U03/U04. Own `campaigns/phases/compiler.py` and relevant experience schemas/tests. Exclude trainer stepping and RSI.

Introduce a complete DecisionExample with input-target separation, legal typed candidate descriptions and response schema. Eliminate six/eight-token content cuts. Use actual environment event prefixes. Validate context eligibility before training, not by slicing. Add schema identities to data and checkpoint compatibility checks and explicitly reject older incompatible bundles. Preserve complete tool result/error fields and typed timestamps. Unknown temporal scope remains unknown; do not lexically order it.

Accept when distinct legal candidates stay distinct, terminal payloads round trip through every environment, complete training/inference input tokens agree for the same decision, hidden-only changes leave the public input unchanged, and changing received evidence changes the input. Include negative numeric timestamps, sub-microsecond differences where supported, unknown-time evidence and multivalued predicates. No duplicate `mark_conflicts` definitions or legacy dead implementations left in the active module.

## U03 — Connect trained objectives to executable decisions

Prerequisites: U02. Own `learning/k8_scoring.py`, policy/prediction adapters under coordinated episode.py ownership, relevant `ProductionOps.construct_objectives` changes and tests. Exclude episode budget loop and E5 scheduling.

Use the trained action head for candidate selection and the answer decoder for typed submission. Connect world feedback prediction and value evaluation to the identical trained representations. Record every objective's consumer, prefix, target, head, eligibility denominator and gradient path. Keep K8 weights unchanged. Allocate input/target/mask/span tensors on the actual model device; reject overlong and all-illegal inputs before model execution. Do not substitute CPU tensors or a separate decoder path that bypasses S1 gates.

Accept with backward-only tests showing eligible objectives affect their intended parameters and ineligible terms do not. Compare complete production compiler-to-consumer representations. A parameter perturbation in the trained action head must affect policy scores, and a value-head perturbation must affect the declared value consumer. These are connectivity tests, not capability claims. Test CPU now; include a CUDA-marked test for owner E0 qualification, reported NOT RUN locally.

## U04 — Make planning and the episode loop terminate correctly

Prerequisites: U02/U03 interfaces. Own planner/kernel under a single episode.py editor, plus runtime loop tests. Exclude data generation and statistical aggregation.

Carry full state through planner→predictor composition, transition an imagined copy using typed feedback, recompute hypothetical legality, honor terminal predictions, and bind selected roots/paths to stable IDs. Fix cumulative versus per-selection counters. Check exhaustion before prediction. End on zero-call preflight refusal or exhausted nodes; do not spin. Include the parent job deadline. Preserve actual call counts and separate them from hypothetical nodes, emitted events and real actions.

Accept when the independent audit's liveness and evidence-conditioning cases pass; repeated planner selections never decrease cumulative calls; every loop iteration makes permitted progress or terminates; call/node caps are never exceeded; depth-two predictions receive the applied first outcome; legal-list permutation preserves non-tied choices; no imagined record enters received history. Run liveness tests in child processes with explicit deadlines so regressions cannot hang the suite.

## U05 — Restore meaningful E2 comparisons and measurements

Prerequisites: U01–U04. Own `campaigns/phases/e2.py`, E2 report reducers and tests. Exclude episode.py unless the lead grants ownership.

Use actual A/B checkpoint consumers, retain scripted/oracle controls under separate identities, and run matched frozen mechanisms with canonical training exclusion. Join selected node predictions by decision/action/path/horizon and retain invalid/missing outcomes. Test multiple decisions with repeated action names and a chosen depth-two path. Empty feedback is unknown, not an exact-match success. Publish consumed checkpoint/schema identities per mode.

Accept when a spy on the actual A boundary proves A-direct and A-fixed call the A decoder; changing B alone leaves A outputs unchanged; mislabeling a fixture fails learned eligibility; every paired row has the same mechanism/seed set; action/forecast joins match the selected path; all per-mode counts reduce from traces; late rejected cases cannot silently be replaced. Independent reference ceilings remain separate from learned comparison rows.

## U06 — Qualify the real optimizer window and E0 geometry

Prerequisites: U02/U03 data and objective contracts. Own `campaigns/worker.py`, calibration, session/update-window integration, focused tests and needed ops changes under lead coordination. Exclude cognition and RSI policy code.

Trace actual microbatch/accumulation settings into production update windows. One logical optimizer window aggregates all eligible term numerators and denominators before normalization and commits once. Preserve the existing declared normalization convention; make it explicit in the protocol rather than changing it accidentally through token slicing or one-row updates. A one-candidate action loss does not count as discriminative supervision. Disabled terms contribute neither gradient nor denominator.

Time the whole unit: data-to-device, all objective forwards, backward, optimizer/scaler, plus separately measured checkpoint/evaluation overhead. Synchronize CUDA at timing boundaries during owner qualification. Qualify full K8 geometry and heaviest registered mode, including the actual planner response/call envelopes. Tiny resume remains a diagnostic, not full-model qualification. Add pilot and child-resume attempts/commits/exposure into top-level ledger reductions without double counting restored historical counters.

Accept locally with whole-window versus microbatch backward equivalence and fake-clock/counter tests. Accept E0 only when the full-profile CUDA tests, fresh-process resume and timing evidence exist on the owner hardware. Failed/overflow attempts still consume their real time. Missing/incomplete pilot measurements must block protocol freezing. Show that the sum of all E0 commits remains within 128 per worker, including resume and pilot work.

## U07 — Repair trial authority, budget inheritance and retention

Prerequisites: U02/U03/U06 contracts. Own `campaigns/trial_service.py`, E5 trial/support/query helpers, reservation validation tests; exclude proposer control sequencing until U08.

Bind real JobInput or a explicitly validated reservation mapping to the correct API. Check allocation, reservation, job, device, source and deadline correspondence; do not invent missing production authority. A local no-update diagnostic is a distinct explicitly labeled mode. Child trial caps inherit the block/job/campaign deadline and remaining allowance. Preserve failed trial records.

Resolve query/protected references to complete public DecisionExamples. Fail on missing protected tasks; do not evaluate an opaque mechanism ID as if it were the task or silently drop it. Validate published payload existence, parent, schema/config/source identity and reloadability before admitting learned evidence.

Accept with real-session no-step admission tests, expired/mismatched authority rejection, simulated deadline exhaustion across consecutive trials, false checkpoint-ID rejection, protected-reference corruption and protected-input parity. No local optimizer step is needed to establish these code paths. Restore actual weights in a bounded tiny CPU diagnostic only if needed, never claim a trained checkpoint from random weights.

## U08 — Execute the actual proposer/successor chain

Prerequisites: U07 and valid compiler consumers. Own E5 orchestration and method-selection serialization/tests; coordinate helpers with U07. Exclude scheduling ledger reducers owned by U09.

Train P0 before capture; collect a distinct fresh archive for successor training; fork the trained P0; apply selected recipe to P1 and M0 to P_fixed; train both on identical newly admitted data within equal allowance; capture per-task selections independently from P1, P_fixed and frozen P0 before any confirmation outcome. Record exact model/checkpoint/transcript/archive cutoff identities. Keep teacher fallback in a separate control stream.

Accept with a deterministic stateful model double whose outputs change after the designated training boundary: a pretraining M0 choice must not survive when posttraining output is M2; P1 and P_fixed can produce independent selections; confirmation outcomes cannot influence already frozen proposals. Also run the real model boundary with random weights and no updates, accepting invalid proposals as failures rather than teacher successes. Never represent a test double's state change as actual learning.

## U09 — Reconcile schedule, evidence and exported outcomes

Prerequisites: U05–U08. Own E5 event reduction, campaign export/summary integration and associated tests under lead coordination. Enforce 12/12/6 tasks, three methods, 45/45/90-second trial caps and the registered block reserves. Default production must use the full protocol; reduced fixture counts require explicit diagnostic configuration and fixture evidence.

Reduce optimizer commits, attempts, supervised exposure and elapsed charges from all archive, successor and confirmation events. A successful trial count is not an optimizer update count. Enforce phase totals and immutable allocation identity across retries. Keep raw task-level confirmation tables, protected outcomes, failure status and selection provenance. Independent policies use the same measured table; P_fixed and fixed_M0 remain distinct.

Accept with a simulated complete schedule, injected failure/retry at each block, no duplicate events, top-level totals equal to child sums, and a fake 64-character checkpoint rejected from learned evidence. Export missing-payload, partial and failed phases honestly; do not declare experiment success merely because a report file exists.

## U10 — Deliver one coherent qualification bundle

Prerequisites: all prior packages. Lead owns integration, entry-point documentation, notebook consistency and acceptance report. Run the focused changed-path suite once after integration; broaden only for a specific unresolved risk. Then execute the CPU no-update vertical trace for every family/mode, the bounded liveness diagnostics, and complete artifact/consumer parity checks.

Deliver `engineering/reports/INTEGRATED_READINESS_<unique_run_id>/HANDOFF.md` and the acceptance matrix. Preserve all prior receipts and dirty user work. Document data/schema migrations and exact GPU-only checks. Push only scoped committed work. Do not bypass the chief readiness manifest. The chief can clear code readiness after reviewing the connected evidence; the owner then performs the already budgeted GPU qualification and experiment.
