# B2.2 — make the integrated contracts true end to end

Active implementation assignment, 13 September 2026. Baseline: `c3662b3ca097abf6ca5acaa73eb91cd9d074bc7a`. Read root AGENTS, STATUS, [chief review](../reports/B2_CHIEF_20260913/REVIEW.md), the prior B2 build contract, and this packet. The goal is to accept a reliable integrated training path, not add more research features.

## Execution order and ownership

### C1. Preserve and bind data semantics

Own `research/data/`, `research/experience/codec.py`, `sequences.py`, prepared-data helpers in `commands.py`, associated tests. Carry semantic/content identities, family, trainability, pair group and role from manifest through prepared rows, sampler, loss and evaluation. Use explicit validators at each boundary. Refuse ineligible gradient data. Include entry license/provenance/trainability in dataset identity. Describe content dedup honestly; require qualified mechanism-cluster IDs for generalization claims.

Prepare files atomically in a new directory. Hash every prepared split's exact bytes, schema and row inventory. Train/resume/evaluate must verify those identities before consuming rows. Config identity mismatch remains a clear error, but it is not a substitute for data integrity. Preserve controller/measurement/confirmation separation.

Acceptance: the chief probe's pair IDs survive; nontrainable data never reaches gradient batches; a changed token/mask/group/family/split fails validation; policy/provenance changes alter the appropriate identity; invalid prepared data refuses before model allocation. Add an actual prepare->sample->pair-input test, not just independent pair-loss arithmetic.

### C2. Make optimizer boundaries the source of truth

Own trainer/objectives/schedules, loop/controller/replay integration in `commands.py`, `experience/replay.py`. Coordinate commands.py ownership with C1 rather than simultaneous editing.

Honor `grad_accum_steps` in the shared public train/resume path. Normalize accumulated answer gradients once by all eligible targets; define pair-loss weighting over all eligible pairs in the accumulated group. Maintain separate attempted batches, microbatches, consumed targets, replay entries and committed optimizer updates. Checkpoints remain at complete optimizer boundaries; otherwise implement explicit pending-state persistence as a separately reviewed extension.

Use a persisted scheduler for fractional replay proportions, not `round(1/p)` or invocation-local indexes. Controller evaluation follows persisted global update phase. An empty replay attempt cannot count as an optimizer update. Fail explicitly or apply a configured, logged fallback. Synchronize sampler, controller and replay state into the checkpoint snapshot before every periodic or controller-triggered publication. Avoid duplicate publication at the same update while preserving named checkpoint references.

Acceptance: pure trace tests prove uninterrupted and split invocations choose the same data/replay/controller events at non-divisible boundaries, including p=.3; empty replay produces no fictional step; configured accumulation reaches the same normalized gradient as the global batch. Use fake update callbacks for control-flow tests where a real optimizer is unnecessary.

### C3. Close recovery and lineage integrity

Own `research/runtime/checkpoint.py`, `resume.py`, source snapshot helpers and runtime tests. Integrator alone connects changes in commands.py.

Acquire a real writer lease at train/resume start. Publication must verify ownership and expected current parent under a serialized publication boundary. A stale writer cannot infer a new parent from whatever LATEST currently says. Release ownership in finally; crashed-writer recovery requires an explicit policy that does not steal an active lease merely because a long job exceeds a fixed age.

Validate LATEST pointer directory containment, ID, update and manifest/marker association. Bind manifest identities to their content and validate run/config/tokenizer/data/source compatibility before loading. A legitimate source migration is explicit and recorded; a generic HEAD-dirty suffix is not a code snapshot. Remove unrestricted pickle fallback. Ensure valid native checkpoints still restore using supported safe state encodings.

Acceptance: synthetic payload publication tests cover stale writer, stale parent, redirected/mismatched LATEST, missing COMPLETE, data/source mismatch and corrupted content without training. Preserve the last valid checkpoint through interruption. Final learned acceptance must resume from an intermediate checkpoint rather than only a final checkpoint with manually filled state.

### C4. Bind evaluator claims to a protocol and paired raw cases

Own `research/evaluation/`, evaluation CLI path and tests. Use explicit case IDs separate from mechanism clusters. Compare the same cases with equal labels/families/budgets/renderings/protocols, reject duplicates or incomplete pairing, then bootstrap by mechanism. Use the same declared aggregation for point estimates and intervals.

Promotion consumes a validated evidence bundle tied to parent, child, protocol, pool and data identities. Missing uncertainty is insufficient evidence when required by the protocol. Pair criteria are mandatory for paired-goal protocols and explicitly not applicable for others; None must not silently bypass a required gate. Reject nonfinite metrics and malformed thresholds. Required protected-family evidence cannot be substituted by zero or unrelated rows.

Route confirmation outputs to confirmation storage. Preserve pair roles end to end and compute pair metrics from actual CLI outcomes. Resolve generation caps from the evaluation protocol/context budget; do not hard-code eight tokens for all tasks. Correctly stopped byte-level and non-ASCII outputs need independent examples.

Acceptance: mismatched labels/cases cannot produce a delta; incomplete evidence cannot promote; correct case pairing is order invariant; paired CLI rows no longer fail from missing role; confirmation records do not appear in measurement storage. No real model training is needed for most of these checks.

### C5. Runtime resource policy and one final acceptance run

Own smoke ledger/runtime policy, CLI lifecycle and integration receipts. Keep the current build ledger intact: reported **193/200 CPU optimizer updates and 111.234/300 seconds**, zero GPU use. The current chief review adds zero updates. Recheck live ledger before executing anything.

No fresh allocation is implied by this packet. Use non-learning or gradient-only tests for repairs. Reserve the remaining allowance for one final comparison of at most six REAL optimizer updates total: uninterrupted three versus interrupted one plus resumed two, through the actual shared path. Include replay and controller scheduling with a nontrivial boundary; use pure traces to cover additional branches. Count ALL learned updates in subprocesses/tests/retries. If the existing allowance cannot accommodate an informative verification, complete every nondependent repair and report the exact validation still requiring owner allocation. Do not silently reset the ledger or enable the full learned suite.

A run's own resource policy is separate from the build smoke ledger. Record/reserve budgets before work; check deadlines inside the loop; append consumed work and failure/timeout in finally paths. Non-smoke production resume must respect its declared run policy rather than the tiny build ledger. Implement and test this with mocks/fixtures; no production training, paid compute, downloads or accelerator campaign is authorized.

Acceptance: one command exercises verified preparation, configured accumulation, optional pair-input routing, checkpoint, fresh-process restore, inference and evaluation within remaining allowance. Check model/optimizer/RNG/cursor/controller/replay equivalence and the future event sequence, not only current weights. Save immutable identities, exact commands and limitations. A numerical-only discrepancy requires diagnosis and a declared tolerance, never relabeling a divergent stream equivalent.

### C6. Integrate, correct status and return to chief

Update `engineering/reports/B2_2/PROGRESS.md` continuously and finish `HANDOFF.md` with each R1–R7 disposition, exact source/test/receipt identities, residual limits and resource accounting. Preserve prior agent reports as history; don't rewrite old measurements. Resolve contradictory status statements. Push BRAMASTRA normally under existing owner authorization; no force-push or unrelated changes.

The chief will accept B2.2 before choosing data scale or a capability-learning experiment. Completion requires integrated behavior, not a larger count of green isolated tests. The later [complete learner program](../master_program_20260913/README.md) incorporates C1–C6 as M00 and authorizes continued provisional implementation after its dependency checks/report; it supersedes this packet's former stop-and-return boundary. It does not authorize a new training campaign or scientific promotion.

## Multi-agent discipline

Use one integrator and Luna for bounded independent work if available. Shared commands.py edits must be sequential or mediated by the integrator. Runtime publication has one owner. Data and checkpoint changes require an interface agreement before edits diverge. No helper receives its own independent smoke budget.

## Operator prompt

The copyable [AGENT_PROMPT.md](AGENT_PROMPT.md) invokes this complete assignment. This packet and the chief's reproducible findings are authoritative over the previous blanket claim that B00–B12 is complete.
