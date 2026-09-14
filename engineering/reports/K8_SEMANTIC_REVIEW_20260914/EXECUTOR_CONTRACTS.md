# Real execution contracts for BRAMASTRA K8

This is the implementation specification for the next delivery. It supplements [experiment.md](../../../experiment.md), whose treatment weights, sample sizes, evaluation rules and 480-minute allocation remain authoritative. It replaces neither the architecture research objective nor the scientific controls. Its purpose is to make the full campaign's behavior unambiguous enough that an implementation cannot pass by merely emitting expected labels.

## 1. Job, learner and evidence identities

Extend the shared interfaces before parallel phase edits. A job must carry immutable job/slot/phase/arm/seed IDs, physical and local device identities, source/data/tokenizer/configuration/protocol hashes, an authorized reservation, absolute deadline and an explicit update target or time-budget rule. Training phases cannot silently choose four or three updates from function defaults. Context length and optimizer settings come from the frozen protocol, not just a named model profile.

A parent reference contains the exact checkpoint ID, manifest path, payload hash, architecture/config identity and the lineage it belongs to. A string such as E1-B-1701 is a lookup key, not a model state. Resolving it must either return a verified checkpoint record or fail. The learner handle owns model, optimizer, scaler, stream cursor, RNG/controller state and the validated reservation. It exposes its actual current checkpoint/architecture identities; phase code cannot invent them.

A completed training receipt derives successful and attempted updates from durable update events and actual counter deltas. A completed evaluation receipt derives evaluated cases and success from verifier outcomes. Neither accepts a caller-supplied task count as optimizer work. All outputs declare evidence kind: fixture, local integration or learned campaign. Fixture receipts cannot enter accepted campaign aggregates.

## 2. Shared operations must support the actual lifecycle

Replace the inadequate init/snapshot/train/generate-only interface with these responsibilities, using existing repository facilities where possible:

| Operation | Preconditions and result |
|---|---|
| initialize_random | E1 or explicit permitted pilot only; full frozen configuration plus seed; returns learner with reservation attached |
| restore_parent | Hash-validated payload and compatible config/architecture; returns the actual parent model and requested optimizer policy |
| fork_child | Copies verified parent state into an isolated child lineage; declares fresh/preserved optimizer and RNG policy; no sibling mutation |
| construct_objectives | Consumes compiled episode targets and public histories; returns live sums and per-objective denominators without stepping |
| apply_update | Uses one complete accumulation window, one normalization and one explicit finalization; emits actual attempt/commit events |
| evaluate_episode | Runs model decisions through the environment with observed actions/results/costs; invokes independent final verifier |
| publish_checkpoint | Calls the existing runtime save_checkpoint with complete identities, writer token and expected parent; atomic payload publication |
| restore_verify | Loads the artifact in a fresh process and validates state/next stream under the required configuration |

Do not branch on `type(ops).__name__`. Dependency injection should replace expensive operations through the same validated interface, not disable target construction, parent checks or evidence validation. The test implementation must enforce the same shapes, required fields, lineage rules and objective eligibility as production. A permissive recorder alone is not an integration test.

The checkpoint adapter must use the existing `runtime.checkpoint.save_checkpoint` signature, including tokenizer/data/source identities and publication fencing. Do not rename the missing publish_checkpoint import without supplying the other required arguments. Keep distinct child directories for arm/seed/phase and preserve expected-parent checks; no generic `k8` identity can replace source or data hashes.

## 3. Compile real supervision before executing E1

Data compilation produces an immutable sequence of public decision states with separately stored target channels. Token targets include answer/EOS; world targets describe actual next observation after an allowed action; action targets describe the declared teacher policy over legal candidates; value targets follow the specified return convention; pair targets bind actual paired goals and their respective valid answers. None may be a fabricated fixed transition, uniformity loss over two arbitrary token lists, or unconditional zero value chosen for convenience.

The optimizer window owns denominators across all of its microbatches. Count eligibility first, scale each objective sum by its own window denominator and weight, and perform backward without unbounded graph retention. Partitioning the same window into unequal microbatches must yield equivalent gradients in a deterministic check. Disabled terms must not execute extra forwards; enabled eligible terms missing targets must fail before the optimizer boundary.

Training sampling uses explicit split membership and the prescribed family mixture. Substring filename matching is insufficient. Empty data must fail, never synthesize replacement training rows. Preserve canonical mechanism IDs in batch sidecars so evaluation/protected membership remains auditable after rendering. The owner can prepare full data offline; no GPU is needed for the compiler.

## 4. E1: formation is a paired learning comparison

Resolve two identical initial states per seed and record their hash; naming the same seed alone is insufficient verification. Run A and B with the same allowed stream and frozen update target, with A token-only and B the exact declared five-objective package. Counterbalance slots exactly as in the campaign plan. Both arms train the prescribed architecture and context configuration.

At each declared checkpoint fraction publish actual state and evaluate the prescribed held-out mechanisms using free generation and their independent task verifier. Record answers, stopping condition, valid actions, success and cost separately. An EOS flag is not task accuracy. Match comparisons at the latest common completed update if a treatment fails to reach target, while preserving the failure and full resource cost. Do not select checkpoints after looking at protected outcomes.

Local acceptance: build a real tiny compiled batch; run ProductionOps target construction and backward with only the final optimizer step replaced by a no-op; check named gradients and disabled-term execution. Save and load a randomly initialized or otherwise non-updated checkpoint through the real publication API with full identities. This verifies integration without consuming learning allowance.

## 5. E2–E4: restore actual competence before measuring changes

E2 restores the accepted E1 endpoints and freezes them. For each existing control/learned arm, feed the real public goal/history into its adapter, execute queries/actions in the task environment and append only received observations to real history. Imagined planner branches remain separate. Derive action/model-call/node counters from the actual operations. Include the prescribed contradiction, complementary-query and goal-swap comparisons, final task success and conditional uncertainty. A missing parent must fail; reinitializing with the same seed is forbidden.

E3 forks both children from exactly the same E1-B payload. T0 implements the declared tool token-only/no-replay treatment. T1 implements the declared objective package with 75% new-tool and 25% old-task replay exposure. Tool-heldout cases never enter either training stream. Verify actual generated tool arguments, execution outputs and request receipts, then measure new composition success and protected old-family retention. Do not pass the stored correct answer to the tool verifier as the claimed execution output.

E4 forks E1-B independently of E3. S1's actual optimized model is the migrated gated model; S0 has disabled matched slots. Verify gate parameters in the optimizer inventory, migrated state identity, head/segment behavior and additional execution cost on that very handle. A successful migration probe on a discarded tiny model does not establish the architecture being trained. Both variants use the same declared training stream, targets and comparison endpoints.

Local acceptance uses real parent artifacts and real environment transitions with deterministic model-output doubles. Corrupt a parent hash, move a protected case into the training sampler, falsify a tool output and remove a gate parameter; each defect must be rejected at the production boundary.

## 6. E5: measured learning-method selection and successor construction

Use the exact three 45-minute blocks in experiment.md. The job holds one fixed adaptation anchor per seed, separate from proposer weights. Every method trial forks the anchor with identical initial state and support order, applies the declared M0/M1/M2 recipe to the actual trainer and evaluates the resulting learner on the corresponding query/protected cases. Archive rows contain measured outcomes, actual costs, support/query identities and trial checkpoint lineage. Failed trials remain recorded. `run_fixture_generation` is not allowed in this path.

Train P0's method-token prediction from the measured training archive. At the successor decision, capture its actual decoder output, input/archive cutoff, parsed method and full recipe identity before applying it. Fork the same P0 into P1 using the selected recipe and P_fixed using M0; give both the specified same archive and time. P_fixed is not M2. The runtime applying the recipe must own the actual learner, not a fake trainer with a stand-in optimizer.

Before running fresh confirmation trials, capture choices from P1, P_fixed, P0, fixed M0 and deterministic random. Evaluate all against the same measured method-outcome tables. There are actual task rows behind each choice and actual adaptation checkpoints behind each outcome. Compare query success, learning-curve metrics, retention, regret and total cost. A constant `future_outcomes_seen=False` cannot establish isolation: enforce unavailable confirmation outcomes through separate immutable archive capabilities and reject current-task identities at proposal rendering.

Local acceptance runs the real three-block scheduler with stateful deterministic learner doubles that change only when the specified method is applied. Keep all resulting receipts fixture-labeled. Replacing a selected recipe with M0, moving a confirmation row into context or changing the adaptation anchor must make the test fail. A no-training boundary must never produce a positive learned-update receipt. The final real GPU run supplies measured evidence; local tests supply correctness only.

## 7. E6 and evidence acceptance

A complete bundle includes the frozen protocol, source/data identities, actual per-phase outcomes, failure records, comparisons, proposer transcripts and every required restorable parent/successor checkpoint. Export payload bytes to durable owner storage and validate them against the published hashes. In a fresh process, load required checkpoints using their real config/architecture and verify the necessary state. Copying IDs into JSON is not export.

Partial failed-run export is valuable, but its status must say incomplete and list missing artifacts; it cannot satisfy complete-campaign acceptance. Check required artifacts by exact lineage identities, not substrings in job names. Generate a final artifact manifest only after successful writes and independently verify it. The existing E6 metadata-only diagnostic must stop reporting completed when there are no payloads.

## Delivery order and ownership

Integrator owns shared operation/job/receipt interfaces and checkpoint adapters first. A bounded Luna executor may own the compiler and split checks; a second isolated task may own real environment evaluation. Once those interfaces are merged, implement E1 and demonstrate its complete path before parallel E2/E3/E4 work. Assign E5 only after actual parent restoration, recipe application and measured evaluation exist. Integrate E6 last but build checkpoint export primitives first.

Submit `engineering/reports/K8_REAL_EXECUTION_20260914/HANDOFF.md` with one acceptance row for each section, exact commands and artifact identities. Keep already-correct runtime code and preserve previous evidence. Do not pad tokens or introduce new experimental themes before this campaign executes the treatments already designed. No local optimizer updates or additional compute allocation are authorized.
