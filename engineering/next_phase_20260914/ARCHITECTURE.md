# Architecture and algorithms for the operational learner

This is an engineering design, not a claim of experimentally established superiority. It uses the existing BRAMASTRA decoder, objectives, environment families and K8 controls. The intended improvement comes from completing the causal connections among learning, decisions, memory and measured method selection.

## 1. A learner session is the unit of execution

Introduce a session object rather than passing loosely populated dictionaries between unrelated helpers. Its durable identity includes source/data/tokenizer/configuration/protocol hashes, architecture ID, parent checkpoint, seed, allocation ID, reservation ID, phase/slot/job ID and physical device. Its runtime state includes the model, optimizer, scaler, random states, stream cursor, objective-window state, controller and event sequence. The session owns no independent compute allowance: its permission comes from the supervisor reservation.

Session states are created, restored, active, checkpointing, completed or failed. A session cannot finalize without an active, compatible reservation. Admission checks the absolute deadline and update allowance immediately before expensive work and again at the optimizer boundary. The supervisor still enforces a hard process deadline. Rebinding a resumed session validates the original allowance and already charged work; it never resets them.

A child session is a real fork of a verified parent payload with an explicit optimizer/RNG policy. A changed architecture is an explicit migration, not a new seed with the parent's name. Checkpoint publication records expected-parent identity and acquires the actual writer fence. Session events bind update attempts and checkpoint publication so a failure after mutation cannot erase consumed work.

## 2. One complete optimizer-window contract

Let each enabled objective j have summed eligible loss S_j, eligible count N_j and weight w_j. The update objective is sum_j w_j*S_j/N_j for N_j>0. Eligibility is calculated for the entire optimizer window before its microbatch backward calls. A microbatch contributes its sums divided by the window denominators, not its own means. This supports small GPU microbatches without changing the objective.

Separate construction, backward and finalization. Construction produces real target channels from the compiled trajectory. Backward accumulates scaled gradients with the same precision policy for all terms. Finalization unscales once, checks finiteness, clips once, applies the actual scheduled LR, attempts one step, updates the scaler and records actual commit/skip status. Pair loss follows the same window ownership; it cannot be independently normalized or silently applied twice.

The same complete window must have equivalent gradients under different microbatch partitions when stochastic layers are controlled. This is a local backward-only test. Missing eligible targets fail; disabled objectives perform no extra forwards. Zero-eligible windows record exposure without claiming an optimizer update.

## 3. The cognitive kernel operates on observations

Use one episode loop with distinct policy adapters. The loop reads a public observation and goal, renders the current state, asks the adapter for a typed legal action, executes it in the environment, records the actual result/cost and updates state. It stops on environment termination or the declared budget. The independent task verifier decides final success. Exceptions and invalid actions become explicit outcomes with costs; they are not replaced by teacher choices.

The public state has four components: the current goal, a bounded observation history, an evidence workspace and available action/resource budgets. The private generator state is accessible only to the environment/verifier. A model output cannot directly modify the private state or claim a tool succeeded. Real observations and imagined outcomes have different types and storage paths.

Each operation emits an event with session/checkpoint/input identity, action or generation identity, observed result, resource delta and predecessor event. Model-call counts, tool calls and imagined nodes are reduced from these events. Do not assign them as constants after evaluation. The state renderer is shared by training and inference and enforces the full context policy; truncation must not silently discard contradictions or the changed goal.

## 4. Working memory and belief revision

Represent workspace entries as evidence records: entity/property, observed value, observation ID, temporal scope and status (active, superseded, conflicting or derived). Contradictory evidence retains both source observations; a revision changes which claim is currently supported, not history. Derived claims carry the input evidence IDs and whether their derivation came from a symbolic control or a learned prediction.

For K8, workspace construction must be a declared public deterministic transformation or a learned operation whose input format was trained. Do not quietly provide a solved symbolic belief to the learned arm. The reference belief updater remains a separate control. Compare useful working memory by its effect on the already planned contradiction and complementary-query cases, not by the number of entries in its dictionary.

Keep two memory lifetimes distinct. Episodic memory exists within an evaluation case and is reset between independent cases. Replay memory contains permitted training episodes selected by the declared sampler. Evaluation outcomes never enter replay or change the checkpoint. Cross-episode retrieval and longer-term memory are future extensions unless already specified in K8; do not add those exposures to an arm without a new comparison design.

## 5. Planning is a bounded alternative policy

A planning node contains public state, goal, action prefix, predicted outcome, value estimate, remaining budget and provenance. Expansion uses legal actions and the world predictor's declared finite support. Predicted states remain hypothetical. Duplicate public outcomes are merged before normalization. Record support construction because calibration is conditional on it.

Implement the planned depth-two/node-eight search under the same real-action budget as the policy arm. Propagate predicted success/value minus declared action cost to rank candidate actions, using the existing value convention consistently. The proposal is a design hypothesis: model error can make planning worse. Record predicted values and actual resulting outcomes to separate search failure from world-model error.

All compared modes receive the same allowed initial evidence. Run the relevant modes on matched copies of a mechanism/seed, not different randomly assigned cases. Change the goal before rendering the prompt and before verifying success. Derive contradiction and complementary-query cases from actual environment mechanisms. A mode name must select a different declared adapter, with its own trace, rather than label the same free-generation call.

## 6. Skill acquisition and retention use explicit streams

Tool acquisition requires generated arguments, real tool execution and verifiable output state. A tool reply is evidence only after an invocation with a request ID produces it. Bound outputs to the episode directory and make retries idempotent. Verification uses actual output bytes/content, never the stored answer as a proxy for execution.

The T0 and T1 learning streams and objective switches are distinct. T1 receives the declared 75/25 new/replay mixture; T0 receives the declared token-only/no-replay treatment. Measure realized exposure and report small-window rounding honestly. Both start from isolated copies of the same E1-B parent. Held-out composition and protected-old cases remain excluded from training. Evaluate acquisition and retention separately; an average cannot hide forgetting.

E4's migrated model is the actual trained/evaluated handle. Shared blocks, gates, heads, packed-segment masking and optimizer registration must survive the fork and restore. Extra block execution is charged even though it adds few parameters. Test gate-path activation on the actual handle, not a discarded tiny demonstration model.

## 7. Measured improvement is an experiment inside the experiment

Create an explicit TrialRequest containing fixed anchor reference, method recipe, support/query/protected identities, seed/RNG policy, reservation and deadline. Trial execution restores/forks the anchor, applies the method to the real trainer, trains within the trial allowance, evaluates query/protected tasks, publishes its checkpoint and returns measured outcomes/costs. All methods for a task share the same anchor and support order. Failed trials are archive records with failure reasons and real cost.

The archive stores only completed trial records with immutable source and event cutoffs. For local correctness tests, a method-sensitive learner double can generate explicitly fixture-labeled results through the same scheduling boundaries. The production implementation must train and evaluate; it must not dispatch dictionaries into a fixture branch based on shape or class name. Use explicit interfaces and evidence-kind validation.

P0 is trained on measured training-archive method targets. Its actual decoder chooses the successor update recipe from the disclosed M0/M1/M2 language. A host-computed best method can be a teacher label or control, never the model's captured choice. Capture prompt bytes, archive cutoff, raw decoded output, parsing result and checkpoint identity before applying the recipe.

Fork the same P0 into P1 and P_fixed. P1 receives the selected recipe; P_fixed receives M0 with the same training archive and time. Both perform actual method-selection learning updates. Capture P1/P_fixed/P0/fixed/random choices on fresh confirmation tasks before any current-task method outcomes are measured. Score every chooser against the same measured trial table. Beating P0 but not P_fixed supports extra training, not benefit from the selected improvement method.

## 8. Learning to investigate is the research direction

The unifying hypothesis is that a learner can improve by choosing informative interactions, constructing reusable internal state and learning which training changes work on new tasks. K8 tests bounded components of that hypothesis. It does not test unrestricted self-rewriting, broad language competence or indefinite recursive gains.

After valid results, choose the next extension by measured bottleneck: better inquiry targets if observations are ignored; improved state representation if contradictions cause failure; model-calibration work if planning loses to policy; replay policies if tools induce forgetting; a larger diverse meta-task confirmation if successor selection helps. Record alternative explanations and a falsifying outcome before allocating the next experiment. These are future proposals, not extra work silently added to the current eight-hour session.
