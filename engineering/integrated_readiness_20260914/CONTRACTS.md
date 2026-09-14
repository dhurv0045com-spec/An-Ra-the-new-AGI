# Algorithm and data decisions for the connected learner

## Registration delta before owner execution

This packet changes the proposed rule task's information contract and specifies the trained interfaces that the K8 implementation must consume. U10 must update the campaign's data/schema identities, task definition, consumer mapping, timing envelopes and registration validation before generating the final owner bundle. Preserve objective weights, seed count, treatment slots, phase durations and total compute allocation. Do not compare results under the repaired task/schema directly with old incompatible rows as if the tasks were identical. There are no accepted K8 learning results to reinterpret in this review. The in-progress 96-token predictor change is a proposal, not an already accepted timing contract.

## 1. Admit tasks that can be learned from their observations

The base rule task is evaluation under partial observation: publish the rule function, relevant variable roles, negations, threshold and target predicate as the goal specification. Keep sampled variable values hidden until inspected. Revealing the function does not reveal its value on the hidden world. Limit the required relevant observations to a set attainable within the four-inquiry budget. Distractors remain public candidates whose values do not determine the answer.

If the experiment instead asks the learner to discover an unknown function, supply a public finite hypothesis class and legal labeled-example/intervention queries. Require an information sufficiency witness showing that a bounded adaptive query tree distinguishes the answer on the target input. Do not silently substitute induction for evaluation in K8; defer induction as its own future task if it does not fit the existing budget.

For a public transcript h, define the diagnostic version space V(h) as private mechanisms compatible with the received observations. The answer is identifiable when all members of V(h) imply the same answer for the public goal. This is a dataset validator, not an inference component. For the base task either exhibit a legal sequence that produces an identifiable transcript within the budget or reject the task before split freezing with an explicit reason. Use full enumeration only for bounded finite diagnostic spaces; for larger families provide constructive witnesses and adversarial twin tests, without claiming an exhaustive proof.

An irreducibly ambiguous task can be valuable for testing calibrated abstention, but it cannot be mixed into a benchmark that expects a uniquely correct answer without an ambiguity label and corresponding scoring rule. Prefer repairing the base family first. Keep witness/private labels outside learner inputs and distinguish symbolic teachers from learned components.

## 2. One DecisionExample drives every consumer

Version the following logical record: task family/schema, goal, received event prefix, evidence view, public legal actions, action schemas, budgets, chosen teacher action, received next observation, terminal answer if eligible, return target/horizon, split/canonical mechanism identity, source-event references and masks. Public input and target channels must be physically distinct fields. Hash both schemas and the exact rendered input separately from labels.

Derive every decision from the actual environment trace. A transition input stops before its target observation. A final-answer input includes the received observations available at submission. Counterfactual goals carry their own correct answer and input context. Do not label every intermediate state with a hidden constant answer. Keep failed/partial/exploratory trajectories where they provide valid targets; absence of a terminal verdict means the return target is ineligible, not invented zero.

The compiler, trainable scorers, E2 adapters and RSI support/query evaluator must consume this same record. No six-token prefixes, eight-byte actions, ID-only retention prompts or arbitrary truncation. Schema changes invalidate incompatible prepared artifacts and are recorded in new manifests; preserve old files as historical evidence. A test must compare complete actual consumer inputs for the same DecisionExample, not merely a common prefix.

## 3. Use the interfaces that receive training

Adopt the existing trainable candidate action head for bounded inquiry/tool selection. It scores complete legal actions conditioned on the public decision state; training and inference use the same canonical candidate serialization, mask and hidden-state path. Batch candidate branches for efficiency and report candidate sequence count separately from model invocation count. This avoids inventing an untrained free-generation index-code language. A one-candidate softmax provides no discriminative gradient; mark its action target ineligible rather than counting it as action learning.

Submission is a typed action with a family-specific payload contract: rule Boolean, inventory item identifier, program integer, tool result/check schema. Train the answer decoder on the terminal public state and the same compact payload serialization used at inference. Do not coerce a generic `v` to whichever missing key is tested first. A controlled decoder/grammar may restrict well-formed outputs, but its constraints are declared priors and it must not supply the answer.

For A-direct, use A's actual answer decoder without inquiry. For A-fixed, execute the registered fixed inquiry schedule and then call A's decoder on the resulting public state. The random inquiry control also answers through the declared checkpoint if it is labeled model-A. Retain oracle and pure scripted guesses as separate diagnostic controls with truthful names. B's additional objectives and executive operations are the treatment; an unused A model is not an equal comparator.

World prediction must consume the same state/action serialization used by world-transition training and produce the same typed feedback representation. Use the trained value head, with a registered return horizon, for evaluating imagined states; do not request an untrained JSON `success_prob` field. Whether structured prediction uses constrained decoding or a public finite outcome support must be specified and shared by training/evaluation. A finite support must derive from public schemas and ranges, never from the private realized answer.

This is a concrete implementation refinement of K8's existing token/world/action/value/pair objectives, not authorization for a new treatment. Preserve the registered weights. Record the exact consumer mapping and any response-envelope amendment in the protocol before calibration. Remove the unilateral 96-token prediction assumption unless measured full-campaign timing and the amended protocol support it. An unparseable prediction is unknown and cannot generate valid forecast evidence.

## 4. State transitions and stopping are explicit

Carry the actual PublicState into each root prediction. An ImaginedState has its parent, the root public state, applied typed predicted deltas, derived legal actions, remaining hypothetical action budget and provenance. A hash identifies content; it never replaces the content. Apply predicted deltas without mutating received events. If a parent predicts termination, do not continue that branch. Recompute legality from the imagined state and public rules, not by filtering the original list for non-submit actions.

Use at most eight imagined nodes and sixteen model invocations per episode, including invalid generations. Expand admitted roots fairly before children. Node IDs are unique within the episode and decision IDs distinguish repeated actions. Select a root plus a precise path/forecast ID. Make the chosen forecast observable to the evaluator even if another root was scored first.

Use monotone cumulative counters or explicit per-selection deltas, never both. Before dispatch, check the remaining budget; after dispatch, reconcile actual calls. A failed preflight that made zero model calls consumes zero model calls but ends the episode with a specific refusal. Every loop iteration must either execute a budgeted action, spend an actual invocation, or terminate. Guard against zero-progress retries and also enforce the job's wall deadline. A watchdog protects experiments but does not repair incorrect counters.

## 5. Measurements preserve the causal question

Each prediction receipt includes decision ID, node/path ID, action identity, input identity, checkpoint identity, horizon, forecast validity and realized outcome join. Immediate feedback scores use the matching received event; terminal-success scores use a terminal forecast. Empty output is not correct feedback. Missing/invalid forecasts have their own rates, while task failures remain in the task-success denominator. Preserve per-mechanism paired comparisons and both seeds.

Frozen evaluation inventory identities, canonical split exclusion, context admission and mode configuration must be fixed before observing results. E0 may select the predeclared inventory size based on measured throughput; it cannot change semantic eligibility after seeing accuracy. Model-origin means a model produced the output. Learned evidence additionally requires trained compatible parent lineage, valid payloads and the intended treatment execution. Keep these fields separate.

## 6. Measured recursive improvement has a real chain

Implement the chain as explicit states: anchor validated; archive block A measured; P0 trained; P0 decision captured; fresh archive block B measured; trained P0 forked into P1 and P_fixed; each successor trained on the same newly available archive under its own recipe; all per-task confirmation choices captured from the actual P1, P_fixed and frozen P0 checkpoints; only then confirmation trials execute.

Keep P_fixed (a learned proposer trained with M0) distinct from fixed_M0 (an always-M0 controller). Do not copy P0's selections into P1. A failed decoder output is a failed model proposal; a labeled teacher fallback may be shown as an additional control but cannot make a model-proposer success or RSI claim. Bind capture metadata to the exact proposer payload, not the adaptation anchor ID.

Use 12 archive-A tasks, 12 fresh archive-B tasks, and six confirmation tasks per seed with the registered three recipes. Trial caps are 45 seconds in each archive block and 90 seconds in confirmation. Preserve the block overhead/training reserves: block A 27 minutes trials, eight P0 training, ten overhead; block B 27 trials, eight total successor training, ten overhead; confirmation 27 trials, eighteen evaluation/export. These are maxima within the fixed 135-minute E5 phase, not additional allowances.

All trial deadlines are the minimum of the per-trial cap and the parent block/job/campaign deadline. All updates and exposure reduce from child events, including failed attempts, proposer updates and confirmation adaptations. Retention uses the real protected public task inputs and verified references. Missing protected data invalidates the trial rather than shrinking the denominator. A positive RSI claim requires independent fresh P1 improvement over P_fixed and P0 under cost and retention constraints; extra training alone is not the finding.

## 7. Creativity after the base is sound

The information sufficiency witness makes the present experiment interpretable. The later evidence-repair controller can learn to predict when its current answer depends on a missing observation, request a discriminating query, and stop when further inquiry is unnecessary. Train that controller from public traces and withheld evidence pairs only after the base compiler is trustworthy. Evaluate against equal-budget random and fixed inquiry, with irrelevant-evidence and contradictory-evidence controls. Do not add this future controller or another training objective to the current K8 run without a separately frozen protocol. The base must first be capable of learning from what it sees.
