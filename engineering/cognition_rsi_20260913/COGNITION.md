# Cognition as learned operations over grounded state

This specification concerns functional cognition: information processing that supports perception, belief revision, reasoning, goal-directed behavior, memory and learning. It makes no assertion about subjective experience. A natural-language statement such as “I understand” is not a capability measurement.

## 1. The cognitive state

Represent the public cognitive state as a bounded `CognitiveWorkspace` rendered through the shared decoder. It is an externalized working-state interface, not a claim that all internal neural computation is interpretable. Keep learned parameters, hidden activations and inspectable workspace distinct.

| Component | Fields | Update rule |
|---|---|---|
| Goal frame | user goal, public success predicate, constraints, priority, budget | Root goal changes only through the task/owner interface; learned subgoals must reference it |
| Evidence ledger view | public observations, tool results, source status and local aliases | Append actual observations; preserve retractions/corrections without rewriting history |
| Belief set | proposition/hypothesis, confidence/distribution, supporting/conflicting evidence aliases, status | Revise when new admitted evidence or a verified derivation changes support |
| Subgoal graph | parent, dependencies, intended measurable result, status, estimated cost | Propose, validate, execute and retire; cycles and unverifiable completion reject |
| Active agenda | candidate cognitive/external operations, estimated utility/cost, selected operation | Learned executive selection under legal/resource masks |
| Capability estimate | checkpoint-conditioned recent validated performance by operation/family | Derived from allowed training/controller evidence; unknown remains unknown |
| Pending commitments | issued tool/query, expected return type, deadline, remaining work | Exactly-once completion/cancellation through the runtime |

Use episode-local aliases such as `e0` and `b1` for referential binding in tokens. Global task IDs, hidden mechanism IDs, dataset split, source paths and evaluator metadata stay in provenance. A local alias says which public statement is referenced; it must not encode a task answer. Reset aliases and workspace at the protocol's episode boundary.

Each belief has an epistemic status: `observed_report`, `hypothesis`, `derived`, `contradicted`, `retracted` or `unresolved`. An observed report can itself be noisy; do not equate an incoming statement with universal truth. Store source reliability assumptions explicitly when the protocol provides them. A model-generated claim starts as a hypothesis unless an independent derivation checker establishes a stronger status. High model confidence cannot change its provenance class.

## 2. Belief revision and uncertainty

For a finite public hypothesis support H in a qualified task, implement a reference belief update:

`q_next(h) proportional to q_current(h) * p(observation | h, chosen_action)`.

The reference updater is an engineered diagnostic/teacher with declared likelihood assumptions. It sees the same public observations and support as the learner; it must not inspect which hidden hypothesis the environment actually sampled. A zero normalizer means the support/likelihood assumptions are inconsistent with the observation: emit `MODEL_MISMATCH`, retain evidence and allow a broader hypothesis proposal. Do not silently reset confidence to certainty in a convenient alternative.

The learned model predicts hypothesis/proposition distributions or structured belief revisions using the same decoder. Train against qualified public-evidence teacher distributions where exact updates exist, and against future observable consequences elsewhere. Belief confidence is scored by later evidence using proper scores; verbal certainty is not a reward. The teacher's known hypothesis language is a disclosed prior. An open-world hypothesis proposal does not become an exact Bayesian posterior simply because it has a probability field.

Test contradictory evidence, noisy observations, missing evidence, independent facts, correlated repeated reports and task changes. Duplicate retrieval of the same original observation must not count as independent corroboration. Preserve evidence ancestry so the learner can discover that two reports have a common source. Support a revision that reduces confidence or changes the leading hypothesis; retaining every earlier assertion is not consistency.

## 3. Executive operations

Define a typed `CognitiveOperation` with a verb, arguments, referenced evidence, expected observable/checkable result and resource estimate. Initial verbs are `RETRIEVE`, `PREDICT`, `COMPARE`, `DECOMPOSE`, `DERIVE`, `QUERY`, `EXECUTE`, `VERIFY`, `REVISE`, `ABSTAIN` and `SUBMIT`. The runtime also supports cancellation and errors as operational events, not task-solving shortcuts.

The shared model's action scorer selects among legal operation candidates; the same decoder generates arguments when needed. Use candidate-isolated scoring from M02. The host validates types, budgets and scope, then performs the selected operation. It must not secretly replace a weak learned decision with an oracle decision while still labeling the run learned. Record fallbacks and evaluate them separately.

Mental work costs inference tokens, calls and time; queries/tools can additionally cost real interactions. One reasoning step cannot recursively request unbounded reasoning steps. Every operation consumes from the parent ResourceVector and returns its actual delta. Maintain independent limits for workspace size, pending operations, recursion depth and total calls. A no-progress cycle (`restate -> restate -> restate`) must terminate under the allowance and be scored as failure/unfinished work, not hidden behind a long transcript.

The first executive uses supervised qualified traces, then the master's single-window on-policy loss on complete real episodes. The return is task success minus declared costs. Do not reward the number of reasoning steps, confidence words or agreement with an untrained self-critic. A fixed short-chain policy and a direct-answer policy remain strong controls.

## 4. Goals, subgoals and compositional reasoning

The root goal is supplied by the task/owner. The model may decompose it into subgoals whose completion can be checked through a public predicate, executable test or admitted observation. Store dependencies as an acyclic graph. A subgoal's completion is evidence-bearing: `verified`, `failed`, `unknown` or `cancelled`. Model text saying “done” is not the verifier.

Track variable/object bindings explicitly in structured tasks. Renaming an object should preserve its relationships; changing a relationship should change the plan. Tests use equivalent renamings, swapped goals, irrelevant facts and contradictory constraints. Include cases where a familiar procedure's preconditions fail, so blind retrieval is insufficient.

For program/math/logic tasks, allow inspectable intermediate derivations with a bounded independent checker. Correctness credit comes from valid steps and final task outcomes, not imitation of arbitrary prose. A checker may validate local derivations without providing the solution. Declare the checker language and capability as external priors. For open language tasks where such a checker is unavailable, do not present a self-written reasoning trace as a correctness proof.

Planning uses M07's public transition predictions. Deliberation can propose a subgoal or alternate plan; actual branches remain hypothetical until checked or executed. Replanning after surprising feedback must be possible without losing the root constraints. Test interruption/resume in the middle of a multi-subgoal task and compare pending commitments, next selected operation and resource state.

## 5. Episodic and semantic memory

M04 supplies admissible retrieval. Extend it with memory record types: episode trace, verified fact, failed hypothesis, procedure with preconditions, and candidate abstraction. Each has support/contradiction links, scope and source ancestry. These are storage types; the first implementation does not require separate neural modules for each.

The model may propose a reusable rule from multiple episodes. Admission requires training-only verification on cases distinct from the examples used to propose it. Record the examples that generated the rule and counterexamples that restrict it. A statement supported by one episode is an episode-specific hypothesis, not a general law. When a rule fails, revise its preconditions or retract it while retaining the failure evidence.

Distill accepted training knowledge into weights only through the canonical trainer. Compare memory-enabled and memory-free inference from the resulting checkpoint: this separates remembering a retrieved answer from a transferable learned representation. A learned abstraction should improve a new composition or reduce new-task data needs, not merely reproduce the source examples.

## 6. Metacognition and allocating thought

The capability estimate is operational self-knowledge: what this checkpoint tends to get right, when more evidence helps, and when tools or abstention are appropriate. It is not an unverifiable claim of self-awareness. Bind estimates to model version, task regime and evidence pool; mark stale estimates after substantial updates.

Only declared training/controller-derived estimates may enter the model-visible workspace. Query, development-measurement, retention-exam, confirmation and sealed outcomes remain examiner sidecars and cannot be rendered into a later cognitive prompt. A repeatedly used validation summary becomes training feedback and must be reclassified; its old independent status does not survive that use. Serialize the estimate's source-pool identity and enforce eligibility before rendering, including through retrieved summaries.

Define a `DeliberationChoice` over direct answer, more internal computation, retrieval, a real query, a tool check, or abstention. The initial value-of-computation target is obtained from paired **training** continuations of the same public prefix: execute the allowed option under a fixed additional budget and compare validated terminal utility with the direct-answer continuation. This is expensive teacher data, so log every branch cost. The model cannot observe the test continuation outcome before making the test decision.

Train the selector to predict expected utility improvement per declared cost, or rank options by that quantity. At inference, choose an option only when the predicted improvement justifies its cost under the frozen decision rule; otherwise answer/abstain as the protocol permits. Include tasks where more thought helps, does nothing, and makes the answer worse. Evaluate the realized cost-quality curve, not just confidence calibration.

Abstention has a protocol-defined score and cannot be treated as universally successful. On solvable tasks it trades coverage for accuracy; report both. A system that never acts cannot satisfy the acquisition objective through perfect selective accuracy.

## 7. Three learning timescales

Within an episode, the workspace and beliefs change while weights normally stay frozen. Across admitted training episodes, the trainer changes weights and replay preserves capabilities. Across meta-episodes, the system may change how it selects data, allocates learning or proposes a learning-method modification. These are different state transitions and must have separate identities and receipts.

This separation lets the model reason about unfamiliar evidence without retraining on every observation, learn reusable structure from completed experience, and eventually improve its learning policy. Optional fast weights or learned recurrent state can be added later only if the baseline fails a measured retention/adaptation requirement. Do not confuse an external Python dictionary retaining facts with learned long-term memory acquisition.
