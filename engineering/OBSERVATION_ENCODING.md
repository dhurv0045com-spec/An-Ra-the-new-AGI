# Public evidence encoding for the integrated learner

Status: W03 interface decision, 2026-09-08. This specifies the bridge from W02 environments to a shared learner. It is not yet an implemented or evaluated architecture. The discovery prototype's fixed bit-vector encoder remains a separately named diagnostic.

## 1. Why a shared interface is necessary

Switch, inventory and program worlds have different observations and actions. Three hand-written feature pipelines could accidentally supply task-specific reasoning outside the neural core. The integrated experiment therefore needs one representation path whose input is entirely public evidence and whose transformations do not solve the task.

The adapter may validate types, serialize public fields, enumerate explicitly permitted finite candidates and apply legality masks. It may not infer hidden rules, calculate the target, plan through simulator state or attach an oracle's feature vector. Such computations belong in named diagnostic controls, not the deployed learner's encoder.

## 2. Separate provenance from learnable input

Keep semantic/task/episode identities, split labels, generator hashes, timestamps, collection-policy identity and evaluator metadata in the data record for audit and state ownership. Do not serialize them into the learned content stream. They identify evidence; they are not evidence about the task's hidden mechanism.

The learned input contains:

- Public goal and action schema.
- Public observation values and permitted feedback.
- Previous action and its observed consequence.
- Remaining action/inference budget and step number.

The scorer's gold target label is never part of the input. A goal may contain a target **input**, such as which program argument needs a prediction; that is distinct from its unknown answer.

## 3. Initial tokenizer and event representation

Use a deterministic byte encoding with the existing BRAMASTRA 260-token convention or an explicitly versioned equivalent: 256 byte values plus declared structural/special tokens. No tokenizer fitting or pretrained embeddings are needed for this first integrated interface.

Serialize validated public JSON with canonical key order and explicit event role tags. Record the serializer and vocabulary mapping hashes. Preserve list order where it carries meaning. Equivalent object key insertion order must produce the same bytes; reordering a meaningful sequence must not.

One event is the public goal/observation or an action-feedback pair. Its boundary has an explicit EOS/end-event marker. Set finite maximum event length and maximum history length in the model/experiment configuration. An oversized event is a visible encoding error, not silent truncation of the goal or outcome. A deliberate truncation experiment gets its own configuration and reports discarded information.

Byte encoding is an interface choice, not a claim that byte prediction alone produces understanding. It may prove inefficient; compare alternatives only after this baseline makes their assumptions measurable.

## 4. Shared event encoder and state alternatives

Start with a width-128 event encoder with positional information and a designated end-event representation. A small two-layer attention encoder is the default implementation candidate; use four heads and feed-forward width 352, with exact parameter counts derived from the implementation. Padding cannot affect the representation.

The same event encoder feeds two state alternatives:

**R0 full history:** retain the last configured number of encoded events and process them with an attention-based history module. The readout attends to the current goal and all retained events.

**R1 recurrent state:** update a width-128 GRU state with each encoded event; retain no undeclared hidden cross-task state. The readout receives state and current goal.

Keep event encoding, public content, supervised targets and action candidates identical in the initial comparison. Report exact parameter count, event/history capacity, token counts, inference operations and wall time for each variant. Different history modules need not have identical parameter count, but that prevents an unqualified architecture-only attribution; add a capacity-matched comparison if necessary.

Do not begin with a large external memory, learned slots and adaptive recurrence all at once. Add a single mechanism when the delay/overwrite/composition evidence identifies the limitation it is intended to address.

## 5. Action encoding and scoring

Encode each candidate action through the same public serializer/event encoder, using an action role tag. Score it from the candidate representation, goal and current state. The policy returns scores/probabilities over the explicitly supplied legal candidate set.

For the initial finite environments, W02 exposes a finite candidate enumeration. Host enumeration is allowed only when it follows the public action schema and declared bounds. It must not prune candidates by consulting hidden outcomes. If a future domain has an unbounded action space, it needs a separate generative-action experiment; do not disguise an oracle proposal generator as ordinary enumeration.

Add an explicit terminal/submit action where the task requires one. A task-success reward must reflect the requested goal, not merely a correct statement that the goal remains unmet. Invalid actions are rejected under the environment's declared charging rule; their effect on state/time is tested.

Cache candidate encodings only when their serialized content and model version are unchanged. A cache surviving a weight update must be invalidated or recomputed. Record cache cost/benefit when comparing inference throughput.

## 6. Prediction and value outputs

Keep each head's supervision explicit:

- Binary/discrete outcome head predicts an externally defined outcome vocabulary.
- Next-observation head predicts the next public event or explicitly selected observable variables.
- Value head predicts return under the current policy and budget.
- Action head scores the legal action candidates.

Task-specific output schemas are allowed; task-specific solution algorithms inside an output adapter are not. Byte-generative responses are a later language-interface extension and require their own termination, parsing and complete-response scoring.

## 7. Reset and replay behavior

The runtime owns task boundaries. Before the first event of a new task, reset working state and clear scoped caches. Replay sequences identify whether they begin at a true reset or include a declared warm-up prefix. Do not initialize a truncated fragment with a state computed from information unavailable at evaluation.

Different tasks in one batch must not interact through padding, attention, recurrent state or retrieval. Reordering the batch must preserve each task's result. Serialization and sampling tests must cover this independently of a trained model's accuracy.

## 8. W03 acceptance additions

1. All three W02 families pass through one serializer/encoder API without hidden mechanism access.
2. Changing only provenance IDs cannot change content tokens or model outputs.
3. Changing a relevant public goal or observed consequence changes the encoded content; independently specified counterfactual tests check the trained learner uses it when required.
4. Object key order is invariant; meaningful event/list order is preserved.
5. Padding, batch neighbors, old task state and stale candidate caches cannot affect inference.
6. Oversized inputs and unsupported output/action schemas fail explicitly.
7. R0/R1 consume exactly the same eligible public evidence and report their differing capacity/compute honestly.

The chief should accept the interface only after these tests and a bounded learnability pilot. A larger neural model cannot compensate for an adapter that leaks answers, discards relevant observations or changes the task being evaluated.
