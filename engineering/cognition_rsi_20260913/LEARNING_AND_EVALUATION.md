# Teaching and testing cognition and learning-to-learn

Every capability below has a training source, a decision made by the learner, a transfer boundary and a control that can disprove the proposed advantage. Build this mapping into manifests and reports; a named module is not enough.

## 1. Learning signals

| Capability | Admissible supervision | What the model must learn to choose |
|---|---|---|
| Belief tracking | Public-evidence finite-support posterior teacher; actual subsequent observations | Which hypotheses remain plausible and what contradicts them |
| Goal binding | Qualified goal pairs and independently checked answers/actions | Which facts and procedures serve the current goal |
| Decomposition | Verified training task graphs, executable subgoal checks, successful/failed traces | Useful subgoals, dependencies and when to replan |
| Executive control | Qualified cognitive-operation traces; outcome returns with computation cost | Whether to reason, retrieve, query, verify, act or stop |
| Self-monitoring | Held-out-within-training outcomes of predicted confidence and error diagnoses | When its answer is uncertain, what could resolve it, and when to abstain |
| Abstraction | Rules proposed from support cases and checked on different training cases | Reusable relationships and their preconditions |
| Learning to learn | Meta-episodes with support adaptation and independent query measurement | Which learning actions/method proposals help on a novel task |
| Recursive improvement | Complete model-origin proposal and generation receipts | How to propose a useful successor procedure under resources |

Add objective types to the same M05 router. Use masked token likelihood for structured workspace/proposal outputs, distribution cross-entropy for qualified belief/action targets, regression/ranking for measured utility, and the existing complete-episode policy objective where appropriate. Keep separate denominators for beliefs, decisions, token targets, comparisons and meta-episodes. Missing targets are excluded explicitly; no self-generated assertion becomes a truth label by default.

For the initial belief teacher, cross-entropy from teacher q to learner p is `-sum_h q(h) log p(h)`, normalized per eligible belief-update case. Use a public hypothesis support and declare the teacher prior. For predicted gain, use a bounded regression target or ranking loss over genuinely comparable measured proposals; do not supervise on the proposer's own expected gain. Store target-model/verifier/measurement versions.

Do not train long private prose traces merely to make the system appear thoughtful. Structured intermediate state is useful when it improves final behavior and supports checking. Compare it with a direct-answer model at matched compute, and distinguish a useful external scratchpad from a learned reasoning improvement that survives without it.

## 2. Cognitive test families

**Belief revision:** a previously plausible rule receives contradictory evidence; some observations are noisy or correlated copies. Measure proper belief scores, revision direction, recovery and downstream decisions. A frozen-belief control and a latest-observation-only control expose both stubbornness and indiscriminate forgetting.

**Working memory and binding:** multiple objects share surface features; later instructions swap roles or reference earlier evidence after distractors. Measure binding accuracy, lost constraints and task completion across increasing context pressure. Compare full public history, bounded workspace, retrieval and memory-off at declared input budgets.

**Compositional planning:** unfamiliar combinations of familiar operations require a new subgoal graph. Some actions irreversibly spend the task budget; a late observation invalidates a subgoal. Measure valid completion, replanning cost, constraint preservation and novel-composition transfer. A retrieved fixed macro is a control, not the benchmark definition.

**Metacognitive allocation:** paired prefixes include cases where an extra query/check helps, is useless or misleads. The test choice is made before the extra outcome exists. Report success versus total inference/interaction cost, selective accuracy/coverage, unnecessary deliberation and missed useful checks. Compare always-deliberate, never-deliberate and fixed-budget policies.

**Abstraction transfer:** derive a candidate rule from one set of training support cases, then test different objects, surface forms, lengths and compositions. Count counterexamples and restricted preconditions. Compare against memorized episode retrieval and a matched model without abstraction supervision.

**Meta-learning:** novel tasks offer small support sets followed by held-out query cases. Compare learning curves under a shared initialization or a separately declared representation-transfer comparison. Measure zero-shot behavior and improvement per added support experience; a good zero-shot score alone is not fast adaptation.

**Recursive method learning:** compare model-origin method proposals across generations using RSI.md's controls. The claim requires improving future learning or proposal efficiency, authentic successor use and protected retention. Three repeated ordinary fine-tunes do not satisfy it.

## 3. Causal attribution and component ablations

Record the source of every high-level decision. Disable one learned component while retaining the rest of the public path: fixed executive, frozen beliefs, no workspace, no retrieval, no subgoal graph, direct answer, frozen proposer, fixed learning method. These controls distinguish host scaffolding from learned behavior. Optional host fallbacks may keep an operator session usable but cannot be counted as model success without separate reporting.

Measure transfer on different mechanism clusters, not new UUIDs. For memory and abstractions, keep source ancestry so the same example cannot reappear as an apparently new test. For RSI, task-level and meta-level splits both matter; a hidden query label can leak through a method archive or a failure summary even if it never appears in the ordinary training shard.

A cognition demonstration can fail usefully: excellent belief scores but poor action choices suggest executive credit assignment; good in-context reasoning but poor retention suggests consolidation; good fixed-method adaptation but poor proposal selection suggests the meta-learner. Record the narrow diagnosis rather than relabeling aggregate training loss as intelligence.

## 4. Qualification and readiness

The original master's interval, retention and ResourceVector rules still apply. Add cognitive-operation costs, meta-training support exposure, proposal-generation cost, rejected candidate evaluations and compiler/validator failures. Every required quantity must be measured or the comparison marked unsupported. Thresholds and task counts are preregistered in a future allocated experiment, not chosen after inspecting success.

Use four readiness records: **build connected**, **cognition learned under named tests**, **adaptation method improvement supported**, and **bounded recursive improvement supported**. Each has its own prerequisites and evidence. General intelligence additionally requires much broader competence and transfer than these bounded tests alone establish; keep the capability map open and explicit about missing domains.

This extension launches no learned qualification. Implement pure belief-update examples, workspace integrity, operation legality, fake-model executive control, method-language validation and fixture generation transactions now. A future allocation can execute the neural comparisons through those same public paths. Do not spend resources proving a fixture can memorize its own labels.
