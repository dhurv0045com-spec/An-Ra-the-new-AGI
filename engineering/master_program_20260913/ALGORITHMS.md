# Learning, inquiry and improvement algorithms

This file makes implementation choices, not claims of proven superiority. Coefficients, schedules, supports and numerical tolerances are versioned configuration. New objectives default off in compatibility configs; named build fixtures exercise their routes. The executor must implement the equations and their rejection cases before proposing a training campaign.

## A1. Objective units and a single update boundary

For an optimizer window, let `N_tok`, `N_world`, `N_act`, `N_val`, `N_pair`, `N_pg` be the numbers of eligible answer/document tokens, transition tokens, supervised action decisions, value targets, counterfactual groups and on-policy decisions. These denominators are different. Padding and absent supervision contribute to none. EOS is an eligible target where the sequence contract requires stopping.

Define:

```text
L = w_tok   * sum(masked_token_NLL) / N_tok
  + w_world * sum(public_transition_token_NLL) / N_world
  + w_act   * sum(teacher_action_cross_entropy) / N_act
  + w_val   * sum(Huber(V(prefix), observed_return)) / N_val
  + w_pair  * sum(counterfactual_group_loss) / N_pair
  + w_pg    * sum(-stopgrad(advantage) * log pi(chosen_action | prefix)) / N_pg
  - w_ent   * sum(legal_action_entropy) / N_pg
```

Zero eligible examples omit that term with a recorded reason. An enabled objective with no eligible data across an entire planned training stage is a configuration/data error, not a successful zero loss. Huber's transition parameter is configured, initial fixture value 1. Value targets and advantages are detached from the prediction graph. Duplicate supervision must not count a transition token in both `N_tok` and `N_world` unless an explicitly versioned weighting experiment intends it.

Determine window denominators from batch metadata before backward. For each microbatch, backward the weighted sum divided by the **window** denominator for each term. Do not average already-averaged microbatch losses. Accumulate gradients, perform the declared distributed reduction once, check finiteness, clip once, then execute one optimizer step and scheduler step. The existing trainer's raw answer-sum normalization must be migrated consistently; do not retain a second division after introducing pre-normalized multi-objective losses.

In distributed mode specify whether the reducer sums or averages. For a summing reducer divide by global eligible counts; for an averaging reducer multiply local normalized loss by world size or implement an equivalent documented rule. Empty local objective shards still participate in required collectives. Compare a split window with the corresponding global reference using unequal sequence lengths and sparse objective masks. Counters advance on actual events, not loop attempts.

A deterministic unit gradient check must show action supervision reaches `action_head` and shared decoder, value supervision reaches `value_head` and shared decoder, and world prediction reaches decoder/token embeddings. With a feature off, its head must receive no unintended gradient. Use a tiny synthetic forward/backward without optimizer steps; this checks wiring, not learning. Finite-difference checks should be limited to a few scalar entries if needed, never a full model sweep.

## A2. Full-answer learning and goal contrast

Retain masked autoregressive cross-entropy on complete answers and EOS. Ordinary documents use the eligible next-token positions defined by their task type; they do not pretend to contain answer fields. Public transition examples use the same decoder with a different mask and explicitly typed target.

For a qualified counterfactual group with the same public history `h`, two goals `g1,g2` and different validated answers `y1,y2`, define `s(y|h,g)` as mean log probability over eligible answer tokens including EOS. Use both directions:

```text
L_pair(group) = 0.5 * max(0, margin - s(y1|h,g1) + s(y2|h,g1))
              + 0.5 * max(0, margin - s(y2|h,g2) + s(y1|h,g2))
```

Groups with equivalent acceptable answers are excluded and counted. With multiple valid answers, use the protocol's explicit equivalence classes; do not penalize another correct answer. A swapped target is a contrastive scoring input, never a factual observation in the ledger. Preserve pair groups across sharding and accumulation, or join their two sides by identity before computing the term. Do not silently train one-sided pairs.

Test the sign with hand-specified scores: raising the own answer score must not increase the loss, raising a wrong answer score must not decrease it, and a sufficiently satisfied margin yields zero. This is an immediate regression guard against swapping arguments at the existing pair-loss call site. Counterfactual grounding is a proposed inductive pressure; free generation and both-correct evaluation determine whether it actually helps.

## A3. Action and value learning

For K public legal actions, independently encode each `prefix + candidate`, score its terminal hidden state using the existing action head, mask illegal slots, and softmax over legal slots. Teacher cross-entropy uses a target distribution over acceptable actions, not an arbitrary single tie winner. Teacher distributions must sum to one over legal actions. Candidates with identical public action identities are deduplicated before labeling.

Use a protocol-defined extrinsic reward. The first task fixtures use terminal correct submission reward 1, incorrect submission reward 0, and action cost penalty `lambda_cost * actual_cost / initial_budget` on every charged action. A proposed initial fixture coefficient is 0.05; it is not a measured optimum. Environment cost is not itself success reward. Keep raw success, inquiry cost, submission cost and shaped return as separate reported fields.

For a completed finite episode use Monte Carlo return `G_t = sum_{j=t}^{T-1} gamma^(j-t) r_j`, with `gamma=1` in the first bounded tasks. Compute returns from real receipts. For a genuine terminal end, bootstrap value is zero. A time-limit truncation is not termination: either obtain a complete admitted return or exclude it from the initial complete-return objective. Later bootstrapping requires an explicitly separate contract and target-model identity. Do not silently substitute zero for missing future reward.

The value head predicts `G_t` from the public prefix before the action. Train with Huber loss. Store horizon and reward protocol with the target. Value estimates from one cost convention cannot be used in a planner with another. Values can be negative because actions cost resources; an unconditional sigmoid is therefore inappropriate for this return definition.

After supervised action learning, implement a minimal on-policy policy-gradient route. Collect a bounded batch with frozen behavior checkpoint `theta_b`, preserving legal sets and chosen-action log probabilities. Evaluate advantages `G_t - V_b(prefix_t)` using detached behavior values. For the first implementation, process that entire rollout batch in **one** accumulated optimizer window, without intermediate parameter changes or repeated policy epochs. The current model identity at the start must equal the behavior identity. After the update, collect fresh episodes before the next policy-gradient window. Stale trajectories can still supervise world prediction or qualified answers, but cannot silently reenter the on-policy loss.

Do not add importance sampling, clipped multi-epoch policy optimization or a second policy engine until this simple route works. Optional advantage centering/normalization is recorded and disabled for single-decision fixtures. Entropy regularization is confined to legal actions. Tests cover one legal action, all-illegal rejection, absent behavior probability, interrupted episode, stale checkpoint and detached targets.

## A4. Public prediction with uncertainty accounted for

Train next-public-outcome token likelihood on actual observed transitions, including failures and stochastic outcomes. Inputs stop at the chosen action; targets contain only the next admissible public fields. A goal can be an input when it changes the predicted reward, but the physical feedback label cannot be invented from the desired answer. Renderer, masks and public schema have independent identities.

For small worlds, use a public support of possible feedback values. Compute normalized sequence scores for that support, aggregate duplicate parsed outcomes, and retain any explicitly modeled unknown category. For open observations, use bounded samples and retain invalid/unparsed sample mass. Do not repeatedly resample until all outcomes look valid and then report only successes. Calibration is measured on independent measurement worlds; repeatedly tuned calibration data is training-controller data.

A predictive learner is useful to planning only if it predicts **action-dependent differences** on held-out public prefixes. Required controls are action-shuffled conditioning, unchanged-observation prediction and frequency-only prediction. Assess proper scoring loss and calibration as well as top-one feedback accuracy. Low transition loss dominated by structural tokens cannot qualify the model; report content-field loss separately.

The initial recursive expected-return implementation is enabled only for a finite, declared public outcome support with a validated probability-mass contract. Open-text sampling may be implemented and scored as prediction in M03, but does not automatically qualify for M07 recursion. A future sampled planner must carry a separate algorithm identity, an explicit sample estimator and conservative uncertainty/fallback rule, and be reported as approximate. Unknown open-world outcome support is a supported reason to use policy-only behavior.

## A5. Counterfactual inquiry and actual branching planning

The first learned planner performs bounded look-ahead over public predicted outcomes. Its state is a public history plus remaining resource allowance; it is not the private environment state. At a real decision it obtains the current legal action set from the actual public environment adapter. At an imagined node it obtains actions only from a public schema rule or a validated predicted public field. If legality cannot be established, stop that branch at the value estimate; never consult the real environment's future hidden state.

For a node `h`, action `a` and remaining depth `d`, define:

```text
Q_d(h,a) = sum_o p(o | h,a) * [r(h,a,o) + gamma * C_{d-1}(append(h,a,o))]
C_0(h)   = V(h)
C_d(h)   = max_legal_a Q_d(h,a)
```

Terminal branches have continuation zero. An exhausted real task budget is terminal or truncated according to its protocol. Expand a bounded number of root actions using the policy ranking, then outcome branches, maintaining a global node, model-token, model-call and wall-time budget. Beam pruning and sampling are declared approximations. Search depth means actual action/outcome transitions, not a multiplier on generated text length. Root selection must be able to change when predicted feedback or downstream value changes while root policy scores are held fixed.

Invalid or unknown outcome mass receives the protocol's conservative lower-bound return for the remaining horizon, or causes an explicit uncertainty fallback. Do not renormalize it away. The lower bound follows the declared reward range and remaining maximum cost; if no valid bound exists, disable expected-return planning for that protocol. Cache only by complete public prefix, action, model, memory and schema identities. Stop immediately when a hard resource limit is reached, returning the best completed legal root or a declared fixed-policy fallback.

For inquiry-specific tasks, an optional secondary criterion estimates how an observation would change the answer distribution. With a fixed diagnostic answer support `Y` of size greater than one:

```text
I_hat(h,a) = [H(p(Y|h)) - sum_o p(o|h,a) H(p(Y|append(h,a,o)))] / log(|Y|)
score(h,a) = Q_d(h,a) + beta_info * clamp(I_hat(h,a), 0, 1)
```

Use the same frozen model and answer support before/after the hypothetical observation. The information term is an approximation from model predictions, not certified information gain about the true world. Set `beta_info=0` in the main expected-return control; an enabled inquiry treatment is separately identified. Singleton/undefined support disables the information term. An invalid branch cannot produce an information bonus. The search already charges real action costs through reward; imagined compute is additionally reported and bounded.

Complementary-query fixtures are required: the first query can have little immediate answer benefit, while two queries jointly solve the task. Include distractors with high feedback entropy but no goal relevance. These expose greedy query selection and the error of equating surprise with useful information. Compare fixed, random, one-step, depth-two and oracle controls at the same real interaction budget and report additional inference compute. A two-step planner that merely generates longer strings fails this specification.

Search distillation is optional after qualification. Store the search policy, settings and cost as teacher provenance; train the action head on its legal root distribution. Predictions from an unqualified world model must not create authoritative success labels. Always retain an undistilled control to measure whether search knowledge transfers into the fast policy.

## A6. Replay, retention and controlled plasticity

Start with deterministic family-stratified replay using real admitted experience. Keep whole counterfactual groups together. Configure acquisition/replay proportions explicitly, persist fractional scheduling state and record shortfalls. A repeated example increases presentations and supervised targets, not unique experience. Replayed action demonstrations remain teacher-supervised data; old behavior trajectories are excluded from the on-policy policy-gradient term.

Define acquisition gain `A_new = score_child(new) - score_parent(new)` and protected-family retention deltas `R_f = score_child(f) - score_parent(f)`. Report all families, worst-family delta and uncertainty, not just their mean. Replay is accepted as useful only when its cost-matched comparison improves this acquisition/retention tradeoff. Inert weights can retain everything by learning nothing; always report new learning and parameter displacement alongside retention.

The existing engineered plasticity controller remains an optional scheduler. FORM, STABILIZE, EXPAND, REACQUIRE and HOLD transitions consume training-controller metrics only. Use frozen thresholds, hysteresis, minimum exposure and explicit per-family qualification. A previously learned skill can require renewed acquisition plasticity after collapse; historical Arkenstone results do not support universally lowering LR on failure. No controller transition proves a model has improved its own learning algorithm.

Implement optional protected-prefix distillation only after replay works: a frozen accepted parent supplies full-output distributions on approved training anchors; penalize child divergence on eligible answer positions. This is a preservation prior, can impede new learning, consumes parent inference, and must be compared against equal-cost replay. Do not distill sealed answers or claim it removes catastrophic forgetting. Keep its coefficient zero by default.

## A7. Task selection and failure-directed practice

Classify observed failures by independent validators: invalid action, premature stop, wrong goal binding, insufficient evidence, wrong public prediction, forgotten skill, retrieval mismatch or unresolved. A model-written explanation is a proposal, not the failure label. Preserve unresolved cases.

Build a training-only task factory with parameterized mechanisms and difficulty axes. A task proposal contains generator version, public task specification, admissible verifier, expected resource bound and split family. Validate solvability and answer uniqueness/equivalence; deduplicate by mechanism cluster before admission. An executor may generate many proposals, but unverified proposals do not become gold labels.

Use a transparent curriculum first. Maintain moving training-controller success estimates per qualified task bucket; allocate a configured mixture of frontier practice, protected replay and uniform exploration. Define frontier as success within a configured nontrivial interval, initial candidate 0.2–0.8, not a universal optimum. Keep a nonzero uniform component so currently failing families are not permanently abandoned. Persist bucket counts, estimates, sampler RNG and selection reasons. Never use sealed failures to choose the next practice task; if a held-out set is repeatedly used for adaptation, reclassify it and create a new sealed set.

A proposed skill library stores validated public subtask recipes and success conditions, not hidden solutions indexed by task ID. Initially recipes are inspectable action macros with explicit cost and termination; treat them as engineered baselines. Learned option selection/distillation is a later opt-in implementation using the same action model. Expand macros into charged primitive actions so apparent horizon reductions do not hide interaction cost.

## A8. Candidate improvement as a transaction

```text
freeze parent P and eligible memory/data snapshots
propose a bounded change and preregister comparison protocol
collect or select admitted training experience
construct child C from P in an isolated lineage
update C through the canonical trainer under an authorized allocation
evaluate P and C on matched acquisition and protected suites
validate receipts, resource matching, uncertainty and contamination checks
record ACCEPT / REJECT / INCONCLUSIVE with chief decision
publish new parent pointer only for an accepted candidate; otherwise retain P
```

Build this state machine with deterministic fixtures now. Actual learned cycles need allocation and qualification. Idempotent retries reuse a transaction ID and never duplicate updates or evaluation records. A crash cannot publish an unexamined child. Rejected children and failed runs retain compact evidence and their identities; large weights follow the owner's storage policy outside Git.

Every transaction and comparison has `execution_mode`, `evidence_class`, `change_kind` and verified `learned_updates` fields. Classes distinguish construction fixtures, random-model operations and learned measurements. Fixture transactions use a separate registry namespace and cannot write the learned accepted-parent pointer. A weight-improvement claim requires actual recorded child updates, authentic learned-measurement receipts and a chief approval bound to their hashes. Zero-update memory/config changes require their own change kind and corresponding evidence, not a weight-learning label. Synthetic **task data** from a real qualified generator is allowed in learned measurement; it must not be confused with synthetic evaluator/model test doubles.

Separate weight improvement, memory accumulation, task-selection changes and algorithm/code changes in the receipt. The first autonomous proposal space is a versioned allowlist of curriculum/replay/schedule settings with hard resource bounds. General source-code mutation is outside that initial loop; any later method-editing agent must produce a reviewable patch and independent comparison in an isolated branch. The learned model cannot rewrite its evaluator, data eligibility or promotion thresholds. Improving a few settings is a measurable limited capability, not proof of open-ended recursive self-improvement.
