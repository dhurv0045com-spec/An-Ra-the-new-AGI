# Learning and investigation algorithms

Status: engineering specification. Algorithm names here do not imply implementation. The current discovery prototype implements recurrent prediction and supervised imitation of a training-only one-step information-gain teacher. It does not implement the complete actor-critic/planning/consolidation system below.

## A. Observation-driven representation

Initial integrated model: encode each public observation/action/feedback tuple into width 128; update a width-128 GRU state; condition outcome, action and value heads on state and goal. Use explicit masks and reset semantics. A shared action scorer handles variable legal candidates; avoid a separate output neuron per world-specific action identity.

Compare against a full-history control with the same observations. Report exact parameter counts and processing cost. Add a shared recurrent attention block only if delayed-state and composition tests expose a useful weakness of the GRU. Maintain separate experiments for increasing capacity and changing the state mechanism.

Supervised prediction objective is mean negative log likelihood over eligible observed targets. For binary outcomes use binary cross-entropy with logits; for multiple discrete outcomes use categorical cross-entropy. Stop/termination supervision is explicit. Do not mask a padding loss after an invalid infinite computation and assume the resulting gradient is safe.

Useful diagnostic axes: history length, delay, irrelevant events, goal changes, overwritten facts, held-out rule composition and unseen observation rendering. Paired counterfactual goals must produce different answers when required.

## B. Exact one-step teaching: useful bootstrap, insufficient destination

For a training-only posterior over hypotheses `H` and target `Y`, the teacher chooses a legal action maximizing:

`IG(a) = entropy(Y | history) - expected_observation[entropy(Y | history, a, observation)]`.

Use the same hypothesis prior used to generate the training teacher's labels, or explicitly account for a changed prior. Empty posteriors indicate model misspecification or inconsistent data, not certainty. Do not use held-out world truth tables to label training actions.

The prototype imitates ties among maximal-gain actions. Rows with no informative action do not supply a positive action-imitation target. This is a bootstrap assumption, not evidence that the investigator discovers its own learning strategy.

### Exact failure case to preserve

Let hidden independent fair bits be `a,b`, and define `f(x0,x1)=a*x0 XOR b*x1`. The scored target is `f(1,1)`. Querying `(1,1)` is forbidden. Observing `f(1,0)` alone or `f(0,1)` alone gives zero bits about the target, but observing both determines it. A purely one-step gain teacher gives no preference to the first preparatory action.

This is a mathematical limitation of the teaching objective, not proof that a trained neural agent necessarily fails every such task. Preserve an executable exact diagnostic, then test whether a proposed multi-step learner actually overcomes the limitation on held-out constructions.

## C. Multi-step teaching control

Implement a small exact depth-two teacher for bounded finite environments:

`Q2(a) = IG(a) + expected_o[max_legal_b IG(b | history,a,o)]`.

Account for action costs and avoid the scored target. This control establishes whether the environment contains useful delayed information and whether labels can teach it. Exact search is allowed only in small training/diagnostic spaces, with its full computation counted. It is not a scalable deployed AGI planner.

The teacher horizon is capped by remaining real inquiries. Use depth two only with at least two inquiries left; with one left use one-step net gain; with none left disable policy supervision while retaining legitimate prediction targets. Otherwise a teacher can reward plans that the acting learner has no budget to finish.

Compare random collection, coverage, one-step teacher, depth-two teacher and learned policy at equal real inquiries. Report teacher-assisted oracle controls separately from inference using only the learned core.

## D. Learning inquiry policies from outcomes

After supervised bootstrap, train an on-policy discrete actor-critic. First freeze the outcome predictor while training the investigator so the effect of action selection is interpretable. Jointly updating both is a separate arm.

Algorithm references for implementation agents: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) describes the interaction/update approach and surrogate policy objective; [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) motivates the value-based advantage estimator. The settings below are BRAMASTRA's proposed starting choices, not evidence that these methods guarantee useful investigation or AGI.

Initial bounded defaults to calibrate and freeze in the run specification:

| Choice | Initial value / rule |
|---|---|
| Episode horizon | At most eight real inquiries for the first study |
| Discount | 1.0 for these finite bounded episodes |
| Reward | Terminal scored task success minus declared per-inquiry cost |
| Baseline | Goal/state value head |
| Advantage | GAE with lambda 0.95, with terminal/truncation masks tested |
| Policy update | PPO clipped ratio, epsilon 0.2, four minibatch passes per collected rollout batch |
| Entropy weight | 0.01 initially; report realized policy entropy |
| Value weight | 0.5 initially; normalize losses by eligible decisions |
| Gradient clip | Global norm 1.0, finite checks before applying update |
| Bootstrap imitation | Separate initial phase; no hidden test-time teacher |

These are initial engineering settings, not claims of optimality. Tune on declared training/development pilots only. Record collection-policy probabilities, old log probabilities, legal masks, returns and truncation state. No gradient may flow through observed environment labels or the examiner's scoring implementation.

Evaluate a frozen investigator on a fresh strategy-validation pool. Track success and calibration versus real queries, not only reward. A policy can improve a shaped reward while failing the actual task. Include distracting unpredictable observations so novelty seeking is distinguishable from useful investigation.

## E. Predictive world models and bounded planning

Learn next public observation, reward and true termination from observed transitions. Use separate output heads with explicit likelihoods. The initial planner enumerates at most four candidate actions at depth two; depth four is an ablation after short-horizon prediction is calibrated.

Compare four conditions: no planning; learned planner with learned dynamics; oracle transitions with the same planner; learned dynamics with an exact small planner. The last two are diagnostic upper bounds to separate prediction and planning errors. Do not report oracle-assisted performance as learned competence.

Simulated transitions are model predictions, not independent truth labels. Keep real and imagined experience separate. Measure error as a function of rollout depth and domain shift. Reduce planning depth when it worsens real outcomes. Account for the planner's inference compute in every comparison.

## F. Consolidation and replay

Freeze a parent checkpoint before acquisition. Collect experience from an unfamiliar training family. Create three controlled children:

1. New experience only.
2. Equal-count replay of prior and new training states/episodes.
3. Family-balanced replay with the same total update and interaction allowance.

Start with equal-count replay; add priorities only after a measured benefit. State whether optimizer moments are preserved or reset, and apply the same rule to all children. A fresh optimizer fine-tune is not a complete continuation experiment.

Evaluate old-family retention and new-family acquisition on paired fresh worlds. Compare against both the actual parent and the original retained-skill reference. Report worst-family changes and cumulative forgetting, not only mean accuracy. For multiple rounds, preregister practical retention margins and a finite candidate budget; do not loosen a margin after seeing a loss.

Successful acquisition with a hand-written replay schedule is evidence for consolidation, not learned control of consolidation. To test that next step, compare learned experience/replay selection against fixed selection with identical candidate pools, feedback and total compute.

## G. Task generation and self-modification

Task generation comes after useful inquiry and retention. Train or select generators on a learnable frontier, while a fixed independent suite prevents apparent progress from easier generated tasks. Preserve unsuccessful and ambiguous generated cases.

Learning-method changes and code proposals come later. A candidate may propose a change in a bounded experiment but cannot edit its examiner, parent evidence, resource allowance or promotion rule. Execute code in an isolated environment with explicit capabilities and time/memory limits; no inherited credentials or arbitrary host access. Include search cost when comparing the new procedure with its parent.

Initially the chief chooses hypotheses and a fixed orchestrator executes them. Do not call that learned recursive self-improvement. The first learned autonomy claim is narrower: experience chosen by the learner causes independently measured, retained gains.
