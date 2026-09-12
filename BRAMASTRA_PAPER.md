# BRAMASTRA: A From-Scratch Research Architecture for Investigation, Transfer and Durable Self-Improvement

**Research design and preliminary technical report — version 1.0, 9 September 2026**

**Project:** An-Ra / BRAMASTRA

**Status:** Complete design manuscript; integrated-system experiments pending.

**Implementation update, 12 September 2026:** the [B2 build packet](engineering/build_20260912/README.md) is the current execution authority, informed by refreshed cross-branch results. The historical study below is unchanged.

**Evidence baseline:** repository commit `6525332`. This manuscript does not report an achieved AGI system.

## Abstract

An agent that improves on familiar tasks can still fail to investigate unfamiliar mechanisms, transfer knowledge between domains or preserve earlier capabilities. BRAMASTRA is a research architecture for studying these failures as parts of one learning system trained from randomly initialized core weights. The proposed system combines a shared public-evidence representation, recurrent task state, outcome prediction, learned inquiry, bounded planning and persistent experience consolidation. Its central hypothesis is that useful self-improvement requires a coupled loop: the learner selects experiences, those experiences produce transferable parameter updates, and an independent evaluation procedure verifies that the resulting gains survive retention tests. The architecture separates within-task adaptation, across-task learning and changes to the learning procedure, preventing evidence for one from being mistaken for another. We specify information boundaries, data structures, learning objectives, controlled comparisons and a compute-constrained experimental sequence. An existing two-seed synthetic-rule pilot compares one-step and depth-two inquiry teaching under matched conditions. Depth-two teaching ties its control in one seed and improves accuracy by two predictions out of 92 in the other. These observations do not establish a reliable advantage. The paper therefore presents a falsifiable research program toward general intelligence, together with a limited preliminary result, rather than a demonstrated solution or a guarantee that scaling the proposed architecture will produce AGI.

## 1. Research objective and scope

The objective is an agent capable of learning unfamiliar tasks, acquiring reusable knowledge, investigating uncertainty, planning toward goals and improving repeatedly without destructive forgetting. We use “general intelligence” to denote that broad research ambition. We do not define it as a score on the initial environments or as the presence of named architectural components.

The practical constraint is limited computation. The owner reports access to approximately 100 free Kaggle TPU-hours per week; this is a planning envelope, not a verified allocation. A useful program must extract information from small experiments before committing accelerator time. Cheap environments with independently checkable outcomes serve as instruments for identifying failure mechanisms. They do not substitute for later evaluation on independently authored, unfamiliar and practically meaningful tasks.

The core model begins with random weights. “From scratch” does not mean absence of human assumptions: the observation interface, optimization algorithm, training distribution, reward definitions and any symbolic teacher are engineered priors. Every such prior must be disclosed. A pretrained teacher or retrieved solution cannot be silently incorporated and then attributed to the learned core.

The present contribution is the design and operational specification of a connected research program. It is not a claim that recurrent models, active learning, replay or policy optimization are new inventions. The value of the design will depend on whether its experiments isolate and resolve the uncertainties described below.

## 2. Central hypothesis

BRAMASTRA proposes that progress toward broad adaptive competence requires the following chain to work together:

1. Public experience supports a representation of the current task and its uncertainty.
2. The agent chooses actions whose consequences improve its ability to accomplish the goal.
3. Predictive models support useful planning within a bounded computational budget.
4. Selected experience produces reusable parameter changes rather than only temporary task-state changes.
5. Consolidation preserves earlier competence while incorporating new capabilities.
6. An independent examiner distinguishes durable gains from shortcuts, memorization and changes in evaluation difficulty.

This chain is a hypothesis, not a sufficiency theorem. A failure in any link can explain why adding more modules produces a larger system without producing more general intelligence. Conversely, success on all initial tests would justify broader experiments, not an automatic AGI declaration.

The main proposed outcome is **retained transfer per unit total cost**. Acquisition accuracy alone rewards narrow learning; retention alone rewards refusing to learn; raw reward alone can favor a shortcut. We therefore report a vector of acquisition, transfer, retention and resource use before considering any combined score.

## 3. Formal setting

Let a task be sampled from a distribution over partially observed environments. Its hidden mechanism is \(z\), public goal is \(g\), observable history is \(h_t=(o_0,a_0,o_1,\ldots,o_t)\), and remaining interaction allowance is \(b_t\). The agent chooses a legal action according to

\[
a_t\sim\pi_\theta(\cdot\mid h_t,g,b_t).
\]

The environment returns an observation, an explicitly defined task reward and a recorded cost. The evaluator may access hidden truth for scoring; the learner may not. A task distribution is characterized by its mechanism families and transformations, not merely its random seed.

We distinguish three update scales:

- **Task adaptation:** \(s_{t+1}=F_\theta(s_t,o_{t+1},a_t,g,b_{t+1})\), with parameters fixed.
- **Consolidation:** \(\theta_{k+1}=U(\theta_k,D_k,R_k)\), using newly collected data \(D_k\) and qualified replay \(R_k\).
- **Procedure change:** \(U_{k+1}\) or the data-selection procedure changes through an explicitly evaluated proposal.

Better task-state adaptation does not establish better consolidation. A human-written training change does not establish that the model learned to improve its own method. Each experiment names which update scale is under investigation.

For an acquisition round, let \(T_k\) denote transfer performance and \(P_{k,j}\) performance on protected capability family \(j\). A candidate is eligible for acceptance only under a predeclared rule requiring useful transfer improvement, bounded family-level regression and an acceptable total cost. Practical margins and uncertainty procedures must be fixed before confirmation evaluation. This paper does not choose universal numerical thresholds: those require independent pilot variance and task-specific utility judgments.

## 4. System architecture

### 4.1 Shared public-evidence interface

The integrated learner receives goals, observations, action descriptions, feedback and remaining budget through one representation path. Provenance identities remain in the audit record but are excluded from the learned content. A task identifier must not become a lookup key for its answer.

The first proposed interface serializes validated public events deterministically and encodes bytes with a 260-symbol vocabulary, including reserved structural symbols. Object key ordering is canonical; event and list ordering remain meaningful. Oversized inputs fail explicitly rather than silently losing a decisive observation. This interface provides an inexpensive, domain-independent starting point, but its sequence efficiency and suitability for language remain empirical questions.

A proposed width-128 event encoder supplies a shared representation for observations and finite candidate actions. Two state mechanisms are compared: a full eligible-history attention control and a width-128 recurrent state. They receive the same public evidence. Differences in parameters, inference work and memory are reported rather than hidden behind a claim of architectural fairness.

The prototype used in the preliminary study instead consumes fixed-width Boolean observations. Its results do not validate this proposed shared byte interface.

### 4.2 Prediction, policy and value

The task state conditions an outcome predictor, action scorer and value head. A separate predictive component estimates future public observations or state transitions for planning. Output schemas can differ by task, but adapters may not compute task solutions.

For finite action spaces, candidates are enumerated using public schema bounds. Hidden simulator state cannot prune the candidate set. Open-ended language or program generation requires a later generative-action experiment; finite candidate scoring does not solve that problem.

Termination is an action or an explicit environment event. Goal reward must depend on completing the requested task. Correctly announcing that a goal has not been achieved must not count as achieving it.

### 4.3 Working state and persistent experience

Working state is reset at task boundaries. Persistent experience is stored separately and may influence later training or explicitly permitted retrieval. Retrieval access must be held constant across compared systems, and confirmation answers must never enter the retrieval index.

Replay fragments require either a true reset or a declared warm-up prefix. Reconstructing recurrent state from unavailable future information is prohibited. Batch neighbors, padding and stale caches must not alter an episode’s output.

### 4.4 Independent examiner

The examiner owns protected tasks, scoring rules and acceptance decisions. It interacts through the public model interface and cannot be edited by the candidate. The training process receives only the feedback permitted by the active protocol.

Independence is an information and authority boundary, not simply a second agent name. Repeatedly choosing models after reading the same confirmation results turns that pool into development data. Evaluation history must therefore record every candidate and every exposure.

## 5. Learning algorithms

### 5.1 Prediction bootstrap

The initial predictor minimizes negative log likelihood of eligible outcomes:

\[
\mathcal L_{pred}=-\frac{1}{N}\sum_i\log p_\theta(y_i\mid h_i,g_i).
\]

Binary outcomes use binary cross-entropy with logits; categorical outcomes use the corresponding categorical objective. Padding, terminal states and unavailable targets have explicit masks. Prediction loss is a learning signal, not sufficient evidence of task competence.

### 5.2 Inquiry by immediate and delayed information

For a finite training-only posterior over hypotheses, immediate information gain is

\[
I(a;h)=H(Y\mid h)-\mathbb E_o[H(Y\mid h,a,o)].
\]

The posterior must reflect the declared training prior. An empty compatible set signals inconsistency or misspecification, not certainty. The teacher cannot use held-out mechanism tables to label training actions.

An exact depth-two control is

\[
Q_2(a;h)=\max\{0, I(a;h)-c(a)+
\mathbb E_o[\max(0,\max_{b\in A(h,a,o)}(I(b;h,a,o)-c(b)))]\}.
\]

The legal set excludes repeated queries and the scored target. At one remaining inquiry, the teacher reduces to one-step net gain. At zero remaining inquiries, policy supervision is disabled. The zero-valued stop option is a diagnostic utility convention; the current fixed-budget learned rollout still executes its allocated inquiries. Consequently nonzero-cost stopping behavior requires a separate experiment.

The need for delayed information has an exact illustration. Let independent fair bits \(u,v\) define \(f(x_0,x_1)=ux_0\oplus vx_1\), and score \(f(1,1)\) while forbidding that direct query. Either basis observation alone leaves one bit of target uncertainty. Both observations together determine the target. Thus the first useful preparatory query has zero immediate target information but positive two-step value. This demonstrates a limitation of a myopic objective, not a proof that a particular neural agent will fail or succeed.

The supervised policy imitates a distribution over tied best legal actions. Zero-gain rows do not contribute positive imitation supervision. Teacher-search cost is counted separately from model inference. An oracle choosing the right experiment during evaluation is not a learned investigator.

### 5.3 Outcome-trained inquiry

The next proposed stage optimizes actual task return. Initially the predictor is frozen so the effect of policy learning can be isolated. Joint predictor and policy updates form a separate arm.

The policy objective uses a clipped probability ratio:

\[
\mathcal L_{clip}=\mathbb E_t[\min(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)],
\qquad r_t=\frac{\pi_\theta(a_t\mid h_t)}{\pi_{old}(a_t\mid h_t)}.
\]

This follows the PPO family of policy optimization methods [1]. BRAMASTRA’s proposed initial settings are a finite episode discount of 1, advantage-estimation parameter 0.95, clipping parameter 0.2, four minibatch passes, entropy weight 0.01, value-loss weight 0.5 and gradient norm cap 1. These are starting choices to calibrate and freeze, not established optima. Reward is terminal task success minus declared inquiry cost. Rollout probabilities, policy versions, terminal masks and truncation semantics are recorded explicitly.

### 5.4 Predictive planning

The proposed planner evaluates short action sequences through a learned model of public consequences. Its value must be measured against no planning under the same real-interaction allowance, with additional inference cost disclosed. Work on learned world models provides an established methodological precedent for improving behavior using predicted futures [2]; it does not establish that BRAMASTRA’s proposed model is accurate or useful.

Planning tests separate three cases: no planning, learned-model planning and privileged true-model planning. The final case is diagnostic. If true-model planning helps but learned-model planning fails, model error is implicated. If neither helps, the planner or task design may be the limiting factor. Real observations continually correct predictions; a planner cannot treat its own imagined outcomes as verified evidence.

### 5.5 Consolidation and retention

A child update uses a declared mixture of new experience and replay:

\[
\mathcal L_{update}=\alpha\mathcal L_{new}+(1-\alpha)\mathcal L_{replay},
\]

with any distillation or parameter-regularization term introduced as a separately tested intervention. Replay ratios and sampling policies are frozen for each comparison. Every comparison starts from the same parent and uses matched update budgets and clearly specified optimizer state.

Catastrophic forgetting is a recognized continual-learning problem; parameter protection is one established approach [3]. BRAMASTRA does not assume replay or any regularizer solves it. Acquisition gains are evaluated alongside each protected family, and repeated rounds are compared against the original parent as well as the immediately preceding checkpoint.

### 5.6 Experience selection and procedure improvement

The first proposed self-improvement claim is deliberately concrete: experiences selected by the learner produce a child with better fresh-task transfer and acceptable retention than experiences chosen by a fixed policy, under matched budgets.

Selection may use predicted learning progress, uncertainty or failure categories, but these proxies are not acceptance metrics. A learner can select surprising yet useless data. Fixed selection, random selection and replay controls determine whether the selection mechanism contributes beyond additional computation.

Only after this loop works should the agent propose changes to its learning procedure. Proposals execute in isolated, resource-limited experiments and cannot alter the examiner, parent evidence or acceptance rule. Search, rejected candidates and recovery costs belong in the comparison. An externally scripted search is reported as orchestration until the agent demonstrably learns which changes to propose.

## 6. Data structures and reproducibility

The implemented contract layer distinguishes `TaskSpec`, `PublicObservation`, `Action`, `Transition`, `Episode`, `TrainingBatch`, `Checkpoint`, `Experiment`, `Outcome` and `Promotion`. Nested records are immutable after construction; strict validation distinguishes booleans from integers and rejects nonfinite values. Public payloads require explicit reviewed schemas. Merely using a record called “public” does not establish privacy.

Content identities cover canonical records. Tensor identities include dtype, shape, byte order and content. Checkpoint manifests identify weights, optimizer, schedule, random states, sampler, replay state, parent, code, data and runtime. The existence of these fields does not establish that a backend restores them correctly; restore behavior requires its own operational test.

Semantic mechanism identity is separated from surface rendering and query selection. Renaming a mechanism or asking it a different question must not create a supposedly independent confirmation example. Generated-program equivalence is tractable only within declared finite domains; the design does not assume a general program-equivalence oracle.

Each experiment writes a new run directory. Failed runs are retained. Raw outcomes and compact reports are versioned; large datasets and weights stay outside Git with explicit manifests. Snapshots identify the historical implementation even if the runner is later repaired.

## 7. Preliminary study: matched inquiry teaching

### 7.1 Design

The completed D02 pilot used four-bit Boolean mechanisms divided into 73 training, 23 development and 23 unscored confirmation mechanisms. Development contained 14 conjunction, six parity and three threshold cases. Training seeds were 801 and 802. Each seed generated 256 episodes with a two-query horizon and uniform legal history collection.

One-step and depth-two arms shared observations, targets, labels, legal masks, row order, initial model parameters, optimizer settings and sampler initialization. Only teacher gain labels changed. Each learner had 9,090 parameters, width 32, and 500 AdamW updates with batch size 32, learning rate 0.001 and policy weight 0.25. Training started from random weights.

Evaluation used four targets per development mechanism, evaluation seed 17, and budgets zero, one and two. The primary contrast was depth-two learned accuracy minus one-step learned accuracy at two queries. Each arm also evaluated random, coverage and memory-disabled policies using that arm’s predictor. This controls inference-policy comparisons within an arm; it does not isolate shared representation changes between teaching arms.

### 7.2 Results

| Training seed | One-step learned | Depth-two learned | Coverage under either arm | Primary difference |
|---|---:|---:|---:|---:|
| 801 | 61/92 (66.30%) | 61/92 (66.30%) | 61/92 (66.30%) | 0.00 percentage points |
| 802 | 63/92 (68.48%) | 65/92 (70.65%) | 62/92 (67.39%) | +2.17 percentage points |

Descriptive 95% world-cluster bootstrap intervals were [0, 0] and [0, 5.43] percentage points. These intervals are conditional on each trained seed. A degenerate interval over zero observed paired differences is not proof of equivalence. Brier-score differences, depth-two minus one-step, were +0.004166 and −0.004145; lower is better.

The recorded campaign time was 26.783 CPU seconds, including the two seeds. Source snapshots and before/after hashes matched. Independent review recomputed raw correctness and verified unique paired keys and cross-arm labels. The contract suite passed 22 focused tests; the later inquiry suite passed eight, including failure and timeout controls. Those test counts concern correctness, not intelligence.

### 7.3 Interpretation

The study does not establish a reliable depth-two advantage. It demonstrates that the paired experiment can be run and audited, and provides a small directional observation in one seed. Three threshold development mechanisms are insufficient for stable broad family-level conclusions.

The teacher intervention may be weak, difficult to imitate or difficult for the predictor to exploit. Uniform history collection studies imitation coverage rather than autonomous exploration. No original model checkpoints were retained for counterfactual re-evaluation; raw outputs cannot recover unavailable internal policy behavior. Further experiments are deferred. The next specified diagnosis first measures teacher-target disagreement and historical action-sequence changes without additional training.

## 8. Planned experimental program

| Stage | Main question | Required control | Evidence that would reject the proposed mechanism |
|---|---|---|---|
| Environment qualification | Is the task solvable and scored correctly? | Independent finite interpreter and actual baseline rollouts | Oracle access exceeds permitted information, shortcut reward or semantic leakage |
| Shared state | Does memory preserve useful evidence? | Same public history with full-history versus recurrent processing | Gains disappear after equalizing information or are caused by leaked state |
| Inquiry | Does action selection improve task success? | Random, coverage and teacher controls; frozen predictor | Extra inquiry compute fails to improve fresh-task outcomes |
| Planning | Do learned predictions support better decisions? | No planning and privileged-model diagnostic | Planning adds cost without gains or exploits model errors |
| Consolidation | Does acquisition survive updating? | New-only versus replay from the same parent | New gains hide unacceptable protected-family regressions |
| Experience selection | Does the learner choose useful training data? | Fixed selection with matched pools and cost | Improvements are explained by more updates or easier tasks |
| Repeated improvement | Do gains accumulate over rounds? | Original-parent and fixed-procedure references | Drift or forgetting eliminates earlier gains |
| Broader transfer | Does competence cross interfaces and mechanisms? | Independently authored tasks and surface controls | Performance depends on familiar generator templates |

The initial environment families are causal switch systems, stateful inventory tasks and bounded program laboratories. Language and broader practical tasks follow only through interfaces that preserve the same information discipline. No environment package is considered qualified because it returns a hard-coded oracle score.

Confirmation protocols are frozen after development calibration. They specify primary outcomes, practical margins, candidate counts, seed counts, sampling units and rules for incomplete runs. Bootstrap sampling clusters by mechanism. More targets from one mechanism do not create more independent mechanisms; more evaluation rows do not create more training replications.

## 9. Compute and execution strategy

Compute is allocated to resolving uncertainty, not maximizing model size. CPU work first validates task semantics, gradients, masking, identities and resume behavior. Accelerator allocation then requires a concrete learning question and a verified live runtime.

The smallest integrated model is a control, not an asserted sufficient capacity for AGI. Larger candidates are justified by learning curves and diagnosed representation limits. Throughput reports include compilation, padding, evaluation, failed work and checkpoint transfer. Useful updates per wall-time and resource cost matter more than nominal accelerator availability.

Disposable accelerator sessions require durable resume state and fresh-process restore verification. A same-process CPU continuation check is not proof of remote TPU recovery. The owner’s weekly envelope cannot be treated as a verified entitlement or consumed repeatedly by independent packages.

No new training campaign is part of completing this manuscript. The experiment registry and standalone work orders define later execution, once the owner chooses to resume it.

## 10. Limitations and unresolved requirements for general intelligence

The proposed architecture leaves major scientific questions open. A shared byte interface does not establish semantic understanding. A fixed recurrent state may lose information over long horizons. Finite rule environments may teach strategies that do not transfer to language, perception or real-world ambiguity. Learned planning can amplify model error. Replay can preserve obsolete behavior or fail to preserve rare capabilities. An experience-selection policy can exploit its own proxy for progress.

The approach also lacks evidence for open-ended concept formation, robust compositional reasoning, reliable long-horizon execution and autonomous scientific discovery. No theorem connects the proposed components or available compute to AGI. External tools, symbolic teachers and orchestration must remain visible so improvements cannot be credited to a model that did not produce them.

Evidence for broader intelligence would require learning curves on genuinely unfamiliar tasks, transfer between independently designed domains, retained improvement across repeated acquisitions, reliable handling of ambiguity and failure, and external replication under disclosed resource limits. These requirements guide future work; they do not form a universally accepted certification procedure.

## 11. Conclusion

BRAMASTRA frames general intelligence as an unresolved learning-systems problem: choosing useful experience, representing its consequences, turning it into transferable competence and retaining that competence over time. This paper specifies a from-scratch architecture and an experiment sequence for testing those mechanisms without confusing engineering completion with scientific success. The preliminary inquiry result is inconclusive, which directs the next step toward diagnosis rather than scaling. The manuscript is complete as a research design and preliminary report; the broader system and its central hypothesis remain to be tested.

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. *Proximal Policy Optimization Algorithms*. 2017. [Primary paper](https://arxiv.org/abs/1707.06347).
2. Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. *Mastering Diverse Domains through World Models*. [Primary paper](https://arxiv.org/abs/2301.04104). Cited as methodological context, not a BRAMASTRA implementation claim.
3. Kirkpatrick, J., et al. *Overcoming Catastrophic Forgetting in Neural Networks*. [Primary paper](https://arxiv.org/abs/1612.00796).

## Repository evidence and execution documents

- [Architecture](engineering/SYSTEM_ARCHITECTURE.md), [learning algorithms](engineering/LEARNING_ALGORITHMS.md) and [observation encoding](engineering/OBSERVATION_ENCODING.md).
- [Contract acceptance](engineering/reports/W01_REVIEW.md) and [D02 chief review](engineering/reports/D02_REVIEW.md).
- [Immutable D02 manifest](artifacts/bramastra/d02_matched_teaching_20260908_luna/manifest.json), [raw run directory](artifacts/bramastra/d02_matched_teaching_20260908_luna/) and [partial W04 handoff](engineering/reports/W04/HANDOFF.md).
- [Frozen D02 execution design](engineering/work_orders/D02_EXECUTION.md), [next diagnostic design](engineering/work_orders/D02_DIAGNOSIS.md) and [full experiment registry](engineering/EXPERIMENT_REGISTRY.md).

**Authorship note:** This manuscript was drafted through AI-assisted engineering review of the repository. Human authorship, affiliations and publication submission have not been assigned or performed.
