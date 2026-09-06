# An-Ra: a full research blueprint for general intelligence from scratch

**Design revision: 2026-09-06. Status: proposed architecture and research program, grounded in the branch audits below. This document does not report a completed AGI or a demonstrated autonomous self-improvement loop.**

The objective is an AGI that acquires new abilities, investigates unfamiliar situations, acts effectively, and improves its own learning over time. All learned core weights start randomly initialized. The available planning envelope is approximately 100 owner-reported Kaggle TPU-hours per week. Actual hardware, quota, session limits, storage, and throughput remain runtime measurements.

This replaces the earlier blueprint as the **overall design entry point**. [BRAMASTRA.md](BRAMASTRA.md) remains the initial experiment plan; its small experiments are instruments, not the definition of the destination. Read the detailed [Cymek roadmap](docs/bramastra/CYMEK_100X_ROADMAP.md) and [Citadel roadmap](docs/bramastra/CITADEL_100X_ROADMAP.md) for branch-specific work.

There is no established sufficient recipe for AGI. A complete engineering proposal can expose its assumptions, implement them, and test them; it cannot make scientific uncertainty disappear. “100×” below means a falsifiable efficiency ambition against a named baseline, never a promised multiplier of intelligence.

## 1. What the project has been missing

The major missing capability is **a learned, closed loop from uncertainty to experiment to durable, transferable improvement**. A model trained on a human-selected curriculum can become better at that curriculum while never learning how to discover what it needs next. A script that selects another training job is useful automation, but does not by itself supply this capability.

The necessary chain is:

1. Represent a situation and maintain beliefs about things not currently visible.
2. Detect which uncertainties matter for a goal.
3. Select an observation or intervention whose outcome could change a decision.
4. Predict possible outcomes and their consequences.
5. Act, observe real feedback, and revise the representation.
6. Solve the task, including recovering from a failed plan.
7. Consolidate a reusable skill without destroying earlier skills.
8. Learn which learning strategies work in which situations.
9. Demonstrate the gain on fresh tasks that the learner did not choose or grade.

Every arrow needs evidence. More parameters cannot establish the arrows. Better contracts cannot establish them either. Both become valuable when they support a connected learning process.

### Current evidence and its limits

| Component | What exists in the inspected evidence | What is still missing |
|---|---|---|
| Cymek | Model/data/training contracts, data receipts, P35-A definitions and qualification artifacts | A fully materialized P35-A freeze and evidence of an integrated production run; learned investigation and consolidation |
| Citadel | T1D curriculum experiment, runtime bootstrap, calibration, packing and production checkpoint adapter | Real TPU validation/results for the inspected T1D plan; a correct end-of-answer training contract; general learning beyond calculator curricula |
| BRAMASTRA | Random-initialized local training, paired terminal supervision experiments, local continuation checks | Query-sensitive transferable computation, learned exploration, continual acquisition and broad competence |

The [actual BRAMASTRA results](docs/bramastra/RESULTS.md) make the distinction concrete. Terminal supervision repaired complete answers on training cases. Fresh-world transfer remained weak. Increasing binding-world diversity produced 62/128 fresh correct answers, below the 64/128 score of simple visible-value policies, with zero worlds having both counterfactual queries correct. Alternate rendering still failed. Those findings identify problems; they do not establish general reasoning.

The immutable branch audit points are Cymek `28bf57a0d299a2c13a99fe0046616c00a1b8530c` and Citadel `28ff690c04e655c88e4a6b394585e9b3428181ad`. Later branch changes need a new audit. Test receipt counts are software evidence, not intelligence measurements.

## 2. Define the destination without pretending there is one AGI score

The intended system should eventually perform a broad range of cognitive work, learn unfamiliar tasks with limited examples, and operate over extended horizons with a bounded need for human intervention. Early digital environments make this affordable to investigate; they do not establish competence in the physical world or every human domain.

Use a capability profile, with human comparisons where appropriate:

| Dimension | Required assessment | Main confound to exclude |
|---|---|---|
| Breadth | Independently authored tasks in language, mathematics, coding, causal inquiry, planning and practical information use | Many surface variants of one generator |
| Depth | Increasing compositional depth, ambiguity, horizon and abstraction | Memorizing shallow patterns |
| Learning | Success versus permitted observations, interventions and gradient updates on unfamiliar tasks | Training on the evaluation answers |
| Transfer | New mechanisms, combinations, interfaces and domains | Only changing random seeds or number ranges |
| Memory | Long interruptions, overwritten facts and retrieval under distraction | Replaying the answer in the prompt |
| Continual acquisition | Sequential skills with fresh retention tests after each acquisition | An improving mean hiding catastrophic losses |
| Investigation | Useful experiments outperform random and fixed alternatives at equal cost | Reward for novelty or eloquent explanations alone |
| Autonomy | Completed long tasks, recovery rate and human intervention time | A human choosing every next experiment |
| Calibration | Decisions respond appropriately to uncertainty and erroneous predictions | Confident fluent answers mistaken for knowledge |
| Self-improvement | Repeated independently confirmed gains caused by the learner's choices | Evaluator changes or selection of easier tasks |

Set numerical thresholds per experiment before looking at its outcomes. Later broad evaluations need task provenance, difficulty calibration, a specified human reference group, equal tool/time conditions, uncertainty intervals, and external review. No early table of arbitrary pass marks will certify AGI. Separating performance, generality and autonomy follows the useful framing in [Levels of AGI](https://arxiv.org/abs/2311.02462); adopting that framing does not establish that An-Ra meets a level.

## 3. Architecture: one integrated learner, added complexity earned by evidence

### 3.1 Initial engineering choice

Use the hashed Cymek P35 model specification, **35,411,328 parameters**, as the initial integration control once its real training path is verified. Avoid simultaneously maintaining several unrelated medium-size trainers. The existing BRAMASTRA B1 configuration is an alternative experiment, not another mandatory production stack. Keep the tiny local model for inexpensive contract diagnostics.

A 35M model is an experimental substrate. It is not a claim that 35M parameters suffice for AGI. Increase capacity only when learning curves and controlled comparisons support a capacity bottleneck. Dense language-model compute scaling results concern a particular loss regime, not a minimum AGI size: see [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

### 3.2 Shared computation with explicit state

The proposed learner has a shared representation backbone and several output roles:

```text
observable input + goal + prior action/outcome
                    |
             shared encoder/core
                    |
       recurrent working state z_t <--- episodic retrieval
          /         |         \
 action policy   world prediction   value / uncertainty
          \         |         /
           bounded planning computation
                    |
               real action
                    |
          observation and external reward
                    |
       experience store -> replay -> consolidation
```

These are roles, not a requirement for six independent large networks. Share the backbone initially. Add small heads for action selection, outcome prediction, value, and uncertainty only when their supervision is defined.

Compare two architectures:

- **R0:** ordinary causal sequence model with the same visible history, goal and action interface.
- **R1:** R0 with persistent working-state slots updated across environment steps. Start with 32 slots at the core's hidden width and a shared update block. Test one, two and four updates; do not start with a learned halting mechanism.

State updates obey `z_next = F_theta(z, encode(observation, previous_action, previous_feedback), goal)`. Reset state at declared world boundaries. Prevent state or retrieval from crossing evaluation identities accidentally. Position handling, stop tokens and attention boundaries are executable contracts.

Match or report parameters, training FLOPs, observed wall time, context exposure, inference computation, and memory for R0/R1. Recurrence reuses parameters but consumes computation. A recurrent model given four times the processing is not a clean architecture-only comparison.

### 3.3 Prediction and planning

Train the world predictor to estimate observable next-state features, reward, termination and relevant outcomes conditional on an action. Hidden simulator rules may generate training labels for designated supervised diagnostics, but the acting learner must not receive privileged state during ordinary evaluation.

Initial planning enumerates a small legal action set, predicts short trajectories and chooses an action using expected task return and uncertainty penalties. Begin with four candidates and horizon two; test horizon four only after prediction reliability supports it. Record imagined and real transitions separately.

Real observations arbitrate disagreement. Do not train an independent “truth” evaluator solely on the model's own predictions. Reduce imagination horizon or return to real exploration when calibrated prediction error rises. Explicitly evaluate cases where a wrong model makes a longer plan worse.

[DreamerV3](https://arxiv.org/abs/2301.04104) supplies evidence that world-model learning and imagined behavior improvement can work across diverse environments. It does not demonstrate that copying that architecture into this repository will produce AGI under this budget.

### 3.4 Memory at three timescales

| Memory | Content and lifetime | Update mechanism | Required test |
|---|---|---|---|
| Working state | Current world, hypotheses and recent consequences | Forward computation within an episode or declared task | Delay, distraction, overwrite and partial observability |
| Episodic store | Observed events, actions, outcomes, source identities and verified failures | Append/retrieve within an explicit scope | Relevant retrieval versus matched irrelevant or disabled retrieval |
| Consolidated skill | Reusable computation in weights | Replay and optimizer updates | Fresh transfer and old-skill retention after updates |

Memory is not automatically understanding. Compare an unassisted core, the core with retrieval, and the entire planning system. Retrieval may improve system competence while leaving core competence unchanged; report both.

Use provenance and scope for every stored record. Evaluation gold answers stay outside the learner's store. Feedback legitimately observed while solving a new world may remain available within that world. At the next independent world, clear task state unless the protocol explicitly measures cross-world learning.

## 4. Learning from scratch: where the learning signal comes from

Random initialization does not mean the system starts without an environment, a learning algorithm or human-designed interfaces. Those are declared design priors. It means no pretrained weights are imported. Track supervised algorithmic examples, human-written material, and self-generated experience separately. A deterministic arithmetic teacher is a source of task labels; a pretrained model supplying solutions would introduce an additional source of learned capability and requires an explicit change to the from-scratch protocol.

For the P35 integration control, retain the tokenizer named by its frozen specification and explicitly declare it an inherited preprocessing prior. Verify its vocabulary/model artifact hash and training provenance; an artifact hash alone does not establish which documents trained it. If that provenance is unavailable, record the limitation and do not claim a completely independently trained tokenizer. A stricter tokenizer-from-scratch comparison must train one on training-only material, charge its construction cost, and use a new compatible model/data identity. Bind corpus snapshots, collection dates, semantic split manifests and tokenizer hashes to every experiment. Exclude evaluation documents and mechanisms before tokenizer fitting and training-data generation whenever those processes are under project control.

### 4.1 Bootstrap in a learnable environment

1. Verify that an oracle can solve generated worlds and that a simple agent reaches some rewards. Otherwise sparse feedback may be uninformative.
2. Collect initial trajectories with random actions plus a declared coverage heuristic. Include failures and action probabilities.
3. Learn outcome prediction and short-horizon action value from those trajectories.
4. Compare the learned policy against the exact exploration policies that generated its experience.
5. Only then let policy-selected experience dominate collection.

This avoids requiring a capable investigator to appear before it has received a usable learning signal. It also makes the contribution of the bootstrap heuristic measurable.

### 4.2 Objective components

An initial training objective is a weighted combination:

`L = lambda_obs * L_next_observation + lambda_act * L_action + lambda_value * L_value + lambda_text * L_text + lambda_replay * L_retention`

Each term needs a declared dataset, eligible targets, normalization and training phase:

- Observation prediction: cross-entropy for discrete observable variables, or a justified continuous likelihood for continuous observations.
- Action learning: supervised bootstrap actions first; later an explicit actor-critic or offline policy objective with logged behavior probabilities where required.
- Value learning: realized or bootstrapped discounted returns, with truncation distinguished from true termination.
- Text learning: causal language modeling on qualified natural/code material, including correct document and terminal boundaries.
- Retention: replay of earlier training experiences; evaluation examples are excluded.

Do not add raw losses with incomparable scales and conclude a module failed. Log eligible counts, component gradient norms, gradient conflicts and effective sampling shares. Start with separate prediction and policy phases; introduce joint updates only when a matched comparison shows a benefit. Reward hacking and prediction error caused by stochastic noise require dedicated controls.

### 4.3 Fast adaptation versus weight learning

Evaluate fast adaptation with frozen weights across several episodes sharing an unknown mechanism: the learner can update working state from observations, actions and rewards. Evaluate consolidation separately by allowing parameter updates between acquisition rounds. A gain from one is not evidence for the other.

[RL²](https://arxiv.org/abs/1611.02779) motivates learning a fast adaptation procedure in recurrent state through slower optimization across tasks. The project must test whether its task distribution actually teaches a procedure that transfers to different mechanisms.

### 4.4 Language and broader competence

Interactive toy worlds alone are inadequate for AGI. Maintain a second stream of qualified natural language, code, mathematics and explanatory material so the learner can acquire useful representations and interfaces. Couple this stream to actions: interpret an instruction, test a program, examine a table, ask a clarifying query, or verify a factual claim against supplied evidence.

Initially test with compact controlled language. Later introduce independently written paraphrases and documents with distractors. Add visual observations only after the state/action learning loop works and a measured task needs them. A text-only program cannot claim grounded visual or physical competence.

## 5. Build environments that teach discovery instead of template completion

Start with three environment families, adding breadth after transfer is demonstrated:

| Family | Unknown mechanism | Useful investigation | Held-out structure |
|---|---|---|---|
| Switch laboratory | Which switches cause which observable effects, including confounding | Intervene on one variable, compare predictions, revise graph | New graph motifs and compositions, not just renamed nodes |
| Inventory world | Hidden transition rules, overwritten facts and action prerequisites | Inspect, perform a reversible trial, track state, recover | New prerequisite chains, delayed effects and combinations |
| Program laboratory | A small unknown function or faulty program | Select inputs, execute tests, infer or repair behavior | Held-out program syntax-tree structures and operators/compositions |

Expose only observations, legal action schemas, public costs and goals. The simulator owns hidden state and answer keys. Publish `reset`, `step`, observation/action schemas, task identity, mechanism family, termination reason and independent evaluator version.

Split by semantic mechanism before producing text renderings. Deduplicate graph/program structures across splits. Freeze a development pool for design and a separately authored confirmation pool for promotion. New operand bands are extrapolation tests; they do not alone establish new reasoning structure.

Add interventions and controls deliberately:

- Irrelevant information: does the learner ignore it?
- Counterfactual query: does the answer change when the requested entity changes?
- Observational equivalence: can an intervention distinguish two plausible mechanisms?
- Stochastic distraction: does the agent waste its budget chasing unpredictable noise?
- Delayed consequence: does the working state preserve information long enough?
- Broken hypothesis: does the agent revise its plan after disconfirmation?
- Transfer bridge: can a skill learned in one interface help in another without a new hand-written solver?

A generator must be correct, sufficiently diverse, resistant to cheap shortcuts, and capable of producing learnable tasks. Qualification is not a one-time badge: version it when rules, rendering, filtering or task selection change.

## 6. A precise self-improvement loop

### 6.1 Separate the levels of autonomy

| Level | Who selects the learning action? | What it establishes |
|---|---|---|
| Human-led research | Researcher or Codex chooses experiments | Better project engineering |
| Fixed automation | Hand-written controller chooses a scheduled arm | Reproducible execution |
| Learned experience selection | Trained policy chooses observations, tasks or replay | A testable component of learning autonomy |
| Learned adaptation | State or weights change successfully from outcomes | Capability acquisition under a specified protocol |
| Learning-method improvement | System proposes and validates changes to its own learning procedure | Stronger self-improvement, if independently replicated |

Current repository automation is not evidence for the last three levels. Implement them in that order. Source-code modification is a later possible action, not a substitute for learning useful experiments.

### 6.2 Decision objective

The investigator should maximize expected **future task competence gained per total cost**, subject to retention constraints. Uncertainty is an input, not the reward itself. A noisy process can be highly unpredictable and completely useless.

For a proposed learning action `a`, estimate:

`utility(a) = expected transferable gain(a) - cost_weight * total_cost(a) - forgetting_weight * expected_retention_loss(a)`

During training, estimate gains using a training-only probe pool that is disjoint from experience used for gradient updates. Promotion uses a different independently controlled pool. The learner cannot repeatedly query confirmation scores while selecting actions.

Reserve a separate strategy-validation pool of unseen mechanisms to test the investigator itself after its training is frozen. Do not expose those scores during investigator optimization. If they guide a redesign, retire that pool into development and use a fresh pool for the next confirmation. This checks overfitting to the training-only gain probes, in addition to protecting the final parent/child evaluation. Count all candidate-selection rounds in the declared testing budget.

Compare learned selection with uniform sampling, fixed curriculum, recent-failure replay, and a simple learning-progress heuristic. Equalize available candidate tasks, feedback, data and compute. Otherwise the comparison may reward privileged access rather than a learned strategy.

### 6.3 Executable conceptual protocol

```python
champion = initialize_random_core_and_bootstrap()
for round_id in bounded_research_rounds:
    parent = freeze_full_training_state(champion)
    experience = collect_with_learned_investigator(
        parent, training_worlds, real_interaction_budget
    )
    candidate = consolidate(
        parent, experience, prior_training_replay, update_budget
    )
    # Examiner owns these fresh worlds; no training access to their answers.
    comparison = examiner.compare(
        parent, candidate,
        fresh_transfer_worlds=True,
        fresh_retention_worlds=True,
        matched_inference_budget=True,
    )
    archive(parent, candidate, experience, costs, comparison)
    champion = candidate if promotion_rule(comparison) else parent
```

The loop above is a specification, not an implemented runner. Choosing `collect_with_learned_investigator` is exactly the research problem; replacing it with a rule and keeping the name would not solve it.

### 6.4 Promotion and retention

For the first self-improvement study, preregister three acquisition rounds across at least two environment families. Require a positive paired lower confidence bound for the primary transfer improvement and a predefined noninferiority margin for each protected retention family. Choose that margin and sample size from independent pilot variance and practical tolerances before confirmation. Account for repeated candidate testing using a fixed candidate budget and a declared sequential-testing or multiplicity procedure.

Compare each child with its actual parent on the same fresh challenges. Also compare with the original baseline and the best historical retained skill level. Track cumulative degradation: small permitted losses in every round must not silently accumulate into large forgetting.

Archive rejected candidates. A failure can improve a training-only strategy model, but do not feed confirmation answer keys back into the same benchmark and continue calling it held out. After promotion feedback has influenced development, retire that confirmation pool from future independent claims.

Three successful rounds would provide protocol-scoped evidence of bounded self-improvement. They would not demonstrate unlimited recursive improvement, broad AGI or a mathematical guarantee of continued progress.

### 6.5 Learning to propose better tasks and methods

After investigation and consolidation work, add task generation. Select tasks on the learnable frontier: too easy yields little new skill; impossible or broken tasks waste resources. Maintain a fixed independent task suite so a generator cannot make its own progress look better by changing the exam.

[POET](https://arxiv.org/abs/1901.01753) motivates coevolving tasks and solutions with transfer between them. Its results support investigating that mechanism, not the assumption that open-ended generation inevitably reaches general intelligence.

Only after the system demonstrates program understanding and experiment selection should it propose learning-code changes. Run proposed changes as bounded candidate experiments with immutable parent checkpoints, resource caps and independent evaluations. The candidate may not edit the examiner, evidence archive, resource limits or promotion rule used to judge itself. This preserves the meaning of improvement.

Execute candidate code in a disposable isolated process/container or VM appropriate to the runtime, with explicit CPU, memory, accelerator-time and output-size limits. Provide only the permitted training data and tools; keep credentials, evaluator storage and unrelated host files outside its accessible environment. Network access is disabled unless a specific experiment requires a declared endpoint. A supervising process owns timeouts, termination, artifact collection and rollback. Record content hashes outside the candidate's writable area so candidate-written reports cannot replace independently observed evidence. These are execution requirements for the later self-modification experiment, not claims that such isolation exists today.

## 7. Experimental sequence and stop decisions

| Stage | Hypothesis | Minimal decisive comparison | Decision |
|---|---|---|---|
| A: valid learning signal | Complete targets and pack boundaries are correct | Tiny overfit plus counterfactual queries and exact stopping | Repair objective/data if this fails |
| B: useful representation | State tracks hidden consequences | R0 versus R1 on delayed/partially observed worlds | Keep recurrence only for measured benefit |
| C: useful investigation | Chosen actions reduce decision-relevant uncertainty | Learned versus fixed/random selection at equal cost | Diagnose exploration if no gain |
| D: useful planning | Predictions improve real actions | No planning versus bounded planning with equal inference accounting | Shorten/remove planning if model errors dominate |
| E: durable learning | New skills accumulate | Sequential acquisition with/without replay and fresh retention | Fix interference before expanding acquisition |
| F: learning strategy transfer | Investigator transfers to unfamiliar mechanisms | Freeze weights, change mechanism family, allow limited feedback | Expand task diversity if strategy is template-bound |
| G: autonomous improvement | Learner-selected experience repeatedly improves a child | Three independent promotion rounds versus fixed-selection control | Claim only the measured scope |
| H: broad competence | Gains survive independent natural tasks | Independently authored language/code/planning/scientific tasks | Broaden or revise architecture based on failures |

Do not require every proposed benchmark to exist before running Stage A. Conversely, do not launch a large foundation run because software tests pass while its intended learning hypothesis remains uninformative. Cheap pilots select designs; preregistered replications support claims.

Failure diagnosis should separate representation, supervision, optimization, exploration, memory, planning and capacity. For example, an oracle planner with the learned world model tests prediction; a learned planner with true transition access is a diagnostic of planning; neither is deployable performance. A larger model comparison is useful only after the same data/objective is shown to be learnable.

## 8. Budget and the meaning of 100×

### First concrete discovery experiment: SI-01

The initial experiment should test **useful investigation**, before autonomous changes to training code. These are proposed starting settings to freeze after a correctness/throughput pilot, not tuned results:

| Setting | Initial decision |
|---|---|
| Environment | Small deterministic acyclic binary switch networks, generated from declared Boolean operations |
| Training mechanisms | Qualified graphs with three to six variables; semantic graph identities split before rendering |
| Observation | Public variable names, permitted action schema and a declared subset of observed values; no adjacency list or hidden operation labels |
| Intervention | Clamp a permitted variable for one simulator evaluation, then reset to the declared baseline state |
| Goal | Predict a requested outcome under an intervention excluded from the permitted inquiry set for that episode |
| Interaction budget | Compare zero, two, four and eight real inquiries; never let an agent directly query its scored target |
| Core comparison | Frozen-weight R0 versus R1 adaptation; identical training experience for the architecture contrast |
| Selection comparison | Random, fixed coverage, a declared uncertainty heuristic and the learned investigator |
| Main outcome | Paired final prediction accuracy at the same inquiry budget, clustered by hidden mechanism |
| Secondary outcomes | Calibration, redundant inquiries, inference cost, transfer to held-out graph motifs |
| Diagnostics | Privileged oracle and identifiability analysis, reported separately from learner performance |

Qualification must establish that the permitted observations can distinguish enough mechanisms to make the task learnable. Ambiguous worlds need an explicit probabilistic scoring rule or exclusion under a frozen rule; do not punish a learner for information it cannot obtain. Prevent target leakage through variable naming, generator ordering and repeated mechanism IDs.

Bootstrap action learning from a training-only coverage policy, then train a discrete investigator with an on-policy actor-critic objective. Start with realized terminal prediction reward minus a fixed inquiry cost. Choose that cost on training-only pilots and freeze it before the selection comparison. Train the value head on realized returns; introduce imagined rollouts only in a later planning contrast. This makes the first experiment about selecting real observations rather than jointly debugging planning, curiosity rewards and code search.

For the subsequent consolidation comparison, initialize both children from the same parent. One receives investigator-selected training experience; the other receives fixed-policy experience at the same interaction and optimizer-update budgets. Begin with replay sampled equally between new experience and retained training experience, stratified by mechanism family; compare with no replay. The 50:50 starting mix is a hypothesis, not a universal retention solution.

Use two independent training seeds as a minimum directional check, then determine confirmation sample size from separate pilot variance and a preregistered practically meaningful effect. Do not treat the many evaluation worlds from one trained seed as many independent training replications. SI-01 succeeds only if investigation improves fresh task outcomes under its declared comparison. A success advances the larger blueprint; it does not establish broad self-improvement on its own.

### Weekly allocation and efficiency accounting

Use the 100 hours as a weekly upper envelope, not a verified provider entitlement or a promise to consume it all:

| Work | Maximum planning allocation |
|---|---:|
| Systems calibration, correctness and recovery | 10 TPU-hours |
| Qualified experience and world-model experiments | 20 |
| Investigation/policy comparisons | 20 |
| Consolidation and retention | 15 |
| Independent-seed replication | 15 |
| Independent evaluation | 10 |
| Reserve for interruption, compilation and failed runs | 10 |
| **Total** | **100** |

Allocate only to stages that are ready. Generate simple environments, qualify data and run contract tests on CPU where practical. Record compilation, evaluation, upload and restoration inside the budget. Set a global session deadline with time reserved for durable checkpoint publication; a per-arm timeout is insufficient.

Before a stage consumes its full allocation, use at most 5% of that allocation for a calibration pilot. Freeze an execution manifest containing the resulting maximum optimizer updates, eligible targets, unique mechanisms, episodes, real interactions, evaluation worlds, seeds and total deadline. Derive quotas from measured throughput rather than inventing a hardware-independent token promise. If the powered confirmation comparison will not fit, reduce the number of hypotheses or defer it; do not silently reduce sample size and retain the original strength of claim. Unused hours remain unspent or move to the next ready stage through a logged allocation revision. Reserve hours fund named interruptions/recovery, not an unreported search over favorable seeds.

Measure three token counts separately: capacity tokens, real nonpadding tokens, and eligible supervised targets. Report episodes, unique mechanisms, optimizer updates and evaluation examples as well. High capacity-token throughput can coexist with little useful learning.

Define improvement ratios with a fixed task and threshold, for example `baseline total accelerator seconds to target / candidate seconds to the same target`, including evaluation and restoration overhead. If a baseline never reaches the target, report a censored comparison; do not manufacture an infinite speedup.

For the first efficiency baseline, freeze the actual integrated P35 checkpoint/configuration and SI-01 environment manifest, using fixed coverage collection and the same consolidation schedule as the candidate. Publish its measured learning curve, runtime breakdown and fresh-world outcomes before optimizing it. The initial target is a preregistered improvement over random inquiry on the same confirmation distribution, selected using independent pilots and then fixed for both methods. The baseline has not yet been measured: no 100× result can be claimed until both baseline and candidate costs to the identical target exist. Historical calculator and local binding results remain diagnostics, not interchangeable denominators.

Investigate these independent levers:

1. Reduce compilation and host synchronization after profiling.
2. Pack useful content while preserving segment boundaries and target semantics.
3. Avoid repeating nearly identical experiences; select informative examples.
4. Reuse useful representations and retained skills across tasks.
5. Spend planning computation only where it improves outcomes.
6. Stop uninformative experiments early under predefined rules.

Do not multiply optimistic gains measured on different workloads. Amdahl's law gives `S = 1 / ((1-p) + p/s)`: accelerating 90% of runtime by 100× yields only about 9.17× overall. A 100× aspiration is most plausible as a dramatic reduction in wasted research or data, but even that needs an observed baseline and independent capability measurements.

## 9. Implementation boundaries and shared contracts

Use **Cymek for canonical data/model/training state**, **Citadel for runtime experiments and independent evaluation**, and **BRAMASTRA for the integrated learning hypothesis and evidence synthesis**. These are integration responsibilities, not a requirement to keep three diverging implementations forever.

Required interfaces:

| Interface | Required fields / behavior |
|---|---|
| `TaskSpec` | Version, semantic identity, mechanism family, split, generator hash, public schema, hidden-state boundary |
| `Transition` | Task/episode/step IDs, observable input, action, feedback, next observation, termination/truncation, action cost, behavior-policy identity |
| `PackedBatch` | Tokens, segment/position IDs, eligible targets, explicit terminal targets, source counts, sampler cursor |
| `LearnerState` | Parameters, optimizer moments, schedule counter, RNGs, sampler state, working-state reset policy, replay index, parent identity |
| `ExperimentSpec` | Hypothesis, primary contrast, budgets, seeds, code/data/runtime hashes, stop rules and evaluator identity |
| `EvaluationReport` | Per-world outcomes, cluster identities, component and full-system scores, uncertainty, costs, failure reasons |
| `PromotionRecord` | Parent/child hashes, comparison identities, retention constraints, accepted/rejected reason and cumulative history |

Checkpoint restoration must recover the learning process, including replay/sampler continuity, not merely reload weights. A CPU continuation check does not establish TPU or multi-device restoration. Verify each supported backend with actual tensors and a fresh process before declaring that path ready.

## 10. Deliverables in order

**First deliverable:** repair and validate the terminal/packing contract, materialize one qualified data slice, and execute one connected from-scratch update/checkpoint/restore path. This is small enough to verify and important enough to prevent wasted runs.

**Second deliverable:** one interactive world with independent mechanism splits, cheap baselines, a predictive learner, and a frozen-weight adaptation evaluation. Produce results even if the learned agent loses.

**Third deliverable:** a learned investigator that beats simple alternatives on real task success per interaction and transfers beyond its training mechanisms. Keep ordinary language training as a measured supporting stream.

**Fourth deliverable:** sequential acquisition and retention with independently confirmed parent/child gains. Add task generation only after this loop exists.

**Fifth deliverable:** broader task families and natural interfaces, stronger long-horizon planning, independently authored assessments, and capacity scaling justified by learning curves.

The first month should prioritize the first two deliverables, conditional on results. Subsequent months pursue the next gates rather than a calendar promise of AGI. If progress stalls, publish the failed hypothesis and revise the mechanism, not the definition of success.

## 11. Open scientific risks that engineering alone cannot settle

- A small core may not represent the abstractions required for broad transfer.
- Learned exploration can overfit the training family's notion of useful information.
- World-model errors can compound and make planning harmful.
- Continual learning may preserve old scores while losing unmeasured competence.
- Self-generated curricula can collapse toward easy or self-confirming tasks.
- Natural language prediction may not produce grounded causal understanding without appropriate interaction.
- The available compute may suffice for decisive experiments but not broad competence.
- Better local learning strategies do not imply indefinitely accelerating self-improvement.

These are reasons to design discriminating experiments. The most valuable next result is a reliable improvement in one missing part of the chain, followed by evidence that it transfers and survives further learning. That is how this blueprint is to be challenged and improved.
