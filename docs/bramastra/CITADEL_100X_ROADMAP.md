# Citadel: from calculator experiments to an independent discovery laboratory

**Audit date: 2026-09-06. Audited tip: `28ff690c04e655c88e4a6b394585e9b3428181ad`. Status: source audit and proposed work; no new TPU run was performed for this document.**

This roadmap accompanies the [full AGI blueprint](../../AGI_BLUEPRINT.md). Citadel's greatest potential contribution is a trustworthy, efficient experimental environment in which learned investigation and improvement can be distinguished from memorization, automation and measurement errors. “100×” is an efficiency ambition requiring a baseline, not an achieved speedup or intelligence multiplier.

## 1. Credit the current implementation accurately

Citadel has moved substantially beyond its earlier calculator experiment. The inspected tip includes a T1D runner, tiered data, PRE50M readiness checks, pinned runtime bootstrap, a notebook handover, and an adapter for production checkpoint transactions. Describing all of this as missing would be stale.

The inspected T1D plan is still **preregistered with no results**. Local tests and notebook readiness are not evidence of real TPU execution, successful remote restore, or learned generalization. PRE50M denotes readiness for a 50M-token stage, not a 50M-parameter model.

Immutable source anchors:

- [T1D plan](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/28ff690c04e655c88e4a6b394585e9b3428181ad/docs/citadel/experiments/T1D/PLAN.md)
- [Runner and batch assembly](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/28ff690c04e655c88e4a6b394585e9b3428181ad/citadel_tpu/t1d_run.py)
- [Tiered data](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/28ff690c04e655c88e4a6b394585e9b3428181ad/citadel_tpu/tiered_data.py)
- [Calculator encoding/evaluation](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/28ff690c04e655c88e4a6b394585e9b3428181ad/citadel_tpu/calculator_eval.py)
- [PRE50M runner](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/28ff690c04e655c88e4a6b394585e9b3428181ad/citadel_tpu/pre50m.py)

## 2. Fix the answer termination contract before interpreting T1D

The inspected `assemble_batch` calls `cev.encode(t)`, packs those IDs, and marks `answer_spans` as eligible. Ordinary rendered rows end with the answer text. The encoder maps characters to IDs without appending a terminal token. The batch assembly appends no EOS or newline target.

Generation, however, needs an explicit termination event or reaches its token cap. Correct answer characters alone therefore do not guarantee a correct complete response. The [BRAMASTRA experiments](RESULTS.md) independently demonstrate that this distinction can dominate measured success. That is supporting evidence for the contract correction, not a proof that Citadel will reproduce the same numerical result.

Required versioned amendment:

1. Define a single record terminator, preferably the declared EOS ID for this experiment.
2. Append it before packing and include it in the same segment as the answer.
3. Mark the EOS target eligible under the shifted causal loss.
4. Account for the extra token in pack lengths, calibration, budgets and target counts.
5. Declare whether answer/EOS loss is target-averaged or separately weighted. Keep the choice identical across scientific arms.
6. Verify single-character and multi-character answers, adjacent packed records, exactly-full sequences and padding tails.
7. Confirm that a tiny learnable set produces both correct content and correct stopping.

Do not silently change a frozen experiment. Add a new revision and invalidate incompatible prepacked data or calibration receipts. Preserve old results under their old objective.

### Report failure reasons without hiding the stopping problem

Record separate counters for correct answer prefix, complete exact match, EOS, premature EOS, MAX_TOKENS, invalid/nonalphabet output and padding termination. The current `FORMAT_FAILURE` classification uses nonalphabet and PAD counts; that label alone does not include all stopping failures. Report length-cap exhaustion explicitly.

Greedy decoding should be a fixed initial setting. A masked alphabet diagnostic may isolate output-space behavior, but cannot substitute for unrestricted generation when claiming the complete model learned the task.

## 3. Make the five-arm comparison answer the questions it names

The inspected plan uses:

| Arm | Model / treatment | Planned capacity-token budget |
|---|---|---:|
| A | MID about 3.7M, flat answer-CE tiered pool | 8M |
| B | MID, fixed curriculum | 8M |
| C | MID, curriculum plus 40% teacher rows | 8M |
| D | SCALE2 7,378,368 parameters, curriculum | 4M |
| E | MID, masked alphabet diagnostic | 4M |

These arms can be useful, but their interpretations need precision:

- **B versus A:** effect of the complete fixed curriculum schedule under matched declared budget. The schedule is manually specified, not a learned task-selection policy.
- **C versus B:** effect of the teacher-row mixture. Teacher explanations also change sequence lengths, ordinary-example exposure, eligible targets and potentially update counts. The contrast does not isolate an abstract “reasoning” variable.
- **D versus B:** size and training-budget changes together. Record measured compute; add a matched-budget checkpoint comparison or a small learning curve before attributing the difference to scale alone.
- **E versus B:** output restriction and budget differ. Compare E with B at a matched 4M milestone for the restriction contrast; retain 8M B as a separate learning-curve point.

T0/T1 have limited combinatorial spaces and are useful learnability diagnostics. Holding out operand bands in T2+ tests range extrapolation. It is insufficient by itself to claim structural generalization. Add held-out expression structures, operator compositions and independently authored interfaces when making broader claims.

For teacher rows, separately evaluate final answers and intermediate steps. A convincing explanation may accompany a wrong final answer; a correct final answer need not have a faithful generated explanation. Algorithmic teachers are permitted from-scratch supervision, but label their contribution.

The plan's Holm correction needs a defined family of hypotheses, test statistics and units. Threshold differences and “ahead on three tiers” are descriptive unless supported by the declared inference procedure. Use semantic problems, not repeated renderings, as independent units. Repeated DEV checkpoints guide development; they are not multiple independent confirmations.

## 4. Correct the runtime budget and measure the expensive path

Five arms each permitted up to 45 minutes can consume 225 minutes before calibration, PRE50M, evaluation or checkpoint publication. Therefore per-arm timeboxes do not guarantee a session shorter than two hours. Enforce a **global deadline** that reserves time for evaluation and durable checkpoint completion, and mark unfinished arms as incomplete rather than scientific failures.

The inspected code constructs batches and packed layouts on the host, reads scalar values, and copies a gradient-norm contribution to CPU for each parameter. On XLA, these operations can cause costly synchronization. This is a profiling hypothesis, not a measured bottleneck in the present audit.

Measure warm-up and steady-state separately:

- Compilation time and actual compile/recompile counters where available.
- Host data preparation, device transfer and accelerator execution.
- Gradient diagnostics and synchronization overhead.
- Evaluation decoding time, including repeated shape/length effects.
- Checkpoint serialization, upload, verification and restore.
- Capacity/real/eligible tokens per second and useful committed updates per session.

The PRE50M timing heuristic for unexpected recompilation is a useful warning; a slow step is not direct proof that a compiler recompiled a graph. Use compiler/runtime evidence when making that claim.

Then test specific optimizations independently: prepared host batches, stable shapes, consolidated device-side gradient norm reduction, fewer diagnostic synchronizations, and appropriate evaluation batching. Preserve nonfinite detection and update semantics. Compare identical batches and resulting updates before reporting a speed gain.

### The vocabulary opportunity is narrower than 100× end-to-end

T1D's masked arm applies `torch.where` after obtaining full logits. It therefore still pays for the full output projection. A 24,576-to-260 vocabulary reduction would reduce the projection's output dimension by about 94.5×, but does not imply the whole model or training run becomes 94.5× faster.

A genuinely compact calculator head could be a useful diagnostic. It changes the model/tokenizer/interface and must have its own matched comparison. Do not use that specialization to claim broad language capability. For the eventual shared learner, measure whether a task-specific action head provides efficiency without replacing the task with an external solver.

## 5. Evolve Citadel into a discovery laboratory

Calculator training is too narrow to test AGI. Its instrumentation can support interactive experiments if the laboratory exposes three separated surfaces:

1. **Learner surface:** observations, goals, legal actions, action costs and permitted feedback.
2. **Training surface:** training-only generators, replay and optimizer access under declared budgets.
3. **Examiner surface:** hidden mechanisms, answer keys, independent task selection and promotion results.

The learner may investigate a world through actions. It may not read hidden simulator state or modify the examiner. This separation makes gains interpretable, especially when the learner later proposes experiments or code changes.

Implement the master blueprint's environment contract and start with a switch laboratory. At reset, sample an unknown causal mechanism from the correct split. At each step, accept an intervention, charge its cost and return only observable consequences. At termination, assess a held-out prediction or goal achievement.

Run a hierarchy of baselines:

- Random legal actions: establishes an interaction floor.
- Fixed coverage strategy: establishes what simple experiment design achieves.
- Recent-error or information heuristic: establishes a stronger nonlearned alternative.
- Learned investigator: the actual hypothesis under test.
- Privileged oracle: a diagnostic upper bound, excluded from deployable performance.

Keep action and feedback budgets equal. Test a distracting noisy process so curiosity is not rewarded merely for encountering unpredictable events. Test pairs of mechanisms that can only be distinguished by intervention. These experiments address investigation more directly than increasingly long calculator strings.

## 6. Evaluate the whole learning process

Citadel should produce three kinds of report:

**Within-task adaptation:** freeze model weights; measure success after zero, one, two and more permitted observations/interventions. Reset working state according to the task boundary. This tests learning in state.

**Across-task consolidation:** allow training between acquisitions; evaluate fresh transfer and old-skill retention after every child checkpoint. Include a fixed-experience-selection control so extra training is not mistaken for a better learning strategy.

**Method improvement:** later, compare learner-proposed changes to the learning procedure with the parent procedure on fresh tasks under the same total resource allowance. Include the cost of searching for the change, not only the successful final run.

An independent parent/child comparison needs paired worlds, fixed inference budgets, semantic clustering, preregistered primary metrics and retention margins. Keep a cumulative ledger of candidate selection and rejected runs. A chosen best seed is a selection result, not an independent replication.

Maintain both fixed longitudinal benchmarks and rotating fresh confirmation pools. Fixed benchmarks detect regressions; rotating pools limit repeated-test adaptation. Once feedback has informed development, it is development evidence for future rounds.

## 7. What 100× would mean here

The most relevant target is **validated useful experiments per accelerator hour**, together with the cost of reaching a fixed competence level. Count all consumed resources, failed runs, compilation, evaluation, recovery and human intervention. A fast experiment with invalid targets is not useful throughput.

Potential sources of large improvement are avoiding invalid studies, cutting synchronized host overhead, batching evaluation, reusing correct infrastructure, selecting informative experiments, and restoring instead of restarting. Measure each on the same workload. Do not multiply a toy-head projection ratio by a packing ratio and advertise their product as an AGI speedup.

An honest efficiency report contains the baseline commit, candidate commit, hardware/runtime, workload identities, achieved scores and uncertainty, total cost, and failure accounting. If a method improves speed while reducing transfer or retention, report that tradeoff rather than a single winning number.

## 8. Ordered acceptance checklist

**T1 — Correct targets:** explicit terminal supervision and segment-safe packing pass focused tests; a tiny learnable case generates complete answers.

**T2 — Real execution:** pinned runtime executes on the actual accelerator; full state restores in a fresh process; global deadlines protect checkpoint completion.

**T3 — Interpretable calculator study:** arm contrasts and budget differences are accurately reported; results distinguish memorization, extrapolation, formatting and stopping.

**T4 — Interactive laboratory:** hidden mechanism splits, observation/action contracts and independent baselines support a real investigation task.

**T5 — Learned discovery:** a learned investigator produces better real-world task outcomes per interaction than fixed alternatives and transfers to new mechanisms.

**T6 — Independent improvement:** repeated parent/child comparisons show transferable gains with retention under the full cost budget.

T1–T3 make the existing work scientifically usable. T4–T6 connect Citadel to the AGI objective. Broad generality remains a further empirical challenge, not a label awarded for completing the infrastructure.
