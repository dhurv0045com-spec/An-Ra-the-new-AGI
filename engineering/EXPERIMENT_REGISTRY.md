# Experiment registry

Registry revision: 2026-09-07. Entries are proposed unless linked to a completed run. No unrun entry is a result. Packet implementation effort and accelerator consumption are separate budgets.

## Shared protocol

- All core weights start randomly initialized, unless a child explicitly inherits a named from-scratch parent.
- Record all seeds; use at least two training seeds for a directional development check. More evaluation cases from one seed are not more training replications.
- Split semantic mechanisms before rendering. Confirmation sample sizes follow independent pilot variance and a declared minimum useful effect.
- A pilot may consume at most 5% of the proposed stage allocation before the full execution manifest is frozen. If no informative comparison fits, reduce hypotheses or defer execution.
- The weekly 100-hour envelope includes compilation, failed runs, evaluation and recovery. Do not interpret it as authorization for each packet to use 100 hours.
- Current local campaign allowance is bounded CPU experimentation; actual TPU campaigns require a live runtime/quota record and assigned allocation.

## Experiments

| ID | Question and comparison | Primary readout | Scope / gate |
|---|---|---|---|
| D00 | Does the prototype perform the intended updates and restore them? | Parameter/optimizer/sampler agreement and leakage tests | Correctness only; does not establish learning |
| D01 | Does learned query imitation improve prediction over random/coverage? | Paired held-out mechanism success at equal queries | Current development prototype; family/Brier breakdown mandatory |
| D02 | Does depth-two investigation overcome delayed-information failure? | Exact parity diagnostic, then trained transfer versus one-step teacher | Prerequisite for claiming multi-step inquiry benefit |
| D03 | Does recurrent state help under partial observability? | R0/R1 paired success across delay and distraction | Match visible information and report inference compute |
| D04 | Does outcome-trained inquiry beat teacher imitation? | Frozen predictor with PPO policy versus fixed/imitated inquiry | Requires D00 plus usable environments and predictor |
| D05 | Does a learned world model improve real planning? | No planning versus learned planning; oracle decomposition | Requires measured dynamics accuracy and cost accounting |
| D06 | Does new learning survive consolidation? | New-only versus replay at matched updates; transfer and worst-family retention | One round first, then three-round extension |
| D07 | Does experience selected by the learner improve a child? | Learned versus fixed collection, same updates/interactions | Parent/child comparison on fresh independent worlds |
| D08 | Can the intended runtime sustain and restore the actual learner? | Correct updates, useful throughput, fresh-process/remote restore | Operational gate; no intelligence conclusion |
| D09 | Do acquired representations transfer to natural interfaces? | Structured versus independently written instructions/code tasks | Language/data provenance and contamination controls |
| D10 | Does the integrated loop improve repeatedly? | Three bounded acquisitions with fresh transfer/retention and fixed-selection control | Protocol-scoped self-improvement evidence, not AGI certification |

## D01 prototype interpretation

Families: parity, signed conjunction and threshold functions, semantically deduplicated. Initial training excludes threshold; acquisition may add training-threshold worlds. Development scores are reported separately before and after that change. The scored target cannot be queried, and repeat queries are forbidden.

Policies: random legal inquiry, fixed coverage, learned inquiry, predictor-uncertainty inquiry, and memory-disabled control. All initially share the same predictor, so policy differences isolate selection more closely than comparing unrelated models. Shared training of predictor/policy can still affect representation; add a predictor-only training ablation before assigning architectural causes.

The current filtered worlds have positive-label fractions between 1/4 and 3/4. Report this restriction and actual family counts. Do not generalize to arbitrary Boolean functions, all causal worlds or natural reasoning.

## Minimum experiment specification

Before an expensive run, fill [the experiment template](templates/EXPERIMENT.md) with:

1. Hypothesis and a result that would falsify it.
2. Primary contrast and metric, controls and diagnostic-only privileged conditions.
3. Semantic splits, training seed count, cluster unit and planned sample sizes.
4. Parameters, batch sizes, effective targets, interaction allowance, compute limits and stopping rules.
5. Optimizer, schedule, objective weights, replay distribution and inference settings.
6. Data/code/runtime identities and checkpoint requirements.
7. Primary inference method, multiplicity/sequential handling and practical effect/retention margins.
8. Failure taxonomy and policy for incomplete runs.

Never compare arms at different budgets and call the difference a clean effect of one mechanism. Use matched milestones or state the bundled intervention accurately.

## Acceptance is evidence-based

Use paired outcomes and bootstrap by semantic world, preserving all queries from a world together. Report per-family accuracy, Brier score, cost and confidence intervals. Intervals over worlds describe uncertainty conditional on the trained seed; seed variability must be reported separately.

Exploratory intervals are descriptive. For confirmation, freeze candidate count and statistical procedure before observing the confirmation pool. The examiner may return an accept/reject result, but repeated feedback consumes independence; retire pools into development as appropriate.

A task is not “solved” because an agent reports success. Recompute scores from raw predictions and verify task identity, label consistency and matched keys. A correct prefix without required termination is not a correct complete response.
