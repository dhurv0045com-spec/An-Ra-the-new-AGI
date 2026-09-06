# BRAMASTRA experiment plan

Status: proposed experiments, not executed. This file makes [the blueprint](../../BRAMASTRA.md) actionable without pretending unmeasured hardware or unknown data sources are already resolved.

Implementation update, 2026-09-06: a smaller terminal-supervision derivative has been built and run; see [RESULTS.md](RESULTS.md). It uses a 117,312-parameter smoke model and a short constant-LR protocol. It does not close the full B0/B1 experiments specified below. The owner has authorized implementation, experiments, and pushing this work.

## B0: prove that the learning instrument works

**Question:** can a randomly initialized byte-level core fit contextual-selection examples and emit complete answers through the exact training/inference path? This is a learnability/serialization instrument check; memorizing its tiny fixed set can satisfy it. Transfer remains a separate B1 question.

Use the B0 architecture and optimization defaults in the blueprint. Byte IDs are offset by four; PAD/BOS/EOS/UNK are 0/1/2/3. Reserve those IDs exclusively. No modulo character encoding. Decode UTF-8 strictly, and report invalid byte sequences as invalid outputs.

Serialize `BOS + prompt + answer + EOS`. Supervise every answer byte and EOS, excluding prompt and padding for this experiment. Shift labels exactly once. Define the prompt/answer whitespace boundary identically during training and generation. Inference sees no gold answer, target length, relevant-fact pointer, candidate list, or hidden world metadata.

The task supplies several entity-value assignments and asks for one value. Construct complete query-swap groups over the same facts. Use distinct values, balanced query/position assignments, relevant and irrelevant changes, and unambiguous semantics. Test byte round trips and exact parsing. Reject truncation and collisions. Pair members never cross splits.

1. Validate the evaluator on a reference solver and deliberately wrong/truncated/nonterminated outputs. A parser must reject a correct prefix followed by extra non-whitespace content. Do not infer termination from gold length.
2. Check causal masking, target eligibility including EOS, finite gradients, parameter and moment changes, and complete update accounting.
3. Overfit 32 fixed development worlds with all their query variants. Proposed mechanical exit: at least 99% full-answer accuracy and 99% valid termination on this fixed set, reproduced after a clean checkpoint restore. This establishes instrument learnability only. Stop at 5,000 updates or the B0 five-hour bucket, whichever comes first; a miss means diagnose before B1.
4. Train on fresh development worlds and evaluate unseen worlds. Record answer accuracy, both-correct pair accuracy, valid-stop rate, invalid-output rate, max-token rate, and per-position/per-length results. This exploratory step establishes a usable learning regime, not structural transfer.

Do not weaken whole-answer scoring to make the positive control pass. Prefix correctness and teacher-forced token accuracy are diagnostics. If it cannot overfit, inspect label shift, mask, optimization, capacity, and serialization in that order, with evidence. Do not immediately conclude CE or the model family cannot learn.

## B1: the first discriminating data experiment

**Primary question:** does verified contextual-binding training improve complete-answer structural transfer beyond general-data training and format/copy exposure, at a fixed execution budget?

Use one B1 tokenizer trained only on the declared general training pool and generated task records assigned to the training partition. Exclude development, confirmation, and independent-transfer records. Freeze it for all arms. Use the B1 architecture, CE objective, optimizer, context, evaluation path, and paired random initialization across arms. Hold the general-source distribution constant where it appears.

| Arm | Data allocation by registered non-padding token slots | Purpose |
|---|---|---|
| A | 100% general pool | No-task treatment control |
| B | 85% general pool, 15% simple retrieval/copy task records | Controls some task format and output exposure |
| C | 85% general pool, 15% verified multi-entity query-sensitive records | Proposed treatment |

Construct B from the same vocabulary, answer format, approximate record lengths, entity counts, and distractor counts as C. B may explicitly mark the relevant record, with sham markers in C balanced independently of the target. Never train B on randomly wrong answers as the main control: that would compare coherent training against label noise. The simpler control necessarily differs in its information structure; document residual differences and limit the causal claim to this recipe comparison, not a uniquely identified neural mechanism.

Primary training budget is equal executed token positions at a fixed shape and update count. Record non-padding tokens, prompt tokens, supervised targets, answer-plus-EOS targets, raw bytes, unique worlds, repetitions, and measured wall time as secondary quantities. Packing must isolate records, include terminals, and prevent cross-record attention. Actual 85/15 allocations refer to valid data tokens, not padded capacity.

Determine the token/update cap from the B1 calibration, so all three arms fit the 30-hour bucket with margin, at most ten hours per arm. Before launching A/B/C, also budget the decisive pair at the **same training length** in the 20-hour replication bucket. Set the common cap using the slowest measured arm, with interruption margin. If that does not fit, shorten the common registered length before any arm, or schedule replication in another week. The unused ceiling is not a reason to give development arms extra tokens.

Use common evaluation tasks within a seed and common general-data streams where possible. Training sequence seeds and task-world seeds are independent. Inspect development curves at fixed 10%, 25%, 50%, and 100% points. The registered final checkpoint is the primary comparison; do not pick each arm's best-looking checkpoint after examining holdouts.

## Splits and transfer

| Partition | Use | Protection |
|---|---|---|
| Training | Learn parameters/tokenizer | Declared source pool and generator versions |
| Development | Debug, choose a nominated comparison, calibrate power | Results may guide design; never called sealed |
| Confirmation | Test the frozen nominated contrast | Committed worlds/sources, no tuning after results |
| Independent transfer | Test new author/rendering or natural source analogue | Separate construction, task semantics independently checked |

Use separate held-out axes before a compound shift: entities/values, renderings, target positions, number of distractors, and genuinely different relational structures. Replacing names or seeds alone is not structural OOD. The initial binding claim concerns unseen selection settings; it is not a multi-hop reasoning claim.

Include independently authored document/table tasks with exact answers recoverable from their evidence. Audit answer uniqueness and support. Their records and source documents cannot appear in training, tokenizer training, or generator design. Until suitable data exist, report synthetic results as synthetic; do not replace natural transfer with synthetic prose and rename it.

The historical public "sealed" profiles are inspectable in repository code. A prefix such as `SL` and a new seed do not provide independent custody. For confirmation, freeze new fixture hashes and scoring before results, restrict access, and record who constructed and accessed them. Strong independent-transfer claims require independent construction; limited custody supports only a correspondingly limited claim.

## Metrics and decision rules

These are proposed research thresholds. Freeze them before confirmation; they are not universal intelligence criteria.

- Primary: exact complete-answer accuracy and the rate at which **both** members of a query-swap pair are correct on structurally held-out worlds. Require C to improve over both A and B in development before nominating the confirmatory C-versus-B contrast.
- Confirmation: positive lower confidence bound for C-minus-B and at least five percentage points estimated improvement in both primary metrics. Requiring both endpoints is conjunctive. Predefine the two endpoint tests; do not select whichever succeeds.
- Independent natural analogue: positive lower confidence bound for C-minus-B complete-answer accuracy. If absent, keep the result as synthetic transfer only.
- Substrate preservation: upper confidence bound on C's relative held-out byte-normalized NLL increase versus A no greater than 3%. Also report natural-task behavior; a loss bound does not guarantee retained capability.
- Replication: repeat the nominated contrast with a fresh paired initialization/data seed and fresh evaluation worlds at the same training length. Require the same direction and pass the registered confirmation criteria. Report each seed; two seeds provide limited evidence about seed variability.
- Invalid protocol, nonfinite updates, missing continuation state, or data leakage invalidates the affected experiment. A wide interval is inconclusive; a narrowly estimated negligible effect rejects this dose/recipe within the measured regime.

Cluster uncertainty by independent latent world or source document. Queries, counterfactual variants, and paraphrases of one world are not independent observations. Use paired cluster bootstrap intervals for differences, and report discordant world-level outcomes. Set sample sizes using development estimates/simulations for at least 80% power at a nominated ten-point true effect, while retaining the five-point minimum estimated gain. Budget independent worlds, not an inflated count of derived queries. If achievable sample size cannot support the claim, reduce scope or extend evaluation; do not invent a green threshold.

Use two-sided 95% intervals with 10,000 paired cluster-bootstrap resamples and recorded bootstrap seeds. The confirmatory contrast is C-versus-B only; A/B/C comparisons during selection are development results. The complete promotion claim requires both primary endpoints, natural transfer, and substrate preservation to pass together, an intersection-union decision rather than selection among significant endpoints. Separate endpoint claims or additional confirmatory contrasts require Holm adjustment at familywise alpha 0.05. Retention comparisons use simultaneous bounds across their declared families and references.

Before confirmation, use a 128-world development pilot to estimate cluster-level joint outcomes and evaluation throughput. Simulate candidate sizes of 256, 512, 1,024, 2,048, 4,096, and 8,192 independent worlds under nominated ten-point true effects, applying both the positive-bound and five-point estimated-gain rules; choose the smallest achieving at least 80% joint power for the two primary endpoints. Size the natural-source comparison separately at a five-point target effect using document clusters. Record assumptions and sensitivity to pilot uncertainty. Freeze the selected counts before generating or inspecting confirmation outcomes. If no candidate is adequately powered, or inference/I/O cannot fit the evaluation bucket, the confirmatory campaign remains unbudgeted and must be rescheduled or redesigned before launch. The pilot does not certify power or natural data availability by itself.

Do not assign `1/k` chance to unconstrained free text. A `k`-candidate ranking diagnostic may carry an explicit balanced selection null, but its score is secondary. Report positional and token-length bias; do not retrospectively select a scoring normalization using confirmation outcomes.

For later retention experiments, require every previously acquired family's child-minus-reference lower confidence bound to be at least -3 percentage points, using simultaneous/multiplicity-adjusted bounds across the declared retention families. Compare against both the immediate parent and the relevant earlier qualifying checkpoint. This is a provisional practical tolerance, not a promise of zero forgetting.

## Failure branches

| Observation | Next bounded decision |
|---|---|
| Tiny-set overfit fails | Repair instrument or run one optimization/capacity diagnostic; no B1 |
| B0 works; all B1 arms fail even familiar development tasks | Inspect effective exposure, update count, and language substrate; run one longer-training or supervision comparison |
| C beats A but not B | Task/output familiarity remains sufficient explanation; no structural-learning claim |
| C gains only on familiar templates | Redesign train distribution and independent transfer probe; no scale promotion |
| C transfers but harms A's substrate measures | Test one lower dose or replay allocation |
| C transfers and retains but fails replication | Preserve the negative; estimate variance before another experiment |
| C passes confirmation and independent transfer | Proceed to acquisition of state-update skill, retaining binding |
| Later accumulation succeeds, diagnosis policy adds no value | Keep a simpler fixed curriculum; self-diagnosis remains unproved |

One failed synthetic experiment does not refute AGI, Transformers, or CE. Equally, one successful synthetic experiment does not establish any of them as solved.

## Minimum implementation handoff

The owner has accepted continuation into implementation. `bramastra_lab/` now contains a model, data/serialization module, bounded terminal-comparison runner, evaluator, continuation check, and receipts. See its [execution guide](../../bramastra_lab/README.md). The full B0/B1 campaign, target-TPU path, and learned research loop remain subsequent work.

Audit and reuse narrow Cymek model/optimizer/checkpoint utilities where useful; avoid wholesale branch integration. Keep the old namespaces and experiment identities intact. CPU checks cover serialization, masks, independent solver agreement, and receipts; short target-TPU runs establish actual training and restore behavior. Do not substitute CPU test counts for learning evidence.

Each experiment manifest must bind: code commit and dirty-patch hash if any; model/tokenizer hashes; initialization and random seeds; source and generator hashes; splits and pair grouping; serialization/terminal policy; loss masks; optimizer/schedule/batch; exact token/update cap; device topology/dtypes; evaluation/scoring versions; planned contrasts, uncertainty method and sample size; output location; full-resume destination; and stop conditions.

Each result must add: actual consumed counts, raw predictions, stop reasons, checkpoint identity and parent, update/moment verification, train/eval/compile/I/O time, restore receipt, deviations, and scoped verdict. Unfilled fields mean the run manifest is not ready; they do not block writing and checking the B0 implementation.

The immediate next deliverable is a working B0 runner plus its meaningful positive control. No tokenizer tournament, 250M construction, broad benchmark expansion, or self-improvement framework is needed to answer B0.
