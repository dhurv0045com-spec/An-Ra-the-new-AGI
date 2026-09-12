# BRAMASTRA B2: one learner with controlled plasticity

This is the chief's implementation design. The central addition is an **evidence-driven plasticity controller** connecting capability formation, task learning and preservation. It is an unvalidated mechanism, not a promised 10x improvement. The build must work with it disabled.

## 1. One model, one runtime, several learning signals

Extend the existing `TransformerDecoder` through a canonical wrapper in `bramastra_lab/research/models/`. Do not create an unrelated second language model. The wrapper returns token logits, optional hidden states, an action score and a value estimate through typed results. A task-conditioned finite-action head and bounded working memory may share the decoder's representations. New head parameters are counted explicitly; the base decoder's count is never advertised as the entire integrated model's count.

Use random initialization. Core configuration identity includes vocabulary, layers, width, heads, FFN, context limit, normalization, positional encoding and all heads. Tokenizer identity is separate. A 260-symbol byte vocabulary is the compatibility starting point; optional larger physical class spaces are controlled configurations. R1B does not justify changing the baseline to 4096 or treating that number as an architectural law.

This B2 decision supersedes W03's proposed standalone width-128 event encoder as the first integrated implementation. Use the existing causal decoder for the initial shared representation; retain W03's public-information/reset invariants. A second event encoder or GRU is a later optional ablation, not a prerequisite that splits the build into two unrelated learners.

Profiles use the SAME classes and optimizer path: tiny (2 layers, width64, 4 heads, FFN176, context128), development (8 layers, width256, 4 heads, FFN704, context256), and future capacity profile (12 layers, width512, 8 heads, FFN1408, context512). Validate counts analytically and against instantiated tiny/development models when memory permits. Large-profile configuration validation does not require loading it on a 6GB device. Analytical base-decoder counts are 117,312 / 6,493,440 / 38,681,088 respectively; optional heads add parameters. These are derived counts, not live large-model validation. No 500M subject is implied.

## 2. Public event and data path

Canonical input is an episode with explicit goal, public observations, actions, feedback and budget. Serialize through one byte codec. Exclude task IDs, seed, split, generator hash and collection-policy identity from learned tokens; preserve them in the evidence sidecar. Encoder key-order invariance, event-order sensitivity and reset semantics follow `OBSERVATION_ENCODING.md`.

Support two initial batch forms without separate models: ordinary causal token sequences and goal-conditioned trajectories. Each contains `input_ids`, `attention_mask`, `labels`, `loss_mask`, segment/reset boundaries and provenance. Optional `pair_group_id` groups counterfactual examples; it is not a token feature.

Loss masks define the actual supervised denominator, including required EOS. Never equate nonpadding tokens with supervised targets. Keep independent counters for bytes, encoded tokens, semantic examples, presentations, supervised targets, optimizer updates and real interactions. Replays increase presentations, not unique example counts.

Build semantic train/controller/dev/sealed split manifests before rendering. For paired counterfactual goals, both examples stay in one split, and paired minibatching keeps their relation available to the objective. Ordinary document datasets are accepted only through a local manifest with declared license/provenance and trainability status. Missing corpus supply must return `DATA_NOT_READY`; never silently replace it with synthetic fixtures.

## 3. Complete-answer objective and counterfactual grounding

Baseline is masked causal cross-entropy on all eligible answer tokens and EOS. Generation ends on EOS or an explicit cap; report exact answer AND valid stopping. Candidate restriction to correct answers is forbidden in the primary language metric.

Add an optional counterfactual grounding term. For the same public context and goals g1/g2 with different valid answers y1/y2, score full answer-sequence log likelihood, including EOS, under each goal. A margin loss penalizes `score(y_other|g_own) >= score(y_own|g_own)-margin`. Normalize sequence scores by a declared rule, default eligible-token mean, and store the rule in config. Teacher strings originate only from qualified training tasks; never fabricate contrasts by changing meaning without validating answers.

The total initial objective is `L = L_answer + lambda_pair * L_pair`; `lambda_pair=0` is the compatibility control. Default new-feature coefficient remains zero until basic correctness and future qualification. Tie or legitimately identical-answer pairs are excluded with a count, not relabeled negative. Tests must expose query-blind behavior even when aggregate accuracy looks acceptable: both-correct, same-answer rate and goal-swap gap supplement exact/EOS.

This is a proposed way to pressure goal use. It is not demonstrated to improve general reasoning, and the build must preserve an ablation switch.

## 4. Representation telemetry and training-only competition

Implement a `LogitTreatment` interface: `full`, `participating_mask`, `inactive_offset`. Its arguments are determined only by declared training schema/tokenizer identities. All gold targets and structural tokens required for the task must participate. Reject an inconsistent batch rather than masking out its target.

For a physical vocabulary V, active schema A and desired effective participating size K, an optional inactive offset uses `log((V-|A|)/(K-|A|))` when denominators are positive; validate `|A| < K <= V`. This is a mechanistic option inspired by the unexecuted R1C design, not a default training fix. Do not call full-vocabulary behavior improved merely because masked loss falls.

Inference always uses the unmodified full vocabulary for the primary free-generation score. Any schema-limited action evaluation is separately typed. General-language training cannot derive an active vocabulary from the current minibatch and thereby remove difficult competing words; use a reviewed task-level schema or full softmax.

Telemetry, sampled on a fixed training-only diagnostic batch, includes target probability, active/inactive probability mass, target-vs-wrong margins, entropy, core and embedding gradient norms, and parameter displacement. Instrumentation must restore RNG/mode and must not alter gradients or optimizer state before the actual update. Counterfactual gradients are opt-in diagnostics and never added to the training gradient.

## 5. Evidence-driven plasticity controller

Expose a pure function `transition(state, controller_metrics, config) -> decision`. A decision may set a schedule phase, replay mixture or checkpoint request; it cannot directly mutate tensors or access a sealed split. Persist the state and every transition reason in the checkpoint.

States:

| State | Meaning | Permitted behavior |
|---|---|---|
| FORM | No independently qualified capability yet | Base acquisition schedule; inspect formation and representation telemetry |
| STABILIZE | Controller criterion passed for a configured consecutive window | Lower configured LR and preservation replay; do not label this self-improvement |
| EXPAND | New qualified task family introduced | Restore configured acquisition plasticity for selected parameters; preserve protected replay |
| REACQUIRE | Previously qualified capability falls below configured lower threshold | Return to configured reacquisition schedule; do not automatically lower LR |
| HOLD | Invalid metrics, nonfinite computation or resource boundary | Pause updates, persist recoverable state and report cause |

Thresholds, consecutive-window counts, cooldown, maximum transitions and LR multipliers are fixed in the launch config. Enter/exit thresholds differ to prevent chattering. Missing or stale measurements cannot count as passes. Track formation and preservation per capability family; an overall mean cannot override a protected-family failure. Default controller is `disabled`; fixed schedule must remain available for causal comparison.

Controller metrics come from a dedicated repeatedly used training-controller pool, never development-measurement or sealed outcomes. This feedback is part of the training procedure and its cost/data exposure must be declared. Reacquisition is motivated by ARK-010 but is not proven superior in this integration. A learned controller may be considered later; this first version is explicitly engineered.

## 6. Persistent learning and bounded memory

Build an append-only experience ledger with semantic provenance, quality/ambiguity flags, collection-policy version and per-transition costs. Replay samples whole groups/fragments under a deterministic cursor. Preserve reservoir state or an equivalent exact sampling state across checkpoints. Retention starts with fixed stratified replay; learned prioritization remains optional.

Working memory is scoped to an episode. Optional episodic retrieval can select previously observed TRAIN experiences using a deterministic baseline and later a learned selector. No examiner answers or current-task hidden state enter retrieval. Deduplicate copies of the same mechanism before measuring transfer. Retrieval results include content identities and origin, and the inference report states whether retrieval was enabled.

The later self-improvement loop creates a candidate from a parent, collects experience, updates through the same trainer, evaluates acquisition/retention and either publishes a new parent or rejects the candidate. Failure does not overwrite the last accepted model. For this build, prove orchestration on tiny fixtures; do not run a repeated learning campaign.

## 7. Training, checkpoint and recovery semantics

Use one canonical writer. At each optimizer boundary: aggregate gradients with the actual global supervised-target denominator, unscale if needed, check finite values, clip with dtype-aware tolerance, step once, advance scheduler/cursors once, and record counters. Reduction must occur at the accumulation boundary; never repeatedly reduce an already accumulated buffer. CPU equivalence tests precede any distributed backend claims.

Checkpoint publication is atomic: write payload and manifest in a temporary sibling, flush, hash and validate, then publish a completion marker. Restore rejects incomplete or mismatched payloads. Include model/optimizer/scheduler/scaler, all RNG streams, data/replay cursors, controller, tokenizer/schema identities, schedule counters and parent identity. Rotation preserves referenced milestones. Durable storage beyond a disposable session remains required for cloud readiness.

The decisive correctness check is interrupted versus uninterrupted NEXT UPDATE using the real integrated trainer in a fresh process, comparing model state, optimizer state, sampler and counters. Same-process serialization is insufficient. Tiny smoke must not use a fake trainer that bypasses this path.

## 8. Public commands and operational states

Implement `python -m bramastra_lab.research.cli` with `inspect`, `prepare-data`, `train`, `resume`, `evaluate`, `infer` and `package` subcommands. These commands do not exist at the start of this packet. Provide exact argument examples once implemented; `--help` must expose them without loading a model or starting CUDA.

`inspect` validates config, installed backend, corpus supply and predicted memory and produces explicit statuses: CONFIG_VALID, DATA_NOT_READY, DEVICE_UNVERIFIED, DOES_NOT_FIT, LOCAL_SMOKE_PASSED or TRAINING_READY_FOR_DECLARED_DEVICE. Report dimensions separately: a CPU-ready run is not TPU-certified. Large capacity requires a later owner-approved hardware test. `train` never launches solely because a generated readiness file says true.

The runnable deliverable accepts an operator-supplied local corpus and config, trains random weights, restores a run and produces complete responses. It is not acceptable to finish with a CLI that only calls the old experiment scripts.
