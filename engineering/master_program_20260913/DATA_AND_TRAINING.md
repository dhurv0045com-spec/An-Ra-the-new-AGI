# Data structures, preparation and training program

The system's useful experience is the limiting resource as much as its parameter count. Build the data interfaces before increasing model size. A training stage listed here is a future executable recipe; it is not authorization to run it or a claim that its corpus is available.

## 1. Canonical records

Use versioned, immutable serializable records and content hashes. JSON is suitable for small receipts; large token arrays/shards can use a documented binary format with manifests. Hash the actual bytes plus schema/renderer identity. Timestamps and file locations are provenance, not semantic identity. Preserve raw source identity alongside normalized-content identity.

| Record | Required contents | Critical invariant |
|---|---|---|
| `SourceRecord` | source hash, source location, license/provenance, eligibility, document boundaries | Missing permission/trainability is not inferred from readability |
| `TaskSpec` | generator/schema version, public goal, public action/feedback schema, initial budget, verifier identity | Hidden mechanism lives in a separate environment-only object |
| `SplitMembership` | example content ID, mechanism cluster ID or explicit unknown, pair group, split, assignment version | Related mechanisms/groups cannot cross the intended generalization boundary |
| `ObservedTransition` | episode/step IDs, before/after public views, action, legal-set ID, costs, real feedback, terminal/truncation, behavior metadata | References a real receipt; no imagined outcome accepted |
| `SupervisionRecord` | source transition/group, objective type, target, eligibility mask, teacher/verifier identity | Targets are justified by observed data or declared qualified supervision |
| `PredictedOutcome` | public payload, prediction probabilities/sampling metadata, validity, predictor/schema identity | Cannot be deserialized as observed evidence |
| `MemoryContext` | eligible ordered record IDs, payload, scope, index/retriever IDs, token cost | Contains no forbidden split, hidden answer or cross-episode residue |
| `ObjectiveBatch` | tensors below, counts, source references and objective settings | Counts equal the actual eligible tensor elements |
| `CandidateTransaction` | parent/child identities, proposal, data/memory snapshots, allocation, protocol, state, execution mode, evidence class, change kind, learned-update count and receipts | Fixture namespace cannot publish a learned parent; real publication requires hash-bound chief approval |
| `ResourceVector` | schema/version, allocated and consumed quantities, units, hardware/runtime identity, measurement status | Unknown is distinct from zero; comparisons use the protocol's explicit resource predicate |

Mechanism cluster IDs must come from a justified generator equivalence rule or a curated mapping. A text hash cannot detect every rephrasing or semantically identical program. For natural corpora, use document/source-group splits and report the remaining contamination uncertainty. Do not generate a fake semantic hash merely to satisfy a required field. Mark unknown equivalence and restrict the associated claim.

The resource schema covers real inquiry/submission cost, optimizer updates, encoded tokens, supervised targets, teacher calls/time, retrieval calls/tokens, inference calls/tokens, imagined nodes, device seconds and elapsed seconds. Integer counters are exact and floating quantities are finite/nonnegative with declared units. Every relevant layer returns deltas to one accounting owner; parent scopes aggregate child deltas without charging them twice. The run receipt contains both allocated caps and actual consumption, including failures.

## 2. Tensor and mask contract

Let B be examples/decisions, T token positions, K padded candidate slots, and D hidden width. Ordinary token batches use `input_ids[B,T]`, `labels[B,T]`, `padding_mask[B,T]`, `segment_ids[B,T]`, `token_loss_mask[B,T]` and an objective kind. Teacher forcing is causal: each label is predicted from the preceding eligible prefix. Test the shift using a short explicit token sequence, including BOS/EOS and padding.

Action batches use `candidate_input_ids[B,K,T]`, corresponding padding masks, `legal_mask[B,K]`, candidate identity sidecars, `teacher_probs[B,K]` when available, `selected_action[B]`, `behavior_logprob[B]` and `decision_eligible[B]`. Values use independently encoded action-free prefixes, `returns[B]`, `return_eligible[B]`, reward protocol ID and horizon. On-policy batches also carry behavior checkpoint identity and detached behavior values. A missing teacher distribution does not become a one-hot label from the model's own choice.

World batches render the real before-view plus action as input and next-public-outcome as target, with `world_loss_mask[B,T]` and field spans for separate scoring. Pair batches retain two qualified goal/answer variants and group eligibility. An `ObjectiveBatch` may contain several sub-batches rather than a giant mostly empty tensor; the objective router performs explicit dispatch and preserves per-objective counts.

Keep IDs, split names, generator parameters, teacher access flags and source paths in sidecars, never embedded as answer hints. Use strict integer/bool/finite floating types. Reject NaN returns, negative counts, impossible action indices, duplicated action identities, inconsistent EOS masks and invalid probability sums before device allocation where possible.

## 3. Preparation is an evidence-preserving compiler

Preparation validates source manifests, assigns or verifies splits, admits supervision, renders public content, constructs masks, shards deterministically, hashes outputs, and writes an atomic prepared manifest. Source mutation invalidates preparation. Resume binds prepared bytes, tokenizer, public renderer, split plan, objective settings and memory snapshot.

Make the transition conversion explicit:

```text
observed episode
  -> validate ordering, costs and terminal semantics
  -> obtain public prefixes without future events
  -> derive only justified answer/action/value/transition targets
  -> preserve pair and mechanism groups
  -> emit objective records plus rejection counts
  -> deterministic prepared shards and manifest
```

A failed episode still supplies valid transition observations. An incorrect submission cannot be relabeled as a correct answer target. An invalid action can train feedback prediction and failure analysis; teacher policy supervision must not endorse it. Partial episodes may lack return labels while retaining other eligible objectives. Expose per-objective admitted/rejected counts and reasons so the agent cannot hide unusable data behind an overall row count.

M06 bootstraps from a versioned seed ledger fixture produced by the **existing real environment rollout and collector**, using a fixed policy and zero model updates. M01 defines the schema and M06 creates/adopts this fixture with generator, public protocol and content identities, strictly bounded episode counts, and actual interaction accounting. This is valid generated environment data, not a fake evaluator result. M08 later supplies integrated learned-policy collection through the same format. Consequently M06 does not depend on future M08 implementation.

## 4. Task factory and domains

Start by qualifying the current switch, inventory and program laboratory families, then extend along explicit axes. Each generated family needs a public interface, independent validator, equivalence/split rule, difficulty definition, oracle diagnostic and strong nonlearned baselines.

| Family | Variation and transfer boundary | What it tests |
|---|---|---|
| Hidden-rule inquiry | New rule mechanisms; irrelevant features; delayed complementary queries | Choosing useful evidence and revising predictions |
| Inventory/resource tasks | New resource graphs, action dependencies and costs | Feasible planning and budget accounting |
| Small program laboratory | New compositions/AST templates, variable renaming, longer dependency depth | Executable reasoning and systematic composition |
| Grounded instruction pairs | Same world/history with different valid goals | Whether decisions and answers follow the goal |
| Document-assisted tasks | New source documents, explicit citation/answer checks, distractor retrieval | Using external memory with source grounding |
| Bounded tool tasks | Small local file/table/code operations in disposable fixtures | Acting through tools and checking consequences |

Do not launch unrestricted programs generated by a model. Initial code tasks use a bounded interpreter or isolated disposable runner with time/output/filesystem limits and no network. These limits are part of the task interface and cost report. Task success is checked by independent tests or public goal conditions, not an untrained language judge. This is necessary for reliable reward, not a substitute for broader future environments.

Hold out combinations, mechanisms, lengths and representations separately. Changing only seed or surface spelling may measure interpolation; label it accordingly. Every transfer task must specify what training experience could plausibly help and what information must be newly inferred. Include impossible/ambiguous instances as a separately scored abstention protocol, with validated ambiguity labels.

## 5. Curriculum stages and exit evidence

| Stage | Training content and active objectives | Required evidence before the next learned stage |
|---|---|---|
| T0: runtime truth | No learning campaign; typed fixtures and bounded existing smoke | Identity, masks, accumulated update, restore, counters and evaluator checks pass |
| T1: shared representation | Qualified documents and simple grounded sequences; token and transition objectives | Held-out content prediction beats frequency controls; complete-answer stopping works; no source leakage |
| T2: grounded decisions | Teacher-qualified training episodes; token, world, action, value; optional pair term | Trained heads beat fixed/random controls on declared measurement mechanisms; goal-pair and value checks pass |
| T3: useful inquiry | Frozen-policy collection and the A3 on-policy route; optional bounded planning | Higher task success per real interaction or lower cost at matched success, with inference cost disclosed |
| T4: continual acquisition | Sequential new families, fixed replay, optional controller/distillation | New acquisition plus bounded protected-family regressions over repeated transitions |
| T5: broader transfer | Documents, executable programs, longer tasks and tool episodes | Cross-family/composition transfer beyond matched single-family controls |
| T6: repeated improvement | Isolated child proposals and independent qualification | Multiple accepted cycles with new-task gains, retention and complete accounting; no evaluation-driven leakage |

Exit thresholds and uncertainty rules must be filled into the actual experiment protocol before its first run. Do not reuse development outcomes to tune a threshold and then call the same outcome confirmation. A blocked stage does not prevent building later-stage software with fixtures. It does prevent advertising later capability as learned.

The first offline multi-objective fixture config should explicitly exercise token/world/action/value terms with unit coefficients and pair/PG/entropy disabled. It exists to verify routing, not prescribe production relative weights. Subsequent qualification protocols compare a small preregistered coefficient set against the answer-only control, using gradient scale and objective denominators to diagnose domination. The chosen production coefficients remain unresolved until evidence exists.

Keep current AdamW/optimizer implementation and declared LR controls unless a measured defect requires change. Existing default LR `1e-3` is not automatically suitable for every stage or profile. Store the selected LR, warmup, clip norm, decay, accumulation, effective examples/tokens per update, and objective mixture in each launch manifest. Changing any of these creates a new candidate identity, not an exact resume.

## 6. From-scratch corpus and tokenizer decisions

No pretrained core weights are allowed. Training a tokenizer on permitted training text would itself introduce a learned data prior and requires a versioned comparison; the current fixed byte tokenizer avoids that additional training step. It can require longer token sequences, so compare throughput and bytes/semantic units as well as token count. Do not import a pretrained tokenizer and silently treat the whole system as free of external priors.

Prepare local manifests for qualified text, code and structured examples. Record license/provenance, source grouping, exclusions, deduplication policy, quality filters and trainability. The present real-corpus state remains `DATA_NOT_READY` until actual sources exist and pass validation. Implement source-adapter and manifest support now; never substitute tiny synthetic fixtures for a missing corpus while claiming pretraining completion.

Suggested sampling categories are language/document prediction, executable reasoning, grounded interaction, and retention replay. Their fractions are configuration, sum to one, and refer to a declared sampling unit. Equal example counts do not imply equal token or compute fractions. Report realized tokens, supervised targets, source examples, presentations and interactions for each category. Avoid a fixed giant mixture chosen without acquisition evidence; increase one axis at a time so a failure remains diagnosable.

## 7. Accelerator readiness and efficient scaling

Preserve the tiny/development/future-capacity configurations from B2. Add no nominal 500M readiness badge. The chief's evidence review found that Cymek's larger-output arithmetic run failed generalization and its local frozen 500M microstep did not fit the tested device; those are reasons to measure the intended BRAMASTRA path directly.

Build a preflight that reports exact device/runtime versions, precision, parameter/state memory, sequence shape, real accumulation behavior, peak observed memory, tokens/second, checkpoint bytes, export/restore time and remaining resource allowance. Verify a fresh-process resume from durable storage on the actual backend before a disposable session is called training-ready. Cloud upload without restoring optimizer/RNG/sampler state is insufficient.

Forecast using measured end-to-end throughput at the intended shape. `estimated_training_seconds = planned_encoded_tokens / measured_tokens_per_second`, plus separately measured evaluation, checkpoint, compile and recovery overhead. Report the measurement interval and uncertainty; do not extrapolate CPU or tiny-shape performance linearly to a much larger TPU model. Record any recomputation, bucketing or precision option as part of the configuration.

The owner's roughly 100 TPU-hours/week is an unverified planning envelope. A future proposal can reserve, for example, 15% for profiling/recovery, 55% for primary comparisons, 20% for confirmation and 10% for evaluation, but these are suggested planning fractions, not a live quota or allocation. Prefer a few informative paired runs to a broad sweep that cannot answer its own question. Never use all available time merely because it exists.

## 8. Checkpoint and reproducibility expansion

Beyond B2.2's corrected model/optimizer/RNG state, bind objective-router version, window plan, curriculum state, replay cursor, behavior-policy snapshot, pending collection episode state if resumable, memory index snapshot, planner configuration and candidate transaction state. Default checkpoint boundaries occur between optimizer windows and between real actions. If mid-window or mid-action recovery is unsupported, reject that mode and resume from the last committed boundary with explicitly accounted lost work.

Exact resume is one claim; a deliberate change of objective, new head, memory snapshot or dataset is a fork/migration claim. The latter creates a child with declared initialization and resets only the explicitly affected state. Tests must distinguish these cases. Checkpoint loading remains safe and identity-bound; the new program must not reintroduce unrestricted pickle fallback or bypass writer fencing.
