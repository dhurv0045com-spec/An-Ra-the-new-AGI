# Data, state and evidence contracts

Status: schema specification for W01. Implement versioned strict records and validators before treating this as an enforced interface. Prototype records are adapters, not implicit compliance.

## 1. Identity rules

Use SHA-256 of canonical serialized content. Canonical JSON uses UTF-8, sorted keys, no nonfinite values and a declared numeric/string representation. Tensor identities include dtype, shape, byte order and contiguous bytes. Human-readable names do not substitute for content identities.

Separate semantic identity from rendering identity. Two differently worded descriptions of the same rule/program/graph belong to one semantic cluster. Split clusters before generating surfaces. Store the split algorithm/version and immutable manifest.

## 2. Core records

| Record | Required fields | Invariants |
|---|---|---|
| `TaskSpec/v1` | Semantic ID, family, generator hash, public schema ID, split, difficulty descriptors, resource limits | Hidden parameters live outside the public record |
| `PublicObservation/v1` | Task/episode/step IDs, observable values, legal-action schema, feedback, remaining budget | No arbitrary objects, callables, private metadata or answer fields |
| `Action/v1` | Episode/step IDs, action kind, typed arguments, policy identity | Schema-valid, authorized by environment, charged exactly once |
| `Transition/v1` | Observation, action, next observation, reward, cost, termination, truncation, behavior probability | Termination and time-limit truncation are distinct; behavior probability documented or unavailable |
| `Episode/v1` | Ordered transitions, task semantic ID, collection policy, seed, reset scope | Monotonic steps, consistent task, no repeat/forbidden target query in restricted tasks |
| `TrainingBatch/v1` | Tensor content IDs, source IDs, masks, lengths, target counts, sampler cursor | Padding excluded, no cross-episode state leakage, targets shifted exactly once |
| `Checkpoint/v1` | Model spec/weights, optimizer, schedule, RNGs, sampler, replay cursor/index, parent, data/runtime identities | Restore validates actual content, not only descriptive manifest equality |
| `Experiment/v1` | Hypothesis, arms, allocation, seeds, splits, metrics, stop rules, code/data/runtime identities | Changes after freeze create a revision |
| `Outcome/v1` | World/target/arm/budget, prediction, label, cost, failure reason, timing | Score recomputed from raw prediction; duplicate or mismatched paired identities rejected |
| `Promotion/v1` | Parent/child identities, examiner protocol, transfer/retention comparisons, cumulative history, decision | No candidate-owned claim can override examiner result |

## 3. Tensor interfaces

The discovery prototype's rows are a reduced exploratory format, not canonical `Outcome/v1`. Its adapter must mark absent timing, cost and failure details as unavailable or derive them from actual independent run evidence. It must not invent measurements to satisfy a schema.

For discrete rule experiments:

- Observations: `[B,T,D_obs]` float or explicitly typed token representation.
- Lengths: `[B]` integers in `[0,T]`; suffix padding cannot alter state or prediction.
- Working state: `[B,D_state]`, with explicit reset ownership.
- Goal/target features: `[B,D_goal]`; target **features** are permitted, target **answer** is not.
- Candidate actions: `[B,A,D_action]` or a documented shared `[A,D_action]` table.
- Legal mask: `[B,A]` boolean; sampling gives illegal actions exactly zero probability.
- Policy logits: `[B,A]`; no-legal-action states terminate explicitly.
- Binary prediction logits: `[B]`; multi-output domains need named heads/schema identities.
- Rewards, values and continuation masks: `[B,T]`, with terminal/truncation semantics fixed.

For language batches, include token IDs, segment IDs, positions, eligible loss mask and explicit terminal target. Count capacity, nonpadding input and supervised target tokens separately.

## 4. Experience storage and sampling

Append immutable episode shards and publish a manifest only after complete writes. Index by semantic task, domain, success/failure, difficulty, policy version and collection round. Store a replay selection identity, seed and cursor; replay must resume without silently changing its distribution.

Initial replay is uniform within family with declared family weights. Prioritized replay is a later ablation: record sampling probabilities and use correction where the learning objective requires it. Do not preferentially retain successes and then describe the store as representative experience.

Use complete episodes as the default sampling unit for meta-learning. If truncating histories, record what state is available before the fragment. Do not treat unrelated fragments as one continuous world.

## 5. Split boundaries

Keep distinct pools for gradient training, training-only gain probes, development, strategy validation, and confirmation. Do not create separate names that point to overlapping semantic IDs. Once a pool's feedback changes model design, it becomes development evidence for future claims.

Out-of-family evaluation withholds a mechanism family entirely from initial training. After that family becomes acquisition data, the same family is no longer unseen-family transfer; use held-out mechanisms within it and label the change.

All record exports must allow the examiner to verify that source examples were excluded from learner data. Keep a contamination report, including unknown provenance. A hash proves identity, not correctness, diversity, ownership or lack of contamination.

## 6. Checkpoint transaction

Write payloads to a unique temporary location, compute hashes, write a complete manifest, then atomically publish a completion marker or use an equivalent backend transaction. Interrupted writes are never loadable checkpoints. Preserve the last committed parent until the new state is durable.

Verification ladder: same-process CPU continuation; fresh-process CPU continuation; actual accelerator continuation; multi-device continuation when supported; remote durable restore after session loss. Each receipt names its level. The current prototype's same-process CPU check cannot satisfy the later levels.

## 7. Run directory

```text
artifacts/bramastra/<unique_run_id>/
  manifest.json
  source_snapshot/          # exact source bytes or an immutable complete source archive
  data_manifest.json       # shard/split/generator and tensor identities
  runtime.json
  events.jsonl             # append-only progress, failures and consumed budget
  <arm>/
    training.json
    outcomes.jsonl
    metrics.json
    checkpoint_manifest.json
  comparison.json
  decision.json
```

Large payloads may live outside Git under manifest-referenced storage. Document availability. Never say an externally restorable artifact exists merely because its local filename is in a receipt.
