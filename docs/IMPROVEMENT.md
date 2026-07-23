# How An-Ra Improves

Updated: 2026-07-23  
Purpose: define how the repository becomes more capable without turning into a
pile of impressive names, hidden regressions, or incompatible model families.

## The improvement law

An-Ra improves when a change increases useful capability per unit of compute
while preserving stability, reproducibility, and rollback. “The code runs” is
an implementation result. “The model is better” requires a controlled result.

```text
idea
  → explicit problem
  → bounded implementation
  → frozen-parent comparison
  → capability + stability + cost evidence
  → promote, revise, or retire
```

## What matters first

The strongest near-term improvement is not adding another subsystem. It is
training one coherent V4 dense checkpoint and proving that it resumes exactly.
Without a useful language core, results from memory, agents, correction, MoE,
or moonshots are hard to interpret.

Priority order:

1. Protected 181M dense language foundation.
2. Instruction behavior and verifiable correction.
3. Retrieval and long-term memory.
4. Permissioned tools and agents.
5. Reversible domain adapters.
6. Proven model growth.
7. Sparse compute, hybrid sequence models, modalities, and world interaction.

## Improvement questions by subsystem

### Dense training

Ask: does validation improve by source without output collapse, repetition,
copying, or residual instability? Measure tokens, loss by source, gradient and
activation health, generation diversity, context use, and checkpoint
continuity.

### MTP

Ask: from the same parent and data order, does predicting +2/+3 future tokens
improve held-out capability enough to justify its memory and throughput cost?
MTP is implemented, but not promoted.

### MoE

Ask: can sparse upcycling increase useful capacity without exceeding the T4
budget or destabilizing load balance? The current eight-expert geometry would
raise the system to roughly 1.12B parameters, so it is disabled rather than
quietly attached to the 181M baseline.

### MoD and native routing

Ask: can the model skip token-layer computation while matching or improving
quality and reducing real wall-clock work? A router score alone is not value.

### RIM, ESV, DSTP, and HAL

Ask one question per mechanism. Does it change trained behavior positively, or
is it only a neutral initialized path? HAL remains more inspectable as an
external bounded policy until transformer-integrated modulation proves value.

### Self-correction

Ask: on verifiable errors, does
`understand → retrieve → plan → candidate → verify → revise/abstain`
increase correctness more than its extra tokens and latency? Store verifier
outcomes and failed revisions, not self-reported success.

### Retrieval and memory

Ask: does external evidence improve factuality and updateability without
cross-session leakage or unattributed claims? Knowledge should enter through
provenance-bearing context before it is baked into weights.

### Agents and tools

Ask: does the model choose the right tool, use a valid typed contract, respect
permissions and limits, verify the result, and recover from failure? Tool access
is a capability boundary, not a benchmark decoration.

### Model growth

Ask: does the 500M child preserve parent logits and behavior before continued
training? Only after parity should additional parameters be credited for an
improvement.

## Experiment design

A useful architecture pilot declares:

- parent checkpoint hash;
- source commit and model profile;
- one changed variable;
- identical seed 1301 and data order for the first comparison;
- token and wall-clock budget;
- primary capability metric;
- stability and memory limits;
- promotion and rollback rules.

Use another seed only when the first measured difference is too close to
decide. Three seeds are experimental replication, not three permanent models
and not three parallel ways of training every session.

## Stop rules

Automatically pause or reject on:

- NaN or Inf;
- persistent loss explosion;
- collapsed or highly repetitive generation;
- missing remote durability acknowledgement;
- corrupted or incompatible checkpoint;
- duplicate token window;
- wrong source commit or tokenizer hash;
- major validation regression;
- stale worker attempting to commit canonical weights.

Stopping early protects evidence and compute. It is not lack of ambition.

## Invention without chaos

New technology is welcome when it has a repository-specific purpose. A
moonshot belongs in `pilot`, behind a hard-off canonical flag, with its own
budget and evidence. It can be radical internally while remaining reversible
operationally.

Good invention targets current bottlenecks:

- more learning from the same high-quality tokens;
- better long-context use without quadratic waste;
- verifier-guided correction that learns from outcomes;
- memory consolidation without silent catastrophic forgetting;
- sparse capacity whose measured compute is genuinely sparse;
- function-preserving growth between resource budgets;
- multimodal grounding only after language and evidence are stable.

## Current improvement backlog

The executable forward order is in `TODO.md`. The next decisive evidence is:

1. a short protected T4 canary;
2. a forced cross-worker exact handoff;
3. a coherent 200M-token milestone;
4. matched dense-versus-MTP evidence;
5. an audited SFT corpus and real post-training run;
6. parent-parity proof for the 500M child.

## What “better” does not mean

- More parameter files.
- More subsystem names enabled together.
- Lower aggregate loss with poor source-specific behavior.
- A successful smoke test presented as intelligence.
- A generated answer saying it reasoned or corrected itself.
- An architecture copied from a paper without matching An-Ra’s budget.
- An irreversible checkpoint with no lineage.

## Live truth sources

- Priorities and incomplete work: `TODO.md`
- Lifecycle and claimed costs: `runtime/subsystem_catalog.py`
- Architecture constraints: `docs/engineering/V4_ARCHITECTURE_GATE.md`
- Evaluation: `training/eval_v2.py`, `evaluation/ibs.py`
- Promotion and rollback: `evaluation/promotion.py`,
  `inference/adapters.py`
- Experiment evidence: `runtime/evidence_stream.py`,
  `evaluation/thirdeye_adapter.py`
- Historical decisions: `docs/engineering/ENGINEERING_LOG.md`

This file provides the reasoning framework. A promotion decision must cite the
specific signed run and evidence that earned it.
