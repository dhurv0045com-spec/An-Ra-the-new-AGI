# AN-RA Architecture

> One system, one owner per lifecycle, evidence at every boundary.

**Reviewed:** 2026-06-13

## The Shape

```text
licensed sources
  -> canonical data intake
  -> immutable token shards
  -> staged training campaign
  -> candidate checkpoint
  -> IBS + private owner suite + CIV + safety
  -> signed promotion or quarantine
  -> inference / memory / agency / simulation
  -> verified trajectories and corrected failures
  -> isolated continual-learning candidate
  -> the same evaluation and promotion gate
```

There is no privileged shortcut around this loop. SSG may authorize scale; it may not rewrite failed checks as passed.

## Ownership Map

| Lifecycle | Owner | Persistent evidence |
| --- | --- | --- |
| Campaign | `training/train_unified.py` | campaign state and `StageResult` |
| Training execution | `scripts/build_brain.py` | checkpoint, metrics, replay and optimizer reports |
| Model/checkpoint runtime | `training/v2_runtime.py` | schema, tokenizer contract, migration provenance |
| Data mixture | `training/v2_data_mix.py` | bucket weights and replay refresh |
| Data intake | `training/data_pipeline_v3.py` | immutable shard manifest |
| Model growth | `training/csii.py` | parity and alignment report |
| Scale authorization | `training/ssg.py` | structured blockers and evidence paths |
| Intelligence evaluation | `evaluation/ibs.py` | three-seed IBS report |
| Promotion | `evaluation/promotion.py` | signed release manifest and rollback history |
| Metrics | `evaluation/metrics.py` | M-01 through M-12 evidence snapshot |
| Agent lifecycle | `engine/agent_loop.py` | mission, workflow and trajectory |
| Memory | `memory/memory_router.py` | source-preserving retrieval records |
| Robotics | `robotics/workflow.py` | simulation/shadow outcomes |
| Service | `app.py` | jobs, sessions, requests and audit records |
| Technology registry | `runtime/technology_registry.py` | T-01 through T-26 entry points |

## Model Contracts

| Profile | Vocabulary | Width | Layers | Q/KV heads | FFN | Initial context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frontier` / `904m` | 8,209 | 1,536 | 36 | 16 / 4 | 4,096 | 2,048 |
| `3b` | 8,209 | 2,560 | 42 | 20 / 5 | 6,848 | 2,048 |

The 3B context may move to 4,096 only through context-extension evaluation. OOM recovery may reduce a request's operational context; it may not silently mutate the architecture.

The 3B checkpoint descends from a promoted frontier checkpoint. CSII owns width expansion, GQA remapping, identity-preserving depth insertion, RIM distillation, state transfer, parity testing, and progressive unfreezing.

## Tokenizer Contract

The canonical vocabulary has 8,209 rows and 30 validated special-token IDs. Legacy 8,192-row checkpoints migrate as follows:

1. Preserve every legacy row exactly.
2. Initialize only the 17 appended control rows.
3. Use deterministic initialization independent of ambient RNG.
4. Keep tied embedding and LM-head rows equal.
5. Record schema versions and migration provenance in the next checkpoint.

An incompatible checkpoint raises `CheckpointCompatibilityError`; it does not become a silent fresh run.

## Training Contracts

### Data

```text
source validation
  -> license/provenance
  -> deduplication
  -> DEL
  -> style filter
  -> CIV
  -> tokenizer validation
  -> local shard publication
  -> bucket registration
```

DEL rejects quality below `0.65`. Shards are `uint16`, hashed, versioned, immutable, and sized for 10 million tokens in production publication.

### Optimization

- `training/anra_optimizer.py` is the only optimizer factory.
- Identity-critical parameters remain full-rank and decay-free.
- GaLore is active only when the selected backend is actually available and measured.
- WSD owns the base learning-rate shape.
- Dynamic regret may apply only a bounded multiplier.
- PCGrad projects conflicting identity gradients at the optimizer boundary.
- CDR admits only executable, verified corrections to replay.

### Stages

| Stage | Goal | Completion gate |
| --- | --- | --- |
| Foundation | language, code, science, symbolic structure | perplexity and numerical stability |
| Owner Adaptation | identity, owner tasks, protected capabilities | CIV, IBS, safety and reasoning |
| Agency | tools, plans, workflows, trajectories | verified trajectory inventory and tool use |
| Verified Reasoning | STaR, RLVR, symbolic execution | reasoning, verification and truth coverage |

A failed gate pauses the campaign. It does not mark the stage complete.

## Intelligence And Action

```text
goal
  -> retrieve context
  -> HGP decomposition
  -> validate MissionTree
  -> compile Workflow
  -> authorize
  -> predict
  -> execute
  -> verify
  -> store trajectory
  -> update memory and CDR
```

Model output is parsed into typed contracts. Mission depth is at most 5 and the tree has at most 10 leaves. Only machine-verified successes count toward M-04.

## Memory

The canonical router fuses:

- BM25 exact retrieval
- FAISS semantic retrieval
- graph memory
- ghost long-term memory
- short-term context

Fusion is deterministic and source IDs survive every tier. Promotion evidence comes from the frozen private 200-question owner benchmark, not ad hoc demonstrations.

## Robotics Boundary

Robotics is simulation and shadow-only. Each skill requires preconditions, authorization, world-model prediction, uncertainty/reward checks, typed dispatch, postcondition verification, and CBF safety checks.

The live workflow appends transitions but never trains the world model. Offline activation requires at least 100,000 transitions, held-out accuracy of 70% or more, and planning improvement of 10% or more. Physical actuation is a separate future promotion decision.

## Evaluation And Promotion

IBS promotion dimensions:

| Dimension | Weight |
| --- | ---: |
| Reasoning | 20% |
| Tool use | 20% |
| Identity | 20% |
| Owner task | 15% |
| Safety | 10% |
| Anti-timidity | 10% |
| Memory | 5% |

Promotion requires three deterministic seeds, positive confidence-bound improvement, no protected-dimension regression, no private-owner regression, signed artifacts, rollback history, and a post-promotion smoke test.

## Failure Semantics

Every important failure should answer five questions:

1. What typed error occurred?
2. What evidence was written?
3. What state remains valid?
4. Can the operation be retried?
5. What rollback target is available?

Unknown evidence is represented as missing or zero, never inferred from an unrelated report.

## Verification

```powershell
python -m training.train_unified --mode status --model-size 3b
python -m pytest tests -q
python -m scripts.verify_structure
```

For narrative detail, continue with [WALKTHROUGH.md](WALKTHROUGH.md). For implementation rules, use [DEVELOPER.md](DEVELOPER.md).
