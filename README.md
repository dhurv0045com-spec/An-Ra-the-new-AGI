# AN-RA

> Sovereign, owner-shaped intelligence: trained from explicit data contracts, measured by reproducible evidence, and promoted only through enforceable gates.

AN-RA is a research system, not a chatbot wrapper. It joins a custom transformer, identity continuity, verified reasoning, hybrid memory, typed agency, simulation-only robotics, evaluation, promotion, rollback, and continual learning into one architecture.

## What Is Real Today

- Canonical model profiles: `25m`, `frontier`/`904m`, and `3b`.
- Exact transformer counts: `904,535,040` and `2,918,251,520`.
- Canonical tokenizer contract: 8,209 tokens with deterministic 8,192-to-8,209 checkpoint migration.
- Four-stage campaign orchestration through `training/train_unified.py`.
- Function-preserving frontier-to-3B growth through CSII.
- DEL, SADL, OGRS, WSD, PCGrad, CDR, RLVR, IBS, SSG, promotion, and rollback contracts.
- Hybrid BM25, semantic, graph, ghost, and short-term memory routing.
- Typed HGP mission trees, workflow execution, verification, and trajectory evidence.
- Robotics restricted to simulation and shadow execution.
- Authenticated service surfaces in `app.py`.
- Reachability contracts for T-01 through T-26 and evidence schemas for M-01 through M-12.
- Versioned C-01 through C-07 cognitive services with consent, encryption, evidence maturity, and separate extension checkpoints.
- Honest Colab T4 runtime classes and a pre-allocation training preflight.

The included 7,500-record causal curriculum is deterministic synthetic development data. It is explicitly marked non-promotion-grade until independently verified licensed source evidence is published.

This does **not** mean a 3B release is automatically trainable or promoted. Missing checkpoints, dataset manifests, hardware profiles, IBS evidence, growth parity, and release evidence remain explicit blockers.

## Five-Minute Orientation

```powershell
python -m scripts.verify_structure
python -m training.train_unified --mode status --model-size 3b
python -m training.train_unified --mode preflight --model-size 25m
python -m pytest tests -q
```

The status command is intentionally strict. A useful result may be a list of blockers.

For a small integration exercise:

```powershell
python -m training.train_unified --mode session --model-size 25m --max-steps 2
```

## Canonical Ownership

| Concern | Canonical owner |
| --- | --- |
| Campaign/session lifecycle | `training/train_unified.py` |
| Individual training execution | `scripts/build_brain.py` |
| Model construction/checkpoints | `training/v2_runtime.py` |
| Data selection and replay mix | `training/v2_data_mix.py` |
| Intelligence measurement | `evaluation/ibs.py` |
| Promotion and rollback | `evaluation/promotion.py` |
| Scale authorization | `training/ssg.py` |
| Agent lifecycle | `engine/agent_loop.py` |
| Robotics workflow | `robotics/workflow.py` |
| Persistent API | `app.py` |
| Technology reachability | `runtime/technology_registry.py` |
| Cognitive services | `cognition/services.py` |
| T4 preflight | `training/preflight.py` |

The rule is simple: improve the owner. Do not add a competing path beside it.

## Model Lineage

| Profile | Width | Layers | Q/KV heads | Context | Transformer parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `25m` | compact | compact | compact | development | integration profile |
| `frontier` / `904m` | 1,536 | 36 | 16 / 4 | 2,048 | 904,535,040 |
| `3b` | 2,560 | 42 | 20 / 5 | 2,048, gated to 4,096 | 2,918,251,520 |

The 3B model is a child of a promoted frontier checkpoint. CSII expands width and depth, verifies parity, aligns against the frozen parent, and quarantines failed growth candidates. It never overwrites the parent.

## Training Spine

```text
source validation
  -> license and provenance
  -> deduplication
  -> DEL
  -> style filter
  -> CIV gate
  -> tokenizer validation
  -> immutable local shards
  -> registered training buckets
  -> SADL/OGRS sampling
  -> optimizer + WSD + PCGrad
  -> CDR replay
  -> stage evaluation
  -> candidate promotion or quarantine
```

Live dataset streaming during training is not part of the canonical campaign. Published shards are local, hashed, versioned, and immutable.

## Promotion Philosophy

AN-RA distinguishes three states:

1. **Implemented**: code and tests exist.
2. **Measured**: an evidence artifact exists for a named checkpoint.
3. **Promoted**: the candidate passed capability, identity, safety, owner, deployment, and rollback gates.

Imports and log lines prove only reachability. They do not prove capability.

## Service

Core interfaces:

- `POST /generate`
- `POST /goal`
- `POST /session`
- `GET /status`
- `GET /health`
- `GET /cognition/status`
- `GET /training/preflight`

Set `ANRA_OWNER_TOKEN` for bearer authentication. Training, session, goal, memory, sovereignty, and robotics actions are owner-protected surfaces.

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

## Repository Map

```text
anra/          paths and package bridges
training/      campaign, runtime, data, optimization, growth, continual learning
evaluation/    IBS, promotion, memory benchmarks, metric evidence
engine/        agent facade, verification, trajectories, feature flags
memory/        hybrid retrieval and provenance
robotics/      typed simulation/shadow workflows and world model
inference/     generation, cache, speculative and serving helpers
identity/      CIV, ESV, HAL and identity contracts
runtime/       registries, technology reachability, recovery, audit helpers
cognition/     causal, epistemic, owner-model, experiment, synthesis, consolidation, debate
scripts/       executable training and operator entry points
tests/         architecture, behavior, migration, recovery and promotion tests
docs/          operating model, architecture, development, research and history
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Developer Guide](docs/DEVELOPER.md)
- [Operator Manual](docs/OPERATOR.md)
- [Complete Walkthrough](docs/WALKTHROUGH.md)
- [Vision](docs/VISION.md)
- [Master Goals](docs/planning/MASTER_GOALS.md)
- [Engineering Log](docs/engineering/ENGINEERING_LOG.md)
- [V3 implementation status](docs/V3_ZERO_TO_HERO_IMPLEMENTATION.md)
- [Cognitive architecture and T4 readiness](docs/COGNITIVE_ARCHITECTURE.md)

## Engineering Constitution

Before a change ships, identify its owner, interface, schema, caller, persisted state, metric, failure behavior, rollback behavior, tests, and migration path.

No fake metric. No hidden fallback. No silent fresh start. No duplicated subsystem. No checkpoint mutation without provenance. No capability claim without executable evidence.
