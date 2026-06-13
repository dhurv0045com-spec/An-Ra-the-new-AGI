# AN-RA V3: Zero to Evidence

**Current architecture review:** 2026-06-13
**Meaning of this document:** implementation map, not a claim that every training or promotion gate has passed.

## Implemented Core

- Profiles `25m`, `frontier`/`904m`, and `3b`.
- Exact transformer counts:
  - Frontier: `904,535,040`
  - 3B: `2,918,251,520`
  - 3B with separately reported identity modules: `2,925,174,103`
- Canonical 8,209-token tokenizer and deterministic legacy checkpoint migration.
- RMSNorm, GQA, RoPE/YaRN, SwiGLU, LBA, RIM, ESV, HAL, DSTP, MoD/MCR, tied embeddings, and checkpointing contracts.
- Four-stage campaign state machine with pausing gates.
- CSII function-preserving frontier-to-3B growth and parity reporting.
- DEL, immutable token shards, SADL, training OGRS, WSD, PCGrad, optimizer evidence, CDR, and replay refresh.
- Verified STaR/RLVR routing, typed verifiers, and truth-coverage reporting.
- IBS seven-dimension measurement, SSG blockers, signed promotion manifests, smoke rollback, and continual adapter isolation.
- Typed HGP missions, verified trajectories, hybrid memory, robotics activation boundaries, and persistent API state.
- T-01 through T-26 architectural reachability.

## Canonical Commands

Inspect 3B readiness:

```powershell
python -m training.train_unified --mode status --model-size 3b
```

Run a small architecture session:

```powershell
python -m training.train_unified --mode session --model-size 25m --max-steps 2
```

Run verification:

```powershell
python -m pytest tests -q
```

## Four-Stage Campaign

| Stage | Minimum campaign volume | Primary gate |
| --- | ---: | --- |
| A: Foundation | 5B tokens | perplexity below 12 and stable numerics |
| B: Owner Adaptation | 10B tokens | CIV above 0.85, IBS above 50%, no safety/reasoning regression |
| C: Agency | 3B trajectory tokens | at least 1,000 verified trajectories and tool-use above 60% |
| D: Verified Reasoning | 3B tokens | reasoning above 70%, STaR verification above 90% |

The minimum total is 21B campaign tokens. Dataset inventory and stage evidence decide whether a real run may proceed.

## What Still Requires Real Evidence

- Licensed local token inventory and immutable manifests at campaign scale.
- A promoted frontier checkpoint and signed release manifest.
- Three-seed IBS and private-owner evaluation results.
- Target-hardware optimizer, memory, and throughput profiles.
- A successful frozen-corpus growth parity artifact.
- Stage-scale training runs and candidate checkpoints.
- A frozen 200-question private memory benchmark.
- Robotics activation evidence from at least 100,000 simulation transitions.
- Production service telemetry sufficient for uptime and recovery metrics.

## Definition of Done

V3 is complete only when code, evidence, and promoted artifacts agree. The architecture may be implemented while a campaign remains blocked; that is correct behavior, not a failure to hide.

Every future upgrade must strengthen a canonical owner and its lifecycle. It must not arrive as a patch, hidden fallback, demo-only branch, duplicated path, or unmeasured claim.
