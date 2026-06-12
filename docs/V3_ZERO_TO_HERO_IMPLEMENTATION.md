# AN-RA V3 Zero-to-Hero Implementation

Date: 2026-06-12

## Implemented

- Canonical 8,209-token frontier, 3B, and 8M draft configurations.
- Exact parameter accounting:
  - Frontier transformer: 904,535,040
  - 3B transformer: 2,918,251,520
  - 3B identity-augmented system: 2,925,174,103
  - Draft plus shared ESV predictor: 8,004,291
- Pure ESV prediction with explicit `commit_state`.
- Logit-bounded attention, native SDPA GQA, bounded RIM, DSTP, and sparse contextual MCR.
- IBS-50, private-suite loading, three-seed capability promotion, deployment promotion, SSG, and adaptive capability ladder.
- AdamW, AdamW8bit, Adafactor, Muon, GaLore, Q-GaLore, and SCALE adapter surface with honest fallback reporting.
- PCGrad, LPGA research prototype, blockwise QAT, memory profiler, distributed estimates, and SADL.
- Deterministic local shards, DEL, canonical DFC validation, CDR, provenance, licensing, and distillation intake checks.
- Four resumable stages: foundation, owner adaptation, agency, and verified reasoning.
- HGP, calibrated competence, learning-progress curiosity, causal proof memory, verifier search, OGRS, CSII, and reversible continual adapters.
- Tiered/AWKC KV retention, prefix cache, speculative-decoding gates, and KV backend benchmark contract.
- FastAPI endpoints for goals, plans, memory, evaluation jobs, training candidates, robotics workflows, and event streams.
- Typed robotics observations and skills, workflow execution, domain randomization, sim-to-real gates, and uncertain GRU world model.
- Frozen multimodal encoder projector for future vision/audio experiments.

## Operator Commands

Verify the repository:

```powershell
python -m pytest -q
```

Inspect the staged campaign:

```powershell
python -m scripts.train_v3 status
```

Dry-run a frontier foundation campaign:

```powershell
python -m scripts.train_v3 run --dry-run --stage foundation `
  --config config/anra_frontier.yaml --optimizer galore --device cuda
```

Run a small smoke campaign:

```powershell
python -m scripts.train --config config/tiny.yaml --max_steps 100 `
  --optimizer adamw --device cpu
```

## Evidence Still Required

The code does not claim completion of expensive empirical milestones:

- Train and promote the 904.5M frontier checkpoint.
- Profile GaLore and LPGA on an actual T4.
- Run the approximately 60B-token distributed 3B campaign.
- Demonstrate three-seed IBS and owner-suite promotion.
- Benchmark FP16, KIVI, TurboQuant, KVarN, and sliding-window KV backends.
- Demonstrate speculative speedup and acceptance thresholds.
- Collect at least 100,000 simulation transitions before world-model activation.
- Pass randomized simulation, shadow, and supervised hardware gates.
- Demonstrate multimodal gains before unfreezing any encoder.

These are operator-run evidence gates, not missing implementation decisions.
