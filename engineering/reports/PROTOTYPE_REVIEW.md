# Chief review: discovery prototype

Review date: 2026-09-07. Status: bounded local prototype, useful for further experiments. It does not implement the canonical integrated research architecture or demonstrate AGI.

## Implemented components

`bramastra_lab/discovery/` contains deterministic semantic rule worlds, public inquiry episodes, an exact training-only teacher, a random-initialized recurrent predictor/query selector, demonstration generation, AdamW training, parameter and data fingerprints, local checkpoint continuation, five evaluation policies, world-clustered statistics, parent/child comparisons, and a campaign CLI.

Implementation and review used bounded Sol/Luna assignments. Initial prototype coding began before the owner appointed the chief-engineer role; subsequent quality closure, tests and experiment execution were delegated. Future substantial implementation follows the work-order system.

## Executed development campaign

Each seed used 259 nonconstant, semantically deduplicated worlds after a label-balance filter: 60 conjunction, 126 parity and 73 threshold. Initial training used 129 nonthreshold worlds; acquisition used 50 training-threshold worlds. Evaluation used 44 development worlds with four targets each: 176 targets per policy/budget. The 36 test worlds were not scored.

Model: 35,074 learned parameters, width 64, random initialization. Initial training: 512 demonstration episodes, up to six inquiries, 500 AdamW updates. Each acquisition child: 150 updates. Both children started from the same parent with newly initialized AdamW; this is a matched fine-tuning comparison, not continuation of the parent's optimizer.

The teacher uses only training hypotheses. Learned inference receives public observation history, target input features, candidate features and legal mask; it does not receive the target label or teacher posterior. The current same-process API boundary is not a security-isolated examiner.

Primary development readout at six inquiries:

| Seed | Random | Coverage | Learned inquiry | Learned minus random, world-bootstrap 95% interval |
|---|---:|---:|---:|---|
| 701 | 51.14% | 52.27% | 64.77% | +13.64 points; [3.41, 23.31] |
| 702 | 53.41% | 59.09% | 60.23% | +6.82 points; [-1.72, 15.91] |

Seed 702's interval overlaps zero. Against coverage, the second seed's gain is only 1.14 points, with interval [-4.55, 6.83]. These are exploratory intervals conditional on each trained seed, not a multiplicity-controlled confirmation or strong evidence of a generally superior learning algorithm.

Memory-disabled and uncertainty-selection controls also ran. Inspect the full outcomes before attributing a gain to memory, inquiry learning or representation. There is not yet a predictor-only training ablation.

## Acquisition and retention findings

At six inquiries under the learned policy:

| Seed | Parent | New-only child | Equal-replay child |
|---|---:|---:|---:|
| 701 | 64.77% | 62.50% | 59.09% |
| 702 | 60.23% | 65.91% | 60.23% |

The mean does not establish successful consolidation. For seed 702, equal replay reduced conjunction accuracy from 83.33% to 62.50%, a 20.83-point loss on that family's 48 development queries. New-only acquisition improved threshold performance while reducing parity accuracy. In seed 701, neither child improved the overall learned-policy result. Calibration/Brier scores also need attention, especially on parity.

More extreme worst-family changes exist across the full set of policies and inquiry budgets; they are not interchangeable with the fixed six-query learned-policy comparison above. No promotion was claimed or performed. W06 must test retention explicitly rather than assuming a 50:50 replay mixture is sufficient.

## Exact algorithmic limitation

Independent tests verify a two-coefficient parity case: either first inquiry gives zero information about the scored XOR target; after the first observation, the second supplies one bit. This establishes the inadequacy of one-step information gain for valuing some preparatory actions. It does not establish that a particular neural alternative solves them. W04 is assigned that experiment.

## Verification and operational scope

The delegated discovery suite reported 23 passing tests after prototype closure. Tests cover semantic splits, inquiry legality, vectorized teacher agreement with an independent implementation, padding, model updates, tensor identity, checkpoint mismatch rejection, local continuation, statistics and comparison behavior. Later verification additions must retain their own source/test identities.

Chief integration verification subsequently passed **39 BRAMASTRA tests and 21 subtests**, including the original model/experiment/receipt tests and new discovery tests; import boundaries also passed. Review corrected cross-stage label/family validation and aligned parent/child point estimates with equal-world bootstrap weighting. An unequal-cluster regression and row-order-invariance check were added. Original run source snapshots and comparisons remain unchanged; the corrected current comparison implementation has a different source identity. Each executed development world has four targets, so changing row versus equal-world point weighting does not change those historical point estimates.

All six main training arms mutated parameters and passed the recorded **same-process CPU** next-update continuation check, including optimizer and sampler state. Fresh-process, accelerator and external-storage restore remain unverified.

Recorded campaign elapsed times were approximately 13.95 and 30.40 seconds, excluding preceding software implementation/test work. No TPU hours were used. The small smoke run establishes execution, not capability. A bits-three smoke attempt lacked threshold acquisition worlds after filtering/splitting; it is preserved as a failed configuration, not a negative learning result.

## Evidence and reproduction

- [Seed 701 manifest](../../artifacts/bramastra/discovery_dev_701/manifest.json), [results](../../artifacts/bramastra/discovery_dev_701/result.json), [parent/child comparison](../../artifacts/bramastra/discovery_dev_701/comparison.json).
- [Seed 702 manifest](../../artifacts/bramastra/discovery_dev_702/manifest.json), [results](../../artifacts/bramastra/discovery_dev_702/result.json), [parent/child comparison](../../artifacts/bramastra/discovery_dev_702/comparison.json).

These relative paths are resolved from `engineering/reports`; artifact links are checked during integration. Each run stores exact source snapshots, world/data identities and per-target outcomes. Local `.pt` files are excluded from Git; manifests alone do not make those checkpoints available remotely. Re-run training to reconstruct a local model when weights are unavailable.

```powershell
.venv/Scripts/python.exe -m bramastra_lab.discovery.run --output artifacts/bramastra/YOUR_NEW_RUN_ID --seed 701 --episodes 512 --steps 500 --adapt-steps 150 --eval-worlds 32 --targets 4 --width 64 --budget 6 --threads 2
.venv/Scripts/python.exe -m bramastra_lab.discovery.compare --run artifacts/bramastra/YOUR_NEW_RUN_ID
```

Use a unique output directory. For exact historical reproduction, use the run's source snapshot and recorded environment; current source may include later validation fixes.

## Chief decision

Accept the prototype as a bounded development instrument. Do not promote it to the integrated system or claim durable self-improvement. Prioritize W01/W02/W08 contracts and qualification, W04 delayed-information learning, and W06 retention. Preserve the favorable and unfavorable evidence together.
