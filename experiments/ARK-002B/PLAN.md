# ARK-002B — INDEPENDENT T2 REPLICATION (preregistered; PLAN committed before training)

PLAN_COMMIT_SHA: recorded in RESULT.json after execution; this file is committed
and pushed BEFORE any training run of this experiment.

## Question
Does the memorize-first -> generalize-later transition (ARK-001/002a, seed 13)
reproduce on independent seeds under a commutation-free frozen dataset manifest?

## Design
- Dataset: FROZEN manifest (experiments/ARK-002B/TASK_MANIFEST.json,
  split_sha256 0dd930569704..., generated with dataset_seed 13 and frozen BEFORE
  training; commutation-filtered: zero sorted-pair overlap between train and
  test). Dataset membership is IDENTICAL for every run.
- Variance separation: model-INIT seed and data-ORDER seed are independent
  parameters. Runs: (init 29, order 29), (init 47, order 47), plus the
  historical seed 13 result kept for comparison only.
- Model/optimizer/data/budget/cadence identical to ARK-002a: Micro 4L/128w/4H,
  answer-only CE, AdamW 1e-3 (0.9/0.95/1e-8, wd 0.1), batch 64, eval every 200
  steps, wall box 1800s, CPU single-process.
- Metrics: sustained M99/G50/G90/G95 (>=3 consecutive evals), post_mem_delay_90,
  exposure_ratio_90, OOD-AUC after M99, per-position exact, token accounting.

## Predictions
- P-repro: qualitative memorize-first -> generalize-later transition reproduces
  on both seeds (G90 DEMONSTRATED within box on >=1 fresh seed).
- P-quant: G90 dose within ~2x of seed 13's ~9000-step transition.

## Falsification / stop
If neither fresh seed shows any OOD emergence by box end -> transition is
seed/split-unstable; STOP mechanistic claims; diagnose instability.

## Novelty test
Confirms/refutes REPLICATION status of the program's central micro-scale
discovery; no new mechanism claimed by this experiment alone.
