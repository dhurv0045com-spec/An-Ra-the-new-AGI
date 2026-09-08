# D02 bounded execution review

Date: 2026-09-08. Status: completed as a two seed CPU development comparison; this is not full W04 and is not AGI evidence.

## Execution

Run artifact: `artifacts/bramastra/d02_matched_teaching_20260908_luna/`.

Frozen command:

```text
.venv\Scripts\python.exe -c "from bramastra_lab.research.learning.inquiry.run_d02 import run; import json; result=run('artifacts/bramastra/d02_matched_teaching_20260908_luna'); print(json.dumps({'status':result['status'],'elapsed_seconds':result.get('elapsed_seconds'),'source_integrity':result.get('source_integrity'),'results':result.get('results')}, sort_keys=True))"
```

The run completed in 26.78 seconds on CPU with two threads. Configuration was bits=4, 256 episodes, horizon/budget=2, width=32, 500 AdamW updates, batch size 32, learning rate 0.001, policy weight 0.25, evaluation seed 17, four targets per development world, and 1,000 world-cluster bootstrap resamples. The identical semantic inventory contained 73 training, 23 development, and 23 untouched confirmation worlds (family counts in development: conjunction 14, parity 6, threshold 3). Confirmation worlds were not scored.

## Results

The primary contrast is depth-two learned minus one-step learned accuracy at budget 2, pairing world/target keys and evaluating each arm with its own predictor.

| Seed | Accuracy delta | Brier delta | 95% world-cluster interval | Family accuracy deltas (conjunction/parity/threshold) |
|---|---:|---:|---|---|
| 801 | 0.0000 | +0.00417 | [0.0000, 0.0000] | 0.0000 / 0.0000 / 0.0000 |
| 802 | +0.02174 | -0.00414 | [0.0000, 0.05435] | 0.0000 / +0.04167 / +0.08333 |

Both seed directories contain raw prediction/action rows, training histories, summaries, paired comparisons, and per arm tensor identities. Fixed random, coverage, and no-memory controls were evaluated at budgets 0, 1, and 2. The evaluator receives public histories and target features only; teacher scores and posterior state are training-only.

Independent chief recomputation of the raw rows and pair keys passed. At budget 2, seed 801 had 61/92 correct for each learned arm and 61/92 for coverage; seed 802 had 63/92 for one-step learned, 65/92 for depth-two learned, and 62/92 for coverage.

## Controls and integrity

The inconsistent posterior fixture was corrected: unconditioned four-world XOR returns zero gain for the unresolved query, while a separately conditioned posterior returns the intended one-bit gain. Focused inquiry tests passed: `5 passed in 8.71s`.

The runner checks identical non-gain tensors, changed teacher gain tensors, zero remaining labels, equal one-remaining labels, legal actions, target masking, duplicate evaluation keys, label/family agreement, paired initialization fingerprints, paired sampler fingerprints, and source identity stability. Initial model fingerprints matched within each seed: seed 801 `034ffab6310ede7f9cb3fc541b088340cf4270f7dda241008b72d28566dd7542`; seed 802 `81d5efa98359f8529b7dad05c16b9ad721e6d99d1b497a4740b0e50ad0508be3`. Each arm had 9,090 parameters and identical sampler fingerprints within seed.

The focused suite covers teaching horizon, legality, paired tensors, split controls, duplicate/mismatched evaluation pairs, zero-deadline timeout persistence, injected training failure persistence, and source mutation invalidation. It passed 8 tests in 8.95 seconds. The primary campaign artifact predates this runner-only hardening and remains immutable; its recorded source snapshot and completion integrity were independently verified.

Source closure SHA256 identities were captured before and after execution and matched (`source_integrity=true`):

```text
bramastra_lab/discovery/__init__.py 88223e5d990a142a0d7fbfceb0e62872189af681846f1ff29f4a6034562b16f3
bramastra_lab/discovery/curriculum.py 3aee75b525bd3eb17cc2af86d8a4f783f43bc33787a9219fa52a0a8ed26553fa
bramastra_lab/discovery/evaluation.py d3f0bb3d3e4cd6e0a5e4f7e49e9e68da794cb7c1f5722aed5fd81a41972c31fc
bramastra_lab/discovery/learner.py 59acb9004de0238b0a910f45cf087eeda633dca326eb68e8bd6e9e74ed5dea9a
bramastra_lab/discovery/statistics.py a13fb3ab869e2c9092dfafdb19cefc9ac5a2b3a4057faa5e9b7e359c9753db5d
bramastra_lab/discovery/training.py 622e04dae4a07001d90074ad0f8ce738d23a526c7751cd138a7c87de3ee852c7
bramastra_lab/discovery/worlds.py 45b568c49054801fd8ddce8d95047b9fe398dcd7fa8fa79c5992de837877e2b8
bramastra_lab/research/learning/inquiry/__init__.py 715c95e02544d71850617ff60e9c46a3e891dfc18f62458019e2c7f82c205c4f
bramastra_lab/research/learning/inquiry/run_d02.py 930a2c6eba4664789da6175780b380738b8f633d128cfcf3d86a02ad0d981c5f
bramastra_lab/research/learning/inquiry/teaching.py c9c896aaac1cead038f8f99b51a0ab1a8f7cca646746863337a86cce104baeba
```

## Interpretation and limits

The point estimates differ by seed and the seed 802 interval includes zero. The result does not establish a reliable depth-two benefit. Uniform legal histories make this an off-policy teacher-imitation coverage study; it does not test autonomous exploration. Only three threshold development worlds make that family readout unstable. The study is limited to this finite synthetic rule-learning development setting. PPO, outcome-trained policy, distractor transfer, strategy validation, retention, and broader self-improvement remain unrun W04 work.
