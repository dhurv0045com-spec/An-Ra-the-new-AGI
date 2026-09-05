# Evidence audit for BRAMASTRA

Date: 2026-09-05. Read-only inspection of historical branches and receipts; no historical experiments were rerun. Main design: [BRAMASTRA.md](../../BRAMASTRA.md).

## Audited identities

| Branch | Commit |
|---|---|
| BRAMASTRA at start / triquetra | `f23f0af42d90847cf1d2c244160c8203d1995b33` |
| cymek | `3e1b8b26434f3393157de22390fcb7e0071174a8` |
| citadel | `66ceadec0e7c4e6b00f8606345bb1d2d56ef2d6f` |
| esoes | `85f44b7b449f2ee39a0e80203a2d7df04614983b` |

Paths below refer to the named immutable tree. Inspect with `git show COMMIT:path`; current files may differ. Branch tips are not necessarily the source identities used for a historical experiment.

## 1. Citadel: observed complete-answer failure, inconclusive mechanism

T1C raw receipts under `docs/citadel/tpu_receipts/t1c_session/` record:

| Arm | Total parameters | Executed capacity tokens | Objective-supervised tokens | Core exact match | Training-sample exact match |
|---|---:|---:|---:|---:|---:|
| A: whole CE, rich | 1,647,104 | 3,997,696 | 1,783,136 | 0/1,000 | 0/500 |
| B: answer CE, rich | 1,647,104 | 3,997,696 | 559,870 | 0/1,000 | 0/500 |
| C: answer CE, narrow | 1,647,104 | 3,997,696 | 409,705 | 0/1,000 | 6/500 |
| D: answer CE, rich, larger | 3,737,472 | 3,997,696 | 559,870 | 0/1,000 | 0/500 |

The receipts identify Citadel runtime `7f81efc30938e8fb154e4150f64fc8f1384050d9` and Cymek runtime `298c91ac04f756f0833a7edcf63e73af3d5af688`. Reload prediction hashes match. The cross-arm verdict is `INCONCLUSIVE`. The prose brief's 0/500 core denominators disagree with the raw 0/1,000 values.

**Static contract mismatch:** `citadel_tpu/arith_data.py` renders rows without a terminator. `citadel_tpu/t1c_run.py::_batch_tensors` directly encodes those rows and pads them. `298c91a:v5_objectives/causal_lm.py` excludes PAD targets. `citadel_tpu/calculator_eval.py::generate` expects PAD/EOS/newline or stops at eight generated characters. No explicit answer-ending token is supervised by this path. Every arm's recorded core stop histogram is `MAX_TOKENS: 1000`.

This is evidence of an untested stopping requirement, not proof that adding EOS would solve arithmetic. Wrong digit predictions remain real. The experiment cannot isolate lack of arithmetic computation from failure to emit complete, terminated answers; the brief's claim that formatting was ruled out is too strong. Do not retroactively rescore with gold answer lengths or prefix acceptance to create a success. A corrected protocol requires a new run and identity.

**Capacity accounting:** MINI's 24,576-by-64 embedding is 1,572,864 parameters, leaving only 74,240 elsewhere. MID leaves 591,744 outside its embedding. Character inputs used a small part of that vocabulary. Nominal parameter labels therefore obscure the capacity comparison. The rich corpus's 6.5M available rows also exceed the 124,928 rows consumed per arm by a large margin.

**Hardware scope:** the receipt says Colab, PyTorch/XLA 2.9, one XLA device, unspecified TPU generation, and unknown Kaggle limits. Its short calibration reports 3,782 capacity tokens/s for one setting, not B1 useful tokens/s on Kaggle. Do not extrapolate the proposed weekly corpus budget from this number.

## 2. Triquetra: useful controls, weak substrate, no general impossibility result

At `f23f0af`, `output/query_value_evidence_dev.json` records raw candidate rank-one accuracy 25.0% for four candidates and a query-control statistic with a confidence interval spanning zero. `output/query_value_evidence_dev_rep.json` repeats the weak pattern. Candidate ranking and free generation are different readouts; neither licenses a claim that the model internally selected the right answer and merely failed to express it.

The entity/value factorial receipts report stronger repair from repeating the correct value than from repeating its entity-value pair. Those are evaluator-side, answer-bearing interventions. Their development effects challenge the earlier addressing explanation; they do not establish a legal deployed repair policy or a universal mechanism of cognition.

`output/structural_ood_e5.json` and `output/readiness_v2_calibrate_30400.json` retain negative/floor-limited outcomes. The latter is calibration with only 12 cases per primitive; it reports `PRIMITIVE_CANARY_FAILED`. This weak checkpoint cannot distinguish all proposed mechanisms. Preserve it as a negative control rather than using it to ban an architecture or objective.

`AN_RA_PROGRAM.md` also preserves corrections to unsupported claims, including an unverified +0.669-nat query-lift claim. Earlier tiny SFT-transfer summaries are historical leads, not newly verified results in this audit. Broad phrases such as "could not use information in context at all" exceed a narrow failed probe battery.

## 3. ESOES: sound mechanical work, delayed learning question

At `85f44b7`, the specifications acknowledge that many architecture, tokenizer, data, and optimizer decisions remain provisional. Kernel timing, precision, initialization, geometry, and 3–10-update continuation checks are bounded engineering evidence. They do not select a cognitive recipe.

`e3_data_objective/plan.py` restricts its initial cognition fractions to 5%, 15%, and 30%, and requires an `e2_winner_sha256`. It therefore delays the central data test behind architecture selection and initially lacks a zero-cognition control. BRAMASTRA adds that control and a format/copy control before any tournament.

Scorer null tests exposed shortest-token preference. Retain that finding, but do not require a universal candidate-likelihood correction before training with direct, verifiable answer generation. A failed scorer does not refute CE learning.

## 4. Cymek: software worth considering, not inherited scientific authority

At `3e1b8b2`, executable model/training/data/checkpoint components and bounded canary receipts exist. This contradicts older documents that still describe the entire trainable path as absent. Its binding v2 generator offers useful paired examples and shortcut controls. Neither production-path existence nor generator qualification demonstrates learned transfer.

Reuse only the pieces needed by the next experiment after checking their contracts. Pin exact source identities. Do not merge entire branches to obtain a small utility. Historical state summaries and nominal counts need reconciliation with their current receipts before citation.

## 5. External evidence used narrowly

- [Hoffmann et al., Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) studies how model/data scale relates to training compute. It informs budgeting; it does not establish an AGI threshold or the right scale for An-Ra.
- [Ye et al., Physics of Language Models, Part 2.1](https://arxiv.org/abs/2407.20311) motivates controlled synthetic investigations of reasoning. Such investigations do not substitute for transfer outside their generators.
- [Scialom et al., Fine-tuned Language Models are Continual Learners](https://arxiv.org/abs/2205.12393) reports continual task acquisition in its pretrained setting. This motivates measuring acquisition and retention separately; its success cannot be assumed for our randomly initialized small core.
- [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) and [the TPU v5e-8 announcement](https://www.kaggle.com/product-announcements/607202) establish why historical hardware assumptions must be checked against the current allocation. Neither verifies this owner's available weekly quota.

## Consequences

Keep provenance, actual-weight-update checks, full continuation state, counterfactual controls, held-out boundaries, and preserved negatives. Reopen model size, tokenizer size, data dose, training objective alternatives, architecture exclusions, and the old phase ordering. Require a complete-answer positive control before interpreting the next model failure.
