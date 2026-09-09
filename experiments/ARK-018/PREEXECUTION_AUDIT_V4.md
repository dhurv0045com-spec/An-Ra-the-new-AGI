# ARK-018 V4 — PRE-EXECUTION AUDIT

## Verdict

**READY_FOR_OPERATOR_GPU_SMOKE / IMPLEMENTED / NOT EXECUTED / EXTERNAL DATA NOT YET RUNTIME-BOUND.**

The scientific plan, score definition, objective probe battery, data/model implementation, resumable runner and pinned Colab launcher are present. No scientific result exists yet. The first operator run must still pass the full dataset-hash preparation gate and CUDA smoke before any model update is scientifically admissible.

## Frozen execution

Pinned execution commit used by the notebook:

`fb0420b7a46521a5f14d125564078ca1c6336d78`

Pinned launcher:

`experiments/COLAB/arkenstone_ark018_science_birth_v4.ipynb`

Final wrapper:

`experiments/ARK-018/run_ark018_science_birth_v4.py`

Scientific implementation:

- `run_ark018_science_birth_v3.py`
- `ark018_v3_common.py`
- `ark018_v3_binding_fast.py`

The notebook runs `python -m py_compile` on all four implementation files before data preparation and then runs a CUDA smoke gate before full training.

## External science source check

Frozen operator path from the provided Drive screenshot:

`/content/drive/MyDrive/genisis-arkenstone/data_15.parquet`

The folder spelling is intentionally `genisis-arkenstone`.

Expected upstream object is Common Pile consolidated Parquet `peS2o/data_15.parquet` with frozen SHA256:

`b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5`

The runtime must stream the full Drive object and match this hash before tokenizer creation or model training. A mismatch fails closed.

The implementation requires Parquet column `text`, records schema/rows/nulls/package versions, and does not rely on the Drive UI size as evidence of identity.

## Birth source check

Frozen repository Birth Book:

- `experiments/ARK-018/ARK018_BIRTH_BOOK.md`
- bytes: `17,906,590`
- SHA256: `8f17d092897f41df45100d227ecf7a2391ed5a5cfd3d3bcf373ad86c94b0c4f4`

The runtime recomputes both before preparation.

## Causal-design checks

PASS by source review:

- primary arms remain `SCIENCE_ONLY`, `BIRTH_NATURAL_2PCT`, `BIRTH_REHEARSAL_10PCT`, and `SCIENCE_REPLAY_10PCT_CONTROL`;
- natural arm has exactly 2 Birth slots per 100, at 25 and 75;
- rehearsal arm has exactly 10 Birth slots per 100;
- matched replay control has exactly 10 small-science replay slots at the same positions;
- normal science batches are deterministic from model seed + absolute step, so shared science steps are matched;
- Birth/replay batches advance sequentially through their frozen token streams;
- replay-control token count must match Birth tokens within 0.1%; implementation currently truncates only the final selected replay document as needed to achieve exact token matching;
- all arms receive exactly 32 x 256 = 8,192 prediction targets/update;
- both model seeds are frozen: `31801`, `31902`;
- all four arms for one seed load the same initial model state;
- all matched arms use one frozen post-tokenization horizon;
- no result-dependent early stopping changes the pretraining horizon.

## Data/evaluation firewall checks

PASS by source review:

- scientific split is by normalized-document SHA256 into 90/5/5 TRAIN/CONTROL/SEALED;
- exact duplicate texts therefore cannot cross split boundaries;
- tokenizer is trained only on scientific TRAIN;
- tokenizer is built twice and byte-identical canonical serialization is required;
- Birth Book never enters scientific CONTROL/SEALED caches;
- objective `BIRTH_BOOK_PROBES.json` is committed before execution;
- `BirthContentScore` is prospectively defined as SEALED Birth probe accuracy;
- CONTROL Birth probes remain diagnostic;
- SEALED scientific/Birth outcomes never select arm schedule, seeds or horizon.

## Model/optimization checks

Frozen conventional proxy:

- 8,192 vocab;
- context 256;
- d_model 384;
- 10 blocks;
- 6 heads;
- FFN 1,536;
- tied embedding/head;
- pre-norm causal decoder;
- AdamW `(0.9, 0.95)`, eps `1e-8`, weight decay 0.1;
- peak LR 3e-4, 2% warmup, cosine to 3e-5;
- clip norm 1.0.

The CUDA smoke requires parameter count in the planned 20–25M range, finite forward/backward, exact effective target count, same-runtime deterministic next-update reproduction, and Drive checkpoint round-trip.

## Maximum-information measurements included

The full runner collects all of the following without altering the primary treatment:

1. scientific CONTROL/SEALED NLL, perplexity and token accuracy every 500 steps;
2. Birth Book NLL diagnostic;
3. frozen paraphrased/multiple-choice Birth content probes;
4. narrow algorithmic OOD probes beyond major Birth Book table ranges;
5. full-model displacement from initialization at evaluation milestones;
6. exact per-update pre-clip gradient norm, LR, source identity and source token counts;
7. deterministic projected science-vs-Birth gradient alignment every 1,000 steps with named parameter tensors;
8. post-pretraining controlled temporary-binding acquisition and matched HIGH-vs-LOW retention;
9. secondary deterministic SciQ subset when network access is available;
10. complete receipt, checkpoint and result provenance.

## Durability / Colab disconnect handling

Drive root is frozen to:

`/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1/`

Preparation/token caches, per-seed initialization states, per-arm checkpoints, partial results and final receipts are durable in that directory.

Training checkpoints are written at least every 1,000 steps and compact partial result JSON at every 500-step evaluation. Re-running the notebook is designed to resume completed/partial work rather than restart the campaign.

A full 2-seed x 4-arm study is **not required to fit one Colab session**. Reducing arms or seeds to make one runtime look complete would weaken the experiment; exact resume is the intended execution path.

## Known limitations / red team

- This audit cannot certify the user's actual Drive bytes before Colab runtime hashes them. Screenshot filename/size is not cryptographic binding.
- CUDA smoke has not yet executed on the operator T4; therefore current status is READY_FOR_SMOKE, not GPU-validated.
- Birth content probes primarily measure internalization of book information. They are not consciousness/selfhood measures.
- Algorithmic OOD and temporary binding are narrow controlled transfer diagnostics, not general reasoning benchmarks.
- SciQ is secondary and contamination is not assumed absent.
- This ~20–25M proxy is a realism bridge, not evidence for a production-scale training law.
- Projected gradient/update diagnostics are explicitly labelled projected; only milestone full-model displacement is claimed exact.

## Green-light condition

The operator has a green light to **open the notebook and Run all**. The notebook itself withholds the green light to *training* until:

1. exact science hash preparation passes;
2. static compile passes;
3. CUDA smoke passes.

Only then does it enter the full four-arm/two-seed campaign.
