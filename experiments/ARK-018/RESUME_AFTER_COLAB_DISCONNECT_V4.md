# ARK-018 V4 — Resume after Colab disconnect

## Purpose

This document defines an **operational resume only**. It does not modify the preregistered ARK-018 treatment, model, data schedule, seeds, evaluation, learning-rate schedule, horizon, or claim boundary.

The original scientific execution remains frozen at commit:

`fb0420b7a46521a5f14d125564078ca1c6336d78`

The resume launcher must check out that exact commit before executing `run_ark018_science_birth_v4.py`.

## Durable Google Drive state

Input corpus:

`/content/drive/MyDrive/genisis-arkenstone/data_15.parquet`

Durable experiment root:

`/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1/`

Subdirectories used by the frozen runner:

- `prepared/` — tokenizer and token caches;
- `checkpoints/` — exact model/optimizer/scaler/RNG checkpoints;
- `results/` — partial/final receipts and result ZIP.

The frozen runner already implements cross-session resume. `prepare_all()` reuses the prepared Drive cache when its bound science/Birth identities match. `train_arm()` loads the durable checkpoint for an arm when present and resumes from its recorded optimizer step. Completed post-training evaluations are also reused when their result JSON exists.

## Important checkpoint cadence

Pretraining partial result JSON is written every 500 optimizer steps, but the heavy model+optimizer checkpoint is written every 1,000 optimizer steps (and at final horizon).

Therefore a result export showing an arm at step 6,500 does **not** imply that the exact resumable model state exists at step 6,500. If the last durable checkpoint is step 6,000, the next Colab session correctly replays steps 6,001–6,500 and then continues. Replaying from the last exact checkpoint is preferable to attempting an invalid reconstruction from metrics alone.

## Resume behavior

The dedicated launcher `experiments/COLAB/arkenstone_ark018_resume_v4.ipynb`:

1. mounts Google Drive;
2. checks the exact science input and durable experiment root;
3. inspects result JSON plus checkpoint step numbers and prints a per-seed/per-arm resume table;
4. clones the repository and checks out the frozen execution commit `fb0420...`;
5. compiles the frozen implementation;
6. invokes the same V4 runner in `--mode all`.

Although `--mode all` is used, completed work is not retrained: the frozen runner loads completed checkpoints/results and its training range is empty for arms already at the horizon. Incomplete arms resume from their last exact checkpoint; untouched arms start normally.

No `--force-prepare` is used.

## Operator rule

Do not delete or rename the existing `ARK018_SCIENCE_BIRTH_V1` Drive directory. Do not edit checkpoints or partial JSON. Reopen the resume notebook with a T4 GPU and use **Runtime → Run all**.

After all eight seed×arm pretraining runs and post-training evaluations finish, the expected final artifact is:

`/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1/results/ARKENSTONE_ARK018_SCIENCE_BIRTH_RESULTS.zip`

A Colab disconnect is not itself scientific failure as long as execution restarts from a receipt-bound exact checkpoint.