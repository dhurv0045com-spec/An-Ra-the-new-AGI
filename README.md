# AN-RA `iterate900`

`iterate900` is the AN-RA branch for one focused experiment:

**train and measure a 900M-class AN-RA frontier model on a T4 GPU, while keeping
the full feature stack from the larger experimental system.**

This branch is not for 25M training and it is not for the old 3B model. The only
public trainable profile is:

```text
frontier
```

## Model Specs

Current built model size:

```text
908,098,891 parameters
```

Core transformer accounting:

```text
904,535,040 transformer parameters
```

Architecture:

| Item | Value |
| --- | --- |
| Profile name | `frontier` |
| Model class | `CausalTransformerV2` / `causal_transformer_v3` registry |
| Vocabulary | `8,209` tokens |
| Embedding size / hidden size | `1536` |
| Transformer layers | `36` |
| Query attention heads | `16` |
| KV heads | `4` |
| Head dimension | `96` |
| Context length | `2048` |
| Base sequence length | `2048` |
| Target sequence length | `2048` |
| FFN / SwiGLU hidden size | `4096` |
| Dropout | `0.0` |
| RMSNorm epsilon | `1e-5` |
| Embeddings / LM head | tied |
| Gradient checkpointing | enabled |
| HAL | enabled |
| ESV | enabled |
| RIM | enabled |
| DSTP | enabled |
| MoD routing | enabled |

MoD layers:

```text
4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34
```

Training defaults:

| Item | Value |
| --- | --- |
| Micro batch size | `2` |
| Gradient accumulation | `16` |
| Effective batch size | `32` |
| Default session length | `90` minutes |
| Max mixed examples | `4096` |
| Precision | bf16 when supported, otherwise mixed precision fallback |
| Checkpoint | `anra_frontier_900m.pt` |

## What This Branch Is For

Use this branch when you want to run real AN-RA frontier experiments and answer:

- Is the loss improving?
- Which subsystem is active?
- Which feature appears to help or regress?
- Did HAL, ESV, RIM, DSTP, MoD, cognition, memory, agents, and evaluation remain connected?
- What should the next experiment be?

ThirdEye is connected for evidence and subsystem analysis. It does not fake
results before training. Before a campaign runs, it may show some training or
runtime systems as missing activation evidence. After training/evaluation, those
reports become the comparison record.

## Google Colab T4 Setup

Use this when you want the simple notebook-style workflow: open Colab, select a
T4 GPU, run the cells, watch losses, and save checkpoints.

1. Open Google Colab.
2. Go to `Runtime -> Change runtime type`.
3. Select `T4 GPU`.
4. Mount Google Drive.
5. Clone this branch.
6. Install requirements.
7. Run bootstrap/preflight.
8. Start training.
9. Run ThirdEye evaluation after the session.

The bootstrap installs AN-RA plus the optional ThirdEye evidence SDK, so the
notebook can run the report commands in a fresh Colab runtime.

Colab setup cell:

```python
from google.colab import drive
drive.mount("/content/drive")

REPO = "/content/An-Ra-the-new-AGI"
BRANCH = "iterate900"

![ -d "$REPO/.git" ] || git clone --branch "$BRANCH" --single-branch https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git "$REPO"
%cd $REPO
!git fetch origin "$BRANCH"
!git checkout "$BRANCH"
!git pull --ff-only origin "$BRANCH"

!python scripts/colab_bootstrap.py \
  --repo "$REPO" \
  --drive-root /content/drive/MyDrive/AnRa \
  --install \
  --model-size frontier
```

## Train On T4

Recommended direct training command:

```python
%cd /content/An-Ra-the-new-AGI

!python scripts/build_brain.py \
  --data_path training_data/anra_training.txt \
  --checkpoint_path anra_frontier_900m.pt \
  --model-size frontier \
  --max_minutes 90
```

You should see training output with step, loss, best loss, learning rate, and
checkpoint progress. This is the closest path to the training flow you used
before, but locked to the 900M `frontier` profile.

Unified dispatcher command:

```python
%cd /content/An-Ra-the-new-AGI

!python -m training.train_unified \
  --mode session \
  --model-size frontier \
  --prepare_data never \
  --data_path training_data/anra_training.txt \
  --checkpoint_path anra_frontier_900m.pt \
  --session-minutes 90
```

## Evaluate With ThirdEye

Quick report without building the full model:

```python
!python scripts/evaluate_with_thirdeye.py --profile quick --without-model
```

Full activation/evidence report with the 900M model:

```python
!python scripts/evaluate_with_thirdeye.py --profile quick
```

Reports are written under:

```text
output/v2/thirdeye/reports/anra/
```

Important outputs:

- `decision-scorecard.md`
- `scientific-report.md`
- `evidence-bundle.json`
- `decision-dashboard.html`

## Existing Notebook

The Colab notebook for this branch is:

```text
notebooks/AN_RA_T4_TRAINING.ipynb
```

Use that notebook if you want the old experience: open it in Colab, run cells,
watch the loss on screen, and let the branch handle setup/training/evaluation.

## If T4 Runs Out Of Memory

900M training on a T4 is tight. The branch already enables gradient
checkpointing. If Colab still OOMs:

```python
!python scripts/build_brain.py \
  --data_path training_data/anra_training.txt \
  --checkpoint_path anra_frontier_900m.pt \
  --model-size frontier \
  --batch_size 1 \
  --max_minutes 90
```

Keep `--model-size frontier`. Do not use `25m` or `3b`; this branch rejects
them on purpose.
