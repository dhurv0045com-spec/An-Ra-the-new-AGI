# AN-RA `iterate900`

`iterate900` is the AN-RA branch for one focused experiment:

**train and measure a 900M-class AN-RA frontier model on Colab accelerators,
while keeping the full feature stack from the larger experimental system.**

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

## Runtime Paths

This branch now has two explicit training paths:

| Runtime | Notebook | Trainer | Use when |
| --- | --- | --- | --- |
| T4 CUDA GPU | `notebooks/AN_RA_T4_TRAINING.ipynb` | `scripts/build_brain.py` | You selected a T4 GPU in Colab |
| TPU / PyTorch-XLA | `notebooks/AN_RA_TPU_TRAINING.ipynb` | `scripts/build_brain_tpu.py` | You selected a TPU runtime in Colab |

The two paths use the same `frontier` model profile, tokenizer contract, data
mixture, checkpoint name, Drive checkpoint folder, HAL/ESV/RIM/DSTP/MoD model
features, and ThirdEye evidence hooks. They differ only in runtime mechanics:
CUDA mixed precision for T4, PyTorch/XLA device loading, optimizer stepping, and
checkpoint saving for TPU.

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
  --batch_size 1 \
  --max_minutes 90
```

## Google Colab TPU Setup

Use this when Colab gives you a free TPU runtime. TPU is not CUDA, so do not run
the T4 notebook on it. Use the TPU notebook:

```text
notebooks/AN_RA_TPU_TRAINING.ipynb
```

The notebook does this automatically:

1. Mounts Google Drive.
2. Clones or updates the `iterate900` branch.
3. Installs PyTorch/XLA for TPU.
4. Installs AN-RA without overwriting the TPU torch stack.
5. Downloads/prepares the current training data profile.
6. Runs the TPU bootstrap report.
7. Starts training with the dedicated XLA trainer.
8. Saves `anra_frontier_900m.pt` locally and mirrors it to:

```text
/content/drive/MyDrive/AnRa/v2/checkpoints/anra_frontier_900m.pt
```

Direct TPU command:

```python
%cd /content/An-Ra-the-new-AGI

!python scripts/build_brain_tpu.py \
  --data_path training_data/anra_training.txt \
  --checkpoint_path anra_frontier_900m.pt \
  --model-size frontier \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --optimizer adafactor \
  --max_minutes 180 \
  --log_every 1
```

TPU notes:

- The first optimizer step can take several minutes because XLA compiles the
  graph. After compilation, step logs should appear regularly.
- The TPU path defaults to `adafactor` because AdamW optimizer state is very
  large for a 900M model. You can use `--optimizer adamw`, but it needs much more
  memory and creates larger checkpoints.
- ThirdEye intelligence telemetry is attempted by default. If it adds too much
  overhead during a baseline run, set `ANRA_THIRDEYE_INTELLIGENCE=0`.
- This is not an architecture change. It is the same 900M-class frontier model
  running through PyTorch/XLA.

## Data Requirements

The checked-in starter data is enough to test the pipeline, but it is not enough
to make a 900M model broadly capable.

Tokens are the unit of training exposure. A token is a small piece of text/code,
and every optimizer step teaches the model from a batch of tokens. Loss is a
measurement of prediction error on the current training distribution; it is not a
complete intelligence score, and the goal is not simply "loss = 0.1". A very low
loss on a small repeated dataset can mean memorization. For this branch, the goal
is: broad data, stable identity, improving held-out evals, and ThirdEye evidence
showing which subsystems helped.

Use these rough targets:

| Goal | Token target | Meaning |
| --- | ---: | --- |
| Smoke test | `1M - 10M` tokens | Proves the trainer runs |
| First useful experiment | `100M - 500M` tokens | Loss curves and feature checks become meaningful |
| Serious continued pretraining | `2B - 5B` tokens | Good for comparing AN-RA features on T4 sessions |
| Near compute-optimal 900M training | `18B - 20B+` tokens | Around 20 tokens per parameter; too large for a quick single-T4 run |

A single Colab T4 is useful for iterative experiments, continuation runs, and
feature comparisons. It is not a fast way to complete a full 20B-token pretrain
from scratch. For that, use many sessions, remote workers, or a larger GPU pool.

## Data Philosophy

AN-RA should not be trained as a pile of random internet text. It should be
trained as layers:

| Layer | Meaning | How it is used |
| --- | --- | --- |
| Identity | Who AN-RA is, its purpose, style, continuity, owner facts, and principles | Replayed every session so broad data does not erase identity |
| Skill | Code, math, science, writing, tool use, verification, debugging | Trains capability |
| World Knowledge | High-quality educational/web text, science, history, philosophy, biotech, neuroscience, robotics | Gives breadth |
| Reasoning | Math, logic, causal examples, verifier traces, DFC records | Teaches structured thought |
| Agency | Task traces, planning, self-checking, memory use, safe autonomy | Teaches action loops |
| Evidence | Evaluations, hard examples, replay corrections, ThirdEye reports | Teaches the next experiment |

The identity layer should appear in every session, but not as blind repetition of
one tiny file. The trainer mixes identity, owner, teacher, symbolic, replay, and
frontier DFC examples; hard examples and corrections are fed back as replay so
the model continues from previous sessions without only memorizing yesterday's
data.

Recommended domain mix for AN-RA:

| Domain | Target Share | Purpose |
| --- | ---: | --- |
| High-quality web / education | `35% - 45%` | General language, facts, explanations |
| Math and formal reasoning | `10% - 15%` | Stepwise problem solving |
| Code | `10% - 15%` | Tool use, structure, debugging, exactness |
| Science / papers / summaries | `10% - 15%` | Technical concepts and hypothesis language |
| Instruction / dialogue | `10% - 15%` | Usable assistant behavior |
| AN-RA identity / owner data | `5% - 10%` | Identity, style, goals, continuity |
| Verification / DFC / tool traces | `5% - 10%` | Grounded reasoning and ThirdEye evidence |

The built-in downloader already knows these buckets:

```python
%cd /content/An-Ra-the-new-AGI

# See what the Colab-friendly profile will download.
!python scripts/download_training_data.py --profile t4-15gb --dry-run

# Download and convert all current buckets into trainer-readable files.
!python scripts/download_training_data.py --profile t4-15gb --prepare-corpus

# Optional: publish token shards and a licensed token inventory for readiness checks.
!python scripts/download_training_data.py --profile t4-15gb --bucket base --publish-token-shards
```

The notebook runs the `t4-15gb` profile automatically on the first data cell.
Use `FORCE_DATA_REBUILD = True` only when you intentionally want to rebuild the
local Colab data.

Current built-in sources include:

| Source | Domain | Use |
| --- | --- | --- |
| `HuggingFaceFW/fineweb-edu` `sample-10BT` | education/web | best first base corpus |
| `togethercomputer/RedPajama-Data-V2` sample | web | extra broad web text with quality signals |
| `HuggingFaceH4/ultrachat_200k` | dialogue/instruction | conversation behavior |
| `openai/gsm8k` | math | grade-school math reasoning |
| `lighteval/MATH` | math | harder math reasoning |
| `microsoft/orca-math-word-problems-200k` | math | teacher-style math |
| `meta-math/MetaMathQA` | math | math instruction |
| `WizardLMTeam/WizardCoder_evol_instruct_110k` | code instruction | code reasoning |
| `laion/Scientific-Summaries` | science | scientific summaries |

Good next sources to add after the first experiment:

| Source | Why |
| --- | --- |
| `allenai/dolma` | broad open corpus across web, academic text, code, books, and encyclopedic data |
| `bigcode/the-stack-v2` | large source-code corpus with provenance and license metadata |
| FineWeb-Edu larger samples | scale from `10BT` toward `100BT` when storage allows |

Keep a license manifest for every source. For mixed-license instruction
collections, use them only for research unless the source-level licenses are
audited.

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

## Multi-Session Training

A 900M model will not finish in one Colab session. Train it as repeated sessions:

1. Open the notebook.
2. Mount Drive.
3. Confirm the first cell prints `cuda available: True` and an NVIDIA GPU name.
4. Run setup and data cells.
5. Train for `90`, `120`, or `180` minutes.
6. Let the trainer save the checkpoint and reports.
7. End the runtime.
8. Next time, run the notebook again.

If Colab says TPU, v5e, CPU, or `cuda available: False`, stop immediately and
change runtime type to `T4 GPU`. The PyTorch trainer does not train this branch
on TPU.

The trainer restores `anra_frontier_900m.pt` from Google Drive when present and
continues from its saved `global_step`, optimizer, scheduler, scaler, best loss,
and replay state. It also mirrors the frontier checkpoint during long sessions.

Useful session-length command:

```python
!python scripts/build_brain.py \
  --data_path training_data/anra_training.txt \
  --checkpoint_path anra_frontier_900m.pt \
  --model-size frontier \
  --batch_size 1 \
  --max_minutes 180
```

Drive storage note: a 900M training checkpoint can be large because it includes
model weights and optimizer state. With a 15GB Drive limit, keep only the latest
frontier checkpoint and do not store many old checkpoints or raw datasets in
Drive.

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
