# AN-RA `iterate500`

`iterate500` is the practical AN-RA frontier branch for Colab iteration.

It keeps the same AN-RA systems from the larger frontier branch - HAL, ESV, RIM, DSTP, MoD, cognition, memory, agents, runtime, evaluation, and ThirdEye - but uses a 500M-class model so experiments can actually start on constrained Colab runtimes.

The only public trainable profile is:

```text
frontier
```

## Model Specs

Current built model size:

```text
499,167,019 parameters
```

Core transformer accounting:

```text
496,857,600 transformer parameters
```

| Item | Value |
| --- | --- |
| Profile name | `frontier` |
| Model class | `CausalTransformerV2` |
| Vocabulary | `8,209` tokens |
| Embedding / hidden size | `1280` |
| Transformer layers | `28` |
| Query attention heads | `16` |
| KV heads | `4` |
| Head dimension | `80` |
| Context length | `1024` |
| Base / target sequence length | `1024` |
| SwiGLU hidden size | `3456` |
| Dropout | `0.0` |
| Embeddings / LM head | tied |
| HAL | enabled |
| ESV / RIM / DSTP / MoD | enabled |
| Checkpoint | `anra_frontier_500m.pt` |

MoD layers:

```text
4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26
```

## Why This Branch Exists

The 908M branch proved too large for free Colab in practice. `iterate500` is for fast, real experiments:

- Get loss steps running instead of dying before step 1.
- Test all AN-RA subsystems with the same wiring as the larger frontier branch.
- Let ThirdEye collect evidence across many short sessions.
- Use findings to improve data, HAL, replay, and training before scaling back up.

## Colab T4

Use:

```text
notebooks/AN_RA_T4_TRAINING.ipynb
```

Direct command:

```python
%cd /content/An-Ra-the-new-AGI

!python scripts/build_brain.py \
  --data_path training_data/anra_training.txt \
  --checkpoint_path anra_frontier_500m.pt \
  --model-size frontier \
  --batch_size 1 \
  --optimizer adafactor \
  --max_minutes 180
```

## Colab TPU

Use:

```text
notebooks/AN_RA_TPU_TRAINING.ipynb
```

Direct command:

```python
%cd /content/An-Ra-the-new-AGI

!python scripts/build_brain_tpu.py \
  --data_path training_data/anra_training.txt \
  --checkpoint_path anra_frontier_500m.pt \
  --model-size frontier \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --optimizer adafactor \
  --max_minutes 180 \
  --log_every 1
```

TPU notes:

- The TPU trainer is a dedicated PyTorch/XLA path, not the CUDA trainer copied onto a TPU runtime.
- PyTorch gradient checkpointing is disabled on TPU/XLA because the Colab XLA torch build does not expose `torch.xla` for `torch.utils.checkpoint`.
- RIM spectral-norm parametrizations are materialized before TPU training to avoid XLA memory blowups while preserving the current normalized weights.
- T4 remains the safer debug path. TPU may be faster after compilation if memory holds.

## Resume

The checkpoint is mirrored to:

```text
/content/drive/MyDrive/AnRa/v2/checkpoints/anra_frontier_500m.pt
```

Train sessions sequentially. Do not run T4 and TPU at the same time on the same checkpoint.

## Data

The same data philosophy is kept from the larger frontier experiments:

| Layer | Purpose |
| --- | --- |
| Own / identity | Preserve AN-RA identity and owner-specific continuity |
| Teacher / instruction | Usable assistant behavior and reasoning traces |
| Symbolic / verifier | Math, code, logic, and checkable tasks |
| Replay / corrections | Hard examples and failures from prior sessions |
| DFC science | Structured hypothesis, observation, verification, and correction traces |

For Colab preparation:

```python
!python scripts/download_training_data.py --profile t4-15gb --prepare-corpus
# or on TPU:
!python scripts/download_training_data.py --profile tpu --prepare-corpus
```

## ThirdEye

ThirdEye is still enabled by default:

```text
ANRA_THIRDEYE_INTELLIGENCE=1
```

It records optimizer signals, activation/gradient/update signals, HAL hormone signals, feature activation audits, and report bundles. If telemetry overhead becomes a baseline problem, disable it temporarily:

```text
ANRA_THIRDEYE_INTELLIGENCE=0
```

## Green Flag

Use this branch for the next real Colab experiment. It is not the final scale target; it is the branch designed to make iteration work.
