# 45O - Ouroboros

**Layer 16/19: `ouroboros`**

Ouroboros is the recursive reasoning layer. It gives An-Ra a slower path for harder prompts: read, reason, verify, then answer.

## Current Role

```text
ordinary prompt
  -> generation runtime

hard prompt
  -> adaptive pass selection
  -> semantic pass
  -> logic pass
  -> adversarial/check pass
  -> final response
```

The NumPy path is the practical CPU-friendly runtime. Torch training support exists for deeper milestone work.

## Main Files

| File | Role |
| --- | --- |
| `ouroboros_numpy.py` | CPU-friendly recursive reasoning path |
| `ouroboros.py` | Torch wrapper path |
| `adaptive.py` | Dynamic pass count |
| `pass_gates.py` | Pass specialization gates/losses |
| `weight_sharing.py` | Parameter sharing checks |
| `train_ouroboros.py` | Component training script |
| `test_ouroboros.py` | Smoke and behavior tests |

## Use

From this folder:

```bash
python test_ouroboros.py
python train_ouroboros.py
```

From the repo root, the milestone path is:

```bash
python -m training.train_unified --mode train
python scripts/train_ouroboros.py
```

## Design Contract

Ouroboros should not become daily overhead by default. It belongs where depth is worth the cost:

- complex reasoning
- self-review
- repair synthesis
- milestone refinement
- hard-example analysis

Fast paths should stay fast. Deep paths should earn their extra passes.
