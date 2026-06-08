# 45O — Ouroboros

**Component 16/19 · `ouroboros`**

Named for the serpent eating its tail: **reason about reasoning**. Multi-pass refinement when a first answer is not good enough — with pass weights driven by HAL emotional state, not a fixed "think harder" knob.

---

## Pass model

| Pass | Character | When weighted up |
| --- | --- | --- |
| **0** | Fast intuitive answer | High endorphin — trust first instinct |
| **1** | Critical review | Default |
| **2** | Deep constraint reasoning | High cortisol / adrenaline — slow down |

HAL sets weights each turn. Under threat or uncertainty, the stack shifts toward pass 2 automatically.

**Gates:** `pass_gates.py` can reject an answer and loop until confidence criteria hit (milestone configs).

---

## Implementations

| Path | File | Notes |
| --- | --- | --- |
| NumPy | `ouroboros_numpy.py` | CPU-friendly, default for smoke |
| Torch | `ouroboros.py` | Heavier, GPU-oriented |
| Adaptive | `adaptive.py` | Chooses depth from prompt difficulty |

Checkpoint: `anra_v2_ouroboros.pt` (may be absent on fresh clone).

---

## Commands

```bash
# Integrated status
python anra.py --phase3-status

# Local tests
cd "phase3/ouroboros (45O)"
python test_ouroboros.py
python train_ouroboros.py   # milestone training helper
```

---

## When to use Ouroboros

| Daily session | Milestone |
| --- | --- |
| Usually off or light | Full multi-pass + gate checks |
| Keep train loop fast | Pair with sovereignty before promote |

Do not put Ouroboros on the critical path for `train_unified --mode session` unless you mean to pay the cost every day.

---

## Integration

Feeds milestone training and hard-prompt inference. Weights exposed to HAL (`hal.py`) for pass blending.

See [`PHASE3_INTEGRATION.md`](../PHASE3_INTEGRATION.md) for full pipeline position.
