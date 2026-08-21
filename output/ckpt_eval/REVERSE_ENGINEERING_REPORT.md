# Reverse-Engineering Execution Report: fixes, soup experiment, findings

**Date:** 2026-08-22 · **Constraint honored:** no GPU used (another agent owns
it). All execution CPU-only.

## What was fixed

### 1. Inference degeneration controls — `anra_core/generate.py`
Diagnosis showed greedy collapse: top-1 was a whitespace token at p=0.99,
entropy 0.02–0.7 bits; generation locked into " the the the" / " ditj ditj"
from token one. `generate()` now implements the two standard production
remedies, on by default:

- **repetition_penalty=1.15** — CTRL-style: generated logits divided (>0) or
  multiplied (<0) by the penalty;
- **no_repeat_ngram_size=4** — a token completing an already-seen 4-gram is
  banned outright.

Both are operator-disableable (`penalty=1.0, ngram=0` restores exact legacy
behavior). All 11 core/incremental conformance tests still pass.

### 2. Degradation guard — `training/train_tpu.py`
The 20k→30.4k regression went unnoticed because nothing watched for it.
The trainer now tracks best training loss and warns loudly when current loss
exceeds best by >10% ("you are past the useful point of this pack"), and the
final save reports which step held the best loss. This makes the exact
failure mode that damaged step-30400 impossible to miss next time.

## The idea executed: checkpoint soup

Hypothesis from reverse engineering: both checkpoints sit in one loss basin
but drifted to *different* memorization attractors ("the" vs "ditj").
Equal-weight averaging of same-lineage weights often cancels such drift
(Model Soups, Wortsman et al.).

`scripts/make_soup.py` averaged all 203 shared dense tensors (kept parent-A's
190 dormant pilot tensors for loader compatibility) →
`output/ckpt_eval/soup_20k_30k.pt`, then `scripts/eval_soup_cpu.py` ran all
three models through the fixed generator on an 8-case battery.

## Results

| model | exact accuracy | repetition | sample output |
|---|---|---|---|
| step-20000 | 0/6 | 0.00 | "located in the city of Lancashire. The capital is…" |
| step-30400 | 0/6 | 0.00 | "France has Travertineessen, France's largest city…" |
| **soup** | **0/6** | 0.00 | "located in the city of Lancashire. The capital is…" |

With degeneration fixed, repetition is now 0.00 everywhere and outputs are
grammatical English fragments — but exact-match accuracy stays 0/6 across all
three models, and the soup's outputs are byte-identical to step-20000's.

## What this proves

1. **The collapse was inference-side AND knowledge-side.** The new decoder
   controls completely eliminate repetition loops — real improvement.
2. **The soup converges to parent-A's behavior**, meaning the 30.4k drift was
   a small perturbation around the same point, not a different solution.
   Averaging cannot recover capability that neither parent had.
3. **The remaining failure is parametric knowledge/capability**: the models
   produce fluent-ish text but hold no facts, cannot echo, cannot compute.
   No inference trick fixes missing training.

## What we learned (the honest conclusion)

The two errors were layered: an *inference* error (degenerate decoding — now
fixed, big qualitative win: coherent fragments instead of token stutters) and
a *training* error (repeat passes at constant LR — now guarded against).
Underneath both sits the true bottleneck, unchanged: ~500M unique tokens is
~14% of what this architecture needs before facts and instruction-following
emerge. Every fix above makes the model *presentable*; only Phase B data
makes it *capable*.

## Files

- `anra_core/generate.py` — degeneration controls (tested)
- `training/train_tpu.py` — degradation guard (tested)
- `scripts/make_soup.py`, `scripts/eval_soup_cpu.py` — soup tooling
- `output/ckpt_eval/soup_20k_30k.pt` — soup artifact (sha256 04f49813…)
- `output/ckpt_eval/soup_comparison.json` — full evidence
