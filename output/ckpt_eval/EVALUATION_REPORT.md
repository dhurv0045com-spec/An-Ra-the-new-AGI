# Checkpoint Evaluation Report: step-20000 vs step-30400 (local RTX 4050)

**Date:** 2026-08-22 · **Machine:** RTX 4050 Laptop 6GB, torch 2.11.0+cu128
**Protocol:** identical 21-prompt battery + sampling-health + logit diagnostics;
models run strictly sequentially with full VRAM release between checkpoints
(measured 4.91 GiB free after each release; 0.68 GiB allocated per model).

## Headline numbers

| Domain | step-20000 | step-30400 |
|---|---|---|
| Conversation (mean repetition) | 0.74 | **0.88 worse** |
| Factual exact-match | 0/4 | 0/4 |
| Instruction/echo exact-match | 0/4 | 0/4 |
| Arithmetic exact-match | 0/3 | 0/3 |
| Tool-call request emitted | 0/2 | 0/2 |
| Code continuation repetition | 0.92 | 0.79 |
| Mean greedy collapse | " the the the the…" | " ditj ditj ditj…" |
| Sampling distinctness (4 samples) | 4/4 | 4/4 |

## What each model actually does

**step-20000** collapses to `" the"` loops on nearly every prompt. Greedy top-1
is a whitespace token at p=0.99, entropy 0.1–0.7 bits — the distribution is a
spike. It is not "thinking then choosing wrong"; the argmax *is* the degenerate
token from position one.

**step-30400** collapses to `" ditj"` loops — same failure shape, different
attractor token, higher confidence still (margin up to 8.26, entropy down to
0.02 bits on "capital of France"). On echo tasks it is *more* uncertain than
20k (10.2 bits ≈ near-uniform over 32k vocab) and picks garbage like `'ton'`.
Sampling stays diverse in both — the damage is concentrated in the greedy head.

## Reverse-engineered failure chain

1. **First-token attractor.** Both models put ~99% probability on ` `
        (whitespace) after any sentence-like prompt. Generation never escapes.
2. **Attractor drift, not learning.** Between 20k→30.4k the attractor moved
        from a common word (`the`) to a corpus fragment (`ditj`). The TPU phase
        did not add capability; it re-tuned which memorized token wins.
3. **Confidence increased as quality fell.** Margins grew (4.88→8.26) while
        outputs got worse — classic repeat-pass memorization sharpening the
        same wrong spike. This confirms the WSD-schedule diagnosis.
4. **No domain survived.** Chat, facts, echo, arithmetic, tool-calls, code:
        all zero. The substrate cannot perform *any* of these behaviors yet,
        so neither checkpoint can chat or call tools.

## What this proves / does not prove

Proves: both checkpoints are below the language floor; degradation 20k→30.4k
is real, measured, and mechanistically consistent with constant-LR repeat
passes; the step-20000 artifact is the better anchor.

Does NOT prove: the corpus is bad; the architecture is bad; or more data can't
fix it. GPT-2-class models need billions of diverse tokens before greedy
generation stops collapsing — this pair has seen ~500M unique tokens once.

## Recommended next action

Unchanged and now evidence-backed: resume Phase B from **step-20000**, fresh
packs through `train_tpu.py` (WSD decay aligned to each pack), gate on the
capability probe before any SFT.
