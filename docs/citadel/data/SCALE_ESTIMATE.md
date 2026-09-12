# SCALE ESTIMATE

## Current first-party corpus
- 6.42M unique rows, ~96.8 MB
- Estimated BPE tokens: ~25-50M (depends on tokenizer)

## 500M demand
- 500M consumed training tokens
- Expected unique supply from current corpus: ~25-50M tokens
- Replay factor at 500M: ~10-20x (PATHOLOGICAL for single-epoch)
- Verdict: **DATA_VOLUME_IS_A_BOTTLENECK** for 500M

## What this means
The current arithmetic corpus cannot supply 500M unique tokens.
Multi-epoch replay would be required, which risks memorization without
generalization (the exact failure T1D demonstrated).

## Extrapolations (ESTIMATES ONLY)
| Scale | Storage | Unique tokens | Replay at 500M |
|---|---|---|---|
| 100 MB | ~25-50M tokens | ~25-50M | 10-20x |
| 1 GB | ~250-500M tokens | ~250-500M | 1-2x |
| 5 GB | ~1.25-2.5B tokens | ~1.25-2.5B | <1x (no replay needed) |

## Bottleneck order
1. Corpus generation/acquisition (getting real diverse data)
2. Near-dedup processing throughput
3. Contamination scan throughput
4. Packing throughput
5. Storage (not a bottleneck: 500M tokens ≈ 2-5 GB text)

## Clear label
ALL NUMBERS ABOVE ARE ESTIMATES FROM SAMPLE MEASUREMENTS.
NOT measured end-to-end production throughput.
