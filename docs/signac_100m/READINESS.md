# Signac 100M readiness

## Current verdict

**Architecture and qualification package: ready for bounded Kaggle TPU canaries. A production 100M training run is not ready or authorized.** The repository's production trainer still blocks XLA execution, and the corpus, Citadel evaluation, multi-replica resume, and measured TPU-fit gates have no qualifying receipts.

## Evidence by gate

| Gate | State | Evidence or remaining work |
|---|---|---|
| Primary model contract | Implemented | M102 is exactly 101,790,080 parameters; the model receipt is derived from `ModelSpec`. |
| TPU architecture comparison | Implemented as candidates | M102, same-depth FFN-aligned 99,332,480, and fully tiled 100,303,104 parameter geometries are emitted with content hashes and analytic costs. None is promoted by estimated tiling alone. |
| CPU model-shape smoke | Passed locally | All three exact geometries instantiate and return `[1, 4, 24576]` logits through the V5 core. This says nothing about accelerator performance. |
| Kaggle device preflight | Notebook ready; target unrun | Requires PJRT TPU and eight visible global devices, and records runtime versions, topology, and a tiny collective/BF16/RNG smoke receipt. |
| 4,096-token canary | Notebook ready; target unrun | One synthetic update per candidate; records compile-inclusive first-update latency. It does not report reliable peak memory or steady-state throughput. |
| Model/Adam restore canary | Notebook ready; target unrun | M102 128-token canary serializes model and Adam state, restores fresh objects, and checks one deterministic next update. It does not cover the production sampler, per-rank XLA RNG, or multi-process state. |
| Production XLA trainer | Blocked | Current V5 backend rejects XLA; rank-safe checkpoint publishing, XLA RNG restore, distributed eligible-token validation, and efficient checkpoint-boundary audit hashing remain unqualified. |
| Qualified training corpus | Blocked | No supplied manifest passes source/license, tokenizer, split, contamination, pack, and real-token review for this run. |
| Citadel capability evaluation | Blocked | No ready, hash-bound evaluator receipt and preregistered sensitivity-gate result is supplied. |
| Production run authorization | Blocked | The static gate stays `training_authorized: false`, even when receipt-shaped files are present for review. |

## Candidate geometries

| Candidate | Width × depth | Query/KV heads | FFN | Parameters |
|---|---:|---:|---:|---:|
| M102 primary | 640 × 20 | 10 / 5 | 1,600 | 101,790,080 |
| Depth-preserving TPU challenger | 640 × 20 | 10 / 5 | 1,536 | 99,332,480 |
| Fully tiled TPU challenger | 768 × 12 | 12 / 6 | 2,176 | 100,303,104 |

The 640-wide FFN challenger preserves M102 depth and changes only the FFN width. The fully tiled challenger changes several architectural variables and must win a matched systems and multi-seed capability comparison before promotion. At one sequence per replica and context 4,096, the naively materialized BF16 attention-score tensor is 335.5 MB per active layer for the 640-wide candidates and 402.7 MB for the 768-wide candidate; the target kernel may use a different allocation strategy.

## Reproduce local checks

```powershell
python -m pytest tests/test_signac_100m.py tests/test_v5_mutation_canary.py tests/test_v5_target_preflight.py -q
python tools/signac_100m_preflight.py --target tpu
```

The first command covers static contracts and portable canary/preflight checks. The second is expected to return `BLOCKED` until target, data, and evaluation evidence are supplied. A passing Kaggle notebook canary is still not production-run authorization.
