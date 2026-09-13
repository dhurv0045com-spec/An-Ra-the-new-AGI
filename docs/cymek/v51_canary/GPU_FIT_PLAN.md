# GPU FIT PLAN (Rung B, T4)

**Target:** 42,092,544 parameters — must fit a single Colab T4 (16 GiB HBM) with headroom for optimizer state, activations, checkpointing and evaluation. Numbers from `tools/next_core_compute_model.py` (exact params; analytic memory) and the local RTX 4050 preflight.

## Memory model (adopted precision layout: BF16 compute + FP32 master/moments)

| Component | Bytes | Value |
|---|---|---|
| FP32 master parameters | 4 B/param | 168.4 MB |
| FP32 AdamW moments (m, v) | 8 B/param | 336.7 MB |
| BF16 compute copy | 2 B/param | 84.2 MB |
| FP32 gradients (transient) | 4 B/param | 168.4 MB |
| **Static total** | ~18 B/param | **~0.72 GiB** |
| Activations (batch 8 × ctx 2,048, 10 layers, ckpt off) | empirical | ~1.5–2.5 GiB |
| Activations (activation checkpointing ON) | empirical | ~0.5–0.9 GiB |
| Checkpoint staging + evaluation workspace | — | ~0.5 GiB |
| **Peak projected** | — | **~2.5–4 GiB of 16 GiB** — ≥ 4× headroom |

## Throughput model

Rung A on CPU measured ~780–810 tokens/s (10.2M params). Rung B on a T4: scaling by FLOPs/token (∝ ~4.2× params) and T4 BF16 throughput, expect ~2,000–5,000 tokens/s at batch 8 × 2,048 ctx with activation checkpointing. The full Rung-B canary budget (120 updates × 8,192 real tokens/update equivalent, or the same 4096-token update shape) fits a single session comfortably; the launcher still checkpoints every 16 updates and supports exact resume across sessions (§35).

## Preflight calibration procedure (before any substantive Rung-B training)

1. `--mode preflight --rung B --cuda --bfloat16` — one real update through the production backend on CUDA; records runtime dtype receipt, peak allocated/reserved VRAM, step time.
2. Batch/accumulation selection from the MEASURED fit; scientific variables (tokens/update, schedule, seeds) never move for utilization.
3. Checkpoint save/restore timing measured at the chosen batch.

## Local hardware note (Task-3 execution environment)

RTX 4050 Laptop 6 GiB (5,920 MiB free at audit) — sufficient for the bounded preflight/smoke only; the substantive Rung-B campaign is operator-executed on a T4 via the pinned notebook. CPU: 8C/16T with ~15.3 GB RAM (2.9 GB free at audit) — Rung A CPU runs sized accordingly.
