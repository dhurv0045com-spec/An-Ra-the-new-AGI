# TPU_MILESTONES.md

TPU-first production sequence. CUDA evidence is reference only. Each milestone
is preregistered (`experiments/T<n>/PLAN.md`) and receipted with machine-readable
JSON. No milestone is skipped; no receipt is hand-typed.

```text
M0 env probe → M1 single-device one-update → M2 calculator checkpoint
→ M3 8-device data-parallel → M4 full-stream readiness → 5B capability
```

## M0 — Environment probe

Spec: `tpu/TPU_ENVIRONMENT.md`. Receipt: `TPU_ENVIRONMENT.json`.
Gate: `probe_pass == true`, else ABORT (see environment doc).

## M1 — Single-device TPU one-update (experiment T0)

Sequence (smallest possible, no dataset, no large model):

```text
instantiate tiny model → move to XLA → load tiny token batch
→ forward → loss → backward → optimizer step
→ xm.mark_step / xm.optimizer_step correctly
→ verify parameters changed → checkpoint → reload → identical inference
```

Receipts: `TPU_ONE_UPDATE.json`, `TPU_RESUME.json`.
Pass gate: params hash changed, optimizer stepped once, grad norms finite,
post-clip norm ≤ 1.0 + tolerance, checkpoint reload bit-identical inference,
environment block embedded. Any failure → IMPLEMENTATION_FAILURE, never a
scientific verdict. Full preregistration: `experiments/T0/PLAN.md`.

## M2 — Calculator checkpoint (experiment T1)

Smallest useful An-Ra/Cymek-compatible model on the deterministic calculator
canary through the real TPU infrastructure. Simple train/held-out arithmetic,
measured before and after. Required evidence:

```text
untrained accuracy, training loss curve, held-out accuracy,
tokens consumed, updates, wall time, tokens/sec,
checkpoint hash, reload accuracy
```

Receipts: `TPU_CALCULATOR_CHECKPOINT.json`, `TPU_THROUGHPUT.json`.
This checkpoint is experimental infrastructure proof, not intelligence.
Full preregistration: `experiments/T1/PLAN.md`.

## M3 — 8-device data-parallel (experiment T2, preregistered after M1 passes)

Correctness first, distribution second. Verify:

```text
same model initialized on replicas; data sharded correctly;
gradients synchronized; optimizer state correct;
global token accounting exact-once (no eightfold duplication);
checkpoint produced from canonical state
```

Receipt: `TPU_8_DEVICE.json` with per-rank `token_contribution` list summing
to `global_tokens` (reuse `v5_training/distributed.py` schema). Bring-up stays
on bucket 512 fixed-batch until M3 passes.

## M4 — Full-stream readiness (preregistered after M3 passes)

Buckets 1024/2048/4096 re-enabled one at a time with recomp-count benchmark;
mixture → packing → cursor → training contracts exercised end-to-end on a
short stream; durability/remote envelopes unchanged. Only then is any 5B-token
discussion admissible.

## What this does NOT authorize

No 5B corpus work, no large-model training, no sealed fixtures, no CUDA-based
production claim. PyTorch/XLA is the stack unless device evidence forces a
revisit. No JAX/TF rewrite under this plan.
