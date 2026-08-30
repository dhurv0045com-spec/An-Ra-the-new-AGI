# E2 architecture static plan

This package freezes the bounded P35 experiment structure before any model is
built. Shape arms are within one percent parameters and hold MHA, QK norm,
context, vocabulary, data order, and evaluation constant. Separate fractional
arms isolate GQA/QK norm and context length.

Run:

```text
python -m e2_architecture.plan --output artifacts/e2/static_plan.json
```

The expected status is `BLOCKED_E1_INPUTS` until real tokenizer, corpus, and
model-constructor hashes exist. Static FLOP/cache figures are planning evidence,
not target-accelerator throughput or cognition evidence.

An optional bounded CUDA probe is also available:

```text
python -m e2_architecture.device_benchmark --output artifacts/e2/local_cuda_attention.json
```

It reports isolated causal-SDPA latency and memory for the registered
MHA/GQA/QK/context cases, compares native and repeated-K/V GQA implementations,
and checks native GQA against explicitly repeated K/V heads. It remains a kernel
microbenchmark, not full-model or cognition evidence.

`e2_architecture.block_benchmark` goes one level deeper without training: it
constructs each exact P35 shape stack, including tied embeddings/logits, RoPE,
affine QK norm, MHA, SwiGLU, cross-entropy, and backward, then verifies exact
parameter counts and finite gradients. It deliberately performs no optimizer
update. Use small matched sequences on local CPU/CUDA; repeat the same receipt
on the target TPU stack before architecture selection.

`e2_architecture.signal_benchmark` compares the provisional normal(0.02)
initialization with the same paired draws plus `1/sqrt(2L)` residual-output
scaling. It records per-layer activation RMS and block-gradient RMS across
multiple seeds after one forward/backward. It performs no optimizer update and
can only support an initialization canary, not predict learned cognition.

`e2_architecture.qk_norm_benchmark` isolates what QK normalization controls. It
uses paired projection draws at 0.25×/1×/4× scale, sampled causal queries, RoPE,
and a proxy backward to measure attention-logit scale, normalized entropy,
concentration, and gradient finiteness across contexts. Its gate asks whether
QK norm removes attention-distribution sensitivity to Q/K weight scale while
the unnormalized control exposes the perturbation. It performs no optimizer
update and cannot establish that selective attention improves cognition.

`e2_architecture.precision_benchmark` loads identical scaled-residual weights
into exact FP32 and BF16 P35 stacks, then compares logits, cross-entropy, and
four representative gradients after one backward pass. Its limits are fixed in
source and its receipts bind both model and initialization implementations. It
performs no optimizer update; passing is local numerical-parity evidence, not
evidence of long-run BF16 training stability.
