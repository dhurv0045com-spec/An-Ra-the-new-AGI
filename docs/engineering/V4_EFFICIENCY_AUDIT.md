# V4 compute and memory audit

Status: implementation evidence as of 2026-08-11. This is not capability or
AGI evidence. It distinguishes optimizations that preserve a checkpoint's
function from architecture experiments that require a frozen-parent trial.

## Findings

| Area | Current evidence | Decision |
|---|---|---|
| Attention | PyTorch SDPA is the canonical path. V4 uses 14 query heads and 2 KV heads at 181M, and 20/2 at 500M. | Keep. GQA lowers KV projection and cache size. Do not replace a working fused kernel without device benchmarks. |
| Exact KV cache | The old exact backend appended with `torch.cat` on every generated token, reallocating and copying its complete history. | Replaced by `anra-exact-kv-cache/v1`, a lossless preallocated cache. `legacy-float` is the rollback switch. |
| TurboQuant | Real 4-bit packed pilot, but dequantizes before SDPA and is lossy. | Keep pilot-gated. It cannot replace exact cache until checkpoint-specific distribution and behavior parity plus throughput evidence pass. |
| Activation checkpointing | Implemented per block and enabled for the 500M profile. | Keep. It trades additional compute for lower activation memory and does not change checkpoint tensors. |
| Sequence packing | Raw causal data already uses fixed token windows with no padding waste. | Keep. SFT uses assistant-only weights and cannot be repacked casually without preserving prompt boundaries and loss masks. |
| AdamW state | Full precision moments are a large part of training memory. | Do not silently quantize. An 8-bit optimizer can change the training trajectory and needs a named frozen-parent pilot. FSDP optimizer sharding is the safer large-cluster direction after DDP correctness proof. |
| MoD | Training evaluates `ffn(x)` for all tokens before applying its hard mask, so it does not deliver the claimed training compute saving. | Do not promote. Redesign around gathered-token execution and prove quality/throughput independently. |
| MTP | Adds 1,607,424 parameters and extra vocabulary losses. | Named pilot only. It may improve data efficiency but is not function-preserving and must beat dense continuation on held-out behavior per useful compute. |
| MoE | The current upcycle adds about 941M parameters to 181M and executes Python expert dispatch. | Disabled. It is not a practical 181M or 500M efficiency mechanism in its current form. |
| Sliding attention | The current implementation builds a dense boolean window mask for each applicable layer. | Retain for checkpoint compatibility. A fused-window kernel is a future device-specific pilot; ordinary mask caching is smaller than the exact-cache allocation defect fixed here. |
| Native controls | RIM, ESV, DSTP, and MoD tensors exist in the checkpoint ABI but are disabled unless promoted. | Do not describe installed tensors as active intelligence or efficiency. |

## Exact preallocated-cache contract

`CausalTransformerV2.enable_kv_cache(backend="float")` now selects
`preallocated-exact-v1`. It allocates one bounded contiguous K tensor and V
tensor per layer at the model context limit, writes only new values, and returns
occupied views to the unchanged SDPA path. Sliding layers retain the same
attention visibility through their existing mask rather than physically
shifting storage after every token. It has no parameters, persistent buffers,
optimizer state, or checkpoint keys. Clearing a request resets its logical
position; disabling the cache releases it. The previous dictionary-plus-
`torch.cat` implementation remains available as `backend="legacy-float"` for
immediate rollback.

For one 2,048-token decode, a full-attention legacy layer copies
`2 * Hkv * D * T(T+1)/2` elements. A 1,024-token sliding legacy layer copies
the growing prefix and then a bounded 1,025-token concatenation each step. The
preallocated path writes `2 * Hkv * D * T` in either case. With V4's 2 KV heads
and 64-dimensional heads, this is:

- 537,133,056 versus 524,288 elements per layer;
- 403,046,400 versus 524,288 elements for a 1,024-window sliding layer;
- about 7.79 billion versus 9.44 million elements across the actual 181M
  hybrid pattern (four full and fourteen sliding layers).

That analytical 181M ratio is about 825.6x less append-copy traffic at a full
2,048 tokens. It is not a tokens/second claim: the exact cache reserves full
context storage even for sliding layers, and kernel launch, masked SDPA,
sampling, allocator, and host overhead still require measurement on the actual
GPU. Focused tests prove token parity and close logits with the rollback
backend, stable production storage without shifts, bounded standalone eviction,
reset behavior, and the analytical accounting.

## Remaining promotion evidence

- Benchmark exact-static against legacy-float on the trained V4 checkpoint at
  prompt lengths 128, 512, 1,024, and 2,048, reporting latency, tokens/second,
  peak allocated/reserved VRAM, and cache telemetry.
- Re-run the existing KV parity gate on the trained checkpoint and both the
  RTX 4050 and intended server GPU class.
- Keep TurboQuant separate: compression and exact-cache allocation stability
  answer different questions.
- Do not infer better language or reasoning from this change. It makes repeated
  decoding cheaper; capability still comes from data, optimization, and proven
  post-training.
