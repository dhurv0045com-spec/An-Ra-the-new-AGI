# An-Ra Core Architecture: Verified Current Contract

## Truth status

This document replaces earlier claims that were not reproducible. Core vNext is
not promoted merely because a synthetic test passes. It is a compatibility-
preserving execution boundary currently undergoing real-checkpoint validation.

## Four responsibilities

| Responsibility | Owns | Must not own |
| --- | --- | --- |
| Neural Program | V4 learned parameters and deterministic mathematics from IDs to logits | UI, tools, sampling policy, optimizer policy |
| Core Executor | device/dtype/profile, batching, incremental cache allocation, state validation | learned weights mutation or system policy |
| Connector | representation assembly, context selection, decoding/sampling, verification, orchestration | arbitrary Core tensor/state mutation |
| Training/Evaluation | data, objective, updates, checkpoints, evaluation and promotion | inference-time system policy |

The Core is not synonymous with all intelligence. It is the exact learned
substrate plus the machinery required to execute it correctly. Deterministic
validation and state lifecycle are Core concerns even though they are not
learned. Learned routing or recurrence, if later adopted, belongs to neural
semantics; hardware scheduling belongs to the Executor.

## V4 compatibility truth

- Dense executable parameters: **180,093,312**.
- Historical artifact ABI: **181,132,071**.
- Difference: **1,038,759** dormant pilot tensors (MoD, ESV, RIM, and depth
  controls). They are validated as explicit historical inventory and are not
  silently ignored.
- Canonical tokenizer: a V4 32,768-ID representation contract. Information
  representation is fundamental; this exact tokenizer is required only because
  the present embeddings were trained against its IDs.
- A trained artifact has a whole-file hash, normalized dense parameter hash,
  architecture schema hash, and tokenizer contract. These are different
  identities and must remain distinguishable.

## State and execution truth

CoreState is process-local, executor-owned, and bound to model schema,
parameters, representation, profile, batch geometry and capacity. It supports
homogeneous batches; mixed-length batching, state serialization, cross-device
reuse and profile conversion are explicitly unsupported. Full-attention layers
have context-dependent decode cost. Sliding attention bounds visibility but not
an automatic Connector-level context policy.

## Promotion evidence required

An exact Executor refactor needs real-artifact load, tokenizer binding, dense
parameter identity, forward/prefill/decode parity, failure-atomic state tests,
state isolation, training-gradient conformance, and reproducible workload-
matched prefill/decode/memory benchmarks. A neural change is a new model
lineage: it needs training and behavioral evaluation instead of old-logit
parity. Approximate profiles need explicit numerical and behavioral bounds.

## Negative requirements

Core does not know about users, conversations, personalities, tools, browser
sessions, retrieval vendors, long-term memory policy, business logic, or
promotion authority. Connector may request supported operations and diagnostics
but cannot mutate arbitrary weights or internal state. Future self-improvement
uses propose -> isolated experiment -> evaluate -> promote/rollback; no
Connector proposal changes production weights directly.

## Current honest conclusion

The durable architectural direction is a small semantic Core port plus
capability-negotiated extensions. It is not a promise that every future neural
architecture uses a Transformer or KV cache. V4 implements the current port as
next-token logits with an in-memory cache. Core software cleanup improves
reliability, efficiency, and research flexibility; it does not by itself make
the present checkpoint more capable.
