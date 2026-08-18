# An-Ra Core API Reference

Runtime/package version: `0.5.0`.

## Semantic API

`CoreExecutor` accepts a homogeneous batch of integer representation IDs and
returns raw V4 prediction logits. For V4 the result is shaped
`[batch, sequence, vocabulary]`. Probabilities, sampling, retries, text
streaming, retrieval, and tool decisions are Connector concerns.

`CoreExecutor.from_checkpoint(...)` requires a safe V4 artifact by default:
CPU-first `weights_only=True` loading, dense key/shape validation, and a
verified tokenizer representation contract. `legacy_unverified=True` is an
explicit forensic escape hatch, not normal production behavior.

`forward(ids)` executes stateless full sequence prediction. `create_state`
creates an in-memory handle. `prefill`, `forward_step`, and `forward(ids,
state)` advance it. `reset_state`, `rollback_state`, `fork_state`, and
`release_state` are executor-mediated lifecycle operations.

## State validity

A state is bound to one executor owner, architecture schema hash, dense
parameter identity, representation identity, execution profile, batch size,
and capacity. Mismatch is a typed error; no state is silently reinterpreted.
The supported representation is an in-memory, same-process, homogeneous cache.
Serialization, mixed-length batches, cross-device transfer, and cross-profile
reuse are unsupported capabilities.

## Identity domains

- Architecture: V4 mathematical schema and dense count.
- Parameters: normalized dense tensor digest.
- Checkpoint: whole artifact digest and lineage metadata.
- Representation: tokenizer vocabulary/probe contract.
- Runtime: Core implementation version.
- Execution profile: device/dtype/category.

## Training interface

`AnRaCore` is a differentiable PyTorch module. Training owns objectives,
optimizers, updates, distributed strategy, durability, evaluation and
promotion. Core owns only the learned mathematical program and its parameter
schema. Gradient descent is a V4 capability, not a permanent definition of
An-Ra learning.
