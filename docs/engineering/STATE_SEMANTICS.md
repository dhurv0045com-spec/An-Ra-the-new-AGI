# Core State Semantics

`CoreState` is executor-owned acceleration state. For V4 it contains
preallocated key/value backing storage and a logical prefix length; it is not a
conversation object or persistent memory.

## Guarantees

- States have independent storage and can be interleaved.
- A failed neural call does not commit logical tokens. Writes beyond the
  committed prefix are scratch and are overwritten on retry.
- Reset, rollback, fork, and release are explicit executor operations.
- Logical occupied bytes and reserved backing bytes are distinct measurements.
- V4 supports homogeneous batches only; all rows share a logical length.

## Invalidity

State is invalid for a different executor, architecture, parameters,
representation, execution profile, batch geometry, capacity, or lifecycle.
The executor rejects it with typed failure information.

## Scope limits

State is process-local and not serializable or portable across profiles,
hardware, implementations, or restarts. Full-attention V4 layers retain
context-dependent decode cost. Sliding attention bounds neural visibility; it
does not decide what conversation content to discard. Connector manages
context; Core enforces capacity and reports overflow.
