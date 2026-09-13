# CS-TRANSFER-001 failure modes and falsification controls

The experiment is useful only if the V4096↔V24576 contrast is causally interpretable. These are the failure modes that can invalidate that interpretation before any result is discussed.

## FM-01 — tokenization/segmentation confound

**Failure:** each physical-vocabulary arm uses a different tokenizer or receives different token IDs.

**Consequence:** a performance gap cannot be attributed to physical class space.

**Control:** one frozen V24576 production tokenizer renders the candidate surface once; only rows with all content IDs `<4096` survive; exact selected token rows are hash-bound and shared by both arms.

## FM-02 — same seed but different shared initialization

**Failure:** constructing V4096 and V24576 independently under the same seed changes RNG consumption because their embedding shapes differ. Later transformer tensors then differ.

**Consequence:** model initialization becomes a hidden second treatment.

**Control:** initialize one V24576 reference, copy every non-embedding tensor byte-for-byte to V4096, and copy embedding rows 0..4095 exactly. Regression tests explicitly demonstrate that naïve same-seed construction is confounded.

## FM-03 — parameter-count compensation

**Failure:** widen/deepen V4096 to match V24576 total parameters.

**Consequence:** class-space and model capacity change together.

**Control:** width/depth/FFN/attention are identical. The parameter difference is intentionally the physical tied-row treatment.

## FM-04 — low-ID filtering creates a shortcut

**Failure:** selecting only examples representable below ID 4096 biases answers/templates enough that a trivial heuristic solves the task.

**Control:** fresh latent worlds, split isolation, deterministic outcome-blind hash ranking after eligibility, family-level heuristic attacks, minimum acceptance 15%, and fail-closed shortcut threshold `<0.35` before GPU execution.

## FM-05 — low-ID acceptance is too small

**Failure:** production tokenizer maps too much of a family above ID 4095, forcing a tiny or distorted selected subset.

**Control:** preparation fails if any family/split has acceptance below 15% or lacks the preregistered row count. Do not relax this threshold after observing model behavior. If preparation fails, the experiment requires a prospective design amendment or a different causal instrument.

## FM-06 — development evaluation controls training

**Failure:** arm inclusion, endpoint, LR, data, or thresholds change after seeing the formation curve.

**Control:** fixed 480-update endpoint; fixed development checkpoints; no early stopping; all eight arms mandatory unless an engineering failure makes the campaign inconclusive.

## FM-07 — sealed leakage

**Failure:** sealed rows influence debugging/training or are consumed repeatedly.

**Control:** sealed identity is hash-bound at preparation. Finalization writes `SEALED_CONSUMPTION.json` with `STARTED` before sealed inference. A crash after that marker is fail-closed and requires a fresh sealed identity rather than a second look.

## FM-08 — checkpoint published without matching training evidence

**Failure:** resume can continue from a checkpoint whose trace/evaluation history is missing or contradictory.

**Control:** training receipt is persisted before each checkpoint publication. On resume, trace rows ahead of LATEST are trimmed and replayed; a checkpoint ahead of the durable trace is rejected. If a crash happens after a fixed checkpoint but before its development evaluation, the restored checkpoint is evaluated before any further update.

## FM-09 — paired arms use different update streams

**Failure:** sampler seeds, selected token rows, pack bytes, or epoch mapping differ by arm.

**Control:** order seed is paired; pack manifest is global and immutable; the arm identity binds the same pack SHA. Stream-layout cardinality drift fails closed.

## FM-10 — V4096 sees out-of-range content

**Failure:** any content target/prompt ID is `>=4096`.

**Control:** CPU preparation rejects it; persisted rows are deterministically regenerated and compared byte-for-byte before run; model embedding lookup provides a final hard failure rather than silently remapping IDs.

## FM-11 — output metric reproduces Canary-v2 prefix ambiguity

**Failure:** decoded-string and token-prefix metrics disagree due to tokenizer formatting.

**Control:** primary CS-TRANSFER metric is exact **token** answer plus EOS against preregistered answer token IDs. Decoded text is not authoritative. Prefix-token, answer-token ignoring EOS, and EOS-stop rates are secondary diagnostics only.

## FM-12 — neither arm forms capability

**Failure:** an apparent small-vocabulary advantage is measured in a regime where both models are broadly untrained.

**Control:** in every arm/seed, at least one of composition or termination must reach endpoint exact-token+EOS `>=0.50`; otherwise verdict is `INCONCLUSIVE_FORMATION`.

## FM-13 — post-hoc vocabulary promotion

**Failure:** a controlled low-ID result is interpreted as proof that a V4096 production tokenizer is superior.

**Control:** claim ceiling explicitly forbids production tokenizer replacement. A supported result requires replication at the ~40M rung and a later segmentation/tokenizer transfer test.

## FM-14 — operator notebook runs branch HEAD instead of frozen science

**Failure:** later notebook/docs commits silently change executable bytes.

**Control:** use the two-commit operator pattern: freeze one exact scientific executable SHA first; create the notebook later; notebook checks out detached science SHA and verifies critical Git blob hashes before any Drive write or GPU update.

## FM-15 — engineering repair changes scientific meaning

**Failure:** a bug is repaired after partial execution by altering data/treatment/thresholds while keeping the same experiment identity.

**Control:** engineering-only fixes must be documented and shown not to change frozen scientific semantics. Any scientific change requires a prospective amendment/new executable identity before further outcomes are observed.
