# CYMEK REQUIRED CHANGES — 500M campaign readiness

Citadel does not edit/push Cymek. Each change below specifies file,
function, current behavior, required behavior, why, test, and acceptance.
Classification: **BLOCKING** (PRE500M cannot go green without it) /
**RECOMMENDED** / **OPTIONAL**.

---

## BLOCKING

### B1. Materialize the production corpus subset for the campaign
- File(s): new `v5_data/acquire_production.py` (or extend the existing
  acquire path); `v5_data/data_status.py`.
- Current: mixture 65/20/15 is DECLARED in `run_spec`/`training_spec`; no
  manifest-bound bytes exist (`PRE500M.data_readiness` → DATA_NOT_READY;
  BRAMASTRA also flagged P35-A hashes `PENDING_MATERIALIZATION`).
- Required: materialize a qualified corpus subset whose per-source unique
  runnable tokens cover the 500M allocation (65% natural / 20%
  code-math-formal / 15% verified-cognition), emit `DataManifest` +
  `PackManifest` with real hashes, and record the cross-source union after
  dedup (receipt-local unique sums are not acceptable — `build_data_status`
  currently sums without union).
- Why: 500M consumed tokens need real supervised bytes; DECLARED ≠
  MATERIALIZED.
- Test: manifest hash stable across two independent builds; union-dedup
  receipt; contamination scan fail-closed.
- Accept: `data_readiness(state="RUNNABLE")` on the emitted manifests with
  zero blockers.

### B2. Wire the top-level production training entry point
- File(s): new `v5_training/production_entry.py` + `anra_v5` CLI verb.
- Current: components are all CONNECTED and certified, but no single entry
  wires corpus manifest → tokenizer → packing → sampler/cursor → batch →
  `ProductionTrainingBackend` → checkpoint transactions. Every execution so
  far hand-wired its inputs (miniature, canaries, T1D, PRE50M smoke).
- Required: one fail-closed entry point consuming only receipt-bound
  identities (campaign spec, data/pack manifests, tokenizer artifact,
  model spec), resumable from the latest committed generation, emitting
  step receipts + milestone publications.
- Why: the 500M campaign must run the audited production path, not another
  hand-wired script.
- Test: deterministic dry-run (tiny token budget, compressed milestones)
  through the SAME entry: start → train → checkpoint → evaluate →
  disconnect simulation → fresh restore → continue → cross milestone →
  final receipt.
- Accept: dry-run green; no test-only alternate state machine.

### B3. Freeze the tokenizer production artifact
- File(s): `v5_tokenizer/adapter.py` identity + a frozen artifact under
  `artifacts/`.
- Current: interface frozen (24,576 byte-BPE) but
  `artifact_sha256: None` (`PROVISIONAL_CANDIDATE_E1_IDENTITY_REQUIRED`).
- Required: freeze the E1-qualified artifact hash into the training spec;
  the 500M campaign binds data token counts, packing, training, evaluation,
  and resume to exactly that artifact.
- Why: tokenizer identity change = campaign identity change.
- Test: identity round-trip + zero-unknown on the campaign probe corpus.
- Accept: `TokenizerReceipt.assert_valid()` against the frozen artifact.

## RECOMMENDED

### R1. Execute the canonical token-space WSD schedule at least once
- File(s): `v5_training/schedule.py` + production entry.
- Current: `lr_at` (warmup 0→50M, stable 3e-4 to 4.5B) is token-based,
  unit-tested, and already 500M-compatible — but every executed run bound
  `bounded_warmup_schedule` (constant LR) instead.
- Required: the 500M campaign binds the canonical `lr_at` schedule and the
  PRE500M smoke executes ≥1 update under it.
- Why: the campaign's LR claims must be executed evidence, not unit tests.
- Accept: step receipt records `schedule(cumulative_tokens)` matching
  `lr_at`.

### R2. Qualification evidence must not be omittable
- File(s): `v5_data/data_status.py` (+ CLI).
- Current: an empty qualification map produces no unqualified-family
  blocker; `build_data_status` sums receipt-local unique counts without
  cross-receipt union.
- Required: missing qualification = explicit blocker; report the
  cross-source UNION after dedup, not the sum.
- Why: a data-ready flag can currently omit an essential cognition
  qualification check (BRAMASTRA audit, confirmed by inspection).
- Accept: empty-qualification input yields a FAIL status.

### R3. Milestone publication helper in Cymek's trainer
- File(s): `v5_training/trainer.py`.
- Current: no milestone-crossing logic.
- Required: adopt Citadel's pure `crossed_milestones(previous, new)`
  semantics (token-based, first-transaction-crossing publishes, resume-safe)
  — reference implementation `citadel_tpu/milestones.py`, boundary-tested.
- Accept: boundary table 49,999,999→50,000,000 etc.; no duplicate
  publication after resume.

## OPTIONAL

### O1. Frozen evaluation sets at token milestones
- File(s): `v5_evaluation/` new frozen-set builder.
- Why: §21 evaluation points (0/10M/25M/50M/100M/200M/350M/500M) want
  immutable eval fixtures with termination-behavior capture.

### O2. Automatic stability stop
- File(s): `v5_training/trainer.py` abort rules.
- Why: auto-stop on persistent nonfinite/exploding gradients/corruption
  exists piecewise; centralize into one documented stop policy with
  counters.
