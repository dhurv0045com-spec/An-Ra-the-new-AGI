# An-Ra V5 Code and Infrastructure Blueprint

Status: **Ground Blueprint v0.3 implementation contract**
Scope: design and executable contracts; no production trainer or main run authorization

This is the canonical bridge from research decisions to code. `V5_MASTER_BLUEPRINT.md` owns the scientific question; this file owns module boundaries, interfaces, commands, artifacts, failure behavior, and the order in which implementation becomes trustworthy.

## 1. Build target

The provisional V5-A center is represented by `v5_contracts.model_spec.V5A_250M`:

| Field | Value |
|---|---:|
| parameters | 250,216,960 |
| layers × width | 26 × 896 |
| query / KV heads | 14 / 7 |
| head dimension | 64 |
| SwiGLU width | 2,368 |
| QK normalization | affine RMS normalization; E2-gated |
| vocabulary | 24,576, E1-gated |
| context | 4,096 native, E2-gated |
| token budget | 5,000,000,000, corpus-audit-gated |
| idealized compute | 7.50651 EFLOP (`6ND`) |

This exact shape is a center candidate requested for implementation planning, not a frozen winner. E2 may change the shape while preserving the 250M target envelope. Any change must produce a new model-spec hash and parameter receipt.

## 2. Dependency rule

Dependencies point inward toward contracts; no layer may import a layer to its right:

```text
v5_contracts
    ↑
data / tokenizer / model primitives
    ↑
objectives / packing / training state
    ↑
trainer orchestration

e0/evaluation  ─────────────── independent consumer of immutable artifacts
promotion      ─────────────── independent consumer of evaluation receipts
connector      ─────────────── deployment peer, never evaluator or trainer
```

Forbidden imports are CI-enforced:

- training cannot import sealed fixtures, promotion code, or Connector runtime;
- evaluation cannot mutate training data, model state, or checkpoint lineage;
- model code cannot import trainer, checkpoint, tools, permissions, or benchmark truth;
- promotion cannot load mutable recovery checkpoints;
- E0 evaluation generators cannot be imported by training generators.

## 3. Repository topology

Existing now:

```text
v5_contracts/                 framework-independent model/run/lineage schemas
e0_cognition/                 development evaluation and training-generator research
e1_tokenizer/                 artifact-bound tokenizer static audit
artifacts/e0/                 reproducible compact E0 receipt
artifacts/v5/                 reproducible 250M implementation receipt
tests/                        contract/property tests only
blueprints/                   authoritative implementation and experiment designs
```

Authorized only after their owning experiment passes:

```text
v5_data/
  contracts.py               document, source, license, filter, split identities
  ingest/                    source-specific adapters; raw bytes remain external
  filter/                    deterministic quality/safety/license decisions
  dedup/                     exact, MinHash/LSH, benchmark contamination
  cognition/                 training-only executable generators
  mixture.py                 immutable token allocations and curriculum phases
  pack.py                    deterministic sharded pack writer
  cursor.py                  exact sampler position and replay state

v5_tokenizer/
  adapter.py                 one frozen encode/decode interface
  identity.py                artifact/config/corpus hashes

v5_model/
  config.py                  accepts only validated ModelSpec
  embedding.py
  attention.py               GQA/full-attention implementation
  block.py                   pre-norm attention + SwiGLU residual block
  core.py                    pure tensor-in/logits-out model
  initialize.py              deterministic init and tied-weight assertion

v5_objectives/
  causal_lm.py               universal loss
  query_swap.py              compiled only if E3 promotes it

v5_training/
  optimizer.py               optimizer construction after device placement
  schedule.py                token-indexed global schedule
  state.py                   complete serializable training state
  step.py                    pure one-update function
  checkpoint.py              canonical single-writer transaction
  distributed.py             topology and collective boundary
  runner.py                  orchestration, signals, recovery, durable handoff

v5_evaluation/
  adapter.py                 immutable checkpoint → model-scoring interface
  representation.py
  addressing.py
  transformation.py
  selection.py
  realization.py
  substrate.py
  receipt.py

v5_promotion/
  gates.py                   preregistered conjunctive gates
  decide.py                  signed promote/reject/inconclusive receipt
```

No placeholder package is created merely to resemble this tree. A package is introduced only when its gate authorizes executable work.

## 4. Contract registry

Every durable artifact has one schema, a canonical JSON serialization, and a SHA-256 identity.

| Artifact | Required identity and invariants | Writer | Reader |
|---|---|---|---|
| ModelSpec | exact dimensions, parameter receipt, config SHA | research/freeze review | constructor, trainer, evaluator |
| TokenizerReceipt | artifact/config/corpus/probe hashes, vocab, special IDs | E1 | packer, model, evaluator |
| SourceManifest | source/license/acquisition/raw hashes | ingest | filtering, audit |
| DataManifest | filter/dedup/split/tokenizer hashes, tokens by source | pack builder | trainer, audit |
| PackManifest | shard hashes, sequence lengths, exact tokens, cursor schema | pack builder | sampler/trainer |
| RunSpec | model/tokenizer/data identities, optimizer/schedule/seed/topology | freeze review | runner |
| CheckpointManifest | full lineage fields defined in `v5_contracts.lineage` | canonical writer | restore/evaluator |
| EvaluationReceipt | checkpoint, evaluator, fixtures, metrics, CIs | evaluator | promotion |
| PromotionDecision | immutable checkpoint/eval hashes and failed gates | independent promotion process | registry/deployment |

Unknown schema versions, missing hashes, inconsistent token totals, or identity mismatches fail closed. There is no “best effort” restore.

## 5. Data pipeline transaction

```text
authorized source bytes
→ immutable SourceManifest
→ deterministic filtering decisions
→ exact + near-duplicate clusters
→ benchmark-contamination scan
→ tokenizer encoding
→ deterministic document split
→ cognition-mixture allocation
→ immutable pack shards + PackManifest
```

Implementation requirements:

1. Split assignment is a hash function of source identity before tokenization or quality outcome inspection.
2. Dedup clusters never cross-split by copying one member into multiple splits; the cluster receives one split.
3. Token accounting is exact after the frozen tokenizer; estimates cannot enter a run receipt.
4. Cognition examples preserve internal graph/provenance but serialize only permitted model text.
5. Every pack shard is content-addressed and immutable.
6. Sampler order is derived from `(run_seed, epoch, shard_hash)` and persisted as a cursor, not reconstructed from wall time.

The 5B center allocation is provisionally 3.25B natural, 1.0B code/math/formal, and 0.75B verified cognition tokens. E3 may change ratios; it may not waive provenance.

## 6. Model construction contract

The future constructor takes a validated `ModelSpec` and exposes only:

```text
initialize(spec, seed) -> CoreParameters
forward(parameters, token_ids, positions, attention_mask) -> logits
parameter_receipt(parameters) -> names/shapes/count/hash
```

It must not construct an optimizer, load checkpoints, inspect task family, access tools, or decide generation policy. Constructor certification compares executable tensor names/shapes/count to the pure receipt and asserts the tied embedding/output storage identity.

## 7. Training-step contract

The smallest trusted unit is one explicit update:

```text
StepInput(
  parameters,
  optimizer_state,
  batch,
  global_update,
  cumulative_tokens,
  schedule_state,
  rng_state,
  sampler_cursor,
) -> StepOutput(
  new_parameters,
  new_optimizer_state,
  scalar_metrics,
  gradient_receipt,
  new_counters,
  new_rng_state,
  new_cursor,
)
```

The optimizer is created after final device placement. Before a run is trusted, a canary proves live-parameter ownership, finite nonzero gradients, parameter-hash change, exact optimizer-step increment, moment change, tied-weight preservation, and one-time token/cursor advancement.

Learning-rate schedules are indexed by cumulative training tokens from zero, not notebook-local step or data-pack number. Resuming, changing a session, or migrating a pack cannot rewarm the schedule silently.

## 8. Checkpoint commit protocol

Only one logical writer advances a lineage. Distributed workers participate in a collective snapshot, but one coordinator commits it:

1. barrier at a completed optimizer update;
2. snapshot tensors, optimizer, counters, cursor, RNG, curriculum, topology, and all identity hashes;
3. write to a unique temporary generation;
4. fsync/close and compute file/object hashes;
5. independently load and validate manifest plus tensor shapes;
6. atomically publish immutable milestone or rotate recovery pointer;
7. upload to durable storage;
8. download into a clean location and restore-canary it;
9. only then record `durable=true` in the lineage registry.

Recovery A/B may rotate. Milestones never mutate. A `latest` name is only a pointer to a content hash. Promotion points to an existing milestone and never rewrites weights.

Raw full-resume storage planning for 250.22M parameters is approximately:

- BF16 model: 0.50 GB decimal;
- FP32 master parameters: 1.00 GB;
- Adam first and second moments: 2.00 GB;
- metadata/RNG/cursors: small relative to tensors;
- expected uncompressed full-resume artifact: roughly 3.5–4.1 GB depending on framework serialization and whether gradients are retained.

These are planning values. Target-hardware serialization measurement replaces them before freeze.

## 9. Evaluation and promotion services

Evaluation runs asynchronously against immutable milestones. Its adapter must provide:

```text
score_candidates(model, context, query, candidates) -> per-candidate log scores
generate_free(model, prompt, generation_spec) -> text
generate_constrained(model, prompt, candidates) -> text
```

The evaluator produces separate representation, address, transform, selection, and realization records. The statistical protocol hash from `e0_cognition.preregistration` is bound into every receipt.

Promotion is a separate process. It requires integrity, sealed/fresh OOD, natural transfer, substrate retention, and worst-family gates. It has no API for “promote latest.” Assisted results are separate columns and cannot satisfy a raw-Core gate.

## 10. Commands and artifact flow

Available now:

```text
python -m unittest discover -s tests -v
python -m e0_cognition.certify --output artifacts/e0/development_certificate.json
python -m v5_contracts.certify --output artifacts/v5/implementation_contract.json
python -m e1_tokenizer.audit --receipt <candidate.json> --artifact <tokenizer> --output <audit.json>
python -m e0_cognition.sealed --fixture <external-suite.json> --custody-id <id> --output artifacts/e0/sealed_commitment.json
```

The sealed command refuses fixtures located inside the repository and emits only a commitment, never seed/cases/answers.

Future command surface after gates:

```text
anra-v5 data ingest|filter|dedup|pack|verify
anra-v5 tokenizer train|audit|freeze
anra-v5 experiment plan|run|compare|close
anra-v5 train canary|start|resume|status
anra-v5 checkpoint verify|restore-canary|publish
anra-v5 evaluate milestone|fresh-replication
anra-v5 promote decide|show
```

Each command emits JSON to stdout, diagnostics to stderr, a nonzero failure code, and one immutable receipt path. Interactive notebook cells become thin callers of these commands; business logic never lives only in a notebook.

## 11. CI gates

CI is split by cost:

- `contract`: pure schema/parameter/hash tests;
- `e0`: deterministic generation, causal pairs, solver agreement, baselines, statistics, receipt reproduction;
- `e1-static`: probe schema and tokenizer-receipt audits;
- `imports`: dependency-boundary scan;
- `cpu-canary`: future tiny forward/backward/save/restore;
- `accelerator-canary`: manually authorized target-device real-update and exact-resume checks;
- `main-training`: never a CI job.

Committed generated receipts are reproduced byte-for-byte in CI. A source change without the corresponding receipt change fails.

## 12. Implementation milestones

| Milestone | Deliverable | Acceptance gate | Compute |
|---|---|---|---|
| M0 | contracts + E0/E1 harness | current CPU suite and receipts pass | CPU |
| M1 | E0 sealed commitment + natural custody | full E0 exit checklist | CPU, external custodian |
| M2 | E1 tokenizer candidates | identity/static Pareto + matched P35 evidence | CPU + bounded accelerator |
| M3 | pure model constructor | exact tensor receipt and forward parity tests | CPU/one accelerator canary |
| M4 | data/pack pipeline | deterministic rebuild, contamination, cursor receipt | CPU/storage |
| M5 | one-step trainer | real-update and optimizer ownership canaries | target accelerator |
| M6 | exact resume/durability | uninterrupted versus restore equivalence; clean download | target accelerator/storage |
| M7 | E2–E5 experiments | preregistered decisions and scale transfer | bounded accelerators |
| M8 | frozen training spec | all hashes/commands/gates fixed | review |
| M9 | V5-A main run | only after explicit authorization | expensive |

No milestone may be skipped because later code appears to work.

## 13. Immediate execution backlog

1. Obtain an independent custodian for the real T2 suite; generate it outside Git and commit only the hash.
2. Add source-disjoint natural evaluation manifests and legal provenance, not synthetic prose labeled “natural.”
3. Produce real 16k/24k/32k tokenizer candidates and encoding receipts against the external E1 corpus.
4. Close E0, then execute E1 static Pareto filtering.
5. After E1 authorizes it, implement only the P35 constructor needed for matched tokenizer/architecture experiments.
6. Do not implement the V5-A production trainer until E5 and freeze review.

## 14. Definition of implementation readiness

“Ready to implement the clean stack” means E0 is closed, E1 has a frozen tokenizer interface, model/data/checkpoint schemas are versioned, and target-framework/device choices are recorded. “Ready for main training” additionally requires E2–E5 replication, exact-resume/durability canaries, a frozen corpus, and an explicit authorization decision.

Current state: **contracts and research harness ready; production stack not authorized; main training not authorized.**
