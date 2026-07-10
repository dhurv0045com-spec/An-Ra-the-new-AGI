# Model Recovery and Training Blueprint

## Purpose and honest scope

This is the operating blueprint for turning the repaired An-Ra training system
into a useful, measured language-model program. It is based on the completed
500M checkpoint forensic record and the executable master plan. It does **not**
claim that a particular architecture, parameter count, or training run will
produce AGI. "Full potential" here means that every component is trained,
measured, gated, and retained only when it improves independently measured
capability, reliability, efficiency, or safety.

The legacy 500M checkpoint is a baseline artifact, not a starting point to
blindly resume. It loaded exactly but failed the GPU behavioral gate with 0.0%
coherence against an 80% threshold. Its historical `best_loss` is training
EMA, not held-out validation. See
[`CHECKPOINT_FORENSICS.md`](CHECKPOINT_FORENSICS.md) for immutable evidence.

## Executive decision

1. Preserve `anra_frontier_500m.pt` unchanged and never promote it.
2. Do not reuse its optimizer state or select recipes by its legacy loss.
3. Establish a reproducible scratch baseline first, then compare a **fresh
   optimizer continuation** against scratch at identical verified token counts.
4. Scale only configurations that pass the same immutable validation,
   behavioral, reproducibility, and rollback gates across three seeds.
5. Treat advanced subsystems (MoD/router, RIM, DSTP, MTP, MoE, latent reasoning,
   world models, continual learning) as ablations or pilot branches, not as
   assumptions.

## What failed previously

| Area | Verified failure or evidence | Corrective rule |
|---|---|---|
| Capability measurement | `best_loss=0.32788` was minimum training-loss EMA, while validation reused training data and could not block checkpoint publication. | Select only on immutable validation plus behavioral/verifier gates. |
| Training scale | The recorded upper bound is only `6,927 × 8 × 1,024 = 56.75M` target positions, or at most 0.114 positions/parameter. | Log exact non-padding tokens and train a deliberate token budget. |
| Data objective | Repeated `H: prompt / ANRA: answer` scaffolding mixed easy prompt prediction with answer prediction. | Explicit token masks now separate answer/scaffold loss; add per-domain reporting before campaign launch. |
| Gradient checkpointing | A late-bound layer index changed DSTP temperature during backward recomputation; RIM spectral normalization also advanced state on recomputation. | Keep the fixed parity harness mandatory before every architecture/training change. |
| Routing | Router context was not passed; no balance or z-loss trained the old router. Saved router context vectors are all zero. | Measure route entropy/load/context sensitivity and compare routed versus dense controls. |
| Temperature controls | DSTP and layer-temperature controls were buffers rather than learned values in the historical model. | Either intentionally freeze and document them, or make them parameters with tests proving optimizer updates. |
| Data boundary | Raw-shard training selected the validation dataset. | Keep the new loader-identity fail-closed assertion and source-hash split checks. |
| Artifact integrity | The legacy schema lacked full corpus/validation/token metadata and used permissive loading. | Every new checkpoint must carry exact manifests, token counts, validation history, tokenizer fingerprint, config hash, and tensor disposition proof. |
| Behavior | The completed RTX 4050 audit had 0.0% coherence, 97.0% diagnostic EOS failure, and only 3.0% native acceptance. | A model that cannot pass the behavioral gate is undertrained regardless of scalar loss. |

## Training target and token budget

The legacy checkpoint contract is 499,167,047 parameters. The repaired schema-7
candidate is 499,167,075 parameters because its 28 per-layer temperature
controls are now genuinely trainable. The program should use two explicit scales:

| Stage | Verified-token budget | Purpose | Promotion requirement |
|---|---:|---|---|
| Tiny smoke | 5M–20M | Prove loss decreases, exact resume, no NaN/Inf, data and checkpoint contracts. | Tiny-corpus overfit + checkpoint-on/off gradient parity. |
| Pilot ladder | 50M and 150M, three seeds per cell | Choose tokenizer, optimizer, routing and curriculum candidates. | Pre-registered forecast; held-out benefit versus dense AdamW baseline. |
| Foundation campaign | 20B Phase A + 6B Phase B anneal | Learn broad language/code/math capability. | Validation, behavior, data-quality, throughput and recovery gates. |
| Capability curriculum | 2.5B reasoning + 1B conversation + 0.5B tools | Add useful task behavior only after a stable foundation. | Domain-specific verifiers and protected general capability. |

The full planned curriculum is 30B tokens. It is a decision target, not proof
of intelligence. The first scaling law must be measured from the 50M/150M
ladder; stop or revise the campaign if observed validation loss/capability
falls materially outside the pre-registered forecast. Record **actual
non-padding tokens**, unique-token coverage, source/domain token counts, and
tokens/second in every checkpoint.

The 6GB RTX 4050 is suitable for unit tests, tokenizer/data validation,
small-model pilots, and inference gates. It is not a practical full-training
device for a 500M model campaign. Use the existing cluster/TPU route or a
larger rented accelerator only after the small-scale gates demonstrate that the
recipe is worth the compute.

## Data: quality, legality, and split discipline

Build immutable, content-addressed shards before training. The target mix from
the existing data program is:

| Source class | Share |
|---|---:|
| Educational web foundation | 55% |
| Permissively licensed code | 15% |
| Verified math | 12% |
| Science and technical text | 8% |
| Verified general instructions | 5% |
| Verified DFC | 3% |
| Identity/replay material | 2% |

Required data pipeline, in this order:

1. License allowlist and provenance record for every raw source.
2. Language, PII, malware/secret, repetition, boilerplate, quality, and
   domain-specific filters; parse code where applicable.
3. Exact and near-duplicate removal before splitting.
4. Train/validation/test allocation by source hash before tokenization. No
   shard, source family, or deduplicated document may cross a boundary.
5. Immutable manifests with raw/source hashes, filter versions, token counts,
   tokenizer hash, shard hashes, and split membership.
6. Held-out tokenizer fertility and round-trip tests before accepting V4.

V3's measured English fertility is 2.518 tokens/word, far above the 1.35 goal.
The append-only V4 migration preserves IDs 0–8208 and is the path to improve
that tax, but it must be chosen by the registered 150M pilot on representative
campaign data—not merely by a small local corpus result. Synthetic data and
model self-training must stay verifier-backed and capped at 30% of each phase;
never let synthetic output become a self-confirming majority.

## Codebase changes that remain mandatory

### Non-negotiable invariants

- `scripts/build_brain.py` must reject unknown or validation training datasets.
- Checkpoint payloads must distinguish `best_training_loss` from
  `best_validation_loss`, include loss semantics, full manifests, tokenizer
  identity, exact token counters, seed, config, code revision, and evaluation
  evidence.
- Resume is allowed only from a complete optimizer boundary and matching
  model/tokenizer/data-profile contracts.
- Every checkpoint migration must account for all tensors; no silent
  `strict=False` acceptance.
- Every activation-checkpoint path must pass logits/loss/all-gradient parity
  against non-checkpointed execution.
- Failed evaluation, manifest, signature, or rollback verification blocks
  publication and promotion.

### Architecture policy

Start with the simplest proven dense baseline: the current 500M GQA/SWA
backbone with fixed, tested attention and tokenizer contracts. Instrument it
before adding complexity. For each layer/module, record activation RMS,
attention entropy, gradient RMS, update ratio, NaN/Inf counts, parameter norm,
router load/entropy, and latency/memory cost.

Adopt an architectural component only when its three-seed pilot improves a
predefined protected score at matched tokens and compute:

| Component | Required evidence | Decision if it fails |
|---|---|---|
| Router/MoD | Context changes routes; no collapse; balance/z metrics stable; improves held-out capability. | Run dense path. |
| RIM/DSTP/temperature controls | Gradient parity, optimizer-update proof where trainable, no numerical regression. | Freeze/remove from campaign config. |
| Muon optimizer | Better validation/capability per token than AdamW baseline in three-seed pilot. | Dense AdamW fallback. |
| V4 tokenizer | Frozen-prefix migration, clean round trip, representative fertility gain, >=1.3x effective compute in pilot. | Proven append-only 16k fallback or V3 baseline. |
| MoE/upcycling | Better capability per active FLOP with stable expert load and reproducible routing. | Dense model. |
| MTP/think/reasoning heads | Verifier-backed task gain without loss of protected general benchmarks. | Disable. |
| Moonshots | Their own published acceptance and kill criteria. | Keep off the critical path. |

No component receives credit because it sounds cognitive. The useful
architecture is the smallest configuration that wins its protected ablation.

## Launch sequence

### 0. Freeze the experiment contract

For each run, emit a signed/hashed manifest before launch containing: seed,
model config, code commit, tokenizer and data manifests, optimizer/scheduler,
batch/sequence settings, target tokens, evaluation suite hashes, expected
curve, stop criteria, checkpoint cadence, and rollback parent. Register the
forecast before compute begins.

### 1. Prove the trainer on a tiny corpus

Run a deliberate overfit test on a tiny clean corpus. It should overfit only
that corpus, replay exactly after restart, and never contaminate validation.
Run identical mini-jobs with activation checkpointing on/off; logits, loss and
all gradients must match within the established tolerance. Deliberately inject
wrong manifest, tokenizer, validation loader, interrupted accumulation, and
corrupt checkpoint cases; each must fail closed.

### 2. Establish the dense scratch baseline

Train 50M then 150M dense AdamW runs across three seeds. Produce:

- train, answer-only, validation, and per-domain losses over token count;
- 200-prompt behavioral gate samples with saved token IDs and stop reasons;
- math/code/format verifier scores and calibration;
- throughput, peak memory, exact-resume and kill-minus-nine recovery evidence;
- data mixture, duplicate rate, contamination checks, and all manifest hashes.

Do not advance because loss falls. Advance only if the model forms coherent
sentences, passes the behavioral threshold, and protects validation quality.

### 3. Run the factorial pilot, then freeze one recipe

Use `training/pilot_factorial.py` to pre-register the model/tokenizer/optimizer
cells and three seeds. Change one causal factor at a time relative to the dense
baseline; include continuation-versus-scratch as an explicit comparison. A
winner needs a statistically and practically meaningful held-out benefit,
stable training, and no protected-metric regression. Record negative results;
they prevent repeated waste.

### 4. Scale in curriculum phases

Scale the frozen winner through the 30B curriculum only with live gates between
phases. Phase A/B trains foundation competence; later reasoning, conversation,
and tool phases must preserve a protected general suite. Checkpoint only after
complete optimizer boundaries and successful immutable-validation evaluation.
At every phase boundary, require a new behavior report, tokenizer/data/config
identity check, rollback drill, and human review of failure samples.

### 5. Post-train only after a competent base exists

SFT, RLVR/GRPO, STaR, DPO, self-distillation, adapters, and continual replay
are refinements—not substitutes for foundation data. Each must beat a finite
ablation and retain a replay mixture from protected foundation data. Never train
on unverified model traces or user data without provenance, consent, filtering,
and contamination controls.

## Evaluation and release gates

Loss is a diagnostic, not a release decision. A candidate must pass all of:

1. Exact checkpoint/tensor/tokenizer/data lineage and complete manifests.
2. Immutable validation and untouched test performance, stratified by domain.
3. Deterministic replay plus a fixed behavioral suite; save prompts, outputs,
   token IDs, decoding settings, judgments, and verifier traces.
4. Safety/reliability tests for tool execution, isolation, prompt injection,
   data leakage, and rollback.
5. Three-seed reproducibility for any recipe or architecture promotion.
6. Performance evidence: measured throughput, latency, memory, and cost—not
   estimates substituted for measurements.
7. Signed release bundle, independently verified rollback, and an explicit
   owner authorization for irreversible deployment actions.

## What success means

The near-term success criterion is not "AGI." It is a model that reliably
speaks, follows formats, uses verifiable tools safely, preserves lineage, and
improves on held-out tasks under reproducible evaluation. Broader intelligence
claims require sustained external evidence: robust generalization, calibrated
uncertainty, safe tool behavior, adversarial testing, human evaluation, and
independent replication. Until then, report the measured capability and the
remaining uncertainty plainly.

## Immediate next actions

1. Keep the failed 500M checkpoint frozen as a forensic baseline.
2. Finish and validate the licensed corpus acquisition; do not train from the
   current insufficient local slice.
3. Produce immutable train/validation/test manifests and build V4 only after
   representative fertility evidence.
4. Run the tiny trainer proof, then the 50M/150M three-seed dense baseline.
5. Launch only pre-registered pilot cells; promote only measurable winners.
6. Scale the selected recipe only after throughput and recovery drills pass.

## Related records

- [`CHECKPOINT_FORENSICS.md`](CHECKPOINT_FORENSICS.md) — artifact identity,
  exact load proof, and verified legacy defects.
- [`../IMPROVEMENT.md`](../IMPROVEMENT.md) — recovery gates and data
  program.
- [`../planning/MASTER_UPGRADE.md`](../planning/MASTER_UPGRADE.md) — master
  workstreams, pilot registry, and campaign phases.
- [`../planning/IMPLEMENTATION_TODOS.md`](../planning/IMPLEMENTATION_TODOS.md)
  — live execution state.
