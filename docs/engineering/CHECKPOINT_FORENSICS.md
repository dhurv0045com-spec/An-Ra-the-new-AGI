# An-Ra 500M Checkpoint Forensics and Recovery Record

This document is the durable evidence record for the legacy 500M checkpoint.
It separates verified defects from hypotheses and defines the experiments that
must pass before any continuation or replacement model is promoted.

The corresponding execution roadmap is
[`MODEL_RECOVERY_AND_TRAINING_BLUEPRINT.md`](MODEL_RECOVERY_AND_TRAINING_BLUEPRINT.md).

## Immutable artifact identity

| Field | Value |
|---|---|
| Checkpoint | `C:\Users\ankit\Downloads\anra_frontier_500m.pt` |
| Size | 2,000,680,247 bytes (1.863 GiB) |
| SHA-256 | `648354a42d68c22769450a3aaa249e93689b21fbe72e68b07dcc15c6f7f4d393` |
| Recorded source commit | `e8d90d9dbfb5694477f142e72d53f010af175682` |
| Source lineage | The recorded commit exists and is an ancestor of the current branch. |
| Training state | step 6,927; 8 sessions; `best_loss=0.32787953184122776` |
| Model contract | 499,167,047 parameters at runtime; 8,209-token V3 vocabulary |
| Data metadata | `t4-cached`; `bucket_packed_v1` |

The original artifact is evidence and must remain unchanged outside git. Any
migration or recovery candidate must be written to a new content-addressed
artifact and retain this hash as its parent lineage.

## Structural load verdict

The current schema-6 runtime safely loaded the schema-4 checkpoint and proved:

- all 608 target tensors are accounted for;
- zero missing, unexpected, or shape-mismatched tensors;
- exact core and exact native loads;
- all legacy vocabulary rows are preserved;
- embedding and LM-head shapes are both `[8209, 1280]`;
- all 500 external tokenizer probes match fingerprint `db1075ad...`.

Therefore malformed language is **not explained by a truncated weight file or
a current tensor-shape mismatch**. The checkpoint remains legacy evidence: it
does not embed its vocabulary hash, 500-probe fingerprint, validation history,
token count, or complete restorable corpus manifests.

## Saved-weight pathology profile

`scripts/profile_checkpoint_pathologies.py` inspected the original state and
wrote content-hashed evidence to
`output/v2/checkpoint_pathology_profile.json` (report hash
`9fb39287d45d7e49556e098f074ede57a9b47ac8f11a72d9f6df959a8a9f29b4`).
The 608 serialized tensor entries contain zero NaN or Inf values. Serialized
state elements are not reported as unique parameters because tied/aliased
weights and parametrization state appear under multiple keys.

Measured architecture state:

- every one of the 12 saved router `context_weights` vectors is exactly zero;
- residual-depth scales range from 0.36115 to 1.63752;
- DSTP temperatures exactly follow the initialization curve from 1.35 down to
  0.65;
- `layer_temperature_bias` is exactly 1.0 in all 28 layers;
- saved ESV state is `[0.0, 0.0, 0.0]`;
- RIM strengths are finite and unsaturated, ranging from -0.17346 to 0.15292.

The profile raises `router_context_dormant` and `residual_scale_extreme` as
high-severity investigation alerts. It does not claim that scale magnitude
alone proves the cause of incoherence; that requires the registered ablation.

## Behavioral observations

- The owner reports that interactive answers could not reliably form a
  sentence despite low displayed loss.
- A deterministic CUDA diagnostic smoke for “Explain why 2 + 2 equals 4.”
  produced `ANRA: The number of ways to create` after 16 tokens. The runtime
  classified the trace as `quality_state=rejected`.
- The completed RTX 4050 recovery audit ran 600 cache-off generations on
  2026-07-10: 200 diagnostic, 200 native, and 200 deterministic-replay prompts
  with greedy decoding and seed 0. The prompt-suite SHA-256 is
  `0bcd88f6c4d77fd7265f371ae3e6b0865f7988830b88dca44c4457c6858449b9`.
  The schema-6 result was **failed: undertraining** at 0.0% coherence against
  the 80% threshold. After the named schema-7 repair, the complete CUDA replay
  again failed at 0.0% coherence; diagnostic/native/replay acceptance were all
  0.0% and diagnostic/native EOS failure was 100%. Finite activations and
  deterministic replay passed. The final atomic report is
  `output/v2/stream_a_forensics.json` (generated 2026-07-11 02:51:32 local
  time). This rules out promotion or a
  claimed recovery of this checkpoint; it does not prove that every individual
  architecture alert caused the failure.

## Why the low loss is not evidence of language ability

### F-1 — `best_loss` is training EMA, not held-out validation (verified)

At the recorded source commit, `scripts/build_brain.py` updates `best_loss`
from the minimum exponential moving average of optimizer-step **training
loss**. The checkpoint contains no `best_validation_loss` and an empty
validation history under the current compatibility loader. Immutable
validation shards and validation history were added later in recovery commit
`bdfc1a8`.

The historical `quick_eval_loss` was also passed the training dataset `ds`.
Startup evaluation defaulted to zero examples, and session-end evaluation used
the first 100 windows from that same dataset. Its “validation” docstring did
not make it independent validation.

### F-2 — the training exposure was far too small for 499M parameters (verified bound)

The source configuration used microbatch 1, accumulation 8, context 1,024,
and at most 4,096 mixture examples per session. Given step 6,927, even assuming
every accumulated window was completely full, the run could have processed at
most:

`6,927 × 8 × 1,024 = 56,745,984 target token positions`.

That is at most 0.114 target positions per parameter and far below the current
10B/30B-token pilot/campaign scale. The real count is likely lower because
windows contain padding and the checkpoint did not record `tokens_seen`.
Repeated epochs over a small sampled mixture can drive training CE down while
leaving free generation incoherent.

### F-3 — the objective contains easy repeated scaffolding (verified)

Every conversational example used the repeated template
`H: {prompt}\nANRA: {answer}`. Prompt tokens retained weight 1.0 while answer
tokens were normally weighted 1.75. The reported scalar therefore mixes
memorized prompt/scaffold prediction with answer prediction. It is not
answer-only loss, sentence coherence, instruction following, or held-out
generation quality.

## Verified historical architecture and pipeline defects

### F-4 — gradient-checkpoint backward used the wrong layer temperature (critical)

Gradient checkpointing was enabled. In commit `e8d90d9`, the checkpointed
closure captured loop variable `i` late and called `_dstp_temperature(i)`.
Forward execution used the current layer, but backward recomputation occurred
after the loop advanced, so layers 0–26 could be recomputed with layer 27’s
DSTP temperature. This violates checkpoint recomputation equivalence and can
produce gradients for a different function than the forward pass.

Recovery commit `bdfc1a8` fixed this by binding `layer_i_: int = i` in the
closure and using `_dstp_temperature(layer_i_)`. New training must prove
checkpointed/non-checkpointed forward and gradient parity before launch.

### F-5 — routing lacked its claimed controls and anti-collapse loss (high)

The checkpoint-era MoD path did not pass `RouterContext`, leaving contextual
routing weights functionally unused. It also had no balance loss or router
z-loss. Recovery commit `bdfc1a8` added context-aware routing telemetry,
balance/z regularization, bounded native regularization, and explicit runtime
modes. The legacy checkpoint never received those training signals.

### F-6 — validation could not block checkpoint publication (high)

The trainer persisted and synchronized the checkpoint before compact
evaluation. Compact-eval exceptions were converted to an evaluation report but
did not invalidate the already-published training artifact. A low training
loss could therefore become the visible success signal even when behavioral
evaluation failed or was absent.

### F-7 — session ends changed the effective optimizer batch (medium)

The historical trainer forced an optimizer step when a timed session ended
with fewer than all eight accumulation microbatches. This creates variable
effective batches and scheduler steps across interruptions. The recovery
pipeline removed this behavior and keeps checkpoints on the last complete
optimizer boundary.

### F-8 — permissive loading hid risk, but did not damage this artifact (resolved)

The checkpoint-era loader used `strict=False` without today’s complete tensor
disposition proof. That could make partial loads look successful. The new
forensic proof shows this specific checkpoint has no missing or mismatched
target tensors, so permissive loading is a historical process defect rather
than the cause of this artifact’s incoherence.

### F-9 — RIM spectral normalization changed checkpoint recomputation (critical, fixed)

The new forward-and-gradient parity regression found another current training
defect while reconstructing F-4. RIM uses parametrized spectral normalization,
which advances its power-iteration state on every training-mode forward.
Activation-checkpoint backward ran the RIM projection a second time, advanced
that state again, and recomputed a slightly different function. Logits matched
before backward, but RIM gradients differed from a non-checkpointed control.

The runtime now supplies a recomputation context that freezes RIM spectral-norm
state updates only during checkpoint backward recomputation. The original
forward still performs its normal single update. A permanent regression compares
the complete logits, total loss, gradient presence, and every parameter gradient
between checkpointed and plain models; 23 focused architecture tests pass.

### F-10 — advertised trainable temperature controls were buffers (high, fixed in schema 7)

At the checkpoint commit, both `dstp_temperature_log` and
`layer_temperature_bias` were registered buffers. The latter still carries a
source comment saying each block should “learn” its influence, but an all-ones
buffer cannot receive optimizer gradients. The saved profile confirms no
learning occurred: the layer bias is uniformly 1.0 and DSTP is its exact
initial schedule. Recovery code first made DSTP trainable with an anchor
regularizer. Schema 7 now replaces the remaining direct-scale buffer with a
positive log-space parameter, clamps its realized multiplier to `[0.5, 2.0]`,
anchors it to neutral with native regularization, reports it in telemetry, and
places it in the subsystem optimizer group. The named migration converts a
legacy positive scale with `log(scale)` and rejects non-finite or non-positive
values. The 28 newly trainable scalars update the current V3 contract from
499,167,047 to 499,167,075 parameters; legacy artifact identity remains
499,167,047. Focused migration/gradient/optimizer tests and the full 590-test
non-GPU suite pass. This creates a valid pilot candidate; removal versus
trainable-control ablation is still required before a campaign winner is
frozen.

### F-11 — raw campaign training loader selected validation shards (critical, fixed)

The current repaired trainer had a separate boundary defect not present in the
legacy `bucket_packed_v1` run. When `raw_causal_shards_v1` was selected, it
correctly constructed distinct `ds` and `eval_ds` objects, but the default
training-loader branch passed `eval_ds` to `DataLoader`. A future foundation
campaign would therefore optimize directly on its immutable validation shard.

The loader now always receives `ds`. A runtime identity assertion fails closed
if any training loader selects an unknown or validation dataset, and a focused
regression injects the validation loader to prove rejection. No raw-shard
campaign had been launched, so this was prevented before contaminating a new
checkpoint.

### F-12 — validation collapsed answer quality into aggregate CE (high, fixed)

The repaired trainer still returned only aggregate validation cross-entropy.
Although conversational training applied answer weights, validation discarded
both those weights and the identity of answer tokens. Prompt/scaffold tokens
could therefore lower the reported validation loss while answer generation
remained poor—the same class of evidence failure exposed by the legacy
checkpoint.

Conversational packing now carries an explicit boolean answer mask independent
of numeric loss weights through GPU and TPU loaders. Training records exact
answer/scaffold token counts and losses; validation reports total, weighted,
answer-only, and scaffold-only CE with token denominators. Checkpoints expose
`best_answer_validation_loss` separately from total validation loss and retain
the semantics. Raw foundation shards intentionally emit an all-false answer
mask because ordinary causal text has no answer boundary. The full non-GPU
suite passes with 593 tests and one skip.

### F-13 — conversational validation reused training examples (critical, fixed)

After F-11 corrected raw-shard selection, the conversational D/E path still
assigned `eval_ds = ds`. It therefore had no held-out boundary at all. The
pipeline now groups records by declared source/document/content hash when
available, otherwise by normalized source+prompt+answer SHA-256, and assigns
whole groups before tokenization. It emits a content-hashed split manifest with
train/validation group lists, per-bucket counts, and an asserted empty overlap.
GPU and TPU training consume only the training side; immutable validation uses
a distinct dataset instance bound to the split hash.

Stage promotion no longer accepts `protected_regression` or
`validation_regression` claims supplied as booleans/scalars. It compares a
newer candidate against its same-identity baseline, requires every protected
domain, caps total/domain regression at 2%, and requires answer-domain evidence
for conversational/replay stages. Missing, reused, changed-identity, non-finite,
or regressed evidence blocks promotion. Full non-GPU verification: 597 passed,
1 skipped.

## Open hypotheses requiring measurement

- Router collapse or pathological residual/temperature values may already be
  encoded in the learned weights. Inspect per-layer router selection entropy,
  residual scales, ESV state, and activation norms on a fixed prompt set.
- The small `t4-cached` mixture may contain duplicates, narrow templates, or
  source imbalance not reconstructable from the two legacy manifest hashes.
- V3’s measured English fertility tax (2.518 tokens/word) likely reduced
  effective linguistic coverage, but it cannot by itself explain all
  incoherence.
- The displayed historical losses of 0.8–1.3 may refer to other session or
  dashboard values; only `0.3278795` is embedded as this artifact’s
  `best_loss`. No conclusion should combine those numbers without the original
  run logs.

## Recovery and replacement program

1. **Finish baseline behavior.** Complete the 200-prompt diagnostic recovery
   gate and save every output, token ID, stop reason, and coherence judgment.
2. **Profile the legacy weights.** Record activation norms, router entropy,
   residual scales, temperatures, NaN/Inf counts, output entropy, and EOS rate.
3. **Do not blindly continue the old optimizer.** Preserve the checkpoint as
   a baseline. Any continuation must start from a named, tested migration and
   a fresh optimizer unless an optimizer-state audit proves exact compatibility.
4. **Prove the repaired trainer at small scale.** Run checkpoint-on/off
   gradient parity, overfit a tiny clean corpus intentionally, then run at
   least three seeds on a held-out pilot. Training loss and validation loss
   must be separately source-stratified.
5. **Use immutable data boundaries.** Train/validation/test must be split by
   source hash before tokenization, deduplicated across splits, and bound to
   the checkpoint with complete manifest payloads.
6. **Select on behavior, never minimum training loss.** Promotion requires
   held-out CE, coherent-response rate, EOS/format compliance, executable
   math/code verification, contamination checks, and blinded comparisons.
7. **Compare continuation against scratch.** At equal verified tokens and
   compute, compare a repaired continuation with a scratch control. Retain the
   old weights only if they improve held-out capability without protected
   regressions.
8. **Advance the canonical tokenizer only through the registered pilot.** V4
   must preserve IDs 0–8208 and beat V3 on fertility-to-effective-compute with
   three seeds before the main campaign.

## Promotion blockers

The checkpoint must not be called recovered or production-ready until all of
the following are true:

- deterministic coherence ≥80% for the recovery decision and ≥90% for final
  promotion;
- independent validation and test manifests are present and hash-verified;
- checkpointed/non-checkpointed gradient parity passes;
- router/activation telemetry shows no collapse or numerical pathology;
- three-seed training evidence beats both the legacy baseline and ablations;
- the signed release bundle includes checkpoint, tokenizer, corpus,
  configuration, evaluation, and rollback artifacts.

This record is intentionally strict. A low loss on a broken or repeated
training surface is a debugging clue, not an intelligence result.
