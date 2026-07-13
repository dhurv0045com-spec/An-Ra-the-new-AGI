# Reclamation Decision Record

Status: explicit migration objective; final retirement review after evidence
Date: 2026-07-13

## Objective decided today

Replace V3 with the canonical V4 tokenizer as the sole target lineage for new
An-Ra training and serving. The end state is fully V4: new data, checkpoints,
training, evaluation, and inference use V4, and V3 is retired from active use.

The physical V3 deletion happens only after the V4 artifacts exist and the
migration evidence is recorded; deleting V3 before that would leave the
repository unable to load the current checkpoint or reproduce its baseline.

## What “fully V4” means

- Build the canonical append-only 32,768-token V4 from the exact campaign slice.
- Preserve V3 token IDs 0 through 8,208 without modification.
- Prove byte-safe encode/decode round trips and held-out fertility improvement.
- Publish V4-bound train, validation, and test shards with content hashes.
- Bind every V4 launch to its tokenizer hash, vocabulary size, special-token IDs,
  probe fingerprint, and matching data-manifest hashes.
- Train new checkpoints with V4 and serve those checkpoints with V4.
- Do not mix V3 tokenized data or a V3 tokenizer into a V4 run.
- Compare V4 against V3 through the pre-registered baseline and three-seed
  pilot before declaring V4 the winning production lineage.

## V3 disposition: retire and remove from active use

V3 is retained as a read-only compatibility artifact because:

- the existing checkpoint’s embedding and output rows use V3 IDs;
- existing V3 shards and manifests are V3-specific;
- V4’s append-only safety proof depends on the frozen V3 prefix;
- V3 is the control needed to measure V4’s actual benefit.

The intended final action is to remove V3 from active runtime paths and then
delete/archive its files after all of the following are complete: canonical V4
build, V4 fertility gates, V4 three-seed pilot, checkpoint migration/serving
validation, and an owner-approved retirement record. Until then, deletion
would destroy rollback and reproducibility before V4 is usable.

## Files and runtime policy

Current stable baseline:

- `tokenizer/tokenizer_v3.json`
- `tokenizer/tokenizer_v3.json.meta.json`

Current experimental artifact:

- `tokenizer/tokenizer_v4_draft.json`
- `tokenizer/tokenizer_v4_draft.json.meta.json`

Future canonical artifact:

- `tokenizer/tokenizer_v4_32k.json`
- its content-addressed metadata and campaign manifest

The draft V4 is not promoted merely by renaming it. Promotion requires the
campaign-slice, provenance, fertility, round-trip, shard, and pilot gates.

## Decisions to revisit at the end

1. Did V4 improve held-out fertility enough to justify its training cost?
2. Did V4 improve effective compute or downstream capability at matched tokens?
3. Did all three V4 seeds reproduce within the declared variance limit?
4. Can the old checkpoint be retired, or must V3 remain supported for rollback?
5. Should V4 become the repository default and production serving tokenizer?

## Architectural suggestion for later review

One possible simplification would be to treat ThirdEye as the canonical
evidence and reporting plane while retaining An-Ra’s signed launch contract as
the model-specific enforcement boundary. In that arrangement, An-Ra would
authorize and describe a run, ThirdEye would store the normalized run,
artifacts, metrics, telemetry, replication evidence, and reports, and duplicate
An-Ra reporting surfaces could eventually be reduced after parity was shown.

This is only a design suggestion for the final review. It does not prescribe a
merge, deletion, or immediate migration, and it leaves open the possibility of
keeping a local fallback if the external ThirdEye integration is unavailable.

## Immediate implementation rule

Do not point the active runtime at V4 until `tokenizer/tokenizer_v4_32k.json`
and its metadata exist. The current V4 draft is not the final replacement.
Once canonical V4 passes, change the active-tokenizer default, regenerate the
V4 shard family, migrate/newly train checkpoints, switch serving, and remove
V3 references in a single recorded migration.

## Legacy-mechanism reclamation objective

Remove the old broken training mechanisms from active code paths and replace
them with only the corrected implementations. This includes the legacy
checkpoint-continuation path, unstable initialization/residual behavior,
implicit tokenizer selection, unbound data selection, unseeded factorial
launching, shared worker evidence files, and permissive forecast outcomes.

The replacement requirements are:

- fresh scratch lineage for new training; the old checkpoint is never silently
  resumed;
- depth-scaled initialization, bounded residual behavior, stable routing, and
  explicit phase policies;
- an explicitly signed tokenizer/data/checkpoint contract for every run;
- one signed manifest and one deterministic seed per replica;
- isolated per-run metrics, evaluations, checkpoints, and recovery evidence;
- fail-closed forecast and provenance validation.

After dependency search and focused verification prove that no supported path
still requires a legacy mechanism, delete or archive that mechanism rather than
leaving a second implementation available for accidental use. Until that
reclamation gate passes, legacy files may remain only as quarantined migration
references and must not be reachable from the canonical trainer or serving
entry points.

## Three-seed usage clarification

The three-seed mechanism is intended for controlled comparison of candidate
training configurations. It is not required for every ordinary training
session. The factorial cells represent different recipes; the three seeds test
whether each recipe is stable rather than lucky. After a recipe passes its
replication and capability gates, routine training may use the selected recipe
with its normal session seed policy.

## Data-pipeline purpose and replacement

The data pipeline exists to make training inputs complete, legal, traceable,
and repeatable. It replaces the former loose corpus flow in which a large file
could be incomplete, duplicated, weakly licensed, source-mixed, or changed
without the trainer proving what it actually consumed.

The intended output is a trusted data package: a resume-safe corpus and audit
index, source/license manifests, hash-chained audit evidence, deterministic
train/validation/test splits, source-pure token shards, a measured V4 campaign
slice, and signed inventories that bind the trainer to an exact token budget
and source mix. This package does not create intelligence by itself; it makes
the model’s training evidence trustworthy and reproducible.

## Non-negotiable safety rule

No file deletion, ID renumbering, checkpoint surgery, or default-tokenizer
switch is final until the evidence above is recorded and owner-approved.
