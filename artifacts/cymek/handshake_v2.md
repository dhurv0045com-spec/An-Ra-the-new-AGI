# CymekSubjectHandshakeV2 — what Triquetra must adopt

Status: **proposed by Cymek, not yet adopted by Triquetra.**
Direction stays: Cymek hands subjects; Triquetra qualifies cognition.
Cymek does not modify Triquetra; this document is the exact compatibility
specification. No schemas are copied silently: field mappings below are
explicit, and drift in either direction fails closed.

## Why V2 exists

V1 (`anra-v5-core-subject-manifest/v1`) forces
`checkpoint_sha256 == checkpoint_file_sha256`, but a `CheckpointStore`
checkpoint is a content-addressed **object directory** whose identity derives
from its manifest bytes. There is no single "checkpoint file". V1 stays
historical. V2 (`anra-v5-core-subject-manifest/v2`) names exactly what each
hash covers.

## V2 custody fields (new, explicit)

| V2 field | Hashes |
|---|---|
| `checkpoint_object_sha256` | content-addressed object directory identity (store manifest SHA) |
| `checkpoint_manifest_sha256` | store `manifest.json` bytes (equals object identity by construction) |
| `model_payload_sha256` | `model.bin` bytes |
| `optimizer_payload_sha256` | `optimizer.bin` bytes |
| `parameter_sha256` | live parameter bytes recomputed after load (name-ordered, FP32) |
| `training_state_sha256` | `training_state.json` bytes |

All V1 lineage fields are retained unchanged:
`model_spec_sha256`, `tokenizer_artifact_sha256`, `tokenizer_identity_sha256`,
`training_spec_sha256`, `data_manifest_sha256`, `pack_manifest_sha256`,
`optimizer_spec_sha256`, `schedule_spec_sha256`, `curriculum_spec_sha256`,
`source_commit`, `parent_checkpoint_sha256`, `global_update`,
`cumulative_training_tokens`, `stage`, `seed`, `custody`,
`creation_receipt_sha256`.

## What Triquetra must adopt

1. Accept `anra-v5-core-subject-manifest/v2` subjects alongside V1.
2. Extend its manifest validator (`triquetra branch:
   x_factor/manifest_validator.py`, currently the 14-field V1 set) with the
   six V2 custody fields above; reject subjects where
   `checkpoint_object_sha256 != checkpoint_manifest_sha256`.
3. Verify artifacts, not shapes: for a runnable subject, re-hash the
   checkpoint object, both payloads, the tokenizer artifact, and the model
   spec, and compare against the manifest (reference implementation:
   `v5_registry.subject_v2.verify_subject_artifacts`).
4. Keep the observed-only firewall unchanged: V2 changes custody proof,
   never evaluation semantics. Cymek still never trains against Triquetra
   diagnostics without a preregistered treatment.

## Compatibility map (V1 → V2)

- V1 `checkpoint_sha256` / `checkpoint_file_sha256` → V2
  `checkpoint_object_sha256` (= `checkpoint_manifest_sha256`).
- V1 `parameter_sha256` → V2 `parameter_sha256` (same bytes definition) plus
  `model_payload_sha256` (serialized bytes, distinct).
- All other V1 fields map by identical name.
- A V1 subject is NOT auto-upgradable: the payload-level hashes require the
  checkpoint object, which V1 never bound.
