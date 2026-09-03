"""Cymek CoreSubjectManifest validator (MISSION 21).

Triquetra must verify every required identity field before accepting a
Cymek checkpoint for scientific use. No placeholder values. Fail closed.
"""

REQUIRED_FIELDS = frozenset({
    "schema", "checkpoint_file_sha256", "parameter_sha256",
    "model_spec_sha256", "tokenizer_artifact_sha256",
    "tokenizer_identity_sha256", "training_spec_sha256",
    "data_manifest_sha256", "pack_manifest_sha256",
    "source_commit", "cumulative_training_tokens",
    "global_update", "stage", "seed",
})
PLACEHOLDER_VALUES = frozenset({"UNFILLED", "", None, "TODO", "PENDING"})


def validate_manifest(manifest: dict) -> dict:
    missing = sorted(REQUIRED_FIELDS - set(manifest.keys()))
    placeholders = sorted(
        k for k in REQUIRED_FIELDS & set(manifest.keys())
        if str(manifest.get(k, "")).strip() in PLACEHOLDER_VALUES
    )
    return {
        "valid": not missing and not placeholders,
        "missing_fields": missing,
        "placeholder_fields": placeholders,
        "checked_fields": sorted(REQUIRED_FIELDS),
    }
