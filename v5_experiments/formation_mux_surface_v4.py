"""FORMATION-MUX-001 Science-S4 public/sealed surface custody.

Training workers receive only a cryptographically bound public manifest with
training + development rows.  Sealed rows are deterministically regenerated
by the trusted coordinator only after development is frozen and are never
persisted into campaign output.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from v5_experiments.formation_mux_data import FAMILIES
from v5_experiments.formation_mux_surface_v2 import (
    PER_FAMILY_COUNTS,
    TOTAL_COUNTS,
    build_official_surface,
)

PUBLIC_SCHEMA = "anra.formation-mux-public-surface/v4"
EXPERIMENTS = ("CS-MECH-002", "REP-FORM-003A")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sealed_commitment(*, experiment: str, full_surface_sha256: str,
                       sealed_rows: list[Mapping[str, Any]]) -> str:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown FORMATION-MUX experiment {experiment}")
    return _sha(
        {
            "schema": "anra.formation-mux-sealed-commitment/v4",
            "experiment": experiment,
            "full_surface_sha256": str(full_surface_sha256),
            "rows": sealed_rows,
        }
    )


def build_public_surface(*, seed: int, tokenizer: Any) -> dict[str, Any]:
    """Build and commit the full deterministic surface, but persist no sealed rows."""

    if tokenizer is None or getattr(tokenizer, "identity", None) is None:
        raise RuntimeError("official public surface requires the frozen production tokenizer")
    full = build_official_surface(seed=int(seed), tokenizer=tokenizer)
    sealed_rows = list(full["splits"]["sealed"])
    full_sha = str(full["sha256"])
    public_splits = {
        "training": list(full["splits"]["training"]),
        "development": list(full["splits"]["development"]),
    }
    public_screens = {
        split: full["shortcut_screens"][split]
        for split in ("training", "development")
    }
    worst_public = max(
        float(value)
        for split in public_screens.values()
        for family in split.values()
        for value in family.values()
    )
    body: dict[str, Any] = {
        "schema": PUBLIC_SCHEMA,
        "seed": int(seed),
        "families": list(FAMILIES),
        "per_family_counts": dict(PER_FAMILY_COUNTS),
        "total_counts": {
            "training": TOTAL_COUNTS["training"],
            "development": TOTAL_COUNTS["development"],
        },
        "physical_vocabulary": 24_576,
        "tokenizer_artifact_sha256": str(tokenizer.identity.artifact_sha256),
        "full_surface_sha256": full_sha,
        "sealed_commitments": {
            experiment: _sealed_commitment(
                experiment=experiment,
                full_surface_sha256=full_sha,
                sealed_rows=sealed_rows,
            )
            for experiment in EXPERIMENTS
        },
        "shortcut_screens": public_screens,
        "worst_public_shortcut_score": worst_public,
        "splits": public_splits,
        "sealed_rows_persisted": False,
    }
    body["sha256"] = _sha(body)
    validate_public_surface(body)
    return body


def validate_public_surface(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != PUBLIC_SCHEMA:
        raise RuntimeError("public surface schema mismatch")
    claimed = manifest.get("sha256")
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    if claimed != _sha(body):
        raise RuntimeError("public surface manifest hash mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise RuntimeError("public surface splits missing")
    if "sealed" in splits:
        raise RuntimeError("SEALED_FIREWALL_BREACH: public worker surface contains sealed rows")
    if set(splits) != {"training", "development"}:
        raise RuntimeError("public worker surface must contain training + development only")
    if manifest.get("sealed_rows_persisted") is not False:
        raise RuntimeError("public surface must declare sealed_rows_persisted=false")
    if float(manifest.get("worst_public_shortcut_score", 1.0)) > 0.5:
        raise RuntimeError("public surface shortcut screen FAILED")
    commitments = manifest.get("sealed_commitments")
    if not isinstance(commitments, Mapping) or set(commitments) != set(EXPERIMENTS):
        raise RuntimeError("public surface sealed commitments missing")
    if any(len(str(commitments[e])) != 64 for e in EXPERIMENTS):
        raise RuntimeError("public surface sealed commitment malformed")
    for split in ("training", "development"):
        rows = list(splits[split])
        expected_total = TOTAL_COUNTS[split]
        if len(rows) != expected_total:
            raise RuntimeError(
                f"public surface count mismatch {split}: {len(rows)} != {expected_total}"
            )
        for family in FAMILIES:
            observed = sum(1 for row in rows if row["family"] == family)
            expected = PER_FAMILY_COUNTS[split]
            if observed != expected:
                raise RuntimeError(
                    f"public surface family count mismatch {split}/{family}: "
                    f"{observed} != {expected}"
                )
        if any("r0_prompt_ids" not in row or "r0_answer_ids" not in row for row in rows):
            raise RuntimeError(f"public surface missing production-BPE ids in {split}")
        latent_max = max(
            max([*row["prompt_ids"], *row["answer_ids"]]) for row in rows
        )
        if latent_max >= 4096:
            raise RuntimeError(
                f"public latent surface escaped shared 0..4095 region: {latent_max}"
            )


def load_public_surface(path: Any) -> dict[str, Any]:
    from pathlib import Path

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_public_surface(manifest)
    return manifest


def regenerate_sealed_rows(*, public_manifest: Mapping[str, Any], tokenizer: Any,
                           experiment: str) -> list[dict[str, Any]]:
    """Regenerate and verify sealed rows after the experiment's dev aggregate is frozen."""

    validate_public_surface(public_manifest)
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    if str(tokenizer.identity.artifact_sha256) != str(
        public_manifest["tokenizer_artifact_sha256"]
    ):
        raise RuntimeError("sealed regeneration tokenizer identity mismatch")
    full = build_official_surface(
        seed=int(public_manifest["seed"]),
        tokenizer=tokenizer,
    )
    if str(full["sha256"]) != str(public_manifest["full_surface_sha256"]):
        raise RuntimeError("sealed regeneration full-surface commitment mismatch")
    for split in ("training", "development"):
        if _canonical(full["splits"][split]) != _canonical(public_manifest["splits"][split]):
            raise RuntimeError(f"sealed regeneration public split drift: {split}")
    sealed_rows = list(full["splits"]["sealed"])
    observed = _sealed_commitment(
        experiment=experiment,
        full_surface_sha256=str(full["sha256"]),
        sealed_rows=sealed_rows,
    )
    expected = str(public_manifest["sealed_commitments"][experiment])
    if observed != expected:
        raise RuntimeError("sealed regeneration commitment mismatch")
    return sealed_rows
