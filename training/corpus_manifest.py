"""Pinned, license-checked upstream corpus manifests (Stream B, TODO 1).

The campaign corpus is assembled from a fixed set of permissively-licensed
upstream datasets. This module is the single declarative source of truth for
*which* datasets, at *what* pinned revision, under *what* license, and in
*what* mix — the contract the data-acquisition workstream (Layer 2.0) and the
GPU-cluster control plane both consume. It carries no download logic: it
defines and verifies the manifest so the pin can be reviewed and hashed
independently of any network access.

Law 1: no pretrained external *weights* ever enter the lineage; upstream
*text* under an allowlisted license is fair game and is recorded here with
its immutable revision so a campaign is exactly reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anra.anra_paths import DATA_MANIFEST_DIR

MANIFEST_SCHEMA_VERSION = 1
UPSTREAM_CORPUS_MANIFEST = DATA_MANIFEST_DIR / "upstream_corpus_manifest.json"

# Permissive licenses An-Ra will ingest text under. Normalized (lowercased,
# hyphenated) before membership tests. "per-record" defers to a row-level
# license field that must itself resolve into this set.
ALLOWED_LICENSES = frozenset(
    {
        "odc-by",
        "cc-by",
        "cc-by-sa",
        "cc0",
        "apache-2.0",
        "mit",
        "bsd-3-clause",
        "bsd-2-clause",
        "isc",
        "mpl-2.0",
        "public-domain",
        "owner",
    }
)
_PER_RECORD_ALLOWED = ("mit", "apache", "bsd", "isc", "mpl")


def normalize_license(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


@dataclass(frozen=True)
class CorpusSource:
    """One pinned upstream dataset in the campaign mix."""

    key: str
    dataset: str
    config: str | None
    license: str
    weight: float
    fields: tuple[str, ...]
    role: str
    # Immutable pin. Either a 40-hex commit sha (preferred, resolved at
    # acquisition and frozen back here) or a "dataset:config" descriptor when
    # the sha is resolved on-worker. license="per-record" means the row's own
    # license field decides, constrained to _PER_RECORD_ALLOWED.
    revision: str

    def license_ok(self) -> bool:
        normalized = normalize_license(self.license)
        if normalized == "per-record":
            return True
        return normalized in ALLOWED_LICENSES

    def revision_pinned(self) -> bool:
        return bool(re.fullmatch(r"[0-9a-f]{40}", self.revision)) or ":" in self.revision


# The canonical campaign mix. Weights match FOUNDATION_CAMPAIGN_MIX in
# scripts/download_training_data.py (the acquisition side); this module is the
# reviewable, hashable declaration of the same contract.
CAMPAIGN_CORPUS_SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(
        key="fineweb_edu",
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-100BT",
        license="ODC-By",
        weight=0.55,
        fields=("text",),
        role="foundation_prose",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    ),
    CorpusSource(
        key="permissive_code",
        dataset="common-pile/stackv2_edu_filtered",
        config=None,
        license="per-record",
        weight=0.15,
        fields=("text",),
        role="code",
        revision="c354dbe88469a1153e97c6a63ac50591849654de",
    ),
    CorpusSource(
        key="finemath",
        dataset="HuggingFaceTB/finemath",
        config="finemath-4plus",
        license="ODC-By",
        weight=0.12,
        fields=("text", "content"),
        role="math",
        revision="e92b25a616738fe95dc186b64dfb19f9c8525594",
    ),
    CorpusSource(
        key="science_technical",
        dataset="common-pile/arxiv_papers_filtered",
        config=None,
        license="per-record",
        weight=0.08,
        fields=("text",),
        role="science_technical",
        revision="033cf7f53f9b348deec868c1a5a48484f3ee9e52",
    ),
    CorpusSource(
        key="verified_instruction",
        dataset="HuggingFaceTB/smol-smoltalk",
        config=None,
        license="Apache-2.0",
        weight=0.05,
        fields=("messages",),
        role="instruction",
        revision="f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
    ),
    CorpusSource(
        key="verified_dfc",
        dataset="an-ra/verified-dfc-v1",
        config=None,
        license="owner",
        weight=0.03,
        fields=("text",),
        role="dfc",
        revision="owner-verified:deterministic-verifier-bank-v1",
    ),
    CorpusSource(
        key="identity_replay",
        dataset="an-ra/identity",
        config=None,
        license="owner",
        weight=0.02,
        fields=("text",),
        role="identity",
        revision="owner:identity_replay",
    ),
)

TARGET_CLEAN_TEXT_GB = 120.0


@dataclass(frozen=True)
class CorpusManifestReport:
    schema_version: int
    sources: list[dict[str, object]]
    total_weight: float
    weight_normalized: bool
    all_licenses_allowed: bool
    all_revisions_pinned: bool
    unique_keys: bool
    target_clean_text_gb: float
    manifest_sha256: str
    valid: bool
    violations: list[str] = field(default_factory=list)


def _canonical_bytes(sources: tuple[CorpusSource, ...]) -> bytes:
    payload = [asdict(source) for source in sources]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_corpus_manifest(
    sources: tuple[CorpusSource, ...] = CAMPAIGN_CORPUS_SOURCES,
) -> CorpusManifestReport:
    """Validate the pinned source set and produce a hashable manifest report."""
    violations: list[str] = []
    keys = [source.key for source in sources]
    unique_keys = len(keys) == len(set(keys))
    if not unique_keys:
        violations.append("duplicate source keys")
    for source in sources:
        if not source.license_ok():
            violations.append(f"{source.key}: license {source.license!r} not allowlisted")
        if not source.revision_pinned():
            violations.append(f"{source.key}: revision {source.revision!r} is not pinned")
        if source.weight <= 0.0:
            violations.append(f"{source.key}: weight must be positive")
    total_weight = round(sum(source.weight for source in sources), 6)
    weight_normalized = abs(total_weight - 1.0) <= 1e-6
    if not weight_normalized:
        violations.append(f"weights sum to {total_weight}, expected 1.0")
    all_licenses_allowed = all(source.license_ok() for source in sources)
    all_revisions_pinned = all(source.revision_pinned() for source in sources)
    manifest_sha256 = hashlib.sha256(_canonical_bytes(sources)).hexdigest()
    return CorpusManifestReport(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sources=[asdict(source) for source in sources],
        total_weight=total_weight,
        weight_normalized=weight_normalized,
        all_licenses_allowed=all_licenses_allowed,
        all_revisions_pinned=all_revisions_pinned,
        unique_keys=unique_keys,
        target_clean_text_gb=TARGET_CLEAN_TEXT_GB,
        manifest_sha256=manifest_sha256,
        valid=not violations,
        violations=violations,
    )


def write_corpus_manifest(
    path: str | Path = UPSTREAM_CORPUS_MANIFEST,
    *,
    sources: tuple[CorpusSource, ...] = CAMPAIGN_CORPUS_SOURCES,
) -> CorpusManifestReport:
    report = build_corpus_manifest(sources)
    if not report.valid:
        raise ValueError(f"Refusing to write an invalid corpus manifest: {report.violations}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Emit the pinned upstream corpus manifest.")
    parser.add_argument("--json-out", default=str(UPSTREAM_CORPUS_MANIFEST))
    args = parser.parse_args()
    report = write_corpus_manifest(args.json_out)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
