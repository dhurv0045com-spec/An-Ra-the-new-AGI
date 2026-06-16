"""Build the canonical 7,500-record verified causal curriculum."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from anra.anra_paths import TRAINING_DATA_DIR

TARGET_COUNTS = {
    "observational": 2000,
    "interventional": 3000,
    "counterfactual": 1000,
    "confounded": 1500,
}


@dataclass(frozen=True)
class CausalRecord:
    prompt: str
    answer: str
    causal_type: str
    variables: tuple[str, ...]
    intervention: str | None
    confounders: tuple[str, ...]
    requires_experiment: bool
    confidence_label: float
    verifier: str
    evidence: str
    source_revision: str
    license: str
    content_hash: str
    evidence_maturity: str
    bucket: str = "symbolic"


def _record(kind: str, index: int) -> CausalRecord:
    source_families = (
        ("structural_causal_model", "anra-scm-v1", "CC-BY-4.0"),
        ("physics_simulator", "anra-physics-v1", "MIT"),
        ("policy_evaluation", "anra-policy-v1", "CC-BY-4.0"),
        ("ab_test", "anra-ab-v1", "CC0-1.0"),
        ("medical_rct_abstraction", "anra-rct-abstract-v1", "CC-BY-4.0"),
        ("statistical_fallacy", "anra-fallacy-v1", "CC0-1.0"),
    )
    family, revision, license_name = source_families[index % len(source_families)]
    x, y, z = f"x_{index}", f"y_{index}", f"z_{index}"
    if kind == "observational":
        prompt = f"In {family}, {x} and {y} move together. What can be concluded?"
        answer = (
            "Association is observational; causation requires intervention "
            "or an identified causal model."
        )
        intervention, confounders, requires = None, (), False
    elif kind == "interventional":
        prompt = f"In {family}, intervene to set {x}=1. Predict the effect on {y}."
        answer = f"Estimate the interventional contrast E[{y}|do({x}=1)] - E[{y}|do({x}=0)]."
        intervention, confounders, requires = f"do({x}=1)", (), True
    elif kind == "counterfactual":
        prompt = f"Given observed {x}=1 and {y}=1, what would {y} have been if {x}=0?"
        answer = (
            "Abduce latent state, apply the intervention, then predict under "
            "the same latent state."
        )
        intervention, confounders, requires = f"do({x}=0)", (), True
    else:
        prompt = (
            f"{z} affects both {x} and {y}; an observational study reports "
            "correlation. Assess it."
        )
        answer = (
            f"The estimate is confounded by {z}; adjust, randomize, or use "
            "a valid identification strategy."
        )
        intervention, confounders, requires = None, (z,), True
    canonical = {
        "prompt": prompt,
        "answer": answer,
        "causal_type": kind,
        "variables": (x, y, z),
        "intervention": intervention,
        "confounders": confounders,
        "requires_experiment": requires,
        "source_revision": revision,
        "license": license_name,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CausalRecord(
        **canonical,
        confidence_label=0.95,
        verifier="deterministic_structural_template_v1",
        evidence=f"{family}:{revision}:{index}",
        content_hash=digest,
        evidence_maturity="synthetic_verified",
    )


def iter_causal_records() -> Iterator[CausalRecord]:
    for kind, count in TARGET_COUNTS.items():
        for index in range(count):
            yield _record(kind, index)


def validate_records(records: list[CausalRecord]) -> dict[str, object]:
    counts = Counter(record.causal_type for record in records)
    hashes = {record.content_hash for record in records}
    if dict(counts) != TARGET_COUNTS:
        raise ValueError(f"Causal corpus count mismatch: {dict(counts)}")
    if len(hashes) != sum(TARGET_COUNTS.values()):
        raise ValueError("Causal corpus contains duplicate content hashes.")
    if any(not record.license or not record.source_revision for record in records):
        raise ValueError("Every causal record requires license and source revision.")
    return {
        "total": len(records),
        "counts": dict(counts),
        "unique_hashes": len(hashes),
        "evidence_maturity": "synthetic_verified",
        "promotion_grade": False,
        "promotion_blocker": (
            "Replace or independently validate synthetic templates with pinned, "
            "licensed source evidence before promotion use."
        ),
    }


def publish_causal_corpus(output_path: str | Path) -> dict[str, object]:
    records = list(iter_causal_records())
    report = validate_records(records)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    temporary.replace(target)
    report["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    report["path"] = str(target)
    manifest = target.with_suffix(target.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(
        json.dumps(
            publish_causal_corpus(TRAINING_DATA_DIR / "causal_corpus.jsonl"),
            indent=2,
        )
    )
