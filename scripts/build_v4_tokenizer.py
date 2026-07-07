# ruff: noqa: E402
"""Build the canonical 32k append-only V4 tokenizer (Stream B, TODO 4).

Runs the fertility audit at the requested ceiling over a campaign corpus and,
if the audit qualifies (>=1M units, >=15% projected reduction), grows the
frozen V3 base to the ceiling by appending only new rows. Defaults to the
canonical 32,768 ceiling; --ceiling 16384 selects the proven fallback path.

The canonical V4 candidates must come from the >=50MB campaign corpus. This
CLI runs on whatever corpus is provided (so the machinery is provable now);
the corpus itself is the acquisition-blocked input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anra.anra_paths import OUTPUT_V2_DIR, ROOT, V3_TOKENIZER_FILE
from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import (
    V3_BASE_VOCAB_SIZE,
    audit_token_fertility,
    build_append_only_v4,
)
from training.v2_config import (
    CANONICAL_SPECIAL_TOKEN_IDS,
    CANONICAL_V4_VOCAB_SIZE,
    V4_VOCAB_SIZES,
    frontier_parameter_count,
)

V4_BUILD_REPORT = OUTPUT_V2_DIR / "v4_tokenizer_build.json"
BYTE_PROBE = "campaign V4 bytes: cafe λ ∑ \U0001f680 中文"


def _prove_append(base_json: Path, grown_json: Path, ceiling: int) -> dict[str, object]:
    base_payload = json.loads(base_json.read_text(encoding="utf-8"))
    grown_payload = json.loads(grown_json.read_text(encoding="utf-8"))
    grown = SubwordTokenizer.load(grown_json)
    prefix_unchanged = (
        grown_payload["id_to_token"][:V3_BASE_VOCAB_SIZE] == base_payload["id_to_token"]
    )
    ids_stable = all(
        grown.token_to_id.get(token) == token_id
        for token, token_id in CANONICAL_SPECIAL_TOKEN_IDS.items()
    )
    roundtrip_ok = grown.decode(grown.encode(BYTE_PROBE)) == BYTE_PROBE
    return {
        "ceiling": ceiling,
        "grown_vocab_size": len(grown_payload["id_to_token"]),
        "frozen_prefix_unchanged": prefix_unchanged,
        "canonical_ids_stable": ids_stable,
        "byte_roundtrip_ok": roundtrip_ok,
        "parameter_count": frontier_parameter_count(ceiling),
        "all_proofs_pass": bool(prefix_unchanged and ids_stable and roundtrip_ok),
    }


def build_v4(
    corpus_paths: list[Path],
    output_json: Path,
    *,
    ceiling: int = CANONICAL_V4_VOCAB_SIZE,
    max_units: int = 1_000_000,
    base_json: Path = V3_TOKENIZER_FILE,
) -> dict[str, object]:
    if ceiling not in V4_VOCAB_SIZES:
        raise ValueError(f"--ceiling must be one of {V4_VOCAB_SIZES}; got {ceiling}")
    present = [path for path in corpus_paths if path.is_file()]
    if not present:
        return {
            "status": "blocked_on_corpus",
            "ceiling": ceiling,
            "note": "No campaign corpus present; pass --corpus PATH or build the "
            "campaign slice first (scripts/build_campaign_slice.py).",
            "searched": [str(path) for path in corpus_paths],
        }
    audit = audit_token_fertility(
        base_json, present, max_units=max_units, target_vocab_size=ceiling
    )
    if not audit.get("eligible_for_schema_v4", False):
        return {
            "status": "audit_not_eligible",
            "ceiling": ceiling,
            "audit": {k: v for k, v in audit.items() if k != "candidate_tokens"},
            "note": "Audit needs >=1M units and >=15% projected reduction; the "
            "corpus is too small or too redundant to justify V4 growth.",
        }
    build_append_only_v4(base_json, output_json, audit, target_vocab_size=ceiling)
    proof = _prove_append(base_json, output_json, ceiling)
    return {
        "status": "built" if proof["all_proofs_pass"] else "built_with_proof_failure",
        "ceiling": ceiling,
        "output": str(output_json),
        "output_sha256": hashlib.sha256(output_json.read_bytes()).hexdigest(),
        "audit": {k: v for k, v in audit.items() if k != "candidate_tokens"},
        "candidate_count": audit.get("candidate_count", 0),
        "proof": proof,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical 32k V4 tokenizer.")
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        help="Campaign corpus file(s); repeatable. Defaults to the campaign slice "
        "then the local training corpus.",
    )
    parser.add_argument("--ceiling", type=int, default=CANONICAL_V4_VOCAB_SIZE)
    parser.add_argument("--max-units", type=int, default=1_000_000)
    parser.add_argument(
        "--output",
        default=str(ROOT / "tokenizer" / "tokenizer_v4_32k.json"),
    )
    parser.add_argument("--json-out", default=str(V4_BUILD_REPORT))
    args = parser.parse_args()

    if args.corpus:
        corpus = [Path(item).expanduser() for item in args.corpus]
    else:
        corpus = [
            OUTPUT_V2_DIR / "campaign_slice" / "campaign_slice_train.txt",
            ROOT / "training_data" / "anra_training.txt",
        ]
    report = build_v4(
        corpus,
        Path(args.output),
        ceiling=int(args.ceiling),
        max_units=int(args.max_units),
    )
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "built" else 3


if __name__ == "__main__":
    raise SystemExit(main())
