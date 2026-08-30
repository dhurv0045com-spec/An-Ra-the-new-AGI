"""Run deterministic tokenizer perturbation probes over local E1 candidates.

This is development evidence only. It measures byte cost, round-trip safety,
unknowns, and sensitivity to identifier/number/Unicode/spacing variation; it
does not replace the externally custodied E1 corpus or matched model training.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .tournament import CANDIDATE_VOCABULARIES


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_str(gzip.decompress(path.read_bytes()).decode("utf-8"))


def _cases(seed: int, count: int) -> list[tuple[str, str]]:
    generator = random.Random(seed)
    cases: list[tuple[str, str]] = []
    for index in range(count):
        digits = "".join(str(generator.randrange(10)) for _ in range(8))
        decimal = f"{generator.randrange(1, 999)}.{digits}e{generator.randrange(-12, 13):+d}"
        cases.extend(
            (
                ("number", f"value_{index} = -{decimal}; count={generator.randrange(1, 10**9):,}"),
                ("identifier", f"entity_{index:05d}_edge_case_{digits} -> entity_{index:05d}_result"),
                ("nonce", f"DV-{digits[:4]}-{digits[4:]} maps to FR{digits}-G."),
                ("unicode", f"Δx={decimal}; नमूना-{index}; 東京; 🧪"),
                ("spacing", f"x\t:=\tvalue_{index}\nnext_line_{index}()"),
                ("formal", f"∀x∈S_{index}: P_{digits[:3]}(x) ⇒ ∃y R_{digits[3:]}(x,y)"),
            )
        )
    return cases


def sweep(*, artifact_directory: Path, seed: int = 41001, count: int = 64) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cases = _cases(seed, count)
    for vocabulary_size in CANDIDATE_VOCABULARIES:
        artifact = artifact_directory / f"tokenizer-{vocabulary_size}.json.gz"
        tokenizer = _load(artifact)
        family_tokens: defaultdict[str, int] = defaultdict(int)
        family_bytes: defaultdict[str, int] = defaultdict(int)
        family_cases: defaultdict[str, int] = defaultdict(int)
        roundtrip_failures = 0
        unknown_occurrences = 0
        maximum_tokens = 0
        for family, text in cases:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
            byte_count = len(text.encode("utf-8"))
            family_tokens[family] += len(encoded.ids)
            family_bytes[family] += byte_count
            family_cases[family] += 1
            maximum_tokens = max(maximum_tokens, len(encoded.ids))
            roundtrip_failures += decoded != text
            unknown_occurrences += sum(token == tokenizer.token_to_id("<unk>") for token in encoded.ids)
        rows.append(
            {
                "name": f"local-byte-bpe-{vocabulary_size}",
                "vocabulary_size": vocabulary_size,
                "artifact": artifact.name,
                "artifact_sha256": _sha256_file(artifact),
                "cases": len(cases),
                "roundtrip_failures": roundtrip_failures,
                "unknown_occurrences": unknown_occurrences,
                "maximum_tokens_per_case": maximum_tokens,
                "tokens_per_byte": sum(family_tokens.values()) / sum(family_bytes.values()),
                "tokens_per_byte_by_family": {
                    family: family_tokens[family] / family_bytes[family]
                    for family in sorted(family_tokens)
                },
                "cases_by_family": dict(sorted(family_cases.items())),
            }
        )
    return {
        "schema": "esoes-e1-perturbation-sweep/v1",
        "status": "DEVELOPMENT_STATIC_PASS"
        if all(row["roundtrip_failures"] == 0 and row["unknown_occurrences"] == 0 for row in rows)
        else "FAIL",
        "scope": "deterministic local tokenizer perturbations; development-only",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "seed": seed,
        "repetitions_per_family": count,
        "families": ["formal", "identifier", "nonce", "number", "spacing", "unicode"],
        "rows": rows,
        "limitations": [
            "Local legacy sources and generated perturbations are not external E1 custody.",
            "Static token cost and identity do not measure trained-model loss or cognition.",
            "No tokenizer winner is authorized by this sweep.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = sweep(artifact_directory=args.artifact_directory, seed=args.seed, count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "DEVELOPMENT_STATIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
