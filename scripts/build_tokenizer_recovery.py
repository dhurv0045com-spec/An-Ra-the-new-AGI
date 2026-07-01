from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from anra.anra_paths import OUTPUT_V2_DIR, ROOT, V3_TOKENIZER_FILE
from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import (
    audit_token_fertility,
    build_append_only_v4,
    tokenizer_competence_report,
)
from training.v2_config import CANONICAL_SPECIAL_TOKENS, CANONICAL_VOCAB_SIZE

DEFAULT_SOURCES = (
    ROOT / "training_data" / "base_corpus.txt",
    ROOT / "training_data" / "reasoning.jsonl",
    ROOT / "training_data" / "frontier_dfc.jsonl",
    ROOT / "training_data" / "teacher_reasoning_v2.jsonl",
    ROOT / "training_data" / "anra_training.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representative_split(
    paths: list[Path],
    *,
    maximum_bytes: int,
) -> tuple[list[str], str, int]:
    training: list[str] = []
    heldout: list[str] = []
    consumed = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if consumed >= maximum_bytes:
                    break
                encoded_size = len(line.encode("utf-8"))
                consumed += encoded_size
                digest = hashlib.sha256(line.encode("utf-8")).digest()
                if digest[0] < 13:
                    heldout.append(line)
                else:
                    training.append(line)
        if consumed >= maximum_bytes:
            break
    return training, "".join(heldout), consumed


def _write_report(payload: dict[str, object]) -> Path:
    target = OUTPUT_V2_DIR / "tokenizer_recovery.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Build native An-Ra recovery tokenizers")
    parser.add_argument("--mode", choices=("append-v4", "clean-draft"), required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max-mb", type=int, default=100)
    parser.add_argument("--minimum-mb", type=int, default=50)
    parser.add_argument("--max-units", type=int, default=1_000_000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    paths = [Path(value).expanduser().resolve() for value in args.source]
    if not paths:
        paths = [path for path in DEFAULT_SOURCES if path.is_file()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing or not paths:
        raise FileNotFoundError(f"Tokenizer recovery sources are missing: {missing or paths}")

    training, heldout, consumed = _representative_split(
        paths,
        maximum_bytes=max(1, args.max_mb) * 1024 * 1024,
    )
    if consumed < max(1, args.minimum_mb) * 1024 * 1024:
        raise RuntimeError(
            f"Tokenizer recovery requires at least {args.minimum_mb} MB; "
            f"only {consumed / 1024**2:.2f} MB were available"
        )
    if not training or not heldout:
        raise RuntimeError("Deterministic tokenizer train/held-out split is empty")

    if args.mode == "clean-draft":
        output = Path(args.output or ROOT / "tokenizer" / "tokenizer_draft_bpe.json")
        tokenizer = SubwordTokenizer.train_from_texts(
            training,
            vocab_size=CANONICAL_VOCAB_SIZE,
            min_frequency=2,
            special_tokens=CANONICAL_SPECIAL_TOKENS,
        )
        tokenizer.save(output)
    else:
        output = Path(args.output or ROOT / "tokenizer" / "tokenizer_v4.json")
        audit = audit_token_fertility(V3_TOKENIZER_FILE, paths, max_units=args.max_units)
        build_append_only_v4(V3_TOKENIZER_FILE, output, audit)

    with tempfile.TemporaryDirectory(prefix="anra-tokenizer-heldout-") as directory:
        heldout_path = Path(directory) / "heldout.txt"
        heldout_path.write_text(heldout, encoding="utf-8")
        competence = tokenizer_competence_report(output, [heldout_path])

    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": args.mode,
        "output": str(output),
        "output_sha256": _sha256(output),
        "source_paths": [str(path) for path in paths],
        "source_sha256": {str(path): _sha256(path) for path in paths},
        "sampled_bytes": consumed,
        "heldout_characters": len(heldout),
        "competence": competence,
        "passed": bool(competence["competent"]),
    }
    report = _write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Report: {report}")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
