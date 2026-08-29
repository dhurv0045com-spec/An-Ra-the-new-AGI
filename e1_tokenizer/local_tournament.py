"""Train and audit independent E1 tokenizers on a hash-bound local development corpus.

This closes the gap between vocabulary-prefix arithmetic and real independently
trained tokenizer artifacts. The result remains development evidence: local
sources are not the externally custodied, representative E1 corpus and no model
is trained here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .audit import audit_receipt
from .compare import pareto_front
from .probes import PROBES
from .tournament import CANDIDATE_VOCABULARIES


TEXT_SUFFIXES = frozenset({".py", ".md", ".txt", ".jsonl"})
EXCLUDED_PARTS = frozenset(
    {".git", ".venv", ".venv-cuda", ".codex-worktrees", "__pycache__", "artifacts"}
)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


@dataclass(frozen=True, slots=True)
class SourceSpec:
    label: str
    domain: str
    path: Path

    def assert_valid(self) -> None:
        if not self.label or not self.domain:
            raise ValueError("source label and domain cannot be empty")
        if not self.path.exists():
            raise FileNotFoundError(self.path)


@dataclass(frozen=True, slots=True)
class TextRecord:
    source: str
    domain: str
    text: str
    text_sha256: str


def parse_source(value: str) -> SourceSpec:
    parts = value.split("::", 2)
    if len(parts) != 3:
        raise ValueError("source must use LABEL::DOMAIN::PATH")
    source = SourceSpec(parts[0].strip(), parts[1].strip(), Path(parts[2]).resolve())
    source.assert_valid()
    return source


def _source_files(source: SourceSpec) -> tuple[Path, ...]:
    if source.path.is_file():
        return (source.path,)
    files = tuple(
        sorted(
            path
            for path in source.path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES
            and not any(
                part in EXCLUDED_PARTS for part in path.relative_to(source.path).parts
            )
        )
    )
    if not files:
        raise ValueError(f"source directory contains no supported text files: {source.path}")
    return files


def build_records(
    sources: Iterable[SourceSpec], *, holdout_modulus: int = 10, holdout_bucket: int = 0
) -> tuple[list[TextRecord], list[TextRecord], dict[str, Any]]:
    if holdout_modulus < 2 or not 0 <= holdout_bucket < holdout_modulus:
        raise ValueError("invalid deterministic holdout policy")
    training: list[TextRecord] = []
    evaluation: list[TextRecord] = []
    source_rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for source in sources:
        source.assert_valid()
        if source.label in seen_labels:
            raise ValueError(f"duplicate source label: {source.label}")
        seen_labels.add(source.label)
        files = _source_files(source)
        file_rows: list[dict[str, Any]] = []
        source_train = source_eval = source_nonempty = 0
        for path in files:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="strict")
            relative = path.name if source.path.is_file() else path.relative_to(source.path).as_posix()
            nonempty = 0
            for line in text.splitlines():
                if not line.strip():
                    continue
                nonempty += 1
                encoded = line.encode("utf-8")
                text_sha = _sha256_bytes(encoded)
                record = TextRecord(source.label, source.domain, line, text_sha)
                bucket = int(text_sha[:16], 16) % holdout_modulus
                if bucket == holdout_bucket:
                    evaluation.append(record)
                    source_eval += len(encoded)
                else:
                    training.append(record)
                    source_train += len(encoded)
            source_nonempty += nonempty
            file_rows.append(
                {
                    "relative_path": relative,
                    "sha256": _sha256_bytes(raw),
                    "bytes": len(raw),
                    "nonempty_lines": nonempty,
                }
            )
        source_rows.append(
            {
                "label": source.label,
                "domain": source.domain,
                "files": file_rows,
                "nonempty_lines": source_nonempty,
                "training_utf8_bytes_with_repetitions": source_train,
                "evaluation_utf8_bytes_with_repetitions": source_eval,
            }
        )
    if not training or not evaluation:
        raise ValueError("both deterministic training and evaluation splits must be nonempty")
    training.sort(key=lambda row: (row.source, row.text_sha256, row.text))
    evaluation.sort(key=lambda row: (row.source, row.text_sha256, row.text))
    manifest_core = {
        "schema": "esoes-e1-local-corpus-manifest/v1",
        "status": "DEVELOPMENT_ONLY_NOT_EXTERNAL_E1",
        "holdout": {
            "method": "sha256(utf8 line) prefix modulo",
            "modulus": holdout_modulus,
            "bucket": holdout_bucket,
            "duplicate_text_cannot_cross_splits": True,
        },
        "sources": source_rows,
        "training_records": len(training),
        "evaluation_records": len(evaluation),
        "training_utf8_bytes_with_repetitions": sum(len(row.text.encode("utf-8")) for row in training),
        "evaluation_utf8_bytes_with_repetitions": sum(
            len(row.text.encode("utf-8")) for row in evaluation
        ),
        "training_unique_texts": len({row.text_sha256 for row in training}),
        "evaluation_unique_texts": len({row.text_sha256 for row in evaluation}),
    }
    manifest_core["manifest_sha256"] = _canonical_sha256(manifest_core)
    return training, evaluation, manifest_core


def _new_tokenizer(vocabulary_size: int) -> tuple[Any, Any]:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError(
            "the optional `tokenizers` package is required for the local tournament"
        ) from exc
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        # Exact arm sizes are part of E1's parameter-allocation comparison. The
        # small local development corpus cannot fill 16k with min_frequency=2,
        # so every arm uses the same explicit min_frequency=1 policy.
        min_frequency=1,
        show_progress=False,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    return tokenizer, trainer


def _train_candidate(
    records: list[TextRecord], vocabulary_size: int, output: Path
) -> float:
    tokenizer, trainer = _new_tokenizer(vocabulary_size)
    started = time.perf_counter()
    tokenizer.train_from_iterator((record.text for record in records), trainer=trainer, length=len(records))
    elapsed = time.perf_counter() - started
    if tokenizer.get_vocab_size() != vocabulary_size:
        raise RuntimeError(
            f"candidate requested {vocabulary_size} tokens but produced {tokenizer.get_vocab_size()}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = tokenizer.to_str(pretty=False).encode("utf-8")
    output.write_bytes(gzip.compress(serialized, compresslevel=9, mtime=0))
    return elapsed


def _load_candidate(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("the optional `tokenizers` package is required") from exc
    serialized = gzip.decompress(path.read_bytes()).decode("utf-8", errors="strict")
    return Tokenizer.from_str(serialized)


def _encoding_receipt(tokenizer: Any, *, name: str, vocabulary_size: int, artifact_sha256: str) -> dict[str, Any]:
    return {
        "schema": "esoes-e1-candidate-encoding/v1",
        "tokenizer_name": name,
        "vocabulary_size": vocabulary_size,
        "artifact_sha256": artifact_sha256,
        "unknown_token_id": tokenizer.token_to_id("<unk>"),
        "encodings": [
            {
                "probe_id": probe.probe_id,
                "token_ids": tokenizer.encode(probe.text, add_special_tokens=False).ids,
                "decoded_text": tokenizer.decode(
                    tokenizer.encode(probe.text, add_special_tokens=False).ids,
                    skip_special_tokens=False,
                ),
            }
            for probe in PROBES
        ],
    }


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _evaluate(tokenizer: Any, records: list[TextRecord]) -> dict[str, Any]:
    tokens_by_domain: dict[str, int] = defaultdict(int)
    bytes_by_domain: dict[str, int] = defaultdict(int)
    lengths: list[int] = []
    roundtrip_failures = 0
    unknown_occurrences = 0
    literal_unknown_token_occurrences = 0
    unknown_id = tokenizer.token_to_id("<unk>")
    started = time.perf_counter()
    for record in records:
        encoding = tokenizer.encode(record.text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        byte_count = len(record.text.encode("utf-8"))
        token_count = len(encoding.ids)
        tokens_by_domain[record.domain] += token_count
        bytes_by_domain[record.domain] += byte_count
        lengths.append(token_count)
        roundtrip_failures += decoded != record.text
        literal_unknown_token_occurrences += record.text.count("<unk>")
        if unknown_id is not None:
            unknown_occurrences += sum(token_id == unknown_id for token_id in encoding.ids)
    elapsed = time.perf_counter() - started
    total_tokens = sum(tokens_by_domain.values())
    total_bytes = sum(bytes_by_domain.values())
    return {
        "records": len(records),
        "total_tokens": total_tokens,
        "total_utf8_bytes": total_bytes,
        "tokens_per_byte": total_tokens / total_bytes,
        "tokens_per_byte_by_domain": {
            domain: tokens_by_domain[domain] / bytes_by_domain[domain]
            for domain in sorted(tokens_by_domain)
        },
        "line_tokens": {
            "median": statistics.median(lengths),
            "p95": _percentile(lengths, 0.95),
            "maximum": max(lengths),
        },
        "identity_roundtrip_failures": roundtrip_failures,
        "unknown_token_occurrences": unknown_occurrences,
        "literal_unknown_token_occurrences": literal_unknown_token_occurrences,
        "unexpected_unknown_token_occurrences": max(
            0, unknown_occurrences - literal_unknown_token_occurrences
        ),
        "encoding_seconds": elapsed,
        "encoding_megabytes_per_second": total_bytes / elapsed / 1_000_000,
    }


def run_tournament(
    sources: list[SourceSpec], *, output_directory: Path, determinism_vocabulary: int = 24_576
) -> dict[str, Any]:
    try:
        from tokenizers import __version__ as tokenizers_version
    except ImportError as exc:
        raise RuntimeError("the optional `tokenizers` package is required") from exc

    training, evaluation, manifest = build_records(sources)
    output_directory.mkdir(parents=True, exist_ok=True)
    # Remove only legacy uncompressed artifacts written by older versions of
    # this command. Receipts and unrelated files are never touched.
    for vocabulary_size in CANDIDATE_VOCABULARIES:
        for legacy_name in (
            f"tokenizer-{vocabulary_size}.json",
            f"tokenizer-{vocabulary_size}-replica.json",
        ):
            legacy = output_directory / legacy_name
            if legacy.is_file():
                legacy.unlink()
    manifest_path = output_directory / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for vocabulary_size in CANDIDATE_VOCABULARIES:
        name = f"local-byte-bpe-{vocabulary_size}"
        artifact = output_directory / f"tokenizer-{vocabulary_size}.json.gz"
        training_seconds = _train_candidate(training, vocabulary_size, artifact)
        artifact_sha = _sha256_file(artifact)
        tokenizer = _load_candidate(artifact)
        artifact_reload_pass = tokenizer.get_vocab_size() == vocabulary_size
        if not artifact_reload_pass:
            raise RuntimeError(f"reloaded artifact has wrong vocabulary size: {artifact}")
        receipt = _encoding_receipt(
            tokenizer,
            name=name,
            vocabulary_size=vocabulary_size,
            artifact_sha256=artifact_sha,
        )
        receipt_path = output_directory / f"encoding-{vocabulary_size}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        audit = audit_receipt(receipt, artifact_sha256=artifact_sha)
        audit_path = output_directory / f"audit-{vocabulary_size}.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        audits.append(audit)
        rows.append(
            {
                "name": name,
                "vocabulary_size": vocabulary_size,
                "artifact": artifact.name,
                "artifact_sha256": artifact_sha,
                "artifact_bytes": artifact.stat().st_size,
                "training_seconds": training_seconds,
                "compressed_artifact_reload_pass": artifact_reload_pass,
                "identity_audit_status": audit["status"],
                "encoding_receipt_sha256": _sha256_file(receipt_path),
                "audit_receipt_sha256": _sha256_file(audit_path),
                "evaluation": _evaluate(tokenizer, evaluation),
            }
        )

    determinism_path = output_directory / f"tokenizer-{determinism_vocabulary}-replica.json.gz"
    determinism_seconds = _train_candidate(training, determinism_vocabulary, determinism_path)
    original_path = output_directory / f"tokenizer-{determinism_vocabulary}.json.gz"
    deterministic = _sha256_file(determinism_path) == _sha256_file(original_path)
    result = {
        "schema": "esoes-e1-local-development-tournament/v1",
        "status": "DEVELOPMENT_STATIC_PASS"
        if deterministic
        and all(row["identity_audit_status"] == "PASS" for row in rows)
        and all(row["compressed_artifact_reload_pass"] for row in rows)
        and all(row["evaluation"]["identity_roundtrip_failures"] == 0 for row in rows)
        and all(
            row["evaluation"]["unexpected_unknown_token_occurrences"] == 0
            for row in rows
        )
        else "FAIL",
        "scope": "independently trained tokenizers on local development sources; not external E1",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "tokenizers_version": tokenizers_version,
        "trainer": {
            "model": "BPE",
            "byte_fallback": True,
            "pre_tokenizer": "ByteLevel(add_prefix_space=False,use_regex=True)",
            "normalizer": None,
            "min_frequency": 1,
            "special_tokens": list(SPECIAL_TOKENS),
        },
        "corpus_manifest": manifest_path.name,
        "corpus_manifest_sha256": _sha256_file(manifest_path),
        "candidate_rows": rows,
        "canary_pareto_front": pareto_front(audits),
        "determinism": {
            "vocabulary_size": determinism_vocabulary,
            "artifact_sha256": _sha256_file(original_path),
            "replica_sha256": _sha256_file(determinism_path),
            "byte_identical": deterministic,
            "replica_training_seconds": determinism_seconds,
        },
        "limitations": [
            "Local dialogue, synthetic, code, and documentation sources are not representative pretraining data.",
            "Static tokenization does not measure byte-normalized model loss or cognition.",
            "Candidate training uses equal raw text, not equal downstream model FLOPs.",
            "This receipt cannot authorize E2 or replace external E1 custody.",
        ],
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, help="repeat LABEL::DOMAIN::PATH"
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--determinism-vocabulary", type=int, default=24_576)
    args = parser.parse_args()
    sources = [parse_source(value) for value in args.source]
    if args.determinism_vocabulary not in CANDIDATE_VOCABULARIES:
        parser.error("determinism vocabulary must be one tournament arm")
    result = run_tournament(
        sources,
        output_directory=args.output_directory,
        determinism_vocabulary=args.determinism_vocabulary,
    )
    result_path = args.output_directory / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(result_path)}, sort_keys=True))
    return 0 if result["status"] == "DEVELOPMENT_STATIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
