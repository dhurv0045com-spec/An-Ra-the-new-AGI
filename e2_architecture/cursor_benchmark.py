"""Deterministic packed-stream cursor and checkpoint round-trip canary.

This is an infrastructure experiment, not model training.  It proves that a
checkpoint can resume the exact next token stream from a content-addressed
pack, including a batch boundary in the middle of a sequence.  The optional
CPU/CUDA materialization checks that device transfer does not alter the token
ledger or digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "esoes-e2-cursor-resume/v1"
CURSOR_SCHEMA = "cursor/v1"
MANIFEST_SHA256 = "c" * 64
SHARD_HASHES = tuple(hashlib.sha256(f"shard-{index}".encode()).hexdigest() for index in range(7))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(bytes().join(value.to_bytes(4, "little") for value in tokens)).hexdigest()


@dataclass(frozen=True, slots=True)
class CursorState:
    schema: str
    pack_manifest_sha256: str
    run_seed: int
    epoch: int
    sequence_ordinal: int
    token_offset: int
    cumulative_tokens: int

    def assert_valid(self, *, total_sequences: int, sequence_lengths: tuple[int, ...]) -> None:
        if self.schema != CURSOR_SCHEMA:
            raise ValueError("unsupported cursor schema")
        if len(self.pack_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.pack_manifest_sha256
        ):
            raise ValueError("cursor manifest identity must be lowercase SHA-256")
        if self.run_seed < 0 or self.epoch < 0 or self.cumulative_tokens < 0:
            raise ValueError("cursor counters cannot be negative")
        if not 0 <= self.sequence_ordinal <= total_sequences:
            raise ValueError("cursor sequence ordinal is out of range")
        if self.sequence_ordinal == total_sequences:
            if self.token_offset != 0:
                raise ValueError("terminal cursor must have zero token offset")
        elif not 0 <= self.token_offset < sequence_lengths[self.sequence_ordinal]:
            raise ValueError("cursor token offset is out of range")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CursorState":
        expected = {
            "schema",
            "pack_manifest_sha256",
            "run_seed",
            "epoch",
            "sequence_ordinal",
            "token_offset",
            "cumulative_tokens",
        }
        if set(value) != expected:
            raise ValueError("cursor fields do not match the versioned schema")
        return cls(**value)


class DeterministicPack:
    """A tiny content-addressed pack model with explicit sequence boundaries."""

    def __init__(self, *, manifest_sha256: str, run_seed: int, epoch: int = 0) -> None:
        self.manifest_sha256 = manifest_sha256
        self.run_seed = run_seed
        self.epoch = epoch
        self.sequence_lengths = tuple(11 + ((index * 17) % 31) for index in range(7 * 9))
        pairs = [(shard, sequence) for shard in range(len(SHARD_HASHES)) for sequence in range(9)]
        permutation_seed = int.from_bytes(
            hashlib.sha256(f"{manifest_sha256}:{run_seed}:{epoch}".encode()).digest()[:8],
            "little",
        )
        random.Random(permutation_seed).shuffle(pairs)
        self.order = tuple(pairs)

    def initial_cursor(self) -> CursorState:
        return CursorState(
            CURSOR_SCHEMA,
            self.manifest_sha256,
            self.run_seed,
            self.epoch,
            sequence_ordinal=0,
            token_offset=0,
            cumulative_tokens=0,
        )

    def _token(self, sequence_ordinal: int, token_offset: int) -> int:
        shard, sequence = self.order[sequence_ordinal]
        payload = f"{self.manifest_sha256}:{SHARD_HASHES[shard]}:{sequence}:{token_offset}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % 24_576

    def consume(self, cursor: CursorState, token_budget: int) -> tuple[list[int], CursorState]:
        if token_budget <= 0:
            raise ValueError("token budget must be positive")
        cursor.assert_valid(total_sequences=len(self.order), sequence_lengths=self.sequence_lengths)
        if cursor.pack_manifest_sha256 != self.manifest_sha256 or cursor.run_seed != self.run_seed:
            raise ValueError("cursor does not belong to this pack stream")
        tokens: list[int] = []
        ordinal = cursor.sequence_ordinal
        offset = cursor.token_offset
        while len(tokens) < token_budget and ordinal < len(self.order):
            length = self.sequence_lengths[ordinal]
            while offset < length and len(tokens) < token_budget:
                tokens.append(self._token(ordinal, offset))
                offset += 1
            if offset == length:
                ordinal += 1
                offset = 0
        next_cursor = CursorState(
            CURSOR_SCHEMA,
            self.manifest_sha256,
            self.run_seed,
            self.epoch,
            ordinal,
            offset,
            cursor.cumulative_tokens + len(tokens),
        )
        next_cursor.assert_valid(total_sequences=len(self.order), sequence_lengths=self.sequence_lengths)
        return tokens, next_cursor


def _device_digest(tokens: list[int], device_name: str) -> tuple[str, bool, str | None]:
    try:
        import torch
    except ImportError as exc:
        return "", False, str(exc)
    if device_name == "cuda" and not torch.cuda.is_available():
        return "", False, "CUDA is unavailable"
    device = torch.device(device_name)
    tensor = torch.tensor(tokens, dtype=torch.int32, device=device)
    finite = bool(torch.isfinite(tensor.float()).all().item())
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest(), finite, None


def benchmark(*, device: str, run_seed: int, batches: int, tokens_per_batch: int) -> dict[str, Any]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if run_seed < 0 or batches < 3 or tokens_per_batch <= 0:
        raise ValueError("invalid cursor canary dimensions")
    pack = DeterministicPack(manifest_sha256=MANIFEST_SHA256, run_seed=run_seed)
    cursor = pack.initial_cursor()
    uninterrupted: list[int] = []
    split_cursor = cursor
    split_batches = max(1, batches // 2)
    for _ in range(batches):
        chunk, cursor = pack.consume(cursor, tokens_per_batch)
        uninterrupted.extend(chunk)
        if len(uninterrupted) == split_batches * tokens_per_batch:
            split_cursor = cursor

    checkpoint = json.loads(json.dumps(asdict(split_cursor), sort_keys=True))
    restored_cursor = CursorState.from_dict(checkpoint)
    resumed: list[int] = []
    for _ in range(batches - split_batches):
        chunk, restored_cursor = pack.consume(restored_cursor, tokens_per_batch)
        resumed.extend(chunk)
    expected_tail = uninterrupted[split_batches * tokens_per_batch :]
    tampered_manifest = dict(checkpoint)
    tampered_manifest["pack_manifest_sha256"] = "d" * 64
    tampered_offset = dict(checkpoint)
    tampered_offset["token_offset"] = pack.sequence_lengths[split_cursor.sequence_ordinal] + 1 if split_cursor.sequence_ordinal < len(pack.order) else 1
    rejection_checks: dict[str, bool] = {}
    try:
        pack.consume(CursorState.from_dict(tampered_manifest), tokens_per_batch)
    except ValueError:
        rejection_checks["manifest_mismatch_rejected"] = True
    else:
        rejection_checks["manifest_mismatch_rejected"] = False
    try:
        CursorState.from_dict(tampered_offset).assert_valid(
            total_sequences=len(pack.order), sequence_lengths=pack.sequence_lengths
        )
    except ValueError:
        rejection_checks["cursor_offset_corruption_rejected"] = True
    else:
        rejection_checks["cursor_offset_corruption_rejected"] = False

    all_tokens = uninterrupted
    digest, finite, device_error = _device_digest(all_tokens, device)
    checks = {
        "uninterrupted_stream_complete": len(all_tokens) == batches * tokens_per_batch,
        "resumed_tail_matches_uninterrupted": resumed == expected_tail,
        "final_cursor_token_ledger_exact": restored_cursor.cumulative_tokens == len(all_tokens),
        "checkpoint_json_roundtrip": asdict(restored_cursor) == asdict(CursorState.from_dict(asdict(restored_cursor))),
        "device_materialization_finite": finite,
        "device_digest_matches_host": digest == _token_digest(all_tokens),
        **rejection_checks,
    }
    if device_error is not None:
        checks["device_materialization_finite"] = False
        checks["device_digest_matches_host"] = False
    status = "PASS" if all(checks.values()) else ("BLOCKED_CUDA" if device == "cuda" and device_error else "FAIL")
    try:
        import torch

        torch_version = torch.__version__
        cuda_runtime = torch.version.cuda
        device_name = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else platform.processor()
    except ImportError:
        torch_version = None
        cuda_runtime = None
        device_name = platform.processor()
    return {
        "schema": SCHEMA,
        "status": status,
        "scope": "content-addressed packed-stream cursor; JSON checkpoint round-trip; no model training",
        "implementation_sha256": _sha256_file(Path(__file__)),
        "config": {
            "device": device,
            "run_seed": run_seed,
            "batches": batches,
            "tokens_per_batch": tokens_per_batch,
            "split_batch": split_batches,
            "cursor_schema": CURSOR_SCHEMA,
        },
        "pack": {
            "manifest_sha256": MANIFEST_SHA256,
            "shard_count": len(SHARD_HASHES),
            "sequence_count": len(pack.order),
            "total_stream_tokens": len(all_tokens),
        },
        "checks": checks,
        "metrics": {
            "host_token_digest": _token_digest(all_tokens),
            "device_token_digest": digest,
            "final_cumulative_tokens": restored_cursor.cumulative_tokens,
            "serialized_cursor_bytes": len(json.dumps(checkpoint, sort_keys=True).encode()),
        },
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "device_name": device_name,
        "limitations": [
            "Synthetic content-addressed tokens prove cursor continuity, not pack-reader throughput or data quality.",
            "Single-process canary; distributed sampler/all-reduce and remote durable storage remain open.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, default=38001)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--tokens-per-batch", type=int, default=47)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(
        device=args.device,
        run_seed=args.seed,
        batches=args.batches,
        tokens_per_batch=args.tokens_per_batch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
