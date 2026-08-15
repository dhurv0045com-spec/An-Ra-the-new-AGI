# ruff: noqa: E402
"""Build the >=50MB tokenizer campaign slice (Stream B, TODO 3).

The canonical V4 candidates must come from the campaign corpus, measured on
genuinely held-out per-source text. This builder takes already-acquired local
source files (one per campaign source key), splits each source deterministically
into a train slice and a held-out slice by a per-line hash rule (the same rule
family as the canonical V4 validation path), and emits:

- campaign_slice_train.txt   -- concatenated train text (the tokenizer corpus)
- heldout/<source_key>.txt   -- per-source held-out text (fertility measurement)
- campaign_slice_manifest.json -- per-source byte/line counts, output hashes,
  total train size, and the >=50MB gate verdict.

The machinery runs on whatever sources are present; producing the full >=50MB
canonical slice is blocked only on the acquired campaign corpus, not on code.
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

from anra.anra_paths import OUTPUT_V2_DIR, ROOT, get_identity_file
from training.corpus_manifest import CAMPAIGN_CORPUS_SOURCES

CAMPAIGN_SLICE_DIR = OUTPUT_V2_DIR / "campaign_slice"
MIN_SLICE_MB = 50.0
# Per-line held-out selection: sha256(line)[0] in this half-open range. 26/256
# ~= 10% held out, deterministic and stable across runs and machines.
HELDOUT_HASH_CEILING = 26
LARGE_SOURCE_BYTES = 256 * 1024 * 1024
STREAMING_SLICE_MB = 64.0
CAMPAIGN_WEIGHTS = {source.key: source.weight for source in CAMPAIGN_CORPUS_SOURCES}
REPLAY_WEIGHTED_SOURCES = frozenset({"identity_replay"})


def _is_heldout(line: str) -> bool:
    return hashlib.sha256(line.encode("utf-8")).digest()[0] < HELDOUT_HASH_CEILING


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _campaign_key(source: str, fallback: str) -> str:
    lowered = source.lower()
    if "fineweb" in lowered:
        return "fineweb_edu"
    if "stack" in lowered or "code" in lowered:
        return "permissive_code"
    if "finemath" in lowered or "math" in lowered:
        return "finemath"
    if "dolma" in lowered or "science" in lowered or "technical" in lowered:
        return "science_technical"
    if "smol" in lowered or "instruction" in lowered:
        return "verified_instruction"
    if "dfc" in lowered:
        return "verified_dfc"
    if "identity" in lowered or "replay" in lowered:
        return "identity_replay"
    return fallback if fallback in CAMPAIGN_WEIGHTS else "unclassified"


def _record_text_and_key(line: str, fallback: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped:
        return "", "unclassified"
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped + "\n", _campaign_key(fallback, fallback)
    if not isinstance(payload, dict):
        return "", "unclassified"
    if fallback == "verified_dfc" and not (
        payload.get("verified") is True
        or str(payload.get("verifier_status", "")).lower() == "verified"
    ):
        return "", "unclassified"
    text = str(payload.get("text", "")).strip()
    if not text:
        prompt = str(payload.get("prompt", "")).strip()
        answer = str(payload.get("response", payload.get("answer", ""))).strip()
        text = f"H: {prompt}\nANRA: {answer}" if prompt and answer else ""
    return (text + "\n" if text else ""), _campaign_key(
        str(payload.get("source", fallback)), fallback
    )


def build_streaming_campaign_slice(
    sources: dict[str, Path],
    output_dir: str | Path = CAMPAIGN_SLICE_DIR,
    *,
    min_slice_mb: float = MIN_SLICE_MB,
    max_train_mb: float = STREAMING_SLICE_MB,
) -> dict[str, object]:
    """Build a bounded seven-source slice without materializing source files."""
    root = Path(output_dir)
    heldout_dir = root / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    train_path = root / "campaign_slice_train.txt"
    train_tmp = train_path.with_suffix(".tmp")
    total_budget = max(1, int(max_train_mb * 1_048_576))
    heldout_budget = max(64 * 1024, int(total_budget * 0.02))
    quotas = {key: int(total_budget * weight) for key, weight in CAMPAIGN_WEIGHTS.items()}
    stats = {
        key: {
            "train_bytes": 0,
            "heldout_bytes": 0,
            "train_lines": 0,
            "heldout_lines": 0,
            "replayed_bytes": 0,
            "replayed_lines": 0,
        }
        for key in CAMPAIGN_WEIGHTS
    }
    train_hashes: set[str] = set()
    heldout_hashes: dict[str, set[str]] = {key: set() for key in CAMPAIGN_WEIGHTS}
    heldout_streams: dict[str, object] = {}
    heldout_temps: dict[str, Path] = {}
    unclassified_lines = 0
    replay_pools: dict[str, list[str]] = {
        key: [] for key in REPLAY_WEIGHTED_SOURCES
    }

    try:
        for key in CAMPAIGN_WEIGHTS:
            temporary = heldout_dir / f"{key}.tmp"
            heldout_temps[key] = temporary
            heldout_streams[key] = temporary.open("w", encoding="utf-8")
        with train_tmp.open("w", encoding="utf-8") as train_stream:
            for fallback in sorted(sources):
                path = sources[fallback]
                if not path.is_file():
                    continue
                with path.open("r", encoding="utf-8", errors="replace") as source_stream:
                    for line in source_stream:
                        text, key = _record_text_and_key(line, fallback)
                        if not text or key not in CAMPAIGN_WEIGHTS:
                            unclassified_lines += 1
                            continue
                        encoded = text.encode("utf-8")
                        digest = hashlib.sha256(encoded).hexdigest()
                        row = stats[key]
                        if _is_heldout(text):
                            if (
                                row["heldout_bytes"] < heldout_budget
                                and digest not in heldout_hashes[key]
                            ):
                                heldout_streams[key].write(text)
                                heldout_hashes[key].add(digest)
                                row["heldout_bytes"] += len(encoded)
                                row["heldout_lines"] += 1
                            continue
                        if row["train_bytes"] >= quotas[key] or digest in train_hashes:
                            continue
                        train_stream.write(text)
                        train_hashes.add(digest)
                        row["train_bytes"] += len(encoded)
                        row["train_lines"] += 1
                        if key in replay_pools:
                            replay_pools[key].append(text)
            for key, pool in replay_pools.items():
                row = stats[key]
                if not pool:
                    continue
                replay_index = 0
                while row["train_bytes"] < quotas[key]:
                    text = pool[replay_index % len(pool)]
                    train_stream.write(text)
                    encoded_bytes = len(text.encode("utf-8"))
                    row["train_bytes"] += encoded_bytes
                    row["train_lines"] += 1
                    row["replayed_bytes"] += encoded_bytes
                    row["replayed_lines"] += 1
                    replay_index += 1
    finally:
        for stream in heldout_streams.values():
            stream.close()

    train_tmp.replace(train_path)
    per_source: dict[str, dict[str, object]] = {}
    for key, row in stats.items():
        heldout_path = heldout_dir / f"{key}.txt"
        heldout_temps[key].replace(heldout_path)
        per_source[key] = {
            "status": "sliced" if row["train_lines"] or row["heldout_lines"] else "missing",
            **row,
            "heldout_path": str(heldout_path),
            "heldout_sha256": _sha256_path(heldout_path),
            "heldout_disjoint": not bool(train_hashes & heldout_hashes[key]),
        }

    total_train_bytes = sum(int(row["train_bytes"]) for row in stats.values())
    realized = {
        key: int(row["train_bytes"]) / max(1, total_train_bytes) for key, row in stats.items()
    }
    deviations = {key: abs(realized[key] - CAMPAIGN_WEIGHTS[key]) for key in CAMPAIGN_WEIGHTS}
    all_sources_present = all(int(row["train_lines"]) > 0 for row in stats.values())
    mix_verified = all_sources_present and max(deviations.values(), default=1.0) <= 0.02
    train_mb = total_train_bytes / 1_048_576
    all_disjoint = all(bool(row["heldout_disjoint"]) for row in per_source.values())
    manifest: dict[str, object] = {
        "schema_version": 2,
        "mode": "bounded_streaming",
        "train_path": str(train_path),
        "train_bytes": total_train_bytes,
        "train_mb": round(train_mb, 4),
        "train_sha256": _sha256_path(train_path),
        "min_slice_mb": float(min_slice_mb),
        "max_train_mb": float(max_train_mb),
        "meets_min_slice": train_mb >= min_slice_mb,
        "heldout_split_rule": f"sha256(line)[0] < {HELDOUT_HASH_CEILING}",
        "sources": per_source,
        "sources_sliced": sum(row["status"] == "sliced" for row in per_source.values()),
        "all_heldout_disjoint": all_disjoint,
        "campaign_mix_target": CAMPAIGN_WEIGHTS,
        "campaign_mix_realized": realized,
        "campaign_mix_deviation": deviations,
        "campaign_mix_verified": mix_verified,
        "all_required_sources_present": all_sources_present,
        "replay_weighted_sources": sorted(REPLAY_WEIGHTED_SOURCES),
        "unclassified_lines": unclassified_lines,
        "ready_for_v4": bool(train_mb >= min_slice_mb and all_disjoint and mix_verified),
    }
    manifest_path = root / "campaign_slice_manifest.json"
    manifest_tmp = manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest


def split_source(path: Path) -> tuple[str, str, int, int]:
    """Return (train_text, heldout_text, train_lines, heldout_lines)."""
    train: list[str] = []
    heldout: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            (heldout if _is_heldout(line) else train).append(line)
    return "".join(train), "".join(heldout), len(train), len(heldout)


def build_campaign_slice(
    sources: dict[str, Path],
    output_dir: str | Path = CAMPAIGN_SLICE_DIR,
    *,
    min_slice_mb: float = MIN_SLICE_MB,
) -> dict[str, object]:
    if not sources:
        raise ValueError("The campaign slice requires at least one source file")
    if any(
        path.is_file() and path.stat().st_size >= LARGE_SOURCE_BYTES
        for path in sources.values()
    ):
        return build_streaming_campaign_slice(
            sources,
            output_dir,
            min_slice_mb=min_slice_mb,
        )
    root = Path(output_dir)
    heldout_dir = root / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)

    per_source: dict[str, dict[str, object]] = {}
    train_parts: list[str] = []
    total_train_bytes = 0
    for key in sorted(sources):
        path = sources[key]
        if not path.is_file():
            per_source[key] = {"status": "missing", "path": str(path)}
            continue
        train_text, heldout_text, train_lines, heldout_lines = split_source(path)
        train_parts.append(train_text)
        train_bytes = len(train_text.encode("utf-8"))
        heldout_bytes = len(heldout_text.encode("utf-8"))
        total_train_bytes += train_bytes
        heldout_path = heldout_dir / f"{key}.txt"
        heldout_tmp = heldout_path.with_suffix(".tmp")
        heldout_tmp.write_text(heldout_text, encoding="utf-8")
        heldout_tmp.replace(heldout_path)
        per_source[key] = {
            "status": "sliced",
            "path": str(path),
            "train_lines": train_lines,
            "heldout_lines": heldout_lines,
            "train_bytes": train_bytes,
            "heldout_bytes": heldout_bytes,
            "heldout_path": str(heldout_path),
            "heldout_sha256": _sha256_text(heldout_text),
            "heldout_disjoint": bool(
                not (set(train_text.splitlines()) & set(heldout_text.splitlines()))
            ),
        }

    train_text = "".join(train_parts)
    train_path = root / "campaign_slice_train.txt"
    train_tmp = train_path.with_suffix(".tmp")
    train_tmp.write_text(train_text, encoding="utf-8")
    train_tmp.replace(train_path)

    train_mb = total_train_bytes / 1_048_576
    manifest = {
        "schema_version": 1,
        "train_path": str(train_path),
        "train_bytes": total_train_bytes,
        "train_mb": round(train_mb, 4),
        "train_sha256": _sha256_text(train_text),
        "min_slice_mb": min_slice_mb,
        "meets_min_slice": train_mb >= min_slice_mb,
        "heldout_split_rule": f"sha256(line)[0] < {HELDOUT_HASH_CEILING}",
        "sources": per_source,
        "sources_sliced": sum(1 for entry in per_source.values() if entry["status"] == "sliced"),
        "all_heldout_disjoint": all(
            entry.get("heldout_disjoint", True) for entry in per_source.values()
        ),
    }
    manifest_path = root / "campaign_slice_manifest.json"
    manifest_tmp = manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_tmp.replace(manifest_path)
    return manifest


def _default_sources() -> dict[str, Path]:
    """Best-effort local sources so the builder is runnable without arguments."""
    identity_path = get_identity_file()
    canonical = {
        "native_foundation": ROOT / "training_data" / "foundation_records.jsonl",
        "verified_instruction": ROOT / "training_data" / "reasoning.jsonl",
        "verified_dfc": ROOT / "training_data" / "verified_dfc.jsonl",
        "identity_replay": identity_path or ROOT / "training_data" / "identity_replay.txt",
    }
    if canonical["native_foundation"].is_file():
        return {key: path for key, path in canonical.items() if path.is_file()}
    candidates = {
        **canonical,
        "fineweb_edu": ROOT / "training_data" / "anra_training.txt",
        "permissive_code": ROOT / "training_data" / "base_corpus.txt",
    }
    return {key: path for key, path in candidates.items() if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the >=50MB campaign tokenizer slice.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Campaign source as key=path; repeatable. Defaults to local corpora.",
    )
    parser.add_argument("--output-dir", default=str(CAMPAIGN_SLICE_DIR))
    parser.add_argument("--min-slice-mb", type=float, default=MIN_SLICE_MB)
    args = parser.parse_args()

    sources: dict[str, Path] = {}
    for item in args.source:
        if "=" not in item:
            raise ValueError(f"--source must be key=path, got {item!r}")
        key, raw = item.split("=", 1)
        path = Path(raw).expanduser()
        sources[key.strip()] = path if path.is_absolute() else (ROOT / path).resolve()
    if not sources:
        sources = _default_sources()
    if not sources:
        print(
            json.dumps(
                {
                    "status": "blocked_on_corpus",
                    "note": "No campaign source files present; acquire the corpus "
                    "(scripts/download_training_data.py) or pass --source key=path.",
                    "known_sources": [source.key for source in CAMPAIGN_CORPUS_SOURCES],
                },
                indent=2,
            )
        )
        return 3

    manifest = build_campaign_slice(
        sources, args.output_dir, min_slice_mb=args.min_slice_mb
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("ready_for_v4", manifest["meets_min_slice"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
