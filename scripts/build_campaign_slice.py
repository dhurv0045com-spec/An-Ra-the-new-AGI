# ruff: noqa: E402
"""Build the >=50MB tokenizer campaign slice (Stream B, TODO 3).

The canonical V4 candidates must come from the campaign corpus, measured on
genuinely held-out per-source text. This builder takes already-acquired local
source files (one per campaign source key), splits each source deterministically
into a train slice and a held-out slice by a per-line hash rule (the same rule
family as scripts/measure_tokenizer_fertility.py), and emits:

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

from anra.anra_paths import OUTPUT_V2_DIR, ROOT
from training.corpus_manifest import CAMPAIGN_CORPUS_SOURCES

CAMPAIGN_SLICE_DIR = OUTPUT_V2_DIR / "campaign_slice"
MIN_SLICE_MB = 50.0
# Per-line held-out selection: sha256(line)[0] in this half-open range. 26/256
# ~= 10% held out, deterministic and stable across runs and machines.
HELDOUT_HASH_CEILING = 26


def _is_heldout(line: str) -> bool:
    return hashlib.sha256(line.encode("utf-8")).digest()[0] < HELDOUT_HASH_CEILING


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    candidates = {
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
    return 0 if manifest["meets_min_slice"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
