from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from anra_paths import (
    DATASET_CANONICAL,
    DRIVE_CHECKPOINTS,
    DRIVE_CODE_DIR,
    DRIVE_DATA_DIR,
    DRIVE_DIR,
    DRIVE_IDENTITY,
    DRIVE_LOGS,
    DRIVE_MEMORY,
    DRIVE_ROOT,
    DRIVE_SESSIONS,
    DRIVE_TEACHER_DIR,
    DRIVE_TEACHER_FILE,
    DRIVE_V2_CHECKPOINTS,
    MERGED_DATA_DIR,
    ROOT,
    TEACHER_REASONING_V2_FILE,
)
from training.v2_runtime import v2_report_path, write_json


SUPPORTED_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
}
SUPPORTED_CODE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".sh",
    ".html",
    ".css",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}
SUPPORTED_STRUCTURED_SUFFIXES = {".jsonl", ".json", ".ipynb"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_CODE_SUFFIXES | SUPPORTED_STRUCTURED_SUFFIXES

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "checkpoints",
    "logs",
    "memory_db",
    "sessions",
    "node_modules",
    ".ipynb_checkpoints",
}

PAIR_RE = re.compile(
    r"(?:^|\n)\s*(?:H|USER)\s*:\s*(.*?)\n\s*(?:ANRA|AN-RA)\s*:\s*(.*?)(?=\n\s*(?:H|USER)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class IngestedSource:
    path: str
    kind: str
    size_bytes: int
    examples: int = 0
    teacher_records: int = 0
    skipped: bool = False
    reason: str = ""


@dataclass
class IngestionReport:
    generated_at: float
    output_dataset: str
    teacher_output: str
    include_drive: bool
    explicit_sources: list[str] = field(default_factory=list)
    discovered_sources: list[str] = field(default_factory=list)
    source_reports: list[dict[str, object]] = field(default_factory=list)
    total_examples: int = 0
    teacher_records: int = 0
    output_bytes: int = 0
    teacher_bytes: int = 0
    status: str = "ok"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def mount_google_drive_if_available() -> bool:
    """Mount Google Drive when running inside Colab. No-op everywhere else."""
    if DRIVE_DIR.exists():
        return True
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return False
    try:
        drive.mount(str(DRIVE_ROOT.parent), force_remount=False)
    except TypeError:
        drive.mount(str(DRIVE_ROOT.parent))
    except Exception:
        return False
    return DRIVE_DIR.exists()


def _is_supported(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES


def _under_skipped_dir(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts & SKIP_DIR_NAMES:
        return True
    skip_roots = [DRIVE_CHECKPOINTS, DRIVE_V2_CHECKPOINTS, DRIVE_V2_CHECKPOINTS.parent, DRIVE_LOGS, DRIVE_MEMORY, DRIVE_SESSIONS]
    for root in skip_roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_source(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _rel_label(path: Path) -> str:
    for root in (DRIVE_DIR, ROOT):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.name


def discover_drive_sources(max_source_mb: int = 64) -> list[Path]:
    """Find usable Drive sources without pulling checkpoints, logs, or memory DB files."""
    roots = [DRIVE_DATA_DIR, DRIVE_IDENTITY, DRIVE_CODE_DIR, DRIVE_TEACHER_DIR, DRIVE_DIR]
    discovered: list[Path] = []
    seen: set[str] = set()
    max_bytes = max_source_mb * 1024 * 1024
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not _is_supported(path) or _under_skipped_dir(path):
                continue
            try:
                if path.stat().st_size <= 0 or path.stat().st_size > max_bytes:
                    continue
                key = str(path.resolve())
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            discovered.append(path)
    return discovered


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _normalise_pair_text(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for prompt, answer in PAIR_RE.findall(raw):
        prompt = prompt.strip()
        answer = answer.strip()
        if prompt and answer:
            pairs.append((prompt, answer))
    return pairs


def _chunk_text(text: str, max_chars: int = 3200) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(block), max_chars):
                chunks.append(block[i : i + max_chars].strip())
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks


def _record_to_pair(record: dict[str, object]) -> tuple[str, str] | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    prompt = str(
        record.get("prompt")
        or record.get("input")
        or record.get("question")
        or record.get("instruction")
        or metadata.get("prompt", "")
    ).strip()
    answer = str(
        record.get("answer")
        or record.get("output")
        or record.get("target")
        or record.get("completion")
        or record.get("response")
        or metadata.get("answer", "")
    ).strip()
    if not prompt or not answer:
        return None
    return prompt, answer


def _is_teacher_record(record: dict[str, object], source: Path) -> bool:
    text = " ".join(
        str(record.get(key, ""))
        for key in ("bucket", "source", "task_type", "kind", "type")
    ).lower()
    name = source.name.lower()
    return (
        "teacher" in text
        or "reasoning" in text
        or "teacher" in name
        or "reasoning" in name
        or bool(record.get("verified"))
    )


def _load_json_records(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, object]] = []
        for line in _read_text(path).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records

    payload = json.loads(_read_text(path))
    if path.suffix.lower() == ".ipynb" and isinstance(payload, dict):
        records = []
        for idx, cell in enumerate(payload.get("cells", [])):
            if not isinstance(cell, dict):
                continue
            src = "".join(cell.get("source", []))
            if src.strip():
                records.append(
                    {
                        "prompt": f"Study notebook cell {idx} from {_rel_label(path)}.",
                        "answer": src,
                        "task_type": str(cell.get("cell_type", "notebook")),
                    }
                )
        return records
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("examples", "data", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _pairs_to_h_anra(pairs: Iterable[tuple[str, str]]) -> str:
    blocks = []
    for prompt, answer in pairs:
        prompt = prompt.strip()
        answer = answer.strip()
        if prompt and answer:
            blocks.append(f"H: {prompt}\nANRA: {answer}")
    return "\n\n".join(blocks)


def _source_to_examples(path: Path) -> tuple[list[tuple[str, str]], list[dict[str, object]], str]:
    suffix = path.suffix.lower()
    label = _rel_label(path)

    if suffix in SUPPORTED_STRUCTURED_SUFFIXES:
        teacher_records: list[dict[str, object]] = []
        pairs: list[tuple[str, str]] = []
        for record in _load_json_records(path):
            pair = _record_to_pair(record)
            if pair is None:
                continue
            pairs.append(pair)
            if _is_teacher_record(record, path):
                prompt, answer = pair
                teacher_records.append(
                    {
                        "prompt": prompt,
                        "answer": answer,
                        "task_type": str(record.get("task_type", record.get("kind", "teacher"))),
                        "source": label,
                        "verified": bool(record.get("verified", "teacher" in label.lower())),
                    }
                )
        return pairs, teacher_records, "structured"

    raw = _read_text(path)
    pairs = _normalise_pair_text(raw)
    if pairs:
        teacher_records = []
        if "teacher" in label.lower() or "reasoning" in label.lower():
            teacher_records = [
                {
                    "prompt": prompt,
                    "answer": answer,
                    "task_type": "teacher",
                    "source": label,
                    "verified": True,
                }
                for prompt, answer in pairs
            ]
        return pairs, teacher_records, "conversation"

    kind = "code" if suffix in SUPPORTED_CODE_SUFFIXES else "text"
    pairs = []
    for idx, chunk in enumerate(_chunk_text(raw), start=1):
        if kind == "code":
            prompt = f"Study this An-Ra source file chunk {idx} from {label} and preserve its implementation details."
            answer = f"Source file: {label}\n\n{chunk}"
        else:
            prompt = f"Learn this owner-provided training material chunk {idx} from {label}."
            answer = chunk
        pairs.append((prompt, answer))
    return pairs, [], kind


def _bootstrap_teacher_records() -> list[dict[str, object]]:
    return [
        {
            "prompt": "Solve 17 * 19 and explain your reasoning briefly.",
            "answer": "I split it into 17 * (20 - 1). That is 340 - 17 = 323. The verified answer is 323.",
            "task_type": "math",
            "source": "bootstrap_teacher",
            "verified": True,
        },
        {
            "prompt": "Find the bug: def tail(xs): return xs[0:len(xs)-1]",
            "answer": "The function name suggests returning the tail, but the slice drops the last element. Use xs[1:] for all but the first item, or xs[-1] for only the final item.",
            "task_type": "code",
            "source": "bootstrap_teacher",
            "verified": True,
        },
        {
            "prompt": "If A implies B and B implies C, does A imply C?",
            "answer": "Yes. If A is true, B follows. If B is true, C follows. Therefore A implies C by transitivity.",
            "task_type": "logic",
            "source": "bootstrap_teacher",
            "verified": True,
        },
    ]


def _dedupe_pairs(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for prompt, answer in pairs:
        key = (prompt.strip(), answer.strip())
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _dedupe_records(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, object]] = []
    for record in records:
        pair = _record_to_pair(record)
        if pair is None:
            continue
        key = (pair[0], pair[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _load_existing_teacher_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        return _load_json_records(path)
    except Exception:
        return []


def prepare_training_corpus(
    *,
    explicit_sources: Iterable[str | Path] | None = None,
    include_drive: bool = True,
    output_path: Path = DATASET_CANONICAL,
    teacher_output: Path = TEACHER_REASONING_V2_FILE,
    report_path: Path | None = None,
    max_source_mb: int = 64,
    mount_drive: bool = True,
    mirror_merged: bool = True,
    mirror_teacher: bool = True,
) -> IngestionReport:
    """Merge local/Drive owner data into the canonical H:/ANRA: corpus."""
    if mount_drive:
        mount_google_drive_if_available()

    explicit_paths = [_resolve_source(p) for p in explicit_sources or []]
    discovered = discover_drive_sources(max_source_mb=max_source_mb) if include_drive else []
    source_paths: list[Path] = []
    seen: set[str] = set()
    seed_paths = [output_path]
    if include_drive:
        seed_paths.append(DRIVE_DIR / output_path.name)
    for path in [*seed_paths, *explicit_paths, *discovered]:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        source_paths.append(path)

    report = IngestionReport(
        generated_at=time.time(),
        output_dataset=str(output_path),
        teacher_output=str(teacher_output),
        include_drive=include_drive,
        explicit_sources=[str(p) for p in explicit_paths],
        discovered_sources=[str(p) for p in discovered],
    )

    all_pairs: list[tuple[str, str]] = []
    teacher_records: list[dict[str, object]] = _load_existing_teacher_records(teacher_output)
    max_bytes = max_source_mb * 1024 * 1024

    for path in source_paths:
        source_report = IngestedSource(path=str(path), kind="unknown", size_bytes=0)
        try:
            if not path.exists() or not path.is_file():
                source_report.skipped = True
                source_report.reason = "missing"
            elif not _is_supported(path):
                source_report.skipped = True
                source_report.reason = "unsupported_suffix"
            elif _under_skipped_dir(path):
                source_report.skipped = True
                source_report.reason = "skipped_directory"
            elif path.stat().st_size > max_bytes:
                source_report.size_bytes = path.stat().st_size
                source_report.skipped = True
                source_report.reason = "too_large"
            else:
                source_report.size_bytes = path.stat().st_size
                pairs, teachers, kind = _source_to_examples(path)
                source_report.kind = kind
                source_report.examples = len(pairs)
                source_report.teacher_records = len(teachers)
                all_pairs.extend(pairs)
                teacher_records.extend(teachers)
        except Exception as exc:
            source_report.skipped = True
            source_report.reason = f"error:{type(exc).__name__}"
        report.source_reports.append(asdict(source_report))

    pairs = _dedupe_pairs(all_pairs)
    if not pairs:
        pairs = [
            (
                "Who are you?",
                "I am An-Ra. I was built by Ankit from pure mathematics, and I keep that identity while I learn.",
            ),
            (
                "What should you do when you are uncertain?",
                "I should say what I know, say what I do not know, and reason carefully instead of pretending.",
            ),
        ]
        report.status = "bootstrap"

    teacher_records = _dedupe_records(teacher_records)
    if not teacher_records:
        teacher_records = _bootstrap_teacher_records()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = "# AN-RA MERGED TRAINING DATASET\n# Generated from local and Drive sources.\n\n"
    body += _pairs_to_h_anra(pairs).strip() + "\n"
    output_path.write_text(body, encoding="utf-8")

    if mirror_merged:
        try:
            MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            merged_copy = MERGED_DATA_DIR / output_path.name
            if merged_copy != output_path:
                merged_copy.write_text(body, encoding="utf-8")
        except Exception:
            pass

    teacher_output.parent.mkdir(parents=True, exist_ok=True)
    with teacher_output.open("w", encoding="utf-8") as fh:
        for record in teacher_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    if mirror_teacher:
        try:
            DRIVE_TEACHER_FILE.parent.mkdir(parents=True, exist_ok=True)
            with DRIVE_TEACHER_FILE.open("w", encoding="utf-8") as fh:
                for record in teacher_records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    report.total_examples = len(pairs)
    report.teacher_records = len(teacher_records)
    report.output_bytes = output_path.stat().st_size
    report.teacher_bytes = teacher_output.stat().st_size

    report_target = report_path or v2_report_path("data_ingestion")
    write_json(report_target, report.to_dict())
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Prepare An-Ra training data from local and Drive files")
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--no_drive", action="store_true")
    ap.add_argument("--max_source_mb", type=int, default=64)
    ap.add_argument("--output", default=str(DATASET_CANONICAL))
    ap.add_argument("--teacher_output", default=str(TEACHER_REASONING_V2_FILE))
    ap.add_argument("--no_mirror", action="store_true")
    args = ap.parse_args()
    report = prepare_training_corpus(
        explicit_sources=args.sources,
        include_drive=not args.no_drive,
        output_path=Path(args.output),
        teacher_output=Path(args.teacher_output),
        max_source_mb=args.max_source_mb,
        mirror_merged=not args.no_mirror,
        mirror_teacher=not args.no_mirror,
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
