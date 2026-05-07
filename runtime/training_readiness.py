from __future__ import annotations

import json
import importlib.util
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from anra_paths import (
    DATASET_CANONICAL,
    OUTPUT_V2_DIR,
    ROOT,
    TEACHER_REASONING_V2_FILE,
    V3_TOKENIZER_FILE,
    get_teacher_files,
    get_identity_file,
    get_v2_checkpoint,
)
from training.v2_config import EXPECTED_PAD_TOKEN_ID, EXPECTED_SPECIAL_TOKENS, EXPECTED_TOKENIZER_VOCAB_SIZE


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str
    required: bool = True
    path: str | None = None


@dataclass(frozen=True)
class TrainingReadiness:
    generated_at: float
    score: int
    out_of: int
    ready_for_session: bool
    ready_for_milestone: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[ReadinessCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _text_dataset_status(path: Path) -> ReadinessCheck:
    if not path.exists():
        return ReadinessCheck("dataset", "blocker", "canonical training dataset is missing", path=_rel(path))
    if not path.is_file():
        return ReadinessCheck("dataset", "blocker", "dataset path is not a file", path=_rel(path))
    size = path.stat().st_size
    if size < 100_000:
        return ReadinessCheck("dataset", "blocker", f"dataset is too small for V2 training ({size} bytes)", path=_rel(path))
    sample = path.read_text(encoding="utf-8", errors="replace")[:4000]
    if "H:" not in sample or "ANRA:" not in sample:
        return ReadinessCheck("dataset", "blocker", "dataset does not expose H:/ANRA: conversation turns", path=_rel(path))
    return ReadinessCheck("dataset", "ok", f"{size} bytes with H:/ANRA: turns", path=_rel(path))


def _tokenizer_status(path: Path) -> ReadinessCheck:
    if not path.exists():
        return ReadinessCheck("tokenizer", "blocker", "tokenizer_v3.json is missing", path=_rel(path))
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception as exc:
        return ReadinessCheck("tokenizer", "blocker", f"tokenizer cannot be read: {type(exc).__name__}", path=_rel(path))

    token_to_id = blob.get("token_to_id", {}) if isinstance(blob, dict) else {}
    vocab_size = int(meta.get("vocab_size", len(token_to_id)))
    special_tokens = list(meta.get("special_tokens", []))
    pad_id = token_to_id.get("<pad>")
    if vocab_size != EXPECTED_TOKENIZER_VOCAB_SIZE:
        return ReadinessCheck(
            "tokenizer",
            "blocker",
            f"vocab_size={vocab_size}, expected={EXPECTED_TOKENIZER_VOCAB_SIZE}",
            path=_rel(path),
        )
    if special_tokens and special_tokens != EXPECTED_SPECIAL_TOKENS:
        return ReadinessCheck("tokenizer", "blocker", "special token contract does not match V2 config", path=_rel(path))
    if pad_id != EXPECTED_PAD_TOKEN_ID:
        return ReadinessCheck("tokenizer", "blocker", f"pad token id={pad_id}, expected={EXPECTED_PAD_TOKEN_ID}", path=_rel(path))
    return ReadinessCheck("tokenizer", "ok", f"{vocab_size} tokens, pad={pad_id}", path=_rel(path))


def _checkpoint_checks() -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    for name in ("brain", "identity", "ouroboros"):
        path = get_v2_checkpoint(name)
        if path.exists():
            checks.append(ReadinessCheck(f"checkpoint:{name}", "ok", f"{path.stat().st_size} bytes", required=False, path=_rel(path)))
        else:
            checks.append(
                ReadinessCheck(
                    f"checkpoint:{name}",
                    "warn",
                    "missing; training can create it, but resume/promotion cannot use it yet",
                    required=False,
                    path=_rel(path),
                )
            )
    return checks


def _report_checks() -> list[ReadinessCheck]:
    reports = {
        "metrics": OUTPUT_V2_DIR / "v2_session_train_metrics.json",
        "eval_summary": OUTPUT_V2_DIR / "v2_eval_summary.json",
        "curriculum": OUTPUT_V2_DIR / "v2_next_session_curriculum.json",
        "mix_report": OUTPUT_V2_DIR / "v2_dataset_mix.json",
    }
    checks: list[ReadinessCheck] = []
    for name, path in reports.items():
        if path.exists():
            checks.append(ReadinessCheck(f"report:{name}", "ok", f"{path.stat().st_size} bytes", required=False, path=_rel(path)))
        else:
            checks.append(ReadinessCheck(f"report:{name}", "warn", "missing until a session/eval writes it", required=False, path=_rel(path)))
    return checks


def _data_mix_checks() -> list[ReadinessCheck]:
    teacher_files = get_teacher_files()
    frontier = DATASET_CANONICAL.parent / "frontier_dfc.jsonl"
    replay = ROOT / "state" / "failure_replay.jsonl"
    checks = [
        ReadinessCheck(
            "data_mix:teacher",
            "ok" if teacher_files else "warn",
            f"{len(teacher_files)} teacher file(s)" if teacher_files else f"no teacher JSONL found at {_rel(TEACHER_REASONING_V2_FILE)}",
            required=False,
            path=_rel(teacher_files[0]) if teacher_files else _rel(TEACHER_REASONING_V2_FILE),
        ),
        ReadinessCheck(
            "data_mix:frontier_dfc",
            "ok" if frontier.exists() else "warn",
            "frontier DFC examples available" if frontier.exists() else "frontier DFC examples missing",
            required=False,
            path=_rel(frontier),
        ),
        ReadinessCheck(
            "data_mix:failure_replay",
            "ok" if replay.exists() else "warn",
            "failure replay dataset available" if replay.exists() else "failure replay dataset not created yet",
            required=False,
            path=_rel(replay),
        ),
    ]
    return checks


def _alignment_dependency_checks() -> list[ReadinessCheck]:
    identity_file = get_identity_file()
    psutil_available = importlib.util.find_spec("psutil") is not None
    android_runtime = hasattr(sys, "getandroidapilevel")
    psutil_status = "ok" if psutil_available or android_runtime else "warn"
    if psutil_available:
        psutil_detail = "psutil available for sovereignty resource monitoring"
    elif android_runtime:
        psutil_detail = "psutil unsupported on Android; sovereignty will use fallback resource metrics"
    else:
        psutil_detail = "psutil missing; sovereignty resource monitoring is degraded"
    return [
        ReadinessCheck(
            "identity:file",
            "ok" if identity_file is not None else "warn",
            "identity file available" if identity_file is not None else "identity injection has no source identity file",
            required=False,
            path=_rel(identity_file) if identity_file is not None else None,
        ),
        ReadinessCheck(
            "dependency:psutil",
            psutil_status,
            psutil_detail,
            required=False,
        ),
    ]


def _compute_checks(dataset_path: Path) -> list[ReadinessCheck]:
    checks = [
        _text_dataset_status(dataset_path),
        _tokenizer_status(V3_TOKENIZER_FILE),
        ReadinessCheck(
            "device",
            "ok" if torch.cuda.is_available() else "warn",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable; training will run on CPU",
            required=False,
        ),
    ]
    checks.extend(_checkpoint_checks())
    checks.extend(_report_checks())
    checks.extend(_data_mix_checks())
    checks.extend(_alignment_dependency_checks())
    return checks


def assess_training_readiness(dataset_path: Path | None = None, extra_checks: Iterable[ReadinessCheck] = ()) -> TrainingReadiness:
    checks = _compute_checks(dataset_path or DATASET_CANONICAL)
    checks.extend(extra_checks)
    blockers = [check.detail for check in checks if check.status == "blocker" and check.required]
    warnings = [f"{check.name}: {check.detail}" for check in checks if check.status == "warn"]
    penalty = len(blockers) * 4 + min(4, (len(warnings) + 1) // 2)
    score = max(0, 10 - penalty)
    brain_ready = get_v2_checkpoint("brain").exists()
    return TrainingReadiness(
        generated_at=time.time(),
        score=score,
        out_of=10,
        ready_for_session=not blockers,
        ready_for_milestone=not blockers and brain_ready,
        blockers=blockers,
        warnings=warnings,
        checks=checks,
    )
