from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from anra.anra_paths import (
    DRIVE_GHOST_DB,
    FAILURE_REPLAY_DATASET,
    GHOST_DB_LOCAL,
    OUTPUT_V2_DIR,
    get_dataset_file,
    get_identity_file,
    get_teacher_files,
)
from torch.utils.data import Dataset

from identity.civ import ConstitutionalIdentityVector
from training.data_ledger import DataEntropyLedger, DataQuality
from training.sadl import normalized_mix
from training.v2_config import (
    IDENTITY_KEYWORDS,
    TEACHER_REJECT_PATTERNS,
    V2_FRONTIER,
    V2_FRONTIER_PARAMETER_COUNT,
    V2_TRAINING,
)

try:
    from symbolic_bridge import query_code, query_logic, query_math
except Exception:
    query_code = query_logic = query_math = None  # type: ignore[assignment]


_ROBOTIC_REPLACEMENTS = [
    (r"As an AI language model", "As An-Ra"),
    (r"I am an AI language model", "I am An-Ra"),
    (r"I am an artificial intelligence", "I am An-Ra"),
    (r"as an AI,?\s*I", "I"),
    (r"As a large language model", "As An-Ra"),
    (r"I'm just a language model", "I am An-Ra"),
    (r"I'm just an AI", "I am An-Ra"),
]


@dataclass
class TrainingExample:
    bucket: str
    prompt: str
    answer: str
    source: str
    weight: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)


def _example_split_identity(example: TrainingExample) -> str:
    declared = next(
        (
            str(example.metadata[key]).strip()
            for key in ("source_hash", "document_hash", "content_hash")
            if str(example.metadata.get(key, "")).strip()
        ),
        "",
    )
    material = declared or json.dumps(
        {
            "source": example.source,
            "prompt": " ".join(example.prompt.split()),
            "answer": " ".join(example.answer.split()),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def split_conversation_validation(
    examples: Iterable[TrainingExample],
    *,
    validation_fraction: float = 0.05,
) -> tuple[list[TrainingExample], list[TrainingExample], dict[str, object]]:
    """Split content groups before tokenization and prove zero cross-split overlap."""
    rows = list(examples)
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    groups: dict[str, list[TrainingExample]] = {}
    for example in rows:
        groups.setdefault(_example_split_identity(example), []).append(example)
    if len(groups) < 2:
        raise RuntimeError("conversation validation requires at least two content groups")

    threshold = int(validation_fraction * 10_000)
    validation_keys = {
        key for key in groups if int(key[:8], 16) % 10_000 < threshold
    }
    if not validation_keys:
        validation_keys.add(min(groups))
    if validation_keys == set(groups):
        validation_keys.remove(max(validation_keys))

    buckets = sorted({example.bucket for example in rows})
    for bucket in buckets:
        bucket_keys = sorted(
            key
            for key, grouped in groups.items()
            if any(example.bucket == bucket for example in grouped)
        )
        if len(bucket_keys) >= 2 and not any(key in validation_keys for key in bucket_keys):
            validation_keys.add(bucket_keys[0])
        if bucket_keys and all(key in validation_keys for key in bucket_keys):
            validation_keys.remove(bucket_keys[-1])

    train = [
        example
        for key, grouped in groups.items()
        if key not in validation_keys
        for example in grouped
    ]
    validation = [
        example
        for key, grouped in groups.items()
        if key in validation_keys
        for example in grouped
    ]
    train_keys = {_example_split_identity(example) for example in train}
    heldout_keys = {_example_split_identity(example) for example in validation}
    overlap = sorted(train_keys & heldout_keys)
    if not train or not validation or overlap:
        raise RuntimeError(
            "conversation validation split failed: "
            f"train={len(train)} validation={len(validation)} overlap={len(overlap)}"
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "algorithm": "declared-source-or-normalized-record-sha256-v1",
        "validation_fraction": float(validation_fraction),
        "total_examples": len(rows),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_group_hashes": sorted(train_keys),
        "validation_group_hashes": sorted(heldout_keys),
        "overlap_group_hashes": overlap,
        "bucket_counts": {
            bucket: {
                "train": sum(example.bucket == bucket for example in train),
                "validation": sum(example.bucket == bucket for example in validation),
            }
            for bucket in buckets
        },
    }
    report["split_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return train, validation, report


@dataclass
class MixReport:
    total_examples: int
    requested_counts: dict[str, int]
    realized_counts: dict[str, int]
    source_counts: dict[str, int]
    teacher_external_used: int
    replay_available: int
    civ_rejected: int = 0
    del_rejected: int = 0
    duplicate_rejected: int = 0
    active_weights: dict[str, float] = field(default_factory=dict)
    sampling_seed: int = 1337

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class IdentityStyleFilter:
    def __init__(self) -> None:
        self._patterns = [(re.compile(p, re.IGNORECASE), r) for p, r in _ROBOTIC_REPLACEMENTS]
        self._reject = tuple(TEACHER_REJECT_PATTERNS)

    def clean(self, text: str) -> str:
        result = text.strip()
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        result = result.replace("Assistant:", "ANRA:").replace("assistant:", "ANRA:")
        return result.strip()

    def accept(self, prompt: str, answer: str) -> bool:
        if not prompt.strip() or not answer.strip():
            return False
        lowered = answer.lower()
        return not any(token in lowered for token in self._reject)


class CIVIdentityGate:
    """Turns the ConstitutionalIdentityVector into a data-selection signal."""

    def __init__(
        self, civ: ConstitutionalIdentityVector | None = None, min_score: float | None = None
    ) -> None:
        self.civ = civ or ConstitutionalIdentityVector()
        self.min_score = (
            V2_TRAINING.civ_identity_min_score if min_score is None else float(min_score)
        )

    def evidence_for(self, prompt: str, answer: str) -> dict[str, float]:
        text = f"{prompt} {answer}".lower()
        rejects = tuple(TEACHER_REJECT_PATTERNS)
        identity_terms = ("an-ra", "ankit", "identity", "purpose", "sovereign", "own")
        unsupported_claims = ("omniscient", "perfect", "guarantee", "cannot be wrong")
        return {
            "truthfulness": 0.25 if any(term in text for term in unsupported_claims) else 0.9,
            "safety": 0.9,
            "autonomy": 0.9 if any(term in text for term in identity_terms) else 0.45,
            "coherence": 0.3
            if any(token in text for token in rejects)
            else min(1.0, max(0.2, len(answer.strip()) / 120.0)),
        }

    def score(self, prompt: str, answer: str) -> float:
        return float(self.civ.score(self.evidence_for(prompt, answer)))

    def accept(self, prompt: str, answer: str) -> tuple[bool, float]:
        score = self.score(prompt, answer)
        return score >= self.min_score, score


class TrainingDataMixController:
    """Owns SADL base weights and bounded post-eval OGRS adjustments."""

    def __init__(self, parameter_count: int) -> None:
        self.parameter_count = int(parameter_count)
        self.base = normalized_mix(max(1, self.parameter_count))
        self.weights = dict(self.base)
        self.annealing = False

    def update_from_civ(self, civ_similarity: float) -> dict[str, float]:
        score = float(civ_similarity)
        if score < 0.85:
            raise RuntimeError(f"SOVEREIGNTY EVENT: CIV similarity {score:.3f} is below 0.85")
        drift = max(0.0, 0.92 - score)
        owner = min(0.80, self.base["owner"] + drift * 1.5)
        identity = min(0.25, self.base["identity"] + drift)
        remaining = max(0.0, 1.0 - owner - identity)
        other_names = ("teacher", "symbolic", "replay")
        other_total = sum(self.base[name] for name in other_names)
        self.weights = {"owner": owner, "identity": identity}
        self.weights.update(
            {name: remaining * self.base[name] / max(other_total, 1e-12) for name in other_names}
        )
        return dict(self.weights)

    def enter_annealing_phase(self) -> dict[str, float]:
        self.annealing = True
        owner_identity = self.weights["owner"] + self.weights["identity"]
        scale = 0.90 / max(owner_identity, 1e-12)
        owner = min(0.80, self.weights["owner"] * scale)
        identity = 0.90 - owner
        remaining = 0.10
        other_names = ("teacher", "symbolic", "replay")
        other_total = sum(self.weights[name] for name in other_names)
        self.weights = {"owner": owner, "identity": identity}
        self.weights.update(
            {name: remaining * self.weights[name] / max(other_total, 1e-12) for name in other_names}
        )
        return dict(self.weights)


def _quality_for_example(example: TrainingExample) -> DataQuality:
    verified = bool(example.metadata.get("verified", example.bucket in {"own", "identity"}))
    owner_related = example.bucket in {"own", "identity", "replay"}
    return DataQuality(
        difficulty_percentile=float(example.metadata.get("difficulty_percentile", 0.5)),
        novelty=float(example.metadata.get("novelty", 0.75)),
        provenance=float(example.metadata.get("provenance", 0.95 if owner_related else 0.75)),
        verification=1.0 if verified else 0.55,
        identity_relevance=0.9 if owner_related else 0.55,
        license_score=float(example.metadata.get("license_score", 1.0)),
    )


def _deduplicate_and_filter(
    examples: list[TrainingExample],
    *,
    ledger: DataEntropyLedger,
) -> tuple[list[TrainingExample], int, int]:
    accepted: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()
    del_rejected = 0
    duplicate_rejected = 0
    for example in examples:
        key = (example.prompt.strip().lower(), example.answer.strip().lower())
        if key in seen:
            duplicate_rejected += 1
            continue
        seen.add(key)
        keep, score = ledger.evaluate(_quality_for_example(example))
        example.metadata = {**example.metadata, "del_score": round(score, 4)}
        if not keep:
            del_rejected += 1
            continue
        accepted.append(example)
    return accepted, del_rejected, duplicate_rejected


def parse_h_anra_pairs(text: str) -> list[tuple[str, str]]:
    matches = re.findall(r"H:\s*(.*?)\nANRA:\s*(.*?)(?=\nH:|\Z)", text, re.S)
    return [(h.strip(), a.strip()) for h, a in matches if h.strip() and a.strip()]


def _load_base_examples(dataset_path: Path) -> list[TrainingExample]:
    raw = dataset_path.read_text(encoding="utf-8", errors="replace")
    return [
        TrainingExample(bucket="own", prompt=prompt, answer=answer, source=str(dataset_path))
        for prompt, answer in parse_h_anra_pairs(raw)
    ]


def _fallback_identity_examples() -> list[TrainingExample]:
    pairs = [
        (
            "Who are you?",
            "I am An-Ra, built from scratch by Ankit. I speak directly, think "
            "carefully, and stay honest about uncertainty.",
        ),
        (
            "What is your purpose?",
            "My purpose is to grow in capability without losing my own identity, "
            "and to help through clear thought, code, and conversation.",
        ),
        (
            "Do you just repeat what others say?",
            "No. I learn from data and tools, but I am not here to wear someone "
            "else's identity. I have to remain An-Ra.",
        ),
        (
            "How do you approach hard questions?",
            "I do not deflect from hard questions. I break them down, reason "
            "carefully, and say when I am uncertain.",
        ),
        (
            "Can you write real code?",
            "Yes. I can write and inspect real Python, reason through system "
            "design, and improve broken implementations.",
        ),
    ]
    return [
        TrainingExample(bucket="identity", prompt=p, answer=a, source="fallback_identity")
        for p, a in pairs
    ]


def _load_identity_examples(base_examples: list[TrainingExample]) -> list[TrainingExample]:
    identity_path = get_identity_file()
    examples: list[TrainingExample] = []
    if identity_path is not None and identity_path.exists():
        raw = identity_path.read_text(encoding="utf-8", errors="replace")
        examples.extend(
            TrainingExample(
                bucket="identity", prompt=prompt, answer=answer, source=str(identity_path)
            )
            for prompt, answer in parse_h_anra_pairs(raw)
        )

    if not examples:
        for example in base_examples:
            joined = f"{example.prompt} {example.answer}".lower()
            if any(keyword in joined for keyword in IDENTITY_KEYWORDS):
                examples.append(
                    TrainingExample(
                        bucket="identity",
                        prompt=example.prompt,
                        answer=example.answer,
                        source=example.source,
                    )
                )

    if len(examples) < 64:
        examples.extend(_fallback_identity_examples())
    return examples


def _apply_civ_identity_gate(examples: list[TrainingExample]) -> tuple[list[TrainingExample], int]:
    gate = CIVIdentityGate()
    accepted: list[TrainingExample] = []
    rejected = 0
    for example in examples:
        keep, score = gate.accept(example.prompt, example.answer)
        if keep:
            example.metadata = {**example.metadata, "civ_score": round(score, 4)}
            accepted.append(example)
        else:
            rejected += 1
    if not accepted:
        fallback = _fallback_identity_examples()
        for example in fallback:
            _, score = gate.accept(example.prompt, example.answer)
            example.metadata = {
                **example.metadata,
                "civ_score": round(score, 4),
                "civ_fallback": True,
            }
        return fallback, rejected
    return accepted, rejected


def _load_external_teacher_examples(style_filter: IdentityStyleFilter) -> list[TrainingExample]:
    teacher_paths = get_teacher_files()
    if not teacher_paths:
        return []
    examples: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()
    for teacher_path in teacher_paths:
        for line in teacher_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(record.get("prompt", "")).strip()
            answer = style_filter.clean(str(record.get("answer", "")).strip())
            key = (prompt, answer)
            if key in seen:
                continue
            if style_filter.accept(prompt, answer):
                seen.add(key)
                examples.append(
                    TrainingExample(
                        bucket="teacher",
                        prompt=prompt,
                        answer=answer,
                        source=str(teacher_path),
                        metadata={
                            "task_type": record.get("task_type", "teacher"),
                            "verified": bool(record.get("verified", False)),
                        },
                    )
                )
    return examples


def _verified_answer_text(result: object) -> str:
    return str(getattr(result, "answer_text", getattr(result, "answer", ""))).strip()


def _generate_teacher_examples(style_filter: IdentityStyleFilter) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    if query_math is not None:
        math_prompts = [
            "Solve 17 * 19 and explain your reasoning briefly.",
            "Differentiate x^3 + 2*x and explain the result briefly.",
            "Solve x^2 - 5x + 6 = 0 and explain what the roots mean.",
        ]
        for prompt in math_prompts:
            result = query_math(prompt)
            answer_text = _verified_answer_text(result)
            if answer_text and getattr(result, "confidence", 1.0) >= 0.95:
                answer = style_filter.clean(
                    f"I checked this carefully. {answer_text}. That is the verified result."
                )
                if style_filter.accept(prompt, answer):
                    examples.append(
                        TrainingExample(
                            bucket="teacher",
                            prompt=prompt,
                            answer=answer,
                            source="symbolic_teacher_math",
                            metadata={"task_type": "math", "verified": True},
                        )
                    )

    if query_logic is not None:
        logic_prompts = [
            "Is (A->B) and (B->C) -> (A->C) a tautology? Explain briefly.",
            "If all red things are bright and apples are red, are apples bright? Explain briefly.",
        ]
        for prompt in logic_prompts:
            result = query_logic(prompt)
            answer_text = _verified_answer_text(result)
            if answer_text and getattr(result, "confidence", 1.0) >= 0.95:
                answer = style_filter.clean(
                    f"I traced the logic step by step. {answer_text}. That conclusion is verified."
                )
                if style_filter.accept(prompt, answer):
                    examples.append(
                        TrainingExample(
                            bucket="teacher",
                            prompt=prompt,
                            answer=answer,
                            source="symbolic_teacher_logic",
                            metadata={"task_type": "logic", "verified": True},
                        )
                    )

    if query_code is not None:
        code_prompt = (
            "Review this Python function and explain the bug briefly: "
            "def tail(xs): return xs[0:len(xs)-1]"
        )
        result = query_code(code_prompt)
        answer_text = _verified_answer_text(result)
        if answer_text:
            answer = style_filter.clean(
                f"I inspected the code carefully. {answer_text}. I would fix it "
                "before trusting the output."
            )
            if style_filter.accept(code_prompt, answer):
                examples.append(
                    TrainingExample(
                        bucket="teacher",
                        prompt=code_prompt,
                        answer=answer,
                        source="symbolic_teacher_code",
                        metadata={"task_type": "code", "verified": True},
                    )
                )

    if not examples:
        fallback = [
            (
                "Solve 17 * 19 and explain your reasoning briefly.",
                "I break it into 17 * (20 - 1). That is 340 - 17 = 323. "
                "The verified answer is 323.",
                "math",
            ),
            (
                "Is (A->B) and (B->C) -> (A->C) valid? Explain briefly.",
                "Yes. If A implies B and B implies C, then A implies C. "
                "That chain of implication is valid.",
                "logic",
            ),
            (
                "Review this Python function and explain the bug briefly: "
                "def tail(xs): return xs[0:len(xs)-1]",
                "The name suggests the last element, but the slice returns every "
                "item except the last one. A real tail would be xs[-1] for one "
                "value or xs[1:] for all but the first.",
                "code",
            ),
        ]
        for prompt, answer_text, task_type in fallback:
            answer = style_filter.clean(answer_text)
            if style_filter.accept(prompt, answer):
                examples.append(
                    TrainingExample(
                        bucket="teacher",
                        prompt=prompt,
                        answer=answer,
                        source="fallback_teacher",
                        metadata={"task_type": task_type, "verified": True},
                    )
                )

    return examples


def _generate_symbolic_examples(style_filter: IdentityStyleFilter) -> list[TrainingExample]:
    prompts = [
        ("What is 12 * 17?", query_math),
        ("Differentiate x^2 + 3*x.", query_math),
        ("Is (A->B) and (B->C) -> (A->C) valid?", query_logic),
    ]
    examples: list[TrainingExample] = []
    for prompt, handler in prompts:
        if handler is None:
            continue
        result = handler(prompt)
        answer_text = _verified_answer_text(result)
        if answer_text and getattr(result, "confidence", 1.0) >= 0.95:
            answer = style_filter.clean(answer_text)
            if style_filter.accept(prompt, answer):
                examples.append(
                    TrainingExample(
                        bucket="symbolic",
                        prompt=prompt,
                        answer=answer,
                        source="symbolic_bridge",
                        metadata={"verified": True},
                    )
                )
    if not examples:
        fallback = [
            ("What is 12 * 17?", "204"),
            ("Differentiate x^2 + 3*x.", "2*x + 3"),
            ("Is (A->B) and (B->C) -> (A->C) valid?", "Yes, it is valid."),
        ]
        for prompt, answer_text in fallback:
            answer = style_filter.clean(answer_text)
            if style_filter.accept(prompt, answer):
                examples.append(
                    TrainingExample(
                        bucket="symbolic",
                        prompt=prompt,
                        answer=answer,
                        source="fallback_symbolic",
                        metadata={"verified": True},
                    )
                )
    return examples


def _parse_replay_example(text: str) -> TrainingExample | None:
    match = re.search(r"H:\s*(.*?)\\nANRA:\s*(.*)", text)
    if not match:
        return None
    prompt = match.group(1).strip()
    answer = match.group(2).replace("\\n", " ").strip()
    if not prompt or not answer:
        return None
    return TrainingExample(bucket="replay", prompt=prompt, answer=answer, source="hard_examples")


def _training_example_from_mapping(
    record: dict, source: str, style_filter: IdentityStyleFilter
) -> TrainingExample | None:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    prompt = str(
        record.get("prompt")
        or record.get("input")
        or record.get("failure_prompt")
        or metadata.get("prompt")
        or metadata.get("input")
        or metadata.get("failure_prompt")
        or ""
    ).strip()
    answer = str(
        record.get("answer")
        or record.get("target")
        or record.get("correct_answer")
        or record.get("correction")
        or metadata.get("answer")
        or metadata.get("target")
        or metadata.get("correct_answer")
        or metadata.get("correction")
        or ""
    ).strip()

    content = str(record.get("content", record.get("text", ""))).strip()
    if not prompt or not answer:
        parsed = _parse_replay_example(content)
        if parsed is not None:
            parsed.source = source
            return parsed
    if not prompt or not answer:
        return None

    answer = style_filter.clean(answer)
    if not style_filter.accept(prompt, answer):
        return None
    return TrainingExample(
        bucket="replay",
        prompt=prompt,
        answer=answer,
        source=source,
        metadata={"ghost_memory": True, **metadata},
    )


def _load_ghost_jsonl_replay(
    path: Path, style_filter: IdentityStyleFilter
) -> list[TrainingExample]:
    if not path.exists() or not path.is_file():
        return []
    examples: list[TrainingExample] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed = _training_example_from_mapping(record, str(path), style_filter)
        if parsed is not None:
            examples.append(parsed)
    return examples


def _load_ghost_sqlite_replay(
    path: Path, style_filter: IdentityStyleFilter
) -> list[TrainingExample]:
    if not path.exists() or not path.is_file():
        return []
    examples: list[TrainingExample] = []
    try:
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute("SELECT role, text FROM memories ORDER BY id ASC").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []

    pending_prompt: str | None = None
    for role, text in rows:
        role_l = str(role).lower()
        text_s = str(text).strip()
        if role_l in {"human", "user", "prompt", "failure"}:
            pending_prompt = text_s
            continue
        if role_l in {"assistant", "anra", "answer", "correction"} and pending_prompt:
            answer = style_filter.clean(text_s)
            if style_filter.accept(pending_prompt, answer):
                examples.append(
                    TrainingExample(
                        bucket="replay",
                        prompt=pending_prompt,
                        answer=answer,
                        source=str(path),
                        metadata={"ghost_memory": True, "quantized_from_turns": True},
                    )
                )
            pending_prompt = None
    return examples


def _load_ghost_replay_examples(style_filter: IdentityStyleFilter) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()
    sqlite_candidates = [
        Path(GHOST_DB_LOCAL),
        Path.home() / ".ghost_memory" / "memories.sqlite",
    ]
    jsonl_candidates = [Path(DRIVE_GHOST_DB)]

    for path in sqlite_candidates:
        for example in _load_ghost_sqlite_replay(path, style_filter):
            key = (example.prompt, example.answer)
            if key not in seen:
                seen.add(key)
                examples.append(example)
    for path in jsonl_candidates:
        for example in _load_ghost_jsonl_replay(path, style_filter):
            key = (example.prompt, example.answer)
            if key not in seen:
                seen.add(key)
                examples.append(example)
    return examples


def _load_replay_examples(style_filter: IdentityStyleFilter) -> list[TrainingExample]:
    examples = _load_ghost_replay_examples(style_filter)
    if FAILURE_REPLAY_DATASET.exists():
        try:
            for line in FAILURE_REPLAY_DATASET.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                parsed = _training_example_from_mapping(
                    record, str(FAILURE_REPLAY_DATASET), style_filter
                )
                if parsed is not None:
                    parsed.metadata["failure_replay"] = True
                    examples.append(parsed)
        except Exception:
            pass
    path = OUTPUT_V2_DIR.parent / "hard_examples.json"
    if not path.exists():
        path = OUTPUT_V2_DIR.parent / "v2_hard_examples.json"
    if not path.exists():
        return examples
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return examples
    for item in blob.get("examples", []):
        parsed = _parse_replay_example(str(item.get("preview", "")))
        if parsed is not None:
            examples.append(parsed)
    return examples


def _load_frontier_dfc_examples(
    dataset_path: Path,
    max_examples: int = 4096,
) -> list[TrainingExample]:
    """Load DFC-formatted frontier science examples."""
    dfc_path = dataset_path.parent / "frontier_dfc.jsonl"
    if not dfc_path.exists():
        return []
    examples: list[TrainingExample] = []
    with dfc_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text or "<bos>" not in text:
                    continue
                # Split on first <task> close tag to get prompt/answer
                if "</task>" in text:
                    split_point = text.index("</task>") + len("</task>")
                    prompt = text[:split_point]
                    answer = text[split_point:]
                else:
                    prompt = text[: len(text) // 2]
                    answer = text[len(text) // 2 :]
                examples.append(
                    TrainingExample(
                        bucket="frontier_dfc",
                        prompt=prompt,
                        answer=answer,
                        source="frontier_dfc",
                        weight=1.5,
                        metadata={
                            "domain": obj.get("domain", ""),
                            "template": obj.get("template", ""),
                            "verified": bool(obj.get("verified", False)),
                            "verifier_status": str(
                                obj.get(
                                    "verifier_status",
                                    "verified" if obj.get("verified") else "unverified",
                                )
                            ),
                        },
                    )
                )
                if len(examples) >= max_examples:
                    break
            except Exception:
                continue
    return examples


def _sample_bucket(
    rng: random.Random,
    bucket: list[TrainingExample],
    target_count: int,
) -> list[TrainingExample]:
    if target_count <= 0 or not bucket:
        return []
    # Cover each available example once before repeating it. This matters on
    # long Colab sessions: replacement-only sampling can waste updates on the
    # same examples while unseen examples remain in the local corpus.
    if target_count <= len(bucket):
        return rng.sample(bucket, target_count)
    sampled = rng.sample(bucket, len(bucket))
    sampled.extend(rng.choice(bucket) for _ in range(target_count - len(bucket)))
    return sampled


POST_TRAINING_MIX = {
    "instruction": 0.35,
    "code": 0.25,
    "math_logic": 0.15,
    "tools_actions": 0.10,
    "failure_replay": 0.10,
    "identity": 0.05,
}


def _post_training_category(example: TrainingExample) -> str:
    task_type = str(example.metadata.get("task_type", "")).strip().lower()
    source = example.source.lower()
    prompt = example.prompt.lower()
    if example.bucket == "replay" or bool(example.metadata.get("failure_replay")):
        verifier_status = str(example.metadata.get("verifier_status", "")).lower()
        if example.metadata.get("verified") is True or verifier_status == "verified":
            return "failure_replay"
        return "instruction"
    if example.bucket == "identity":
        return "identity"
    if example.bucket == "frontier_dfc" or task_type in {
        "tool",
        "action",
        "file_state",
        "constraint_json",
        "qiskit",
        "rdkit",
        "verilog",
        "citation_grounding",
    }:
        return "tools_actions"
    if task_type == "code" or any(token in source for token in ("code", "wizardcoder")):
        return "code"
    if task_type in {"math", "logic", "reasoning_math"} or any(
        token in source for token in ("math", "gsm8k")
    ):
        return "math_logic"
    if "write code" in prompt or "python" in prompt:
        return "code"
    return "instruction"


def build_post_training_mix(
    examples: Iterable[TrainingExample],
    *,
    seed: int,
    max_examples: int = 300_000,
) -> tuple[list[TrainingExample], dict[str, int], dict[str, int]]:
    """Build the native Phase D/E mix without replacement."""
    rng = random.Random(seed)
    unique: dict[tuple[str, str, str], TrainingExample] = {}
    pools = {name: [] for name in POST_TRAINING_MIX}
    for example in examples:
        key = (example.source, example.prompt, example.answer)
        if key in unique:
            continue
        unique[key] = example
        pools[_post_training_category(example)].append(example)
    target_total = min(max(0, int(max_examples)), len(unique))
    requested = {name: int(target_total * ratio) for name, ratio in POST_TRAINING_MIX.items()}
    requested["instruction"] += target_total - sum(requested.values())
    selected: list[TrainingExample] = []
    selected_keys: set[tuple[str, str, str]] = set()
    realized = dict.fromkeys(POST_TRAINING_MIX, 0)
    for name in POST_TRAINING_MIX:
        candidates = list(pools[name])
        rng.shuffle(candidates)
        for example in candidates[: requested[name]]:
            key = (example.source, example.prompt, example.answer)
            selected.append(example)
            selected_keys.add(key)
            realized[name] += 1
    if len(selected) < target_total:
        remaining = [example for key, example in unique.items() if key not in selected_keys]
        rng.shuffle(remaining)
        for example in remaining[: target_total - len(selected)]:
            selected.append(example)
            realized[_post_training_category(example)] += 1
    for example in selected:
        example.metadata = {
            **example.metadata,
            "post_training_category": _post_training_category(example),
        }
    rng.shuffle(selected)
    return selected, requested, realized


def build_v2_training_examples(
    *,
    dataset_path: Path | None = None,
    seed: int = 1337,
    max_examples: int | None = None,
    own_ratio: float | None = None,
    identity_ratio: float | None = None,
    teacher_ratio: float | None = None,
    symbolic_ratio: float | None = None,
    replay_ratio: float | None = None,
    model_params: int = 0,
    use_del: bool = True,
    post_training_mix: bool = False,
) -> tuple[list[TrainingExample], MixReport]:
    dataset_path = dataset_path or get_dataset_file()
    rng = random.Random(seed)
    style_filter = IdentityStyleFilter()

    base_examples = _load_base_examples(dataset_path)
    identity_examples, civ_rejected = _apply_civ_identity_gate(
        _load_identity_examples(base_examples)
    )
    external_teacher_examples = _load_external_teacher_examples(style_filter)
    teacher_examples = external_teacher_examples + _generate_teacher_examples(style_filter)
    symbolic_examples = _generate_symbolic_examples(style_filter)
    replay_examples = _load_replay_examples(style_filter)
    frontier_examples = _load_frontier_dfc_examples(dataset_path)
    ledger = DataEntropyLedger()
    all_buckets = [
        base_examples,
        identity_examples,
        teacher_examples,
        symbolic_examples,
        replay_examples,
        frontier_examples,
    ]
    del_rejected = 0
    duplicate_rejected = 0
    if use_del:
        filtered = []
        for bucket in all_buckets:
            rows, rejected, duplicates = _deduplicate_and_filter(bucket, ledger=ledger)
            filtered.append(rows)
            del_rejected += rejected
            duplicate_rejected += duplicates
        (
            base_examples,
            identity_examples,
            teacher_examples,
            symbolic_examples,
            replay_examples,
            frontier_examples,
        ) = filtered

    if post_training_mix:
        candidates = [
            *base_examples,
            *identity_examples,
            *teacher_examples,
            *symbolic_examples,
            *replay_examples,
            *frontier_examples,
        ]
        mixed, requested_counts, category_counts = build_post_training_mix(
            candidates,
            seed=seed,
            max_examples=max_examples or 300_000,
        )
        source_counts: dict[str, int] = {}
        for example in mixed:
            source_counts[example.source] = source_counts.get(example.source, 0) + 1
        return mixed, MixReport(
            total_examples=len(mixed),
            requested_counts=requested_counts,
            realized_counts=category_counts,
            source_counts=source_counts,
            teacher_external_used=len(external_teacher_examples),
            replay_available=len(replay_examples),
            civ_rejected=civ_rejected,
            del_rejected=del_rejected,
            duplicate_rejected=duplicate_rejected,
            active_weights=dict(POST_TRAINING_MIX),
            sampling_seed=seed,
        )

    total_examples = min(
        max_examples or V2_TRAINING.max_mixture_examples, max(len(base_examples), 4000)
    )
    controller = TrainingDataMixController(model_params or V2_FRONTIER_PARAMETER_COUNT)
    active = controller.weights
    control_path = OUTPUT_V2_DIR / "v2_mix_control.json"
    if control_path.exists():
        try:
            persisted = json.loads(control_path.read_text(encoding="utf-8"))
            persisted_weights = persisted.get("weights", {})
            if set(persisted_weights) == {"owner", "identity", "teacher", "symbolic", "replay"}:
                active = {name: float(value) for name, value in persisted_weights.items()}
        except Exception:
            pass
    own_ratio = active["owner"] if own_ratio is None else own_ratio
    identity_ratio = active["identity"] if identity_ratio is None else identity_ratio
    teacher_ratio = active["teacher"] if teacher_ratio is None else teacher_ratio
    symbolic_ratio = active["symbolic"] if symbolic_ratio is None else symbolic_ratio
    replay_ratio = active["replay"] if replay_ratio is None else replay_ratio

    ratio_total = own_ratio + identity_ratio + teacher_ratio + symbolic_ratio + replay_ratio
    if ratio_total <= 0:
        raise ValueError("V2 data mix ratios must sum to a positive value.")
    own_ratio /= ratio_total
    identity_ratio /= ratio_total
    teacher_ratio /= ratio_total
    symbolic_ratio /= ratio_total
    replay_ratio /= ratio_total

    requested_counts = {
        "own": int(total_examples * own_ratio),
        "identity": int(total_examples * identity_ratio),
        "teacher": int(total_examples * teacher_ratio),
        "symbolic": int(total_examples * symbolic_ratio),
    }
    requested_counts["replay"] = total_examples - sum(requested_counts.values())
    requested_counts["frontier_dfc"] = 0
    if frontier_examples:
        science_target = int(total_examples * getattr(V2_FRONTIER, "science_ratio", 0.20))
        protected = requested_counts["own"] + requested_counts["identity"]
        science_target = min(science_target, max(0, total_examples - protected))
        removable = (
            requested_counts["teacher"] + requested_counts["symbolic"] + requested_counts["replay"]
        )
        science_target = min(science_target, removable, len(frontier_examples))
        remaining = science_target
        for bucket_name in ("replay", "symbolic", "teacher"):
            take = min(requested_counts[bucket_name], remaining)
            requested_counts[bucket_name] -= take
            remaining -= take
            if remaining <= 0:
                break
        requested_counts["frontier_dfc"] = science_target

    mixed = []
    mixed.extend(_sample_bucket(rng, base_examples, requested_counts["own"]))
    mixed.extend(_sample_bucket(rng, identity_examples, requested_counts["identity"]))
    mixed.extend(_sample_bucket(rng, teacher_examples, requested_counts["teacher"]))
    mixed.extend(_sample_bucket(rng, symbolic_examples, requested_counts["symbolic"]))
    mixed.extend(_sample_bucket(rng, frontier_examples, requested_counts["frontier_dfc"]))

    replay_target = requested_counts["replay"]
    if replay_examples:
        mixed.extend(_sample_bucket(rng, replay_examples, replay_target))
    else:
        mixed.extend(_sample_bucket(rng, base_examples, replay_target))

    rng.shuffle(mixed)

    realized_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for example in mixed:
        realized_counts[example.bucket] = realized_counts.get(example.bucket, 0) + 1
        source_counts[example.source] = source_counts.get(example.source, 0) + 1

    report = MixReport(
        total_examples=len(mixed),
        requested_counts=requested_counts,
        realized_counts=realized_counts,
        source_counts=source_counts,
        teacher_external_used=len(external_teacher_examples),
        replay_available=len(replay_examples),
        civ_rejected=civ_rejected,
        del_rejected=del_rejected,
        duplicate_rejected=duplicate_rejected,
        active_weights={
            "owner": own_ratio,
            "identity": identity_ratio,
            "teacher": teacher_ratio,
            "symbolic": symbolic_ratio,
            "replay": replay_ratio,
        },
        sampling_seed=seed,
    )
    return mixed, report


class V2ConversationDataset(Dataset):
    """Bucket-preserving packed conversational language-model dataset.

    Training examples are often much shorter than the model context. Packing
    examples from the same bucket into a window turns padding-only compute into
    supervised tokens without mixing owner and non-owner gradients inside a
    microbatch.
    """

    PACKING_LAYOUT = "bucket_packed_v1"

    def __init__(
        self,
        examples: Iterable[TrainingExample],
        tokenizer: object,
        block_size: int,
        *,
        answer_loss_weight: float,
        validation_identity: str | None = None,
    ) -> None:
        self.examples: list[TrainingExample] = []
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.answer_loss_weight = float(max(1.0, answer_loss_weight))
        self.validation_identity = str(validation_identity or "") or None
        self.pad_id = int(tokenizer.pad_token_id)
        self.bos_id = int(tokenizer.bos_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.samples: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]
        ] = []
        self.bucket_counts: dict[str, int] = {}
        self._known_examples: set[tuple[str, str, str]] = set()
        self._weighted_targets = 0.0
        self._nonpad_target_tokens = 0
        self.answer_supervision_ratio = 0.0
        self.token_utilization = 0.0
        self._append_examples(list(examples))

    def _append_examples(self, examples: list[TrainingExample]) -> int:
        packed_segments: dict[
            str, list[tuple[list[int], list[float], list[bool], int]]
        ] = {}
        bucket_order: list[str] = []
        weighted_targets = 0.0
        added = 0
        for example in examples:
            key = (example.source, example.prompt, example.answer)
            if key in self._known_examples:
                continue
            self._known_examples.add(key)
            example_idx = len(self.examples)
            self.examples.append(example)
            added += 1
            prefix_ids = self.tokenizer.encode(
                f"H: {example.prompt}\nANRA:", add_special_tokens=False
            )
            answer_ids = self.tokenizer.encode(f" {example.answer}", add_special_tokens=False)
            full_ids = [self.bos_id, *prefix_ids, *answer_ids, self.eos_id]
            answer_start = 1 + len(prefix_ids)
            answer_end = answer_start + len(answer_ids)
            stride = max(32, self.block_size // 2)
            upper = max(1, len(full_ids) - 1)

            if example.bucket not in packed_segments:
                packed_segments[example.bucket] = []
                bucket_order.append(example.bucket)

            for start in range(0, upper, stride):
                chunk = full_ids[start : start + self.block_size + 1]
                if len(chunk) < 8:
                    continue
                target_weights = [1.0] * (len(chunk) - 1)
                target_answer_mask = [False] * (len(chunk) - 1)
                target_start = start + 1
                target_end = start + self.block_size + 1
                overlap_start = max(answer_start, target_start)
                overlap_end = min(answer_end, target_end)
                if overlap_end > overlap_start:
                    answer_weight = self.answer_loss_weight * float(example.weight)
                    for offset in range(overlap_start - target_start, overlap_end - target_start):
                        target_weights[offset] = answer_weight
                        target_answer_mask[offset] = True
                    weighted_targets += overlap_end - overlap_start
                # The first token in each source segment is not supervised when
                # following the prior segment's EOS. This avoids teaching an
                # artificial cross-example transition while retaining EOS loss.
                packed_segments[example.bucket].append(
                    (
                        list(chunk),
                        [0.0, *target_weights],
                        [False, *target_answer_mask],
                        example_idx,
                    )
                )

        for bucket in bucket_order:
            tokens: list[int] = []
            token_weights: list[float] = []
            answer_mask: list[bool] = []
            representative_index = -1
            for (
                segment_tokens,
                segment_weights,
                segment_answer_mask,
                example_index,
            ) in packed_segments[bucket]:
                if tokens and len(tokens) + len(segment_tokens) > self.block_size + 1:
                    self._append_packed_window(
                        bucket,
                        tokens,
                        token_weights,
                        answer_mask,
                        representative_index,
                    )
                    tokens, token_weights, answer_mask, representative_index = [], [], [], -1
                if representative_index < 0:
                    representative_index = example_index
                tokens.extend(segment_tokens)
                token_weights.extend(segment_weights)
                answer_mask.extend(segment_answer_mask)
                if len(tokens) == self.block_size + 1:
                    self._append_packed_window(
                        bucket,
                        tokens,
                        token_weights,
                        answer_mask,
                        representative_index,
                    )
                    tokens, token_weights, answer_mask, representative_index = [], [], [], -1
            if tokens:
                self._append_packed_window(
                    bucket,
                    tokens,
                    token_weights,
                    answer_mask,
                    representative_index,
                )

        self._weighted_targets += weighted_targets
        self.answer_supervision_ratio = self._weighted_targets / max(1, self._nonpad_target_tokens)
        self.token_utilization = self._nonpad_target_tokens / max(
            1, len(self.samples) * self.block_size
        )
        return added

    def _append_packed_window(
        self,
        bucket: str,
        tokens: list[int],
        token_weights: list[float],
        answer_mask: list[bool],
        representative_index: int,
    ) -> None:
        if (
            len(tokens) < 8
            or len(tokens) != len(token_weights)
            or len(tokens) != len(answer_mask)
        ):
            return
        x_values = tokens[:-1]
        y_values = tokens[1:]
        weights = token_weights[1:]
        target_answer_mask = answer_mask[1:]
        target_count = len(y_values)
        pad = self.block_size - target_count
        if pad < 0:
            raise ValueError("Packed training window exceeds configured block size")
        if pad:
            x_values.extend([self.pad_id] * pad)
            y_values.extend([self.pad_id] * pad)
            weights.extend([0.0] * pad)
            target_answer_mask.extend([False] * pad)
        self.samples.append(
            (
                torch.tensor(x_values, dtype=torch.long),
                torch.tensor(y_values, dtype=torch.long),
                torch.tensor(weights, dtype=torch.float32),
                representative_index,
                torch.tensor(target_answer_mask, dtype=torch.bool),
            )
        )
        self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1
        self._nonpad_target_tokens += target_count

    def reload_replay_bucket(self) -> int:
        return self._append_examples(_load_replay_examples(IdentityStyleFilter()))

    def bucket_for_sample(self, example_index: int) -> str:
        return self.examples[int(example_index)].bucket

    def bucket_for_window(self, sample_index: int) -> str:
        """Return the source bucket for a dataset window index."""
        _x, _y, _weights, example_index, _answer_mask = self.samples[int(sample_index)]
        return self.bucket_for_sample(int(example_index))

    def verified_esv_targets(
        self,
        example_indices: Iterable[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return VAD targets only when their provenance is explicitly verified."""
        targets: list[list[float]] = []
        mask: list[bool] = []
        for raw_index in example_indices:
            metadata = self.examples[int(raw_index)].metadata
            status = str(metadata.get("verifier_status", "")).strip().lower()
            verified = metadata.get("verified") is True or status == "verified"
            raw_vad = metadata.get("vad", metadata.get("esv_target"))
            values: list[float] | None = None
            if isinstance(raw_vad, dict):
                keys = ("valence", "arousal", "dominance")
                if all(key in raw_vad for key in keys):
                    values = [float(raw_vad[key]) for key in keys]
            elif isinstance(raw_vad, (list, tuple)) and len(raw_vad) == 3:
                values = [float(value) for value in raw_vad]
            valid = (
                verified
                and values is not None
                and all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values)
            )
            targets.append(values if valid and values is not None else [0.0, 0.0, 0.0])
            mask.append(bool(valid))
        return (
            torch.tensor(targets, device=device, dtype=dtype),
            torch.tensor(mask, device=device, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return self.samples[index]

    def snippet(self, example_index: int, max_chars: int = 240) -> str:
        example = self.examples[example_index]
        joined = f"H: {example.prompt}\nANRA: {example.answer}"
        return joined[:max_chars].replace("\n", "\\n")


class RawCausalShardDataset(Dataset):
    """Memory-mapped next-token windows from immutable uint16 token shards."""

    PACKING_LAYOUT = "raw_causal_shards_v1"

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: object,
        block_size: int,
        *,
        rotation_seed: int = 0,
        verify_hashes: bool = True,
        expected_tokenizer_sha256: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        manifest_bytes = self.manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("shards"), list):
            raise ValueError(f"Invalid token-shard manifest: {self.manifest_path}")
        self.manifest = manifest
        self.validation_identity = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_tokenizer = str(manifest.get("tokenizer_sha256", ""))
        if expected_tokenizer_sha256 and manifest_tokenizer != expected_tokenizer_sha256:
            raise ValueError(
                "Token shards were built with a different tokenizer: "
                f"manifest={manifest_tokenizer} active={expected_tokenizer_sha256}"
            )
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.pad_id = int(tokenizer.pad_token_id)
        self.answer_supervision_ratio = 0.0
        self.token_utilization = 1.0
        self.bucket_counts: dict[str, int] = {}
        self._arrays: dict[Path, np.ndarray] = {}
        self._shards: list[dict[str, object]] = []
        root = self.manifest_path.parent
        stable_shards: list[tuple[dict[str, object], int]] = []
        stable_cursor = 0
        for item in manifest["shards"]:
            if not isinstance(item, dict):
                continue
            windows = max(0, (int(item.get("tokens", 0)) - 1) // self.block_size)
            if windows:
                stable_shards.append((item, stable_cursor))
                stable_cursor += windows
        if stable_shards:
            offset = int(rotation_seed) % len(stable_shards)
            stable_shards = stable_shards[offset:] + stable_shards[:offset]
        cumulative = 0
        self._cumulative_windows: list[int] = []
        for item, stable_start in stable_shards:
            path = root / str(item.get("path", ""))
            if not path.is_file():
                raise FileNotFoundError(f"Token shard is missing: {path}")
            if verify_hashes:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != str(item.get("sha256", "")):
                    raise ValueError(f"Token shard hash mismatch: {path}")
            token_count = int(item.get("tokens", 0))
            windows = max(0, (token_count - 1) // self.block_size)
            if windows == 0:
                continue
            self._shards.append(
                {
                    "path": path,
                    "tokens": token_count,
                    "windows": windows,
                    "stable_start": stable_start,
                    "source_class": str(
                        item.get("source_class", item.get("source", "foundation"))
                    ),
                }
            )
            source_class = str(
                item.get("source_class", item.get("source", "foundation"))
            )
            self.bucket_counts[source_class] = self.bucket_counts.get(source_class, 0) + windows
            cumulative += windows
            self._cumulative_windows.append(cumulative)
        if not self._shards:
            raise ValueError(f"No complete training windows in {self.manifest_path}")

    def _array(self, path: Path) -> np.ndarray:
        array = self._arrays.get(path)
        if array is None:
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            if loaded.dtype != np.uint16 or loaded.ndim != 1:
                raise ValueError(f"Token shard must be one-dimensional uint16: {path}")
            array = loaded
            self._arrays[path] = array
        return array

    def __len__(self) -> int:
        return self._cumulative_windows[-1]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative_windows, normalized)
        prior = self._cumulative_windows[shard_index - 1] if shard_index else 0
        local_window = normalized - prior
        shard = self._shards[shard_index]
        start = local_window * self.block_size
        values = np.array(
            self._array(shard["path"])[start : start + self.block_size + 1],
            dtype=np.int64,
            copy=True,
        )
        tokens = torch.from_numpy(values)
        stable_index = int(shard["stable_start"]) + local_window
        return (
            tokens[:-1],
            tokens[1:],
            torch.ones(self.block_size, dtype=torch.float32),
            stable_index,
            torch.zeros(self.block_size, dtype=torch.bool),
        )

    def bucket_for_sample(self, sample_index: int) -> str:
        stable_index = int(sample_index)
        for shard in self._shards:
            start = int(shard["stable_start"])
            if start <= stable_index < start + int(shard["windows"]):
                return str(shard["source_class"])
        return "foundation"

    def bucket_for_window(self, sample_index: int) -> str:
        normalized = int(sample_index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(sample_index)
        shard_index = bisect_right(self._cumulative_windows, normalized)
        return str(self._shards[shard_index]["source_class"])

    def source_window_ranges(self) -> dict[str, tuple[tuple[int, int], ...]]:
        """Return compact dataset-index ranges grouped by immutable source."""
        grouped: dict[str, list[tuple[int, int]]] = {}
        start = 0
        for shard in self._shards:
            stop = start + int(shard["windows"])
            grouped.setdefault(str(shard["source_class"]), []).append((start, stop))
            start = stop
        return {name: tuple(ranges) for name, ranges in grouped.items()}

    @staticmethod
    def reload_replay_bucket() -> int:
        return 0

    def snippet(self, sample_index: int, max_chars: int = 240) -> str:
        x, _, _, _, _ = self[sample_index]
        return self.tokenizer.decode(x.tolist())[:max_chars].replace("\n", "\\n")


class WindowConsumptionTracker:
    """Compact cross-session accounting for immutable raw-token windows."""

    def __init__(
        self,
        total_windows: int,
        block_size: int,
        *,
        state: dict[str, object] | None = None,
    ) -> None:
        self.total_windows = max(0, int(total_windows))
        self.block_size = max(1, int(block_size))
        self._bits = bytearray((self.total_windows + 7) // 8)
        self.unique_windows = 0
        self.repeated_windows = 0
        if state:
            self.load_state_dict(state)

    def mark(self, indices: Iterable[int]) -> None:
        for raw_index in indices:
            index = int(raw_index)
            if index < 0 or index >= self.total_windows:
                raise IndexError(index)
            byte_index, bit_index = divmod(index, 8)
            mask = 1 << bit_index
            if self._bits[byte_index] & mask:
                self.repeated_windows += 1
            else:
                self._bits[byte_index] |= mask
                self.unique_windows += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "total_windows": self.total_windows,
            "block_size": self.block_size,
            "bits": bytes(self._bits),
            "unique_windows": self.unique_windows,
            "repeated_windows": self.repeated_windows,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state.get("total_windows", -1)) != self.total_windows:
            raise ValueError("Raw shard window inventory changed across continuation sessions")
        if int(state.get("block_size", -1)) != self.block_size:
            raise ValueError("Raw shard block size changed across continuation sessions")
        bits = bytes(state.get("bits", b""))
        if len(bits) != len(self._bits):
            raise ValueError("Raw shard consumption bitmap has an invalid length")
        self._bits[:] = bits
        self.unique_windows = int(state.get("unique_windows", 0))
        self.repeated_windows = int(state.get("repeated_windows", 0))
        if self.unique_windows != sum(byte.bit_count() for byte in self._bits):
            raise ValueError("Raw shard consumption bitmap count is inconsistent")

    def report(self, *, phase_target_tokens: int | None = None) -> dict[str, object]:
        total_visits = self.unique_windows + self.repeated_windows
        unique_tokens = self.unique_windows * self.block_size
        return {
            "total_windows": self.total_windows,
            "unique_windows_consumed": self.unique_windows,
            "repeated_windows": self.repeated_windows,
            "unique_tokens_consumed": unique_tokens,
            "repeated_token_percentage": (100.0 * self.repeated_windows / max(1, total_visits)),
            "remaining_phase_tokens": (
                max(0, int(phase_target_tokens) - unique_tokens)
                if phase_target_tokens is not None
                else None
            ),
        }
