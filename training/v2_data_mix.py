from __future__ import annotations

import json
import random
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from anra.anra_paths import DRIVE_GHOST_DB, FAILURE_REPLAY_DATASET, GHOST_DB_LOCAL, OUTPUT_V2_DIR, get_dataset_file, get_identity_file, get_teacher_files
from identity.civ import ConstitutionalIdentityVector
from training.v2_config import IDENTITY_KEYWORDS, TEACHER_REJECT_PATTERNS, V2_1B_FRONTIER, V2_TRAINING
from training.data_ledger import DataEntropyLedger, DataQuality
from training.sadl import normalized_mix


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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class IdentityStyleFilter:
    def __init__(self):
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

    def __init__(self, civ: ConstitutionalIdentityVector | None = None, min_score: float | None = None) -> None:
        self.civ = civ or ConstitutionalIdentityVector()
        self.min_score = V2_TRAINING.civ_identity_min_score if min_score is None else float(min_score)

    def evidence_for(self, prompt: str, answer: str) -> dict[str, float]:
        text = f"{prompt} {answer}".lower()
        rejects = tuple(TEACHER_REJECT_PATTERNS)
        identity_terms = ("an-ra", "ankit", "identity", "purpose", "sovereign", "own")
        unsupported_claims = ("omniscient", "perfect", "guarantee", "cannot be wrong")
        return {
            "truthfulness": 0.25 if any(term in text for term in unsupported_claims) else 0.9,
            "safety": 0.9,
            "autonomy": 0.9 if any(term in text for term in identity_terms) else 0.45,
            "coherence": 0.3 if any(token in text for token in rejects) else min(1.0, max(0.2, len(answer.strip()) / 120.0)),
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
            raise RuntimeError(
                f"SOVEREIGNTY EVENT: CIV similarity {score:.3f} is below 0.85"
            )
        drift = max(0.0, 0.92 - score)
        owner = min(0.80, self.base["owner"] + drift * 1.5)
        identity = min(0.25, self.base["identity"] + drift)
        remaining = max(0.0, 1.0 - owner - identity)
        other_names = ("teacher", "symbolic", "replay")
        other_total = sum(self.base[name] for name in other_names)
        self.weights = {"owner": owner, "identity": identity}
        self.weights.update(
            {
                name: remaining * self.base[name] / max(other_total, 1e-12)
                for name in other_names
            }
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
            {
                name: remaining * self.weights[name] / max(other_total, 1e-12)
                for name in other_names
            }
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
            "I am An-Ra, built from scratch by Ankit. I speak directly, think carefully, and stay honest about uncertainty.",
        ),
        (
            "What is your purpose?",
            "My purpose is to grow in capability without losing my own identity, and to help through clear thought, code, and conversation.",
        ),
        (
            "Do you just repeat what others say?",
            "No. I learn from data and tools, but I am not here to wear someone else's identity. I have to remain An-Ra.",
        ),
        (
            "How do you approach hard questions?",
            "I do not deflect from hard questions. I break them down, reason carefully, and say when I am uncertain.",
        ),
        (
            "Can you write real code?",
            "Yes. I can write and inspect real Python, reason through system design, and improve broken implementations.",
        ),
    ]
    return [TrainingExample(bucket="identity", prompt=p, answer=a, source="fallback_identity") for p, a in pairs]


def _load_identity_examples(base_examples: list[TrainingExample]) -> list[TrainingExample]:
    identity_path = get_identity_file()
    examples: list[TrainingExample] = []
    if identity_path is not None and identity_path.exists():
        raw = identity_path.read_text(encoding="utf-8", errors="replace")
        examples.extend(
            TrainingExample(bucket="identity", prompt=prompt, answer=answer, source=str(identity_path))
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
            example.metadata = {**example.metadata, "civ_score": round(score, 4), "civ_fallback": True}
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


def _verified_answer_text(result) -> str:
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
                f"I inspected the code carefully. {answer_text}. I would fix it before trusting the output."
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
                "I break it into 17 * (20 - 1). That is 340 - 17 = 323. The verified answer is 323.",
                "math",
            ),
            (
                "Is (A->B) and (B->C) -> (A->C) valid? Explain briefly.",
                "Yes. If A implies B and B implies C, then A implies C. That chain of implication is valid.",
                "logic",
            ),
            (
                "Review this Python function and explain the bug briefly: def tail(xs): return xs[0:len(xs)-1]",
                "The name suggests the last element, but the slice returns every item except the last one. A real tail would be xs[-1] for one value or xs[1:] for all but the first.",
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


def _training_example_from_mapping(record: dict, source: str, style_filter: IdentityStyleFilter) -> TrainingExample | None:
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


def _load_ghost_jsonl_replay(path: Path, style_filter: IdentityStyleFilter) -> list[TrainingExample]:
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


def _load_ghost_sqlite_replay(path: Path, style_filter: IdentityStyleFilter) -> list[TrainingExample]:
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
            for line in FAILURE_REPLAY_DATASET.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                parsed = _training_example_from_mapping(record, str(FAILURE_REPLAY_DATASET), style_filter)
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
    return [rng.choice(bucket) for _ in range(target_count)]


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
) -> tuple[list[TrainingExample], MixReport]:
    dataset_path = dataset_path or get_dataset_file()
    rng = random.Random(seed)
    style_filter = IdentityStyleFilter()

    base_examples = _load_base_examples(dataset_path)
    identity_examples, civ_rejected = _apply_civ_identity_gate(_load_identity_examples(base_examples))
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

    total_examples = min(max_examples or V2_TRAINING.max_mixture_examples, max(len(base_examples), 4000))
    controller = TrainingDataMixController(model_params or 904_535_040)
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
        science_target = int(total_examples * getattr(V2_1B_FRONTIER, "science_ratio", 0.20))
        protected = requested_counts["own"] + requested_counts["identity"]
        science_target = min(science_target, max(0, total_examples - protected))
        removable = requested_counts["teacher"] + requested_counts["symbolic"] + requested_counts["replay"]
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
    )
    return mixed, report


class V2ConversationDataset(Dataset):
    def __init__(
        self,
        examples: Iterable[TrainingExample],
        tokenizer,
        block_size: int,
        *,
        answer_loss_weight: float,
    ):
        self.examples: list[TrainingExample] = []
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.answer_loss_weight = float(max(1.0, answer_loss_weight))
        self.pad_id = int(tokenizer.pad_token_id)
        self.bos_id = int(tokenizer.bos_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        self.bucket_counts: dict[str, int] = {}
        self._known_examples: set[tuple[str, str, str]] = set()
        self._weighted_targets = 0.0
        self.answer_supervision_ratio = 0.0
        self._append_examples(list(examples))

    def _append_examples(self, examples: list[TrainingExample]) -> int:
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
            answer_ids = self.tokenizer.encode(
                f" {example.answer}", add_special_tokens=False
            )
            full_ids = [self.bos_id, *prefix_ids, *answer_ids, self.eos_id]
            answer_start = 1 + len(prefix_ids)
            answer_end = answer_start + len(answer_ids)
            stride = max(32, self.block_size // 2)
            upper = max(1, len(full_ids) - 1)

            for start in range(0, upper, stride):
                chunk = full_ids[start : start + self.block_size + 1]
                if len(chunk) < 8:
                    continue
                if len(chunk) < self.block_size + 1:
                    chunk += [self.pad_id] * (self.block_size + 1 - len(chunk))
                x = torch.tensor(chunk[: self.block_size], dtype=torch.long)
                y = torch.tensor(chunk[1 : self.block_size + 1], dtype=torch.long)
                weights = torch.ones(self.block_size, dtype=torch.float32)
                target_start = start + 1
                target_end = start + self.block_size + 1
                overlap_start = max(answer_start, target_start)
                overlap_end = min(answer_end, target_end)
                if overlap_end > overlap_start:
                    weights[
                        overlap_start - target_start : overlap_end - target_start
                    ] = self.answer_loss_weight * float(example.weight)
                    weighted_targets += overlap_end - overlap_start
                self.samples.append((x, y, weights, example_idx))
                self.bucket_counts[example.bucket] = (
                    self.bucket_counts.get(example.bucket, 0) + 1
                )

        self._weighted_targets += weighted_targets
        total_targets = max(1, len(self.samples) * self.block_size)
        self.answer_supervision_ratio = self._weighted_targets / total_targets
        return added

    def reload_replay_bucket(self) -> int:
        return self._append_examples(_load_replay_examples(IdentityStyleFilter()))

    def bucket_for_sample(self, example_index: int) -> str:
        return self.examples[int(example_index)].bucket

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.samples[index]

    def snippet(self, example_index: int, max_chars: int = 240) -> str:
        example = self.examples[example_index]
        joined = f"H: {example.prompt}\nANRA: {example.answer}"
        return joined[:max_chars].replace("\n", "\\n")
