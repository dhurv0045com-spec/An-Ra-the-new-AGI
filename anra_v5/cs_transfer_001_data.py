"""CS-TRANSFER-001 deterministic shared-token data plane.

The experiment compares physical V=4,096 vs V=24,576 models. The primary
causal requirement is stronger than ordinary semantic matching: paired arms
must consume byte-identical integer token sequences. We therefore render a
fresh V5.1 canary surface with the frozen production tokenizer, retain only
examples whose prompt+answer content lies wholly inside IDs 4..4095, and
freeze the selected token rows before training.

Selection is outcome-blind and hash-ranked. Original latent-world split
membership is preserved; no world can migrate between train/dev/sealed.

AMENDMENT 1: the original preregistration used ``\nAnswer:`` plus a leading
space. Static audit of the committed production-tokenizer probe showed
``Answer`` maps to ID 18224, which makes that delimiter incompatible with the
shared <4096 surface. Before any scientific outcome, Amendment 1 replaced the
delimiter with newline only (observed token ID 202) and removed the leading
answer space. All other scientific variables are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from anra_v5.v51_canary_data import (
    FAMILIES,
    Example,
    build_dataset,
    contamination_screen,
)

PROMPT_SUFFIX = "\n"
ANSWER_PREFIX = ""
COMMON_VOCAB = 4096
MAX_CONTENT_ID = COMMON_VOCAB - 1
MAX_ANSWER_TOKENS = 24
MAX_SEGMENT_TOKENS = 256  # includes BOS/EOS
MIN_ACCEPTANCE = 0.15
SELECT_COUNTS = {"training": 600, "development": 80, "sealed": 120}
SPLIT_ORDER = ("sealed", "development", "training")


@dataclass(frozen=True)
class TokenRow:
    example_id: str
    group_id: str
    family: str
    split: str
    template_id: str
    prompt: str
    answer: str
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]

    @property
    def content_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.answer_ids

    def canonical(self) -> dict[str, Any]:
        out = asdict(self)
        out["prompt_ids"] = list(self.prompt_ids)
        out["answer_ids"] = list(self.answer_ids)
        return out


def _rank(example_id: str) -> str:
    return hashlib.sha256(f"cs-transfer-001|{example_id}".encode()).hexdigest()


def _tokenize(tokenizer, row: Example) -> TokenRow | None:
    prompt_ids = tuple(int(x) for x in tokenizer.encode(row.prompt + PROMPT_SUFFIX))
    answer_ids = tuple(int(x) for x in tokenizer.encode(ANSWER_PREFIX + row.answer))
    content = prompt_ids + answer_ids
    if not prompt_ids or not answer_ids:
        return None
    if any(token < 4 or token > MAX_CONTENT_ID for token in content):
        return None
    if len(answer_ids) > MAX_ANSWER_TOKENS:
        return None
    if len(content) + 2 > MAX_SEGMENT_TOKENS:
        return None
    return TokenRow(
        example_id=row.example_id,
        group_id=row.group_id,
        family=row.family,
        split=row.split,
        template_id=row.template_id,
        prompt=row.prompt,
        answer=row.answer,
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
    )


def _selected_examples(rows: list[TokenRow]) -> list[Example]:
    return [
        Example(
            r.example_id, r.group_id, r.family, "easy", r.split,
            r.prompt, r.answer, r.template_id,
        )
        for r in rows
    ]


def _family_shortcuts(rows: list[TokenRow]) -> dict[str, dict[str, float]]:
    """Simple outcome-blind heuristic audit on the selected surface."""
    colors = (
        "crimson", "blue", "green", "amber", "violet", "scarlet",
        "teal", "coral", "ivory", "jade", "indigo", "rose",
    )
    result: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        fam = [r for r in rows if r.family == family]
        if not fam:
            raise ValueError(f"selected surface missing family {family}")
        answers = [r.answer.lower() for r in fam]
        counts = {a: answers.count(a) for a in set(answers)}
        modal = max(sorted(counts), key=lambda a: counts[a])
        constant = sum(a == modal for a in answers) / len(fam)
        fixed_none = sum(a == "none" for a in answers) / len(fam)
        last_word = 0
        first_color = 0
        answer_len_mode = 0
        lengths = [len(r.answer_ids) for r in fam]
        length_counts = {n: lengths.count(n) for n in set(lengths)}
        modal_len = max(sorted(length_counts), key=lambda n: length_counts[n])
        for r in fam:
            words = r.prompt.replace("?", "").replace(".", "").lower().split()
            if words and r.answer.lower() == words[-1]:
                last_word += 1
            hits = [c for c in colors if c in words]
            if hits and r.answer.lower() == hits[0]:
                first_color += 1
            if len(r.answer_ids) == modal_len:
                answer_len_mode += 1
        result[family] = {
            "constant_answer": constant,
            "fixed_none": fixed_none,
            "last_prompt_word": last_word / len(fam),
            "first_color_in_prompt": first_color / len(fam),
            "modal_answer_token_length_fraction": answer_len_mode / len(fam),
        }
    return result


def _predictive_shortcut_max(shortcuts: dict[str, dict[str, float]]) -> float:
    predictive_names = {
        "constant_answer", "fixed_none", "last_prompt_word", "first_color_in_prompt"
    }
    return max(
        float(value)
        for scores in shortcuts.values()
        for name, value in scores.items()
        if name in predictive_names
    )


def build_shared_surface(*, tokenizer, seed: int, candidate_worlds_per_family: int,
                         select_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Build and validate the shared low-ID surface."""
    counts = dict(SELECT_COUNTS if select_counts is None else select_counts)
    if set(counts) != {"training", "development", "sealed"}:
        raise ValueError("select_counts must define training/development/sealed")
    base = build_dataset(seed=seed, worlds_per_family=candidate_worlds_per_family)

    selected: dict[str, list[TokenRow]] = {s: [] for s in counts}
    acceptance: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in SPLIT_ORDER:
        acceptance[split] = {}
        for family in FAMILIES:
            candidates = [r for r in base["splits"][split] if r.family == family]
            eligible = [x for x in (_tokenize(tokenizer, r) for r in candidates) if x is not None]
            eligible.sort(key=lambda r: _rank(r.example_id))
            needed = int(counts[split])
            rate = len(eligible) / max(1, len(candidates))
            acceptance[split][family] = {
                "candidates": len(candidates), "eligible": len(eligible),
                "selected": needed, "acceptance_rate": rate,
            }
            if rate < MIN_ACCEPTANCE:
                raise ValueError(
                    f"low-ID acceptance below {MIN_ACCEPTANCE:.2f}: {split}/{family}={rate:.3f}"
                )
            if len(eligible) < needed:
                raise ValueError(
                    f"insufficient low-ID rows for {split}/{family}: {len(eligible)} < {needed}"
                )
            selected[split].extend(eligible[:needed])
        selected[split].sort(key=lambda r: r.example_id)

    selected_examples = {
        "splits": {s: _selected_examples(rows) for s, rows in selected.items()}
    }
    screen = contamination_screen(selected_examples)
    if not screen["clean"]:
        raise ValueError(f"selected surface contamination: {screen['collisions']}")

    dev_shortcuts = _family_shortcuts(selected["development"])
    predictive_max = _predictive_shortcut_max(dev_shortcuts)
    if predictive_max >= 0.35:
        raise ValueError(f"selected surface shortcut baseline {predictive_max:.4f} >= 0.35")

    split_hashes: dict[str, str] = {}
    token_stats: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        payload = "\n".join(
            json.dumps(row.canonical(), sort_keys=True, separators=(",", ":"))
            for row in selected[split]
        ).encode()
        split_hashes[split] = hashlib.sha256(payload).hexdigest()
        ids = [token for row in selected[split] for token in row.content_ids]
        token_stats[split] = {
            "rows": len(selected[split]),
            "content_tokens": len(ids),
            "unique_content_ids": len(set(ids)),
            "minimum_content_id": min(ids),
            "maximum_content_id": max(ids),
            "all_lt_4096": max(ids) < COMMON_VOCAB,
            "max_answer_tokens_observed": max(len(r.answer_ids) for r in selected[split]),
            "max_segment_tokens_observed": max(len(r.content_ids) + 2 for r in selected[split]),
        }

    manifest = {
        "schema": "anra-cs-transfer-001-shared-token-manifest/v1",
        "seed": int(seed),
        "candidate_worlds_per_family": int(candidate_worlds_per_family),
        "selection_counts_per_family": counts,
        "families": list(FAMILIES),
        "prompt_suffix": PROMPT_SUFFIX,
        "answer_prefix": ANSWER_PREFIX,
        "common_vocab_size": COMMON_VOCAB,
        "split_hashes": split_hashes,
        "token_stats": token_stats,
        "acceptance": acceptance,
        "contamination": screen,
        "shortcut_baselines_development": dev_shortcuts,
        "predictive_shortcut_max": predictive_max,
    }
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "rows": selected,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
    }


def serialize_rows(rows: list[TokenRow]) -> str:
    return "\n".join(json.dumps(r.canonical(), sort_keys=True) for r in rows) + "\n"


def deserialize_rows(text: str) -> list[TokenRow]:
    rows: list[TokenRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(TokenRow(
            example_id=d["example_id"], group_id=d["group_id"], family=d["family"],
            split=d["split"], template_id=d["template_id"], prompt=d["prompt"],
            answer=d["answer"], prompt_ids=tuple(int(x) for x in d["prompt_ids"]),
            answer_ids=tuple(int(x) for x in d["answer_ids"]),
        ))
    return rows


__all__ = [
    "ANSWER_PREFIX", "COMMON_VOCAB", "MAX_ANSWER_TOKENS", "MAX_SEGMENT_TOKENS",
    "PROMPT_SUFFIX", "SELECT_COUNTS", "TokenRow", "build_shared_surface",
    "deserialize_rows", "serialize_rows",
]
