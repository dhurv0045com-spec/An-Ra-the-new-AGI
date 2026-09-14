"""CS-TRANSFER-001 Amendment-2 direct-token shared surface.

The original natural-language low-ID filter proved infeasible before any GPU
training: the frozen production tokenizer yielded 0.000 acceptance for sealed
identity rows. This revision keeps the physical-vocabulary causal contrast but
constructs the instrument directly in the shared token-ID space 4..4095.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

COMMON_VOCAB = 4096
MAX_CONTENT_ID = COMMON_VOCAB - 1
MAX_ANSWER_TOKENS = 24
MAX_SEGMENT_TOKENS = 256
MIN_ACCEPTANCE = 0.15
SELECT_COUNTS = {"training": 600, "development": 80, "sealed": 120}
SPLIT_ORDER = ("sealed", "development", "training")
FAMILIES = ("identity", "binding", "state_order", "composition", "termination", "missing_info")

COPY, BIND, STATE, COMPOSE, COUNT, MISS = 260, 261, 262, 263, 264, 265
PAIR, ADD, REMOVE, QUERY, SEP, NONE = 266, 267, 268, 269, 270, 271
SYMBOLS = tuple(range(300, 364))
VALUES = tuple(range(400, 432))
COUNT_TOKENS = {n: 500 + n for n in range(3, 9)}
GRAMMAR_IDS = {COPY, BIND, STATE, COMPOSE, COUNT, MISS, PAIR, ADD, REMOVE, QUERY, SEP, NONE,
               *SYMBOLS, *VALUES, *COUNT_TOKENS.values()}
assert min(GRAMMAR_IDS) >= 4 and max(GRAMMAR_IDS) < COMMON_VOCAB


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


def _u64(seed: int, domain: str, counter: int) -> int:
    payload = f"{seed}:{domain}:{counter}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _perm(pool: tuple[int, ...], *, seed: int, domain: str, n: int) -> list[int]:
    ranked = sorted(pool, key=lambda x: hashlib.sha256(f"{seed}:{domain}:{x}".encode()).hexdigest())
    if n > len(ranked):
        raise ValueError("requested permutation exceeds pool")
    return ranked[:n]


def _row_for(*, seed: int, split: str, family: str, ordinal: int, attempt: int) -> TokenRow:
    dom = f"{split}:{family}:{ordinal}:{attempt}"
    syms = _perm(SYMBOLS, seed=seed, domain=dom + ":s", n=10)
    vals = _perm(VALUES, seed=seed, domain=dom + ":v", n=10)
    h = lambda k: _u64(seed, dom, k)

    if family == "identity":
        n = 3 + h(1) % 6
        seq = tuple(syms[:n])
        prompt = (COPY, SEP, *seq, QUERY)
        answer = seq
        template = "token-copy"
    elif family == "binding":
        n = 4 + h(2) % 2
        query_index = h(3) % n
        body: list[int] = [BIND]
        for e, v in zip(syms[:n], vals[:n]):
            body.extend((PAIR, e, v))
        prompt = (*body, QUERY, syms[query_index])
        answer = (vals[query_index],)
        template = "token-binding"
    elif family == "state_order":
        n = 4
        remove_index = h(4) % n
        body = [STATE]
        for e in syms[:n]:
            body.extend((ADD, e))
        body.extend((REMOVE, syms[remove_index], QUERY))
        prompt = tuple(body)
        answer = tuple(e for i, e in enumerate(syms[:n]) if i != remove_index)
        template = "token-state"
    elif family == "composition":
        a, b, c, d, e = syms[:5]
        prompt = (COMPOSE, PAIR, a, b, PAIR, b, c, PAIR, d, e, QUERY, a)
        answer = (c,)
        template = "token-two-hop"
    elif family == "termination":
        n = 3 + h(5) % 6
        seq = tuple(syms[:n])
        prompt = (COUNT, SEP, *seq, QUERY)
        answer = (COUNT_TOKENS[n],)
        template = "token-count"
    elif family == "missing_info":
        n = 4
        unknown = (h(6) % 5) == 0
        body = [MISS]
        for e, v in zip(syms[:n], vals[:n]):
            body.extend((PAIR, e, v))
        if unknown:
            query = syms[n]
            answer = (NONE,)
        else:
            qi = h(7) % n
            query = syms[qi]
            answer = (vals[qi],)
        prompt = (*body, QUERY, query)
        template = "token-abstain-binding"
    else:
        raise ValueError(f"unknown family {family}")

    if len(answer) > MAX_ANSWER_TOKENS or len(prompt) + len(answer) + 2 > MAX_SEGMENT_TOKENS:
        raise ValueError("direct-token row violates length contract")
    if min((*prompt, *answer)) < 4 or max((*prompt, *answer)) >= COMMON_VOCAB:
        raise ValueError("direct-token row escaped common vocabulary")

    canonical = json.dumps({"prompt": prompt, "answer": answer}, separators=(",", ":"))
    exid = hashlib.sha256(f"{split}|{family}|{ordinal}|{attempt}|{canonical}".encode()).hexdigest()[:16]
    gid = hashlib.sha256(f"group|{split}|{family}|{ordinal}|{attempt}".encode()).hexdigest()[:16]
    return TokenRow(
        exid, gid, family, split, template,
        "TOKENS " + " ".join(map(str, prompt)),
        "TOKENS " + " ".join(map(str, answer)),
        tuple(prompt), tuple(answer),
    )


def _predictive_shortcuts(rows: list[TokenRow]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        fam = [r for r in rows if r.family == family]
        if not fam:
            raise ValueError(f"missing family {family}")
        answers = [r.answer_ids for r in fam]
        mode = max(set(answers), key=lambda a: (answers.count(a), a))
        constant = sum(a == mode for a in answers) / len(fam)
        last_prompt = sum(r.answer_ids == (r.prompt_ids[-1],) for r in fam) / len(fam)
        first_symbol = sum(
            r.answer_ids == (next((x for x in r.prompt_ids if x in SYMBOLS), -1),)
            for r in fam
        ) / len(fam)
        out[family] = {
            "constant_full_answer": constant,
            "last_prompt_token_as_full_answer": last_prompt,
            "first_symbol_as_full_answer": first_symbol,
        }
    return out


def _predictive_max(shortcuts: dict[str, dict[str, float]]) -> float:
    return max(v for scores in shortcuts.values() for v in scores.values())


def build_shared_surface(*, tokenizer, seed: int, candidate_worlds_per_family: int,
                         select_counts: dict[str, int] | None = None) -> dict[str, Any]:
    counts = dict(SELECT_COUNTS if select_counts is None else select_counts)
    if set(counts) != {"training", "development", "sealed"}:
        raise ValueError("select_counts must define training/development/sealed")
    identity = getattr(tokenizer, "identity", None)
    if identity is None or int(identity.vocabulary_size) != 24576:
        raise ValueError("expected frozen 24576-entry production tokenizer identity")
    specials = dict(identity.special_token_ids)
    if specials != {"pad": 0, "unk": 1, "bos": 2, "eos": 3}:
        raise ValueError(f"unexpected special token ids: {specials}")

    selected: dict[str, list[TokenRow]] = {s: [] for s in counts}
    seen_content: dict[tuple[tuple[int, ...], tuple[int, ...]], str] = {}
    seen_groups: set[str] = set()
    acceptance: dict[str, dict[str, dict[str, float | int | str]]] = {}

    for split in SPLIT_ORDER:
        acceptance[split] = {}
        needed = int(counts[split])
        for family in FAMILIES:
            rows: list[TokenRow] = []
            ordinal = 0
            attempts = 0
            while len(rows) < needed:
                row = _row_for(seed=seed, split=split, family=family, ordinal=ordinal, attempt=attempts)
                key = (row.prompt_ids, row.answer_ids)
                ordinal += 1
                attempts += 1
                if key in seen_content or row.group_id in seen_groups:
                    if attempts > needed * 100 + 1000:
                        raise ValueError("cannot generate unique direct-token rows")
                    continue
                seen_content[key] = split
                seen_groups.add(row.group_id)
                rows.append(row)
            selected[split].extend(rows)
            acceptance[split][family] = {
                "candidates": needed,
                "eligible": needed,
                "selected": needed,
                "acceptance_rate": 1.0,
                "construction": "direct_common_token_ids",
            }
        selected[split].sort(key=lambda r: r.example_id)

    collisions: list[str] = []
    for i, a in enumerate(SPLIT_ORDER):
        a_content = {(r.prompt_ids, r.answer_ids) for r in selected[a]}
        a_groups = {r.group_id for r in selected[a]}
        for b in SPLIT_ORDER[i + 1:]:
            if a_content & {(r.prompt_ids, r.answer_ids) for r in selected[b]}:
                collisions.append(f"token_content:{a}:{b}")
            if a_groups & {r.group_id for r in selected[b]}:
                collisions.append(f"group:{a}:{b}")
    if collisions:
        raise ValueError(f"direct-token contamination: {collisions}")

    shortcuts = _predictive_shortcuts(selected["development"])
    predictive_max = _predictive_max(shortcuts)
    if predictive_max >= 0.35:
        raise ValueError(f"direct-token shortcut baseline {predictive_max:.4f} >= 0.35")

    split_hashes: dict[str, str] = {}
    token_stats: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        payload = "\n".join(
            json.dumps(r.canonical(), sort_keys=True, separators=(",", ":"))
            for r in selected[split]
        ).encode()
        split_hashes[split] = hashlib.sha256(payload).hexdigest()
        ids = [t for r in selected[split] for t in r.content_ids]
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
        "schema": "anra-cs-transfer-001-shared-token-manifest/v2",
        "surface_revision": "A2_DIRECT_TOKEN_COMMON_SPACE",
        "seed": int(seed),
        "candidate_worlds_per_family_legacy_field": int(candidate_worlds_per_family),
        "selection_counts_per_family": counts,
        "families": list(FAMILIES),
        "common_vocab_size": COMMON_VOCAB,
        "direct_token_construction": True,
        "production_tokenizer_used_for_identity_not_text_filtering": True,
        "grammar_ids_sha256": hashlib.sha256(json.dumps(sorted(GRAMMAR_IDS)).encode()).hexdigest(),
        "split_hashes": split_hashes,
        "token_stats": token_stats,
        "acceptance": acceptance,
        "contamination": {"clean": True, "collisions": collisions},
        "shortcut_baselines_development": shortcuts,
        "predictive_shortcut_max": predictive_max,
    }
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"rows": selected, "manifest": manifest, "manifest_sha256": manifest_sha}


def serialize_rows(rows: list[TokenRow]) -> str:
    return "\n".join(json.dumps(r.canonical(), sort_keys=True) for r in rows) + "\n"


def deserialize_rows(text: str) -> list[TokenRow]:
    rows: list[TokenRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(TokenRow(
            example_id=d["example_id"], group_id=d["group_id"], family=d["family"], split=d["split"],
            template_id=d["template_id"], prompt=d["prompt"], answer=d["answer"],
            prompt_ids=tuple(int(x) for x in d["prompt_ids"]),
            answer_ids=tuple(int(x) for x in d["answer_ids"]),
        ))
    return rows


__all__ = [
    "COMMON_VOCAB", "MAX_ANSWER_TOKENS", "MAX_SEGMENT_TOKENS", "MIN_ACCEPTANCE",
    "SELECT_COUNTS", "TokenRow", "build_shared_surface", "deserialize_rows", "serialize_rows",
]
