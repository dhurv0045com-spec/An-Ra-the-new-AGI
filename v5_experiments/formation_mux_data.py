"""FORMATION-MUX-001 deterministic data surface (both experiments).

Latent semantic worlds are the CS-TRANSFER-001 direct-token grammar ported
from the read-only authority ``cymek-cs-transfer-001 @ b52fe453``
(``anra_v5/cs_transfer_001_data_v2.py``) — hash-ranked deterministic
generation, dedup, and shortcut screens, unmodified. On top of the shared
latent surface, EXPERIMENT B adds exactly two renderings:

    R0 PRODUCTION_BPE      latent grammar rendered to deterministic English
                           text and encoded by the frozen 24,576 tokenizer
    R1 ISOMORPHIC_RENDERING one latent grammar token per output token ID
                           (the CS-transfer surface itself; no composite
                           segmentation)

Both renderings share latent world identity (example_id / group_id / split).
Physical vocabulary stays 24,576 in every arm. The surface is generated ONCE
before GPU workers start and persisted as a canonical-JSON manifest with a
SHA-256; workers only read the manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

# -- latent grammar (ported verbatim in structure from cs_transfer_001_data_v2)

COMMON_VOCAB = 4096
MAX_ANSWER_TOKENS = 24
MAX_SEGMENT_TOKENS = 256
SELECT_COUNTS = {"training": 600, "development": 80, "sealed": 120}
SPLIT_ORDER = ("sealed", "development", "training")
FAMILIES = ("identity", "binding", "state_order", "composition", "termination",
            "missing_info")
COPY, BIND, STATE, COMPOSE, COUNT, MISS = 260, 261, 262, 263, 264, 265
PAIR, ADD, REMOVE, QUERY, SEP, NONE = 266, 267, 268, 269, 270, 271
SYMBOLS = tuple(range(300, 364))
VALUES = tuple(range(400, 432))
COUNT_TOKENS = {n: 500 + n for n in range(3, 9)}
GRAMMAR_IDS = {COPY, BIND, STATE, COMPOSE, COUNT, MISS, PAIR, ADD, REMOVE,
               QUERY, SEP, NONE, *SYMBOLS, *VALUES, *COUNT_TOKENS.values()}

# -- R0 rendering table: latent grammar token -> deterministic English token

GRAMMAR_WORDS = {
    COPY: "copy", BIND: "bind", STATE: "set", COMPOSE: "chain", COUNT: "count",
    MISS: "probe", PAIR: "equals", ADD: "plus", REMOVE: "drop", QUERY: "ask",
    SEP: "then", NONE: "unknown",
}
for _i, _sid in enumerate(SYMBOLS):
    GRAMMAR_WORDS[_sid] = f"s{_i:02d}"
for _i, _vid in enumerate(VALUES):
    GRAMMAR_WORDS[_vid] = f"v{_i:02d}"
for _n, _tid in COUNT_TOKENS.items():
    GRAMMAR_WORDS[_tid] = ["three", "four", "five", "six", "seven", "eight"][_n - 3]

WORD_WHITELIST = set(GRAMMAR_WORDS.values())


@dataclass(frozen=True)
class MuxRow:
    example_id: str
    group_id: str
    family: str
    split: str
    template_id: str
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    r0_prompt_text: str
    r0_answer_text: str

    def canonical(self) -> dict[str, Any]:
        out = asdict(self)
        out["prompt_ids"] = list(self.prompt_ids)
        out["answer_ids"] = list(self.answer_ids)
        return out


def _u64(seed: int, domain: str, counter: int) -> int:
    payload = f"{seed}:{domain}:{counter}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _perm(pool: tuple[int, ...], *, seed: int, domain: str, n: int) -> list[int]:
    ranked = sorted(pool, key=lambda x: hashlib.sha256(
        f"{seed}:{domain}:{x}".encode()).hexdigest())
    if n > len(ranked):
        raise ValueError("requested permutation exceeds pool")
    return ranked[:n]


def _row_for(*, seed: int, split: str, family: str, ordinal: int,
             attempt: int) -> MuxRow:
    dom = f"{split}:{family}:{ordinal}:{attempt}"
    syms = _perm(SYMBOLS, seed=seed, domain=dom + ":s", n=10)
    vals = _perm(VALUES, seed=seed, domain=dom + ":v", n=10)
    h = lambda k: _u64(seed, dom, k)

    if family == "identity":
        n = 3 + h(1) % 6
        seq = tuple(syms[:n])
        prompt, answer, template = (COPY, SEP, *seq, QUERY), seq, "token-copy"
    elif family == "binding":
        n = 4 + h(2) % 2
        qi = h(3) % n
        body: list[int] = [BIND]
        for e, v in zip(syms[:n], vals[:n]):
            body.extend((PAIR, e, v))
        prompt, answer, template = (*body, QUERY, syms[qi]), (vals[qi],), "token-binding"
    elif family == "state_order":
        n = 4
        ri = h(4) % n
        body = [STATE]
        for e in syms[:n]:
            body.extend((ADD, e))
        prompt = (*body, REMOVE, syms[ri], QUERY)
        answer = tuple(e for i, e in enumerate(syms[:n]) if i != ri)
        template = "token-state"
    elif family == "composition":
        a, b, c, d, e = syms[:5]
        prompt = (COMPOSE, PAIR, a, b, PAIR, b, c, PAIR, d, e, QUERY, a)
        answer, template = (c,), "token-two-hop"
    elif family == "termination":
        n = 3 + h(5) % 6
        seq = tuple(syms[:n])
        prompt, answer, template = (COUNT, SEP, *seq, QUERY), (COUNT_TOKENS[n],), "token-count"
    elif family == "missing_info":
        n = 4
        unknown = (h(6) % 5) == 0
        body = [MISS]
        for e, v in zip(syms[:n], vals[:n]):
            body.extend((PAIR, e, v))
        if unknown:
            query, answer = syms[n], (NONE,)
        else:
            qi = h(7) % n
            query, answer = syms[qi], (vals[qi],)
        prompt, template = (*body, QUERY, query), "token-abstain-binding"
    else:
        raise ValueError(f"unknown family {family}")

    if len(answer) > MAX_ANSWER_TOKENS or len(prompt) + len(answer) + 2 > MAX_SEGMENT_TOKENS:
        raise ValueError("direct-token row violates length contract")
    if min((*prompt, *answer)) < 4 or max((*prompt, *answer)) >= COMMON_VOCAB:
        raise ValueError("direct-token row escaped the latent vocabulary")

    r0_prompt = " ".join(GRAMMAR_WORDS[t] for t in prompt)
    r0_answer = " ".join(GRAMMAR_WORDS[t] for t in answer)
    canonical = json.dumps({"prompt": prompt, "answer": answer}, separators=(",", ":"))
    exid = hashlib.sha256(
        f"{split}|{family}|{ordinal}|{attempt}|{canonical}".encode()).hexdigest()[:16]
    gid = hashlib.sha256(
        f"group|{split}|{family}|{ordinal}|{attempt}".encode()).hexdigest()[:16]
    return MuxRow(exid, gid, family, split, template, tuple(prompt), tuple(answer),
                  r0_prompt, r0_answer)


def shortcut_screens(rows: list[MuxRow]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        fam = [r for r in rows if r.family == family]
        if not fam:
            raise ValueError(f"missing family {family}")
        answers = [r.answer_ids for r in fam]
        mode = max(set(answers), key=lambda a: (answers.count(a), a))
        out[family] = {
            "constant_full_answer":
                sum(a == mode for a in answers) / len(fam),
            "last_prompt_token_as_full_answer":
                sum(r.answer_ids == (r.prompt_ids[-1],) for r in fam) / len(fam),
            "first_symbol_as_full_answer":
                sum(r.answer_ids == (next(
                    (x for x in r.prompt_ids if x in SYMBOLS), -1),)
                    for r in fam) / len(fam),
        }
    return out


def build_surface(*, seed: int, tokenizer: Any = None,
                  select_counts: Mapping[str, int] | None = None,
                  ) -> dict[str, Any]:
    """Generate the shared latent surface; attach R0 token ids when the
    frozen production tokenizer is supplied. Deterministic in seed."""

    counts = dict(SELECT_COUNTS if select_counts is None else select_counts)
    if set(counts) != {"training", "development", "sealed"}:
        raise ValueError("select_counts must define training/development/sealed")
    if tokenizer is not None:
        identity = getattr(tokenizer, "identity", None)
        if identity is None or int(identity.vocabulary_size) != 24576:
            raise ValueError("expected frozen 24,576-entry production tokenizer")
        if dict(identity.special_token_ids) != {"pad": 0, "unk": 1, "bos": 2, "eos": 3}:
            raise ValueError("unexpected special token ids")

    splits: dict[str, list[dict[str, Any]]] = {}
    seen_content: set = set()
    seen_groups: set = set()
    screens: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        rows: list[MuxRow] = []
        per_family = counts[split] // len(FAMILIES)
        remainder = counts[split] - per_family * len(FAMILIES)
        for family_index, family in enumerate(FAMILIES):
            needed = per_family + (1 if family_index < remainder else 0)
            ordinal = attempts = 0
            family_rows: list[MuxRow] = []
            while len(family_rows) < needed:
                row = _row_for(seed=seed, split=split, family=family,
                               ordinal=ordinal, attempt=attempts)
                ordinal += 1
                attempts += 1
                key = (row.prompt_ids, row.answer_ids)
                if key in seen_content or row.group_id in seen_groups:
                    continue
                seen_content.add(key)
                seen_groups.add(row.group_id)
                family_rows.append(row)
            rows.extend(family_rows)
        screens[split] = shortcut_screens(rows)
        splits[split] = []
        for row in rows:
            body = row.canonical()
            if tokenizer is not None:
                body["r0_prompt_ids"] = [2, *tokenizer.encode(row.r0_prompt_text)]
                body["r0_answer_ids"] = [*tokenizer.encode(row.r0_answer_text), 3]
            splits[split].append(body)
    worst = max(v for split in screens.values() for fam in split.values()
                for v in fam.values())
    manifest = {
        "schema": "anra.formation-mux-surface/v1", "seed": seed,
        "select_counts": counts,
        "families": list(FAMILIES),
        "physical_vocabulary": 24576,
        "shortcut_screens": screens,
        "worst_shortcut_score": worst,
        "splits": splits,
    }
    manifest["sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in manifest.items() if k != "sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest


def assert_surface_clean(manifest: Mapping[str, Any], *,
                         worst_allowed: float = 0.5) -> None:
    """Fail closed on shortcut/contamination screens at manifest load."""

    claimed = hashlib.sha256(json.dumps(
        {k: v for k, v in manifest.items() if k != "sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if claimed != manifest["sha256"]:
        raise RuntimeError("FORMATION-MUX surface manifest hash mismatch")
    if manifest["worst_shortcut_score"] > worst_allowed:
        raise RuntimeError(
            f"surface shortcut screen FAILED: {manifest['worst_shortcut_score']}")
    # The ported CS-transfer grammar carries intrinsic mode-answer rates up to
    # ~0.35 on missing_info (NONE at 20% plus value modes); the fail-closed
    # bound exists to catch DEGENERATE surfaces, not to reject the historical
    # grammar. Per-family screens stay in the manifest for the audit.


def load_surface(path: str | Path) -> dict[str, Any]:
    from pathlib import Path as _Path
    manifest = json.loads(_Path(path).read_text(encoding="utf-8"))
    assert_surface_clean(manifest)
    return manifest
