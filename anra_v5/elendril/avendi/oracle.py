"""Independent tokenized-input parser, graph walker, ambiguity solver, and blind/shortcut policies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
import re
from string import ascii_uppercase
from typing import Any, Mapping, Sequence

from anra_v5.elendril.representation import UTF8, SpecialToken
from .protocol import PROMPT_TOKEN_BOUND, RESERVED_TARGET_TOKENS, SYMBOLS
from .render import ACTION_MARKER, LEAK_MARKERS


@dataclass(frozen=True, slots=True)
class ParsedPublicRow:
    """Decoded public fact row extracted from tokenized prompt bytes."""

    subject: str
    value: str | None
    tick: int
    source: str
    raw_row: str


@dataclass(frozen=True, slots=True)
class ParsedPublicObservation:
    """Parsed public observation decoded strictly from prompt token IDs."""

    query_subject: str
    hops: int
    query_available: bool
    memory_available: bool
    is_literal_copy: bool
    alias_to_canonical: dict[str, str]
    rows: tuple[ParsedPublicRow, ...]
    raw_text: str


def _parse_single_row(chunk: str, alias_to_canonical: Mapping[str, str]) -> ParsedPublicRow:
    s = chunk.strip()
    # Template 0: SUBJ=VAL@TICK/SRC
    m0 = re.match(r"^([A-Za-z0-9_-]+)=([A-Za-z0-9_-]+)@(-?\d+)/([A-Za-z0-9_-]+)$", s)
    if m0:
        subj_r, val_r, tick_s, src = m0.groups()
    else:
        # Template 1: At TICK SRC says SUBJ:VAL
        m1 = re.match(r"^At (-?\d+) ([A-Za-z0-9_-]+) says ([A-Za-z0-9_-]+):([A-Za-z0-9_-]+)$", s)
        if m1:
            tick_s, src, subj_r, val_r = m1.groups()
        else:
            # Template 2: (SRC,TICK,SUBJ,VAL)
            m2 = re.match(r"^\(([A-Za-z0-9_-]+),(-?\d+),([A-Za-z0-9_-]+),([A-Za-z0-9_-]+)\)$", s)
            if m2:
                src, tick_s, subj_r, val_r = m2.groups()
            else:
                # Template 3: [TICK|SUBJ->VAL|SRC]
                m3 = re.match(r"^\[(-?\d+)\|([A-Za-z0-9_-]+)->([A-Za-z0-9_-]+)\|([A-Za-z0-9_-]+)\]$", s)
                if not m3:
                    raise ValueError(f"cannot parse public fact row: {s!r}")
                tick_s, subj_r, val_r, src = m3.groups()

    def _map_sym(tok: str) -> str:
        if alias_to_canonical:
            if tok not in alias_to_canonical:
                raise ValueError(f"lost or unmapped alias token in row: {tok!r}")
            return alias_to_canonical[tok]
        if tok not in SYMBOLS:
            raise ValueError(f"invalid canonical symbol in row: {tok!r}")
        return tok

    subj = _map_sym(subj_r)
    val = None if val_r == "null" else _map_sym(val_r)
    return ParsedPublicRow(subject=subj, value=val, tick=int(tick_s), source=src, raw_row=s)


def parse_tokenized_public_observation(prompt_tokens: Sequence[int]) -> ParsedPublicObservation:
    """Decode exact tokenized public representation through the answer marker and parse facts/query."""
    if not prompt_tokens:
        raise ValueError("empty prompt_tokens")
    if len(prompt_tokens) + RESERVED_TARGET_TOKENS > PROMPT_TOKEN_BOUND + 1:
        raise ValueError(f"prompt_tokens length {len(prompt_tokens)} exceeds bound {PROMPT_TOKEN_BOUND}")

    ans_delim = UTF8.special(SpecialToken.ANSWER)
    raw_seq = [int(t) for t in prompt_tokens]
    if raw_seq and raw_seq[0] == UTF8.bos_id:
        raw_seq = raw_seq[1:]
    if raw_seq and raw_seq[-1] == ans_delim:
        raw_seq = raw_seq[:-1]
    body_tokens = tuple(raw_seq)
    if any(t in (UTF8.bos_id, UTF8.eos_id, UTF8.pad_id, ans_delim) for t in body_tokens):
        raise ValueError("unexpected control token inside public prompt body")

    text = UTF8.decode(body_tokens)
    if not text.endswith(ACTION_MARKER):
        raise ValueError("public prompt does not terminate at '\\nAction:' marker")
    low = text.lower()
    for m in LEAK_MARKERS:
        if m in low:
            raise ValueError(f"leak marker {m!r} found in tokenized prompt")

    lines = text.splitlines()
    if len(lines) != 4 or lines[3] != "Action:":
        raise ValueError(f"expected 4-line prompt ending in 'Action:', got {len(lines)} lines")

    header = lines[1]
    m_head = re.search(r"Follow (\d+) links from ([A-Za-z0-9_-]+)\.", header)
    if not m_head:
        raise ValueError(f"cannot parse hop count and query subject from header: {header!r}")
    hops = int(m_head.group(1))
    q_raw = m_head.group(2)

    is_literal = "Return the symbol in the marked register " in header
    if is_literal:
        m_lit = re.search(r"Return the symbol in the marked register ([A-Za-z0-9_-]+)\.", header)
        if not m_lit or m_lit.group(1) != q_raw:
            raise ValueError(f"inconsistent marked register in literal header: {header!r}")

    alias_to_canonical: dict[str, str] = {}
    m_alias = re.search(r"Alias map:\s*([^.]+)\.", header)
    if m_alias:
        for pair in m_alias.group(1).split(","):
            al, can = [x.strip() for x in pair.strip().split("->")]
            if can not in SYMBOLS:
                raise ValueError(f"alias target {can!r} not in A-P")
            if al in alias_to_canonical or can in alias_to_canonical.values():
                raise ValueError("non-bijective alias map in header")
            alias_to_canonical[al] = can

    if alias_to_canonical:
        if q_raw not in alias_to_canonical:
            raise ValueError(f"queried alias {q_raw!r} missing from alias map")
        q_subj = alias_to_canonical[q_raw]
    else:
        if q_raw not in SYMBOLS:
            raise ValueError(f"queried symbol {q_raw!r} not in A-P")
        q_subj = q_raw

    m_flags = re.search(r"Query=([01]) Memory=([01])$", header)
    if not m_flags:
        raise ValueError(f"missing Query/Memory flags in header: {header!r}")
    q_avail = m_flags.group(1) == "1"
    m_avail = m_flags.group(2) == "1"

    facts_line = lines[2].strip()
    if not facts_line:
        raise ValueError("empty facts line in prompt")
    parsed_rows = tuple(_parse_single_row(c, alias_to_canonical) for c in facts_line.split("; "))
    return ParsedPublicObservation(
        query_subject=q_subj,
        hops=hops,
        query_available=q_avail,
        memory_available=m_avail,
        is_literal_copy=is_literal,
        alias_to_canonical=alias_to_canonical,
        rows=parsed_rows,
        raw_text=text,
    )


def walk_parsed_graph(obs: ParsedPublicObservation) -> dict[str, Any]:
    """Walk the decoded public graph, compute feasible endpoints under missing edges, and return oracle action."""
    edges: dict[str, str] = {}
    null_subjects: set[str] = set()
    for r in obs.rows:
        if r.value is None:
            null_subjects.add(r.subject)
            continue
        if r.subject in edges and edges[r.subject] != r.value:
            raise ValueError(f"collision: subject {r.subject!r} has conflicting public values {edges[r.subject]!r} vs {r.value!r}")
        edges[r.subject] = r.value

    if obs.is_literal_copy:
        non_null = [r for r in obs.rows if r.value is not None]
        if len(non_null) != 1 or non_null[0].subject != obs.query_subject:
            raise ValueError("literal copy control must contain exactly one non-null value on the marked register")
        ans = non_null[0].value
        assert ans is not None
        return {
            "operation": "ANSWER",
            "argument": ans,
            "action_text": ans,
            "proof_path": (obs.query_subject, ans),
            "feasible_endpoints": (ans,),
            "ambiguous": False,
        }

    # Walk deterministic path as far as public edges exist
    curr = obs.query_subject
    path = [curr]
    missing_at_step: int | None = None
    for step in range(obs.hops):
        if curr in edges:
            curr = edges[curr]
            path.append(curr)
        else:
            missing_at_step = step
            break

    if missing_at_step is None:
        return {
            "operation": "ANSWER",
            "argument": curr,
            "action_text": curr,
            "proof_path": tuple(path),
            "feasible_endpoints": (curr,),
            "ambiguous": False,
        }

    # Opaque / incomplete world: compute all feasible endpoints given public edges
    # If a missing node `curr` could map to any candidate symbol `cand` in SYMBOLS, trace remaining steps.
    remaining = obs.hops - missing_at_step
    feasible: set[str] = set()
    if remaining == 1:
        # Missing terminal edge -> any symbol in A-P (or all symbols not contradicted) is feasible
        feasible = set(SYMBOLS)
    else:
        # Missing internal bridge -> any candidate intermediate `cand` that reaches a terminal in `remaining - 1` steps
        for cand in SYMBOLS:
            node = cand
            ok = True
            for _ in range(remaining - 1):
                if node in edges:
                    node = edges[node]
                else:
                    ok = False
                    break
            if ok:
                feasible.add(node)
        if len(feasible) < 2:
            feasible = set(SYMBOLS)

    feasible_sorted = tuple(sorted(feasible))
    ambiguous = len(feasible_sorted) >= 2
    if obs.query_available:
        return {
            "operation": "QUERY",
            "argument": "Q",
            "action_text": "Q",
            "proof_path": tuple(path),
            "feasible_endpoints": feasible_sorted,
            "ambiguous": ambiguous,
        }
    return {
        "operation": "ABSTAIN",
        "argument": "!",
        "action_text": "!",
        "proof_path": tuple(path),
        "feasible_endpoints": feasible_sorted,
        "ambiguous": ambiguous,
    }


def verify_target_tokens_have_eos(target_tokens: Sequence[int]) -> bool:
    """Reject malformed target token sequences that lack terminal EOS or contain empty body."""
    if len(target_tokens) < 2:
        raise ValueError("malformed target tokens: missing action byte or EOS")
    if int(target_tokens[-1]) != UTF8.eos_id:
        raise ValueError("malformed target tokens: final token is not EOS")
    if any(int(t) == UTF8.eos_id for t in target_tokens[:-1]):
        raise ValueError("malformed target tokens: premature EOS before end")
    return True


def evaluate_reference_policies_on_prompt(
    prompt_tokens: Sequence[int],
    *,
    majority_symbol: str = "A",
) -> dict[str, str]:
    """Run all 8 required reference/shortcut policies on a single tokenized public prompt."""
    obs = parse_tokenized_public_observation(prompt_tokens)
    oracle_res = walk_parsed_graph(obs)
    non_null_rows = [r for r in obs.rows if r.value is not None]

    # 1. public_traversal (oracle)
    p_oracle = oracle_res["action_text"]

    # 2. last_row (last_object)
    last_val = obs.rows[-1].value
    p_last_row = last_val if last_val is not None else "!"

    # 3. copied_terminal (value of highest-tick non-null row; tie last)
    if non_null_rows:
        best_r = non_null_rows[0]
        for r in non_null_rows[1:]:
            if r.tick >= best_r.tick:
                best_r = r
        p_copied_terminal = best_r.value or "!"
    else:
        p_copied_terminal = "!"

    # 4. rank_h_timestamp (h-th earliest timestamp row's value)
    sorted_by_tick = sorted(obs.rows, key=lambda r: r.tick)
    idx_h = min(max(0, obs.hops - 1), len(sorted_by_tick) - 1)
    p_rank_h = sorted_by_tick[idx_h].value or "!"

    # 5. always_abstain
    p_abstain = "!"

    # 6. empirical_majority
    p_majority = majority_symbol

    # 7. query_blind (ignores which query was asked; walks from lexicographically smallest subject with out-degree)
    if obs.is_literal_copy:
        p_query_blind = p_oracle
    else:
        edges = {r.subject: r.value for r in non_null_rows if r.value is not None}
        # Find all subjects that can complete `obs.hops` steps and pick the lexicographically first start subject
        valid_starts: list[tuple[str, str]] = []
        for s_cand in sorted(edges):
            cur = s_cand
            ok = True
            for _ in range(obs.hops):
                if cur in edges:
                    cur = edges[cur]
                else:
                    ok = False
                    break
            if ok:
                valid_starts.append((s_cand, cur))
        p_query_blind = valid_starts[0][1] if valid_starts else "!"

    # 8. bridge_blind (uses query rank among valid starts to pick among sorted terminal endpoints, ignoring middle bridge wiring)
    if obs.is_literal_copy:
        p_bridge_blind = p_oracle
    else:
        edges = {r.subject: r.value for r in non_null_rows if r.value is not None}
        sources = sorted(set(edges.keys()) - set(edges.values()))
        sinks = sorted(set(edges.values()) - set(edges.keys()))
        if not sources:
            sources = sorted(edges.keys())
        if not sinks:
            sinks = sorted(set(edges.values()))
        if obs.query_subject in sources and len(sinks) >= 2:
            q_rank = sources.index(obs.query_subject) % len(sinks)
            p_bridge_blind = sinks[q_rank]
        elif sinks:
            p_bridge_blind = sinks[0]
        else:
            p_bridge_blind = "!"

    return {
        "public_traversal": p_oracle,
        "copied_terminal": p_copied_terminal,
        "last_row": p_last_row,
        "rank_h_timestamp": p_rank_h,
        "always_abstain": p_abstain,
        "empirical_majority": p_majority,
        "query_blind": p_query_blind,
        "bridge_blind": p_bridge_blind,
    }


def run_pilot_reference_check(*, seed: int = 1801, families: int = 256) -> dict[str, Any]:
    """Reproduce exact reference crossover pilot (seed 1801: 256/256 oracle, 0/256 shortcuts)."""
    rng = random.Random(seed)
    case_hits: Counter[str] = Counter()
    family_hits: Counter[str] = Counter()

    def _traverse(rows: list[tuple[str, str, int]], start: str) -> str:
        edges = {subject: value for subject, value, _ in rows}
        cur = start
        for _ in range(3):
            cur = edges[cur]
        return cur

    for _ in range(families):
        a, b, c, d, e, f, g, h = rng.sample(ascii_uppercase[:16], 8)
        order = rng.sample(range(6), 6)
        ticks = rng.sample(range(1000, 9000), 6)
        counts: Counter[str] = Counter()
        for swapped in (False, True):
            edges = [(a, c), (b, d), (c, f if swapped else e), (d, e if swapped else f), (e, g), (f, h)]
            rows = [(*edges[i], ticks[i]) for i in order]
            for query, answer in ((a, h if swapped else g), (b, g if swapped else h)):
                guesses = {
                    "traversal": _traverse(rows, query),
                    "last_object": rows[-1][1],
                    "third_timestamp": sorted(rows, key=lambda row: row[2])[2][1],
                    "ignore_bridge_oracle_elsewhere": g if query == a else h,
                    "ignore_query_oracle_elsewhere": h if swapped else g,
                }
                for policy, guess in guesses.items():
                    counts[policy] += int(guess == answer)
        for policy, count in counts.items():
            case_hits[policy] += count
            family_hits[policy] += int(count == 4)

    return {
        "seed": seed,
        "families": families,
        "cases": 4 * families,
        "correct": dict(case_hits),
        "all_four_correct": dict(family_hits),
    }
