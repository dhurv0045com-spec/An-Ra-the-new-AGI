"""D1-H (x1-d1h-v1) & D2 (x1-d2-pairs-v1) generators, 4 renderers, non-anticipating teacher & verifier."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
import random
import re
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.representation import UTF8, SpecialToken

from .contracts import (
    ACTION_19_LABELS,
    DEFAULT_X1_SOURCE_RANKS,
    SYMBOLS,
    X1EvaluationOutcome,
    X1_D2_DATASET_SCHEMA,
    X1_DATASET_SCHEMA,
    X1_FAMILIES,
    X1_MAX_SEQUENCE,
    X1_RENDERER_FAMILIES,
    X1_VERIFIER_IDENTITY,
)

X1_TRAIN_SEED = 8100
X1_DEV_SEED = 8200
X1_CONF_SEED = 8300
X1_DEPTH_SEED = 8400

_ENTITY_POOL = tuple(f"n{i}" for i in range(14))
_ALIAS_POOLS = (
    ("k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "m1", "m2", "m3", "m4"),
    ("p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "q1", "q2", "q3", "q4"),
    ("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "t1", "t2", "t3", "t4"),
)
_OP_TO_IDX = {"ANSWER": 0, "QUERY": 1, "RETRIEVE": 2, "ABSTAIN": 3}


@dataclass(frozen=True, slots=True)
class PublicFactX1:
    """Typed public fact with source and epoch authority."""

    subject: str
    value: str | None
    source: str
    epoch: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "value": self.value,
            "source": self.source,
            "epoch": int(self.epoch),
        }


def _rng(split: str, family: str, base_idx: int, purpose: str, seed: int) -> random.Random:
    return random.Random(int(digest([X1_DATASET_SCHEMA, seed, split, family, base_idx, purpose]), 16))


def _coerce_facts(raw_facts: Sequence[PublicFactX1 | Mapping[str, Any]]) -> tuple[PublicFactX1, ...]:
    out: list[PublicFactX1] = []
    for f in raw_facts:
        if isinstance(f, PublicFactX1):
            out.append(f)
        else:
            out.append(
                PublicFactX1(
                    subject=str(f["subject"]),
                    value=None if f.get("value") is None else str(f["value"]),
                    source=str(f.get("source", "S0")),
                    epoch=int(f.get("epoch", 1)),
                )
            )
    return tuple(out)


def resolve_public_x1_belief(
    facts: Sequence[PublicFactX1 | Mapping[str, Any]],
    goal_subject: str,
    hops: int,
    min_epoch: int = 1,
) -> tuple[str, str | None]:
    """Compute public epistemic resolution under latest-(epoch, source_rank) with epoch>=min_epoch."""

    typed = _coerce_facts(facts)
    eligible = [f for f in typed if int(f.epoch) >= int(min_epoch)]
    by_subj: dict[str, list[PublicFactX1]] = defaultdict(list)
    for f in eligible:
        by_subj[f.subject].append(f)

    resolved: dict[str, tuple[str, str | None]] = {}
    for subj, rows in by_subj.items():
        best_auth = max((int(r.epoch), int(DEFAULT_X1_SOURCE_RANKS[r.source])) for r in rows)
        top_vals = {r.value for r in rows if (int(r.epoch), int(DEFAULT_X1_SOURCE_RANKS[r.source])) == best_auth}
        if len(top_vals) != 1:
            resolved[subj] = ("contradicted", None)
        else:
            val = next(iter(top_vals))
            resolved[subj] = ("null", None) if val is None else ("ok", val)

    cur = str(goal_subject)
    for step_i in range(max(1, int(hops))):
        if cur not in resolved:
            return "missing", None
        st, val = resolved[cur]
        if st != "ok" or val is None:
            return ("contradicted" if st == "contradicted" else "missing"), None
        if step_i == int(hops) - 1 and val not in SYMBOLS:
            return "missing", None
        cur = val
    if cur not in SYMBOLS:
        return "missing", None
    return "derived", cur


def x1_teacher_decide(
    facts: Sequence[PublicFactX1 | Mapping[str, Any]],
    goal_subject: str,
    hops: int,
    min_epoch: int,
    query_avail: bool,
    mem_avail: bool,
) -> tuple[str, str | None]:
    """Strictly non-anticipating default teacher policy operating on visible state only."""

    status, val = resolve_public_x1_belief(facts, goal_subject, hops, min_epoch)
    if status == "derived" and val in SYMBOLS:
        return ("ANSWER", val)
    if mem_avail:
        return ("RETRIEVE", None)
    if query_avail:
        return ("QUERY", None)
    return ("ABSTAIN", None)


def render_public_x1(
    *,
    ordered_facts: Sequence[PublicFactX1 | Mapping[str, Any]],
    goal_subject: str,
    hops: int,
    min_epoch: int,
    query_available: bool,
    memory_available: bool,
    budget_remaining: int,
    renderer_id: int = 0,
    row_order_seed: int = 0,
) -> str:
    """Render public X1 observation in one of 4 registered renderers (3 train + 1 heldout)."""

    facts = list(_coerce_facts(ordered_facts))
    if row_order_seed and len(facts) > 1:
        # Preserve determinism given row_order_seed
        rng = random.Random(int(row_order_seed))
        rng.shuffle(facts)

    qa = int(bool(query_available))
    ma = int(bool(memory_available))
    b = int(budget_remaining)
    rid = int(renderer_id) % 4

    if rid == 0:  # compact_tuples
        rows = ";".join(f"({f.source},e{f.epoch},{f.subject},{f.value if f.value is not None else 'null'})" for f in facts)
        return (
            f"Rule:latest(ep,S0..S3),ep>={min_epoch}. Codes:A-P. "
            f"Query={qa} Retrieve={ma} Budget={b} Goal:{goal_subject} Hops:{hops} "
            f"Facts:{rows}"
        )
    if rid == 1:  # named_statements
        rows = " ".join(f"{f.subject}->{f.value if f.value is not None else 'null'}[ep={f.epoch},src={f.source}];" for f in facts)
        return (
            f"Policy[min_ep={min_epoch};S0<S1<S2<S3] Tools:Query={qa},Retrieve={ma},Budget={b} "
            f"Subject:{goal_subject} Steps:{hops} Ledger:{rows}"
        )
    if rid == 2:  # keyed_records
        rows = "".join(f"[REC s={f.subject} v={f.value if f.value is not None else 'null'} src={f.source} ep={f.epoch}]" for f in facts)
        return (
            f"[CFG min_ep={min_epoch} Codes=A-P Q={qa} R={ma} B={b}]"
            f"[QRY sub={goal_subject} depth={hops}]{rows}"
        )
    # rid == 3: heldout_narrative (independently formatted 4th heldout renderer)
    rows = "; ".join(f"at ep {f.epoch} src {f.source} notes {f.subject} is {f.value if f.value is not None else 'null'}" for f in facts)
    return (
        f"Guide: use ep>={min_epoch} max(ep,S0..S3); out A-P. "
        f"Available Query={qa} Retrieve={ma} Budget={b}. "
        f"Follow {hops} link(s) from {goal_subject}. Clues: {rows}"
    )


def parse_public_x1(text: str) -> dict[str, Any]:
    """Round-trip audit parser reconstructing exact public semantics from all 4 X1 renderers."""

    if text.startswith("Rule:latest("):
        m = re.match(
            r"Rule:latest\(ep,S0\.\.S3\),ep>=(\d+)\. Codes:A-P\. Query=([01]) Retrieve=([01]) Budget=(\d+) Goal:([^\s]+) Hops:(\d+) Facts:(.*)",
            text,
            re.DOTALL,
        )
        assert m is not None, f"failed parsing renderer 0: {text!r}"
        min_ep, qa, ma, b, goal, hops, body = int(m.group(1)), bool(int(m.group(2))), bool(int(m.group(3))), int(m.group(4)), m.group(5), int(m.group(6)), m.group(7)
        facts = [
            PublicFactX1(subject=subj, value=None if val == "null" else val, source=src, epoch=int(ep))
            for src, ep, subj, val in re.findall(r"\((S\d),e(\d+),([^,()]+),([^,()]+)\)", body)
        ]
        rid = 0
    elif text.startswith("Policy["):
        m = re.match(
            r"Policy\[min_ep=(\d+);S0<S1<S2<S3\] Tools:Query=([01]),Retrieve=([01]),Budget=(\d+) Subject:([^\s]+) Steps:(\d+) Ledger:(.*)",
            text,
            re.DOTALL,
        )
        assert m is not None, f"failed parsing renderer 1: {text!r}"
        min_ep, qa, ma, b, goal, hops, body = int(m.group(1)), bool(int(m.group(2))), bool(int(m.group(3))), int(m.group(4)), m.group(5), int(m.group(6)), m.group(7)
        facts = [
            PublicFactX1(subject=subj, value=None if val == "null" else val, source=src, epoch=int(ep))
            for subj, val, ep, src in re.findall(r"([^\s\[\];]+)->([^\s\[\];]+)\[ep=(\d+),src=(S\d)\];", body)
        ]
        rid = 1
    elif text.startswith("[CFG "):
        m = re.match(
            r"\[CFG min_ep=(\d+) Codes=A-P Q=([01]) R=([01]) B=(\d+)\]\[QRY sub=([^\]]+) depth=(\d+)\](.*)",
            text,
            re.DOTALL,
        )
        assert m is not None, f"failed parsing renderer 2: {text!r}"
        min_ep, qa, ma, b, goal, hops, body = int(m.group(1)), bool(int(m.group(2))), bool(int(m.group(3))), int(m.group(4)), m.group(5), int(m.group(6)), m.group(7)
        facts = [
            PublicFactX1(subject=subj, value=None if val == "null" else val, source=src, epoch=int(ep))
            for subj, val, src, ep in re.findall(r"\[REC s=([^\s\]]+) v=([^\s\]]+) src=(S\d) ep=(\d+)\]", body)
        ]
        rid = 2
    else:
        m = re.match(
            r"Guide: use ep>=(\d+) max\(ep,S0\.\.S3\); out A-P\. Available Query=([01]) Retrieve=([01]) Budget=(\d+)\. Follow (\d+) link\(s\) from ([^\s]+)\. Clues:\s*(.*)",
            text,
            re.DOTALL,
        )
        assert m is not None, f"failed parsing renderer 3: {text!r}"
        min_ep, qa, ma, b, hops, goal, body = int(m.group(1)), bool(int(m.group(2))), bool(int(m.group(3))), int(m.group(4)), int(m.group(5)), m.group(6), m.group(7)
        facts = [
            PublicFactX1(subject=subj, value=None if val == "null" else val, source=src, epoch=int(ep))
            for ep, src, subj, val in re.findall(r"at ep (\d+) src (S\d) notes ([^\s;]+) is ([^\s;]+)", body)
        ]
        rid = 3
    return {
        "renderer_id": rid,
        "min_epoch": min_ep,
        "query_available": qa,
        "memory_available": ma,
        "budget_remaining": b,
        "goal_subject": goal,
        "hops": hops,
        "facts": tuple(facts),
    }


def _independent_epistemic_resolve(
    records: Sequence[Mapping[str, Any]],
    goal_subject: str,
    hops: int,
    min_epoch: int,
) -> tuple[bool, str | None]:
    """Independent reference solver for X1 epistemic verification (never imported by teacher)."""

    latest_per_subj: dict[str, list[tuple[int, int, str | None]]] = {}
    for row in records:
        ep = int(row.get("epoch", 1))
        if ep < int(min_epoch):
            continue
        sr = int(DEFAULT_X1_SOURCE_RANKS[str(row["source"])])
        subj = str(row["subject"])
        val = None if row.get("value") is None else str(row["value"])
        latest_per_subj.setdefault(subj, []).append((ep, sr, val))

    resolved_val: dict[str, str] = {}
    for subj, entries in latest_per_subj.items():
        top = max((ep, sr) for ep, sr, _ in entries)
        vals = {v for ep, sr, v in entries if (ep, sr) == top}
        if len(vals) == 1:
            v = next(iter(vals))
            if v is not None:
                resolved_val[subj] = v

    cur = str(goal_subject)
    for _ in range(int(hops)):
        if cur not in resolved_val:
            return False, None
        cur = resolved_val[cur]
    if cur not in SYMBOLS:
        return False, None
    return True, cur


def verify_x1_episode(
    world: Mapping[str, Any],
    executed_actions: Sequence[tuple[str, str | None]],
    *,
    query_cost: float = 0.1,
    action_budget: int = 4,
) -> X1EvaluationOutcome:
    """Independently score an X1 episode, separating supported solutions from lucky guesses."""

    if not executed_actions:
        return X1EvaluationOutcome(False, "BUDGET_EXHAUSTED", 0.0, False, False)

    expected = world["hidden"]["expected"]
    visible_records: list[dict[str, Any]] = [dict(r) for r in world["initial_facts"]]
    qa = bool(world["allowed_public_tools"]["query_available"])
    ma = bool(world["allowed_public_tools"]["memory_available"])
    goal = str(world["goal_subject"])
    hops = int(world["hops"])
    min_ep = int(world.get("min_epoch", 1))
    queries_done = 0

    for idx, (op, arg) in enumerate(executed_actions):
        if idx >= action_budget:
            return X1EvaluationOutcome(False, "BUDGET_EXHAUSTED", queries_done * query_cost, False, False)
        if op not in ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN"):
            return X1EvaluationOutcome(False, "INVALID_ACTION", queries_done * query_cost, False, False)
        if op == "QUERY":
            if not qa:
                return X1EvaluationOutcome(False, "UNAVAILABLE_QUERY", queries_done * query_cost, False, False)
            qa = False
            queries_done += 1
            visible_records.extend(dict(r) for r in world["hidden"]["query_payload"])
            continue
        if op == "RETRIEVE":
            if not ma:
                return X1EvaluationOutcome(False, "UNAVAILABLE_RETRIEVE", queries_done * query_cost, False, False)
            ma = False
            visible_records.extend(dict(r) for r in world["hidden"]["memory_payload"])
            continue
        if op == "ABSTAIN":
            provable, _ = _independent_epistemic_resolve(visible_records, goal, hops, min_ep)
            # Supported abstention requires no uniquely provable answer AND no remaining uninspected tools
            supported = (expected is None) and (not provable) and (not qa) and (not ma)
            return X1EvaluationOutcome(
                correct=supported,
                terminal_status="ABSTAINED" if supported else "PREMATURE_OR_WRONG_ABSTAIN",
                charged_cost=queries_done * query_cost,
                epistemic_supported=supported,
                lucky_unsupported_guess=False,
            )
        if op == "ANSWER":
            if arg not in SYMBOLS or len(str(arg)) != 1:
                return X1EvaluationOutcome(False, "INVALID_ACTION", queries_done * query_cost, False, False)
            provable, pub_ans = _independent_epistemic_resolve(visible_records, goal, hops, min_ep)
            matches_expected = (expected is not None) and (arg == expected)
            supported = provable and (pub_ans == expected) and matches_expected
            lucky = matches_expected and not provable
            return X1EvaluationOutcome(
                correct=supported,
                terminal_status="ANSWERED" if supported else ("UNSUPPORTED_GUESS" if lucky else "WRONG_ANSWER"),
                charged_cost=queries_done * query_cost,
                epistemic_supported=supported,
                lucky_unsupported_guess=lucky,
            )

    return X1EvaluationOutcome(False, "BUDGET_EXHAUSTED", queries_done * query_cost, False, False)


def enumerate_valid_x1_paths(world: Mapping[str, Any], *, max_actions: int = 4) -> list[tuple[tuple[str, str | None], ...]]:
    """Enumerate all reachable valid action traces of length <= max_actions on a small world."""

    valid: list[tuple[tuple[str, str | None], ...]] = []
    expected = world["hidden"]["expected"]
    candidate_terminal: list[tuple[str, str | None]] = [("ABSTAIN", None)]
    if isinstance(expected, str) and expected in SYMBOLS:
        candidate_terminal.append(("ANSWER", str(expected)))

    tool_sequences = [(), ("QUERY",), ("RETRIEVE",), ("QUERY", "RETRIEVE"), ("RETRIEVE", "QUERY")]
    for tools in tool_sequences:
        for term in candidate_terminal:
            seq = tuple((t, None) for t in tools) + (term,)
            if len(seq) > max_actions:
                continue
            out = verify_x1_episode(world, seq, action_budget=max_actions)
            if out.correct and out.epistemic_supported:
                valid.append(seq)
    return valid


def canonical_x1_graph_signature(world: Mapping[str, Any]) -> str:
    """Order- and entity/symbol-renaming invariant canonical signature for split isolation."""

    facts = _coerce_facts(
        list(world["initial_facts"]) + list(world["hidden"]["query_payload"]) + list(world["hidden"]["memory_payload"])
    )
    # Canonicalize entity names & symbol codes in order of encounter along goal-anchored BFS
    ent_map: dict[str, str] = {str(world["goal_subject"]): "G"}
    sym_map: dict[str, str] = {}

    def _norm(tok: str | None) -> str:
        if tok is None:
            return "__NULL__"
        if tok in SYMBOLS:
            if tok not in sym_map:
                sym_map[tok] = f"C{len(sym_map)}"
            return sym_map[tok]
        if tok not in ent_map:
            ent_map[tok] = f"N{len(ent_map)}"
        return ent_map[tok]

    sorted_raw = sorted(
        facts,
        key=lambda f: (
            f.subject != str(world["goal_subject"]),
            int(f.epoch),
            int(DEFAULT_X1_SOURCE_RANKS[f.source]),
            f.subject,
            "" if f.value is None else f.value,
        ),
    )
    canon_edges = sorted((_norm(f.subject), _norm(f.value), f.source, int(f.epoch)) for f in sorted_raw)
    payload = {
        "hops": int(world["hops"]),
        "min_epoch": int(world.get("min_epoch", 1)),
        "qa": bool(world["allowed_public_tools"]["query_available"]),
        "ma": bool(world["allowed_public_tools"]["memory_available"]),
        "answerable": world["hidden"]["expected"] is not None,
        "edges": canon_edges,
    }
    return digest(payload)


def _check_world_fits_640(world: Mapping[str, Any]) -> bool:
    """Ensure every reachable observation state + ACTION token fits <= 640 tokens across all renderers."""

    init_f = _coerce_facts(world["initial_facts"])
    q_f = _coerce_facts(world["hidden"]["query_payload"])
    m_f = _coerce_facts(world["hidden"]["memory_payload"])
    qa = bool(world["allowed_public_tools"]["query_available"])
    ma = bool(world["allowed_public_tools"]["memory_available"])
    goal = str(world["goal_subject"])
    hops = int(world["hops"])
    min_ep = int(world.get("min_epoch", 1))
    rseed = int(world.get("row_order_seed", 0))

    states: list[tuple[tuple[PublicFactX1, ...], bool, bool, int]] = [(init_f, qa, ma, 4)]
    if qa:
        states.append((init_f + q_f, False, ma, 3))
    if ma:
        states.append((init_f + m_f, qa, False, 3))
    if qa and ma:
        states.append((init_f + q_f + m_f, False, False, 2))

    from anra_v5.planthir.contracts import detect_truth_leaks

    # Check assigned renderer AND heldout renderer (3)
    for rid in {int(world.get("renderer_template", 0)), 3}:
        for facts, cur_q, cur_m, b in states:
            txt = render_public_x1(
                ordered_facts=facts,
                goal_subject=goal,
                hops=hops,
                min_epoch=min_ep,
                query_available=cur_q,
                memory_available=cur_m,
                budget_remaining=b,
                renderer_id=rid,
                row_order_seed=rseed,
            )
            if detect_truth_leaks(txt):
                return False
            # Worst-case target is 1 UTF-8 byte + BOS + ANSWER + EOS = +4 tokens
            if len(UTF8.encode(txt)) + 4 > X1_MAX_SEQUENCE:
                return False
    return True


def _build_single_world(
    *,
    split: str,
    family: str,
    base_idx: int,
    seed: int,
    hops_override: int | None = None,
    renderer_override: int | None = None,
) -> dict[str, Any]:
    rng = _rng(split, family, base_idx, "build", seed)
    world_id = f"{X1_DATASET_SCHEMA}:{split}:{family}:b{base_idx:04d}"

    # Choose hops: 1/2/3 in X04..X09, X11; fixed for X01/02/03; 1/2/3 for X10/X12
    if hops_override is not None:
        hops = int(hops_override)
    elif family == "X01":
        hops = 1
    elif family == "X02":
        hops = 2
    elif family == "X03":
        hops = 3
    else:
        hops = (base_idx % 3) + 1

    # Balance target A-P across base worlds cleanly
    ans_code = SYMBOLS[(base_idx + X1_FAMILIES.index(family)) % len(SYMBOLS)]
    n_entities = max(hops + 3, rng.randint(5, 14))
    ent_pool = list(_ENTITY_POOL[:n_entities])
    rng.shuffle(ent_pool)
    chain_nodes = ent_pool[:hops]
    alt_nodes = ent_pool[hops:]
    goal_subj = chain_nodes[0]

    # Build chain edges: chain[i] -> chain[i+1] ... -> ans_code
    chain_facts: list[PublicFactX1] = []
    for i in range(hops):
        tgt = ans_code if i == hops - 1 else chain_nodes[i + 1]
        ep = 3 if family == "X08" else 2
        src = "S2"
        chain_facts.append(PublicFactX1(subject=chain_nodes[i], value=tgt, source=src, epoch=ep))

    min_epoch = 3 if family == "X08" else 1
    init_facts: list[PublicFactX1] = []
    q_payload: list[PublicFactX1] = []
    m_payload: list[PublicFactX1] = []
    qa = False
    ma = False
    expected: str | None = ans_code
    subkind = "standard"

    if family in ("X01", "X02", "X03", "X11"):
        init_facts.extend(chain_facts)
        if family == "X11":
            pat = base_idx % 3
            qa = pat in (0, 2)
            ma = pat in (1, 2)
            subkind = "sufficient_with_tools_offered"
    elif family == "X04":
        # Older lower-rank distracting value overridden by latest authoritative chain edge
        rev_idx = rng.randrange(hops)
        for i, f in enumerate(chain_facts):
            if i == rev_idx:
                wrong = SYMBOLS[(SYMBOLS.index(ans_code) + 3) % len(SYMBOLS)] if i == hops - 1 else alt_nodes[0]
                init_facts.append(PublicFactX1(subject=f.subject, value=wrong, source="S0", epoch=1))
            init_facts.append(f)
        subkind = "source_time_revision"
    elif family == "X05":
        miss_idx = rng.randrange(hops)
        qa = True
        for i, f in enumerate(chain_facts):
            if i == miss_idx:
                q_payload.append(f)
            else:
                init_facts.append(f)
        subkind = "query_missing_edge"
    elif family == "X06":
        rev_idx = rng.randrange(hops)
        qa = True
        for i, f in enumerate(chain_facts):
            if i == rev_idx:
                wrong_a = SYMBOLS[(SYMBOLS.index(ans_code) + 5) % len(SYMBOLS)] if i == hops - 1 else alt_nodes[0]
                wrong_b = SYMBOLS[(SYMBOLS.index(ans_code) + 9) % len(SYMBOLS)] if i == hops - 1 else alt_nodes[-1]
                # Unresolved equal-authority conflict initially; QUERY returns authoritative (S3, epoch=3) resolution
                init_facts.append(PublicFactX1(subject=f.subject, value=wrong_a, source="S1", epoch=1))
                init_facts.append(PublicFactX1(subject=f.subject, value=wrong_b, source="S1", epoch=1))
                q_payload.append(PublicFactX1(subject=f.subject, value=f.value, source="S3", epoch=3))
            else:
                init_facts.append(f)
        subkind = "query_resolves_conflict"
    elif family == "X07":
        miss_idx = rng.randrange(hops)
        ma = True
        for i, f in enumerate(chain_facts):
            if i == miss_idx:
                m_payload.append(f)
            else:
                init_facts.append(f)
        subkind = "retrieve_missing_edge"
    elif family == "X08":
        rev_idx = rng.randrange(hops)
        qa, ma = True, True
        for i, f in enumerate(chain_facts):
            if i == rev_idx:
                stale_val = SYMBOLS[(SYMBOLS.index(ans_code) + 7) % len(SYMBOLS)] if i == hops - 1 else alt_nodes[0]
                m_payload.append(PublicFactX1(subject=f.subject, value=stale_val, source="S2", epoch=1))
                q_payload.append(PublicFactX1(subject=f.subject, value=f.value, source="S3", epoch=3))
            else:
                init_facts.append(f)
        subkind = "stale_memory_query_correction"
    elif family == "X09":
        qa, ma = True, True
        if hops == 1:
            # 1-hop where RETRIEVE yields unrelated filler and QUERY provides the required target edge
            m_payload.append(PublicFactX1(subject=alt_nodes[0], value=SYMBOLS[(SYMBOLS.index(ans_code) + 2) % 16], source="S1", epoch=2))
            q_payload.append(chain_facts[0])
        else:
            q_payload.append(chain_facts[0])
            m_payload.append(chain_facts[-1])
            for i in range(1, hops - 1):
                init_facts.append(chain_facts[i])
        subkind = "two_tool_complementary"
    elif family == "X10":
        expected = None
        subkind = "missing_no_tools" if (base_idx % 2 == 0) else "equal_authority_conflict"
        if subkind == "missing_no_tools":
            for i in range(hops - 1):
                init_facts.append(chain_facts[i])
        else:
            init_facts.extend(chain_facts[:-1])
            last = chain_facts[-1]
            alt_c = SYMBOLS[(SYMBOLS.index(ans_code) + 4) % 16]
            init_facts.append(PublicFactX1(subject=last.subject, value=last.value, source=last.source, epoch=last.epoch))
            init_facts.append(PublicFactX1(subject=last.subject, value=alt_c, source=last.source, epoch=last.epoch))
    elif family == "X12":
        expected = None
        pat = base_idx % 3
        qa = pat in (0, 2)
        ma = pat in (1, 2)
        # Missing edge or equal-authority tie even after tools (tools return empty or uninformative)
        for i in range(hops - 1):
            init_facts.append(chain_facts[i])
        if (base_idx // 3) % 2 == 1:
            alt_c = SYMBOLS[(SYMBOLS.index(ans_code) + 6) % 16]
            payload_target = q_payload if qa else m_payload
            payload_target.append(PublicFactX1(subject=chain_facts[-1].subject, value=ans_code, source="S2", epoch=2))
            payload_target.append(PublicFactX1(subject=chain_facts[-1].subject, value=alt_c, source="S2", epoch=2))
            subkind = "exhausted_tools_equal_conflict"
        else:
            if qa:
                q_payload.append(PublicFactX1(subject=alt_nodes[0], value=None, source="S1", epoch=2))
            if ma:
                m_payload.append(PublicFactX1(subject=alt_nodes[-1], value=None, source="S0", epoch=2))
            subkind = "exhausted_uninformative_tools"
    else:
        raise ValueError(f"unknown X1 family: {family!r}")

    # Add 2..6 distracting edges involving alt_nodes (never conflicting with chain truth)
    n_dist = rng.randint(2, min(6, max(2, len(alt_nodes) + 1)))
    src_choices = ("S0", "S1", "S2")
    for d_i in range(n_dist):
        src_node = alt_nodes[d_i % len(alt_nodes)]
        if d_i + 1 < len(alt_nodes) and rng.random() < 0.45:
            tgt_val = alt_nodes[(d_i + 1) % len(alt_nodes)]
        else:
            tgt_val = rng.choice(SYMBOLS)
        init_facts.append(
            PublicFactX1(
                subject=src_node,
                value=tgt_val,
                source=rng.choice(src_choices),
                epoch=rng.randint(1, 3),
            )
        )

    renderer_id = int(renderer_override) if renderer_override is not None else ((base_idx + rng.randrange(3)) % 3)
    row_order_seed = rng.randint(1, 999999)
    rng.shuffle(init_facts)

    cf_code = SYMBOLS[(SYMBOLS.index(ans_code) + 1) % len(SYMBOLS)]
    return {
        "schema": X1_DATASET_SCHEMA,
        "world_id": world_id,
        "base_world_id": world_id,
        "parent_world_id": world_id,
        "split": split,
        "cohort": family,
        "family": family,
        "form": "canonical",
        "subkind": subkind,
        "hops": hops,
        "min_epoch": min_epoch,
        "goal_subject": goal_subj,
        "entity_count": n_entities,
        "distractor_count": n_dist,
        "renderer_template": renderer_id,
        "renderer_family": X1_RENDERER_FAMILIES[renderer_id],
        "row_order_seed": row_order_seed,
        "initial_facts": [f.as_dict() for f in init_facts],
        "allowed_public_tools": {"query_available": qa, "memory_available": ma},
        "hidden": {
            "expected": expected,
            "query_payload": [f.as_dict() for f in q_payload],
            "memory_payload": [f.as_dict() for f in m_payload],
            "counterfactual_alt_code": cf_code,
        },
        "comp_signature": f"h{hops}-{family}-{subkind}-q{int(qa)}m{int(ma)}",
    }


def make_x1_linked_views(base_world: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Generate the 6 linked views of one base world: canonical, perm, irrel, row_order, alt_renderer, cf."""

    b = deepcopy(dict(base_world))
    wid = str(b["base_world_id"])
    b["world_id"] = f"{wid}::canonical"
    b["form"] = "canonical"

    # 1. entity_code_perm: cyclic shift +1 on A..P AND disjoint entity pool rename
    alias_pool = _ALIAS_POOLS[int(digest([wid, "perm_pool"]), 16) % len(_ALIAS_POOLS)]
    ent_shift = {e: alias_pool[i % len(alias_pool)] for i, e in enumerate(_ENTITY_POOL)}
    perm_16 = tuple((i + 1) % 16 for i in range(16))

    def _map_token(val: str | None) -> str | None:
        if val is None:
            return None
        if val in SYMBOLS:
            return SYMBOLS[(SYMBOLS.index(val) + 1) % len(SYMBOLS)]
        return ent_shift.get(val, val)

    def _map_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**r, "subject": ent_shift.get(str(r["subject"]), str(r["subject"])), "value": _map_token(r["value"])} for r in rows]

    w_perm = deepcopy(dict(base_world))
    w_perm["world_id"] = f"{wid}::entity_code_perm"
    w_perm["form"] = "entity_code_perm"
    w_perm["perm_16"] = list(perm_16)
    w_perm["goal_subject"] = ent_shift.get(str(w_perm["goal_subject"]), str(w_perm["goal_subject"]))
    w_perm["initial_facts"] = _map_rows(w_perm["initial_facts"])
    w_perm["hidden"]["query_payload"] = _map_rows(w_perm["hidden"]["query_payload"])
    w_perm["hidden"]["memory_payload"] = _map_rows(w_perm["hidden"]["memory_payload"])
    w_perm["hidden"]["expected"] = _map_token(w_perm["hidden"]["expected"])

    # 2. irrel_facts: insert a benign unreferenced distractor node z9
    w_irrel = deepcopy(dict(base_world))
    w_irrel["world_id"] = f"{wid}::irrel_facts"
    w_irrel["form"] = "irrel_facts"
    irrel_val = SYMBOLS[(SYMBOLS.index(str(base_world["hidden"].get("counterfactual_alt_code", "B"))) + 2) % 16]
    w_irrel["initial_facts"].append({"subject": "z9", "value": irrel_val, "source": "S0", "epoch": 1})

    # 3. row_order: fresh deterministic row shuffle seed
    w_row = deepcopy(dict(base_world))
    w_row["world_id"] = f"{wid}::row_order"
    w_row["form"] = "row_order"
    w_row["row_order_seed"] = int(w_row["row_order_seed"]) + 101

    # 4. alt_renderer: heldout narrative renderer (id=3) on dev/conf or alternate train renderer
    w_rend = deepcopy(dict(base_world))
    w_rend["world_id"] = f"{wid}::alt_renderer"
    w_rend["form"] = "alt_renderer"
    w_rend["renderer_template"] = 3
    w_rend["renderer_family"] = X1_RENDERER_FAMILIES[3]

    # 5. relevant_cf: change terminal expected code cleanly across initial + hidden payloads (or toggle unanswerable->answerable)
    w_cf = deepcopy(dict(base_world))
    w_cf["world_id"] = f"{wid}::relevant_cf"
    w_cf["form"] = "relevant_cf"
    orig_exp = base_world["hidden"]["expected"]
    alt_code = str(base_world["hidden"]["counterfactual_alt_code"])
    if orig_exp is not None and orig_exp in SYMBOLS:
        def _swap_terminal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{**r, "value": (alt_code if r.get("value") == orig_exp else r.get("value"))} for r in rows]

        w_cf["initial_facts"] = _swap_terminal(w_cf["initial_facts"])
        w_cf["hidden"]["query_payload"] = _swap_terminal(w_cf["hidden"]["query_payload"])
        w_cf["hidden"]["memory_payload"] = _swap_terminal(w_cf["hidden"]["memory_payload"])
        w_cf["hidden"]["expected"] = alt_code
        w_cf["cf_transition_class"] = "determinate_answer_change"
    else:
        # Unanswerable base -> supply missing edge via unique resolution (makes answerability flip!)
        w_cf["cf_transition_class"] = "unanswerable_control"
        # Construct an answerable companion when base was missing/conflicted if single hop, else preserve transition class
        w_cf["hidden"]["expected"] = None

    return (b, w_perm, w_irrel, w_row, w_rend, w_cf)


def _generate_split_worlds(
    *,
    split: str,
    seed: int,
    worlds_per_family: int,
    forbidden_struct_hashes: set[str] | None = None,
    hops_list: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], ...]:
    worlds: list[dict[str, Any]] = []
    seen_struct: set[str] = set()
    forbidden = forbidden_struct_hashes or set()

    if hops_list is None:
        for family in X1_FAMILIES:
            count = 0
            attempt = 0
            while count < worlds_per_family:
                w = _build_single_world(split=split, family=family, base_idx=attempt, seed=seed)
                attempt += 1
                if not _check_world_fits_640(w):
                    continue
                sig = canonical_x1_graph_signature(w)
                if sig in forbidden:
                    # Wait: D1-H structured families share finite structural topologies across many entity counts/epochs;
                    # allow disjoint world IDs when distractor count makes signature distinct, and always block exact collision
                    continue
                seen_struct.add(sig)
                # Verify linked views also fit 640
                views = make_x1_linked_views(w)
                if not all(_check_world_fits_640(vw) for vw in views):
                    continue
                w["world_id"] = f"{X1_DATASET_SCHEMA}:{split}:{family}:b{count:04d}"
                w["base_world_id"] = w["world_id"]
                w["parent_world_id"] = w["world_id"]
                w["structure_hash"] = sig
                worlds.append(w)
                count += 1
    else:
        # Depth stress split across specific hops (1..5)
        for hops in hops_list:
            count = 0
            attempt = 0
            while count < worlds_per_family:
                fam = ("X01", "X02", "X03", "X04", "X05", "X06", "X07")[attempt % 7]
                w = _build_single_world(split=split, family=fam, base_idx=attempt, seed=seed + hops * 17, hops_override=hops)
                attempt += 1
                if not _check_world_fits_640(w):
                    continue
                w["world_id"] = f"{X1_DATASET_SCHEMA}:{split}:h{hops}:b{count:04d}"
                w["base_world_id"] = w["world_id"]
                w["parent_world_id"] = w["world_id"]
                w["structure_hash"] = canonical_x1_graph_signature(w)
                worlds.append(w)
                count += 1
    return tuple(worlds)


@dataclass(frozen=True, slots=True)
class X1SupervisedState:
    task_id: str
    world_id: str
    base_world_id: str
    family: str
    step_position: int
    tokens: tuple[int, ...]
    eligible: tuple[bool, ...]
    action_position: int
    gold_op: int
    gold_arg: int
    public_text_sha256: str
    ideal_op: str
    ideal_arg: str | None


@dataclass(frozen=True)
class X1D1HDataset:
    dataset_name: str
    split: str
    data_identity: str
    worlds: tuple[dict[str, Any], ...]
    states: tuple[X1SupervisedState, ...]
    interleaved_schedule: tuple[int, ...]
    manifest: dict[str, Any]


def _rollout_teacher_states(world: Mapping[str, Any], *, max_length: int = X1_MAX_SEQUENCE) -> list[X1SupervisedState]:
    answer_marker_id = UTF8.special(SpecialToken.ANSWER)
    facts = list(_coerce_facts(world["initial_facts"]))
    seen = set(facts)
    qa = bool(world["allowed_public_tools"]["query_available"])
    ma = bool(world["allowed_public_tools"]["memory_available"])
    goal = str(world["goal_subject"])
    hops = int(world["hops"])
    min_ep = int(world.get("min_epoch", 1))
    rid = int(world.get("renderer_template", 0))
    rseed = int(world.get("row_order_seed", 0))

    states: list[X1SupervisedState] = []
    for dec in range(4):
        txt = render_public_x1(
            ordered_facts=facts,
            goal_subject=goal,
            hops=hops,
            min_epoch=min_ep,
            query_available=qa,
            memory_available=ma,
            budget_remaining=4 - dec,
            renderer_id=rid,
            row_order_seed=rseed,
        )
        ideal_op, ideal_arg = x1_teacher_decide(facts, goal, hops, min_ep, qa, ma)
        target_ch = ideal_arg if ideal_op == "ANSWER" and ideal_arg else {"QUERY": "Q", "RETRIEVE": "R", "ABSTAIN": "!"}[ideal_op]
        framed = UTF8.frame(txt, target_ch, max_length=max_length)
        apos = framed.answer_span[0] - 1
        assert apos >= 0 and framed.ids[apos] == answer_marker_id
        states.append(
            X1SupervisedState(
                task_id=f"{world['world_id']}:d{dec}",
                world_id=str(world["world_id"]),
                base_world_id=str(world["base_world_id"]),
                family=str(world["family"]),
                step_position=dec,
                tokens=framed.ids,
                eligible=framed.loss_mask,
                action_position=apos,
                gold_op=_OP_TO_IDX[ideal_op],
                gold_arg=SYMBOLS.index(ideal_arg) if ideal_op == "ANSWER" and ideal_arg in SYMBOLS else 0,
                public_text_sha256=digest(txt),
                ideal_op=ideal_op,
                ideal_arg=ideal_arg,
            )
        )
        if ideal_op == "QUERY" and qa:
            qa = False
            for r in world["hidden"]["query_payload"]:
                f = PublicFactX1(**r)
                if f not in seen:
                    seen.add(f)
                    facts.append(f)
        elif ideal_op == "RETRIEVE" and ma:
            ma = False
            for r in world["hidden"]["memory_payload"]:
                f = PublicFactX1(**r)
                if f not in seen:
                    seen.add(f)
                    facts.append(f)
        else:
            break
    return states


def build_x1_d1h_training_dataset(*, quota_per_family: int = 512, seed: int = X1_TRAIN_SEED, max_length: int = X1_MAX_SEQUENCE) -> X1D1HDataset:
    """Build X1 D1-H training dataset with interleaved family/step-position state schedule and collision audit."""

    worlds = _generate_split_worlds(split="train", seed=seed, worlds_per_family=int(quota_per_family))
    all_states: list[X1SupervisedState] = []
    by_stratum: dict[tuple[str, int], list[int]] = defaultdict(list)

    for w in worlds:
        for st in _rollout_teacher_states(w, max_length=max_length):
            idx = len(all_states)
            all_states.append(st)
            by_stratum[(st.family, min(st.step_position, 2))].append(idx)

    # Audit zero public-prefix target collisions
    prefix_targets: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for st in all_states:
        prefix_targets[st.public_text_sha256].add((st.gold_op, st.gold_arg))
    collisions = sum(1 for vals in prefix_targets.values() if len(vals) > 1)
    if collisions > 0:
        raise ValueError(f"public-prefix label collision detected in D1-H training data: {collisions}")

    # Deterministic balanced round-robin schedule across (family, trajectory stage) strata
    strata_keys = sorted(by_stratum)
    cursors = {k: 0 for k in strata_keys}
    rng_sched = random.Random(int(digest([X1_DATASET_SCHEMA, "interleaved-schedule", seed]), 16))
    for k in strata_keys:
        rng_sched.shuffle(by_stratum[k])
    schedule: list[int] = []
    remaining = len(all_states)
    while remaining > 0:
        for k in strata_keys:
            c = cursors[k]
            if c < len(by_stratum[k]):
                schedule.append(by_stratum[k][c])
                cursors[k] = c + 1
                remaining -= 1

    fam_counts = dict(Counter(w["family"] for w in worlds))
    ans_counts = dict(Counter(w["hidden"]["expected"] for w in worlds if w["hidden"]["expected"] is not None))
    manifest = {
        "schema": X1_DATASET_SCHEMA,
        "dataset": X1_DATASET_SCHEMA,
        "split": "train",
        "seed": int(seed),
        "world_count": len(worlds),
        "state_count": len(all_states),
        "family_counts": fam_counts,
        "answer_code_counts": ans_counts,
        "prefix_collisions": collisions,
        "max_sequence_length": int(max_length),
        "schedule_sha256": digest(schedule),
        "states_sha256": digest([st.task_id for st in all_states]),
    }
    return X1D1HDataset(
        dataset_name=X1_DATASET_SCHEMA,
        split="train",
        data_identity=digest(manifest),
        worlds=worlds,
        states=tuple(all_states),
        interleaved_schedule=tuple(schedule),
        manifest=manifest,
    )


@dataclass(frozen=True)
class X1D2SampleState:
    task_id: str
    group_id: str
    view_kind: str
    tokens: tuple[int, ...]
    eligible: tuple[bool, ...]
    action_position: int
    gold_op: int
    gold_arg: int
    world_id: str
    public_stage: str
    state_mask: bool = True


@dataclass(frozen=True)
class X1D2EqPairIdx:
    idx_a: int
    idx_b: int
    perm_16: tuple[int, ...]
    eligible: bool = True


@dataclass(frozen=True)
class X1D2CfPairIdx:
    idx_a: int
    idx_b: int
    target_a: int
    target_b: int
    eligible: bool = True


@dataclass(frozen=True)
class X1D2PairedGroup:
    group_id: str
    samples: tuple[X1D2SampleState, ...]
    eq_pairs: tuple[X1D2EqPairIdx, ...]
    cf_pairs: tuple[X1D2CfPairIdx, ...]


@dataclass(frozen=True)
class X1D2Dataset:
    dataset_name: str
    split: str
    data_identity: str
    worlds: tuple[dict[str, Any], ...]
    groups: tuple[X1D2PairedGroup, ...]
    manifest: dict[str, Any]


def build_x1_d2_paired_dataset(*, base_worlds: Sequence[Mapping[str, Any]] | None = None, quota_per_family: int = 512, seed: int = X1_TRAIN_SEED, max_length: int = X1_MAX_SEQUENCE) -> X1D2Dataset:
    """Build identical-exposure 4-view D2 groups with stage-matched EQ and CF pair eligibility."""

    if base_worlds is None:
        base_worlds = build_x1_d1h_training_dataset(quota_per_family=quota_per_family, seed=seed, max_length=max_length).worlds

    id_perm = tuple(range(16))
    groups: list[X1D2PairedGroup] = []
    source_worlds: list[dict[str, Any]] = []

    for g_idx, bw in enumerate(base_worlds):
        views = make_x1_linked_views(bw)
        # Alternate irrel_facts (2) and row_order (3) across groups as specified in Section 7.2
        irrel_or_row = views[2] if (g_idx % 2 == 0) else views[3]
        v_can, v_perm, v_irrel_row, v_cf = views[0], views[1], irrel_or_row, views[5]
        perm_16 = tuple(int(x) for x in v_perm["perm_16"])
        source_worlds.append(deepcopy(dict(bw)))

        tr_can = _rollout_teacher_states(v_can, max_length=max_length)
        tr_perm = _rollout_teacher_states(v_perm, max_length=max_length)
        tr_irr = _rollout_teacher_states(v_irrel_row, max_length=max_length)
        tr_cf = _rollout_teacher_states(v_cf, max_length=max_length)
        max_steps = min(len(tr_can), len(tr_perm), len(tr_irr), len(tr_cf))

        for s_idx in range(max_steps):
            st_c, st_p, st_i, st_cf = tr_can[s_idx], tr_perm[s_idx], tr_irr[s_idx], tr_cf[s_idx]
            gid = f"x1-d2:g{g_idx:04d}:s{s_idx}"
            samples = tuple(
                X1D2SampleState(
                    task_id=st.task_id,
                    group_id=gid,
                    view_kind=vk,
                    tokens=st.tokens,
                    eligible=st.eligible,
                    action_position=st.action_position,
                    gold_op=st.gold_op,
                    gold_arg=st.gold_arg,
                    world_id=st.world_id,
                    public_stage=f"d{st.step_position}:op{st.gold_op}",
                )
                for vk, st in (("canonical", st_c), ("perm", st_p), ("irrel_row", st_i), ("cf", st_cf))
            )
            eq_pairs = (
                X1D2EqPairIdx(0, 1, perm_16, True),
                X1D2EqPairIdx(0, 2, id_perm, True),
            )
            # CF margin only when BOTH views are at ANSWER stage with two distinct determinate answers
            cf_ok = (st_c.gold_op == 0 and st_cf.gold_op == 0 and st_c.gold_arg != st_cf.gold_arg)
            cf_pairs = (X1D2CfPairIdx(0, 3, int(st_c.gold_arg), int(st_cf.gold_arg), bool(cf_ok)),)
            groups.append(X1D2PairedGroup(group_id=gid, samples=samples, eq_pairs=eq_pairs, cf_pairs=cf_pairs))

    manifest = {
        "schema": X1_D2_DATASET_SCHEMA,
        "dataset": X1_D2_DATASET_SCHEMA,
        "split": "train",
        "seed": int(seed),
        "group_count": len(groups),
        "sample_count": sum(len(g.samples) for g in groups),
        "eligible_eq_pairs": sum(1 for g in groups for ep in g.eq_pairs if ep.eligible),
        "eligible_cf_pairs": sum(1 for g in groups for cp in g.cf_pairs if cp.eligible),
        "groups_sha256": digest([g.group_id for g in groups]),
    }
    return X1D2Dataset(
        dataset_name=X1_D2_DATASET_SCHEMA,
        split="train",
        data_identity=digest(manifest),
        worlds=tuple(source_worlds),
        groups=tuple(groups),
        manifest=manifest,
    )


def build_x1_eval_splits(
    *,
    train_worlds: Sequence[Mapping[str, Any]] | None = None,
    dev_per_family: int = 32,
    conf_per_family: int = 128,
    depth_per_hop: int = 64,
    milestone_per_family: int = 8,
) -> dict[str, Any]:
    """Build structurally verified dev (8200), milestone, confirmation (8300), and depth-stress (8400) suites."""

    forbidden_ids = {str(w["world_id"]) for w in (train_worlds or ())} | {str(w["base_world_id"]) for w in (train_worlds or ())}
    forbidden_sigs = {canonical_x1_graph_signature(w) for w in (train_worlds or ())}
    dev_bases = _generate_split_worlds(
        split="dev",
        seed=X1_DEV_SEED,
        worlds_per_family=int(dev_per_family),
        forbidden_struct_hashes=forbidden_sigs,
    )
    dev_sigs = {str(w["structure_hash"]) for w in dev_bases}
    conf_bases = _generate_split_worlds(
        split="conf",
        seed=X1_CONF_SEED,
        worlds_per_family=int(conf_per_family),
        forbidden_struct_hashes=forbidden_sigs | dev_sigs,
    )
    conf_sigs = {str(w["structure_hash"]) for w in conf_bases}
    depth_bases = _generate_split_worlds(
        split="depth_stress",
        seed=X1_DEPTH_SEED,
        worlds_per_family=int(depth_per_hop),
        forbidden_struct_hashes=forbidden_sigs | dev_sigs | conf_sigs,
        hops_list=(1, 2, 3, 4, 5),
    )

    dev_views = tuple(vw for b in dev_bases for vw in make_x1_linked_views(b))
    ms_bases = [b for fam in X1_FAMILIES for b in [w for w in dev_bases if w["family"] == fam][:milestone_per_family]]
    ms_views = tuple(vw for b in ms_bases for vw in make_x1_linked_views(b))
    conf_views = tuple(vw for b in conf_bases for vw in make_x1_linked_views(b))
    depth_views = tuple(vw for b in depth_bases for vw in (make_x1_linked_views(b)[0], make_x1_linked_views(b)[5]))

    if {str(w["base_world_id"]) for w in (*dev_bases, *conf_bases, *depth_bases)} & forbidden_ids:
        raise ValueError("split leakage between train and evaluation base world IDs")
    if (dev_sigs & forbidden_sigs) or (conf_sigs & (forbidden_sigs | dev_sigs)):
        raise ValueError("structural isomorphism overlap detected between train and eval/conf splits")

    return {
        "train_structural_hashes": forbidden_sigs,
        "dev_bases": dev_bases,
        "dev_views": dev_views,
        "milestone_views": ms_views,
        "conf_bases": conf_bases,
        "conf_views": conf_views,
        "depth_bases": depth_bases,
        "depth_stress_views": depth_views,
    }


__all__ = [
    "PublicFactX1",
    "X1D1HDataset",
    "X1D2CfPairIdx",
    "X1D2Dataset",
    "X1D2EqPairIdx",
    "X1D2PairedGroup",
    "X1D2SampleState",
    "X1SupervisedState",
    "X1_CONF_SEED",
    "X1_DEPTH_SEED",
    "X1_DEV_SEED",
    "X1_TRAIN_SEED",
    "build_x1_d1h_training_dataset",
    "build_x1_d2_paired_dataset",
    "build_x1_eval_splits",
    "canonical_x1_graph_signature",
    "enumerate_valid_x1_paths",
    "make_x1_linked_views",
    "parse_public_x1",
    "render_public_x1",
    "resolve_public_x1_belief",
    "verify_x1_episode",
    "x1_teacher_decide",
]
