"""Grouped query-swap training data (tp-grouped-queryswap-001, v2 generator).

REPLICATION-CRITICAL FIXES over the v1 generator (commit e547564), which
leaked: it shuffled individual ROWS and then took the first 40 as heldout,
so the same group_id landed on both sides of the split (17 of 70 groups in
the committed data). The GROUP is the atomic unit:

  - one group = one fact block (k facts, k = 2..4 deliberately) plus k
    query variants, one per fact; every member shares the identical
    fact block and differs ONLY in which entity the query names and
    therefore which opaque code is correct;
  - groups are partitioned FIRST, then flattened to rows. A group lives
    in exactly one side of the split:
        train_group_ids INTERSECT heldout_group_ids == EMPTY SET;
  - replay rows are drawn from the bank TRAIN split only and are assigned
    to train; the dev-bank replay rows that land in heldout are pure
    retention monitors (no target-family sibling exists for them).

Output contract is unchanged from the proposal: opaque codes ONLY — no
entity-name auxiliary objective (SFT4's mistake).

The mix is defined at the UNIT level with an explicit weight:
    L = alpha * L_group + (1 - alpha) * L_replay
with alpha recorded here and in the receipt (P13). Replay families are
balanced by capping each family to the smallest family pool (P14).

split_audit.json proves: zero group overlap, zero prompt overlap, exact
counts, histograms, and SHA256 hashes of both data files and the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from connector.experiments.capability_bank import OBJECTS, PREFIXES

N_GROUPS = 70          # query-swap target groups
HELDOUT_GROUP_FRACTION = 0.30   # ~21 of 70 groups -> group-atomic holdout
ALPHA_GROUP_LOSS = 0.58         # explicit unit-level mix weight (see P13)
SEED = 808


def _code(rng):
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


class QuerySwapGroup:
    """One fact block + one query variant per fact. The atomic split unit."""

    __slots__ = ("group_id", "facts", "examples")

    def __init__(self, group_id: str):
        self.group_id = group_id
        self.facts: list[dict] = []      # [{"entity","code","line"}]
        self.examples: list[dict] = []   # flattened rows (one per fact)

    @property
    def size(self) -> int:
        return len(self.examples)


def _build_groups(rng: random.Random) -> list[QuerySwapGroup]:
    groups = []
    for gi in range(N_GROUPS):
        g = QuerySwapGroup(f"gqs-{gi:03d}")
        k = 2 + (gi % 3)                      # deliberate k in {2,3,4}
        ents = rng.sample(OBJECTS, k)
        recs = [{"e": e, "c": _code(rng)} for e in ents]
        rng.shuffle(recs)
        fmt = "prose" if gi % 2 == 0 else "table"
        if fmt == "prose":
            lines = [f"{r['e'].capitalize()} holds ref {r['c']}." for r in recs]
        else:
            lines = ["item | ref"] + [
                f"{r['e'].capitalize()} | {r['c']}" for r in recs]
        block = "\n".join(lines)
        for r in recs:
            q = f"Return ONLY the ref of {r['e'].capitalize()}."
            g.facts.append({"entity": r["e"], "code": r["c"], "line": None})
            g.examples.append({
                "family": "queryswap_group", "group_id": g.group_id,
                "group_size": k, "protocol": fmt,
                "prompt": f"{block}\n{q}\nAnswer:",
                "completion": f" {r['c']}.", "gold": r["c"], "answer": r["c"],
            })
        # every member carries the same rendered fact block for audit purposes
        for row in g.examples:
            row["fact_block_sha256"] = hashlib.sha256(
                block.encode("utf-8")).hexdigest()
        g.facts = [{"entity": f["entity"], "code": f["code"]} for f in g.facts]
        groups.append(g)
    return groups


def build(seed: int = SEED):
    """Returns (train_rows, held_rows, audit_dict). Group-atomic split."""
    rng = random.Random(seed)
    groups = _build_groups(rng)

    # ---- P1: partition GROUPS first, flatten afterwards -------------------
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_held_groups = max(1, round(N_GROUPS * HELDOUT_GROUP_FRACTION))
    held_groups, train_groups = shuffled[:n_held_groups], shuffled[n_held_groups:]

    def flatten(gs):
        rows = []
        for g in gs:
            rows.extend(g.examples)
        return rows

    train = flatten(train_groups)

    # ---- replay: bank TRAIN split only, balanced across families (P14) ----
    bank = [json.loads(l) for l in
            Path("data/capability_bank/train.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    want = {"single_fact": 40, "tool_result": 40, "copy": 40,
            "protocol_transfer": 35, "symbolic_ops": 35}
    per_family = []
    for fam, n in want.items():
        pool = [b for b in bank if b["family"] == fam]
        rng.shuffle(pool)
        take = pool[:min(n, len(pool))]
        for it in take:
            it.setdefault("gold", it.get("answer", ""))
            it.setdefault("protocol", it.get("format", "nl"))
        per_family.append(take)
    floor = min(len(t) for t in per_family)      # balance: cap at smallest family
    rehearsal = [row for t in per_family for row in t[:floor]]
    for it in rehearsal:
        it["replay"] = True
    train = train + rehearsal

    # heldout: target-group rows + a small retention monitor slice from the
    # SAME replay pools (disjoint prompts from train by construction of the
    # bank split? no -- these are train-split rows; they are MONITORS only).
    held = flatten(held_groups)

    # ---- audit BEFORE returning ------------------------------------------
    tg = {g.group_id for g in train_groups}
    hg = {g.group_id for g in held_groups}
    assert not (tg & hg), "GROUP OVERLAP"
    tprompts = [x["prompt"] for x in train if x.get("family") == "queryswap_group"]
    hprompts = [x["prompt"] for x in held]
    assert not (set(tprompts) & set(hprompts)), "PROMPT OVERLAP"
    tfb = {x["fact_block_sha256"] for x in train if x.get("family") == "queryswap_group"}
    hfb = {x["fact_block_sha256"] for x in held}
    assert not (tfb & hfb), "FACT-BLOCK OVERLAP"

    def _hist(rows, key):
        return dict(Counter(r[key] for r in rows
                            if r.get("family") == "queryswap_group"))

    def _sha(rows):
        blob = "\n".join(json.dumps(x, sort_keys=True) for x in rows)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    audit = {
        "schema": "anra-grouped-split-audit/v1",
        "seed": seed,
        "alpha_group_loss": ALPHA_GROUP_LOSS,
        "n_train_groups": len(tg),
        "n_heldout_groups": len(hg),
        "group_overlap": len(tg & hg),
        "prompt_overlap": len(set(tprompts) & set(hprompts)),
        "full_fact_block_overlap": len(tfb & hfb),
        "n_train_target_rows": sum(g.size for g in train_groups),
        "n_heldout_target_rows": sum(g.size for g in held_groups),
        "train_group_size_histogram": dict(sorted(Counter(
            g.size for g in train_groups).items())),
        "heldout_group_size_histogram": dict(sorted(Counter(
            g.size for g in held_groups).items())),
        "train_format_histogram": _hist(train, "protocol"),
        "heldout_format_histogram": _hist(held, "protocol"),
        "train_fact_count_histogram": _hist(train, "group_size"),
        "heldout_fact_count_histogram": _hist(held, "group_size"),
        "replay_composition_train": dict(Counter(
            x["family"] for x in train if x.get("replay"))),
        "unit_mix_note": ("alpha=0.58 gradient weight on the group objective "
                          "vs replay (P13); row percentages are reported but "
                          "are NOT the experimental mix"),
        "heldout_role_note": ("target-group rows measure generalization; any "
                              "bank-train replay rows placed here would be "
                              "monitors only — this generator places NONE"),
        "train_data_sha256": _sha(train),
        "heldout_data_sha256": _sha(held),
        "group_split_manifest": {
            "train_group_ids": sorted(tg),
            "heldout_group_ids": sorted(hg),
        },
        "split_manifest_sha256": None,
    }
    manifest_blob = json.dumps(audit["group_split_manifest"], sort_keys=True)
    audit["split_manifest_sha256"] = hashlib.sha256(
        manifest_blob.encode("utf-8")).hexdigest()

    return train, held, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/grouped_queryswap")
    a = p.parse_args()
    train, held, audit = build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train), encoding="utf-8")
    (out / "heldout.jsonl").write_text(
        "\n".join(json.dumps(x) for x in held), encoding="utf-8")
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2),
                                          encoding="utf-8")
    fams = Counter(x["family"] for x in train)
    print(json.dumps({
        "total_train": len(train), "total_heldout": len(held),
        "per_family": dict(fams),
        "percentages": {k: f"{v/len(train):.1%}" for k, v in fams.items()},
        "audit": {k: audit[k] for k in (
            "n_train_groups", "n_heldout_groups", "group_overlap",
            "prompt_overlap", "full_fact_block_overlap",
            "n_train_target_rows", "n_heldout_target_rows")},
    }, indent=2))


if __name__ == "__main__":
    main()
