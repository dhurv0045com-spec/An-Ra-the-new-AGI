"""ARK-014 frozen task contract: order-robust non-arithmetic binding.

Implements the data layer preregistered in experiments/ARK-014/PLAN.md:

- six keys 0..5, six values 0..5, three key=value facts per fact-set;
- fact-set split BEFORE query or order expansion (task seed 4242);
- 400 train fact-sets, 100 held-out fact-sets split by a deterministic
  signature-hash rule into 50 BIND_CONTROL and 50 BIND_SEALED fact-sets;
- four orthogonal diagnostics per held-out split (CANONICAL, ORDER_ONLY,
  QUERY_ONLY, QUERY_ORDER), each enumerating all three queries;
- deterministic order augmentation as a pure function of
  (acquisition_seed, optimizer_step, batch_position, semantic_example_id).

The fact-set universe and the train/held-out membership reproduce the
historical ARK-009 construction exactly; the fixed prior hashes are asserted
at build time. Prompt rendering and symbolic answer computation are declared
fixed priors of the experiment; no external tool answers for the model.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter

TASK_SEED = 4242
TRAIN_FACTSETS = 400
HELDOUT_FACTSETS = 100
CONTROL_FACTSETS = 50
SEALED_FACTSETS = 50
QUERIES_PER_FACTSET = 3
KEYS = range(6)
VALUES = range(6)

# Historical ARK-009 anchor hashes (fixed priors; membership must reproduce).
ARK009_TASK_SEED = 4242
ARK009_TRAIN_FACTSET_SHA256 = "4b4e2e2e6e32448ef70841562e550122232e08a72ec579dc916678d312fb53b4"
ARK009_TEST_FACTSET_SHA256 = "f31bd6586232f77e43cc91e6351a5eada8a654c1fa91e9bb8559270c5a98287a"
ARK009_MANIFEST_SHA256 = "a3625a12996c08495e8a3d416abf1243193cf7bb9d9ac9d10f9c5fdd51ab4cdc"

DIAGNOSTICS = ("CANONICAL", "ORDER_ONLY", "QUERY_ONLY", "QUERY_ORDER")

# Qualification thresholds frozen in experiments/ARK-014/PLAN.md.
QUALIFICATION_THRESHOLDS = {"CANONICAL": 0.90, "ORDER_ONLY": 0.85, "QUERY_ORDER": 0.85}
QUALIFICATION_CONSECUTIVE = 3


def sha_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def fact_signature(facts) -> tuple:
    """Order-independent semantic identity of a fact-set (fixed prior)."""
    return tuple(sorted((int(k), int(v)) for k, v in facts))


def signature_json(facts) -> list:
    return [[int(k), int(v)] for k, v in fact_signature(facts)]


def render_prompt(facts, query: int) -> str:
    """Frozen prompt format shared with the ARK-009 implementation."""
    return "+".join(f"{int(k)}={int(v)}" for k, v in facts) + f"/{int(query)}="


def symbolic_answer(facts, query: int) -> str:
    """Declared fixed prior: the symbolic scoring key for one fact-set."""
    return str(dict((int(k), int(v)) for k, v in facts)[int(query)])


def enumerate_factsets() -> list:
    """All 20 x 120 = 2400 fact-sets in the frozen construction order."""
    factsets = []
    for key_tuple in itertools.combinations(KEYS, 3):
        for val_tuple in itertools.permutations(VALUES, 3):
            factsets.append(tuple(zip(key_tuple, val_tuple)))
    return factsets


def split_control_sealed(heldout_factsets: list) -> tuple[list, list, dict]:
    """Deterministic signature-hash split, applied before any expansion.

    Rule (frozen here before execution): rank held-out fact-sets by
    sha256 over the canonical signature JSON, tie-broken by the signature
    itself; even ranks are BIND_CONTROL, odd ranks BIND_SEALED.
    """
    ordered = sorted(heldout_factsets, key=lambda f: (hashlib.sha256(
        json.dumps(signature_json(f), separators=(",", ":")).encode("utf-8")
    ).hexdigest(), fact_signature(f)))
    control = [f for i, f in enumerate(ordered) if i % 2 == 0]
    sealed = [f for i, f in enumerate(ordered) if i % 2 == 1]
    rule = {
        "algorithm": ("sort held-out fact-sets by sha256(json(canonical signature)), "
                      "tie-break by signature; even rank BIND_CONTROL, odd rank BIND_SEALED"),
        "applied_before_expansion": True,
        "control_sha256": sha_json([signature_json(f) for f in control]),
        "sealed_sha256": sha_json([signature_json(f) for f in sealed]),
        "assignment_sha256": sha_json({
            "control": [signature_json(f) for f in control],
            "sealed": [signature_json(f) for f in sealed],
        }),
    }
    return control, sealed, rule


def _diagnostic_rows(facts, diagnostic: str) -> list[dict]:
    """All three query rows of one diagnostic for one held-out fact-set.

    Semantic invariant enforced by the caller: every row's answer equals the
    symbolic value bound to its query key in the fact-set mapping.
    """
    mapping = dict((int(k), int(v)) for k, v in facts)
    keys_here = [int(k) for k, _ in facts]
    rows = []
    for pos, q in enumerate(keys_here):
        if diagnostic == "CANONICAL":
            shown, asked = facts, q
        elif diagnostic == "ORDER_ONLY":
            shown, asked = tuple(reversed(facts)), q
        elif diagnostic == "QUERY_ONLY":
            shown = facts
            asked = keys_here[(pos + 1) % len(keys_here)]
        elif diagnostic == "QUERY_ORDER":
            # Composite style inherited verbatim from the ARK-009 swap
            # construction: reverse the fact order, then advance the query
            # cyclically in the reversed key order relative to the canonical
            # query being reproduced.
            shown = tuple(reversed(facts))
            rev_keys = [int(k) for k, _ in shown]
            old_pos = rev_keys.index(int(q))
            asked = rev_keys[(old_pos + 1) % len(rev_keys)]
        else:
            raise ValueError(f"unknown diagnostic: {diagnostic}")
        rows.append({
            "diagnostic": diagnostic,
            "facts": [[int(k), int(v)] for k, v in shown],
            "query": int(asked),
            "prompt": render_prompt(shown, asked),
            "answer": str(mapping[int(asked)]),
        })
    return rows


def build_binding_task() -> dict:
    """Build the complete frozen ARK-014 task package and its manifest."""
    factsets = enumerate_factsets()
    if len(factsets) != 2400:
        raise RuntimeError("fact-set universe drift")
    rng = random.Random(TASK_SEED)
    rng.shuffle(factsets)
    train_factsets = factsets[:TRAIN_FACTSETS]
    heldout_factsets = factsets[TRAIN_FACTSETS:TRAIN_FACTSETS + HELDOUT_FACTSETS]
    if len(train_factsets) != TRAIN_FACTSETS or len(heldout_factsets) != HELDOUT_FACTSETS:
        raise RuntimeError("split count drift")

    train_sigs = {fact_signature(f) for f in train_factsets}
    heldout_sigs = {fact_signature(f) for f in heldout_factsets}
    if train_sigs & heldout_sigs:
        raise RuntimeError("binding fact-set leakage between train and held-out")

    # Historical anchor: membership must reproduce the ARK-009 construction.
    if sha_json(sorted(list(train_sigs))) != ARK009_TRAIN_FACTSET_SHA256:
        raise RuntimeError("train fact-set membership drift vs ARK-009 anchor")
    if sha_json(sorted(list(heldout_sigs))) != ARK009_TEST_FACTSET_SHA256:
        raise RuntimeError("held-out fact-set membership drift vs ARK-009 anchor")

    control_factsets, sealed_factsets, split_rule = split_control_sealed(heldout_factsets)
    control_sigs = {fact_signature(f) for f in control_factsets}
    sealed_sigs = {fact_signature(f) for f in sealed_factsets}
    if control_sigs & sealed_sigs:
        raise RuntimeError("CONTROL/SEALED fact-set overlap")
    if control_sigs | sealed_sigs != heldout_sigs:
        raise RuntimeError("CONTROL/SEALED union does not reproduce held-out set")
    if control_sigs & train_sigs or sealed_sigs & train_sigs:
        raise RuntimeError("fact-set crossed a split boundary")
    if len(control_factsets) != CONTROL_FACTSETS or len(sealed_factsets) != SEALED_FACTSETS:
        raise RuntimeError("CONTROL/SEALED count drift")

    def expand_train(groups):
        rows = []
        for facts in groups:
            mapping = dict((int(k), int(v)) for k, v in facts)
            for k, _ in facts:
                rows.append((render_prompt(facts, int(k)), str(mapping[int(k)])))
        return rows

    train_rows = expand_train(train_factsets)
    if len(train_rows) != TRAIN_FACTSETS * QUERIES_PER_FACTSET:
        raise RuntimeError("train expansion drift")

    diagnostics: dict[str, dict[str, list]] = {}
    for split_name, groups in (("BIND_CONTROL", control_factsets), ("BIND_SEALED", sealed_factsets)):
        for diagnostic in DIAGNOSTICS:
            rows = []
            for facts in groups:
                rows.extend(_diagnostic_rows(facts, diagnostic))
            diagnostics.setdefault(diagnostic, {})[split_name] = rows
            if len(rows) != CONTROL_FACTSETS * QUERIES_PER_FACTSET:
                raise RuntimeError(f"{diagnostic}/{split_name} row count drift")
            for row in rows:
                if row["answer"] != symbolic_answer(row["facts"], row["query"]):
                    raise RuntimeError(f"{diagnostic}/{split_name} answer semantics violated")
                if row["prompt"] != render_prompt(row["facts"], row["query"]):
                    raise RuntimeError(f"{diagnostic}/{split_name} prompt rendering drift")

    # Orthogonality audit: which variable each diagnostic changes relative to
    # CANONICAL, verified structurally on the built rows.
    _assert_orthogonal(diagnostics)

    row_hashes = {
        diagnostic: {
            split: sha_json(rows) for split, rows in splits.items()
        }
        for diagnostic, splits in diagnostics.items()
    }

    manifest = {
        "schema": "arkenstone-ark014-binding/v1",
        "task_seed": TASK_SEED,
        "semantics": ("six keys 0..5, six values 0..5, three key=value facts per "
                      "fact-set, one queried value per example"),
        "keys": list(KEYS),
        "values": list(VALUES),
        "facts_per_example": 3,
        "counts": {
            "train_factsets": len(train_factsets),
            "heldout_factsets": len(heldout_factsets),
            "control_factsets": len(control_factsets),
            "sealed_factsets": len(sealed_factsets),
            "train_examples": len(train_rows),
            "rows_per_diagnostic_per_split": CONTROL_FACTSETS * QUERIES_PER_FACTSET,
        },
        "split_rule": split_rule,
        "factset_overlap": 0,
        "train_factset_sha256": sha_json(sorted([signature_json(f) for f in train_factsets])),
        "heldout_factset_sha256": sha_json(sorted([signature_json(f) for f in heldout_factsets])),
        "historical_ark009_anchor": {
            "train_factset_sha256": ARK009_TRAIN_FACTSET_SHA256,
            "test_factset_sha256": ARK009_TEST_FACTSET_SHA256,
            "manifest_sha256": ARK009_MANIFEST_SHA256,
            "membership_reproduced": True,
        },
        "control_factsets": [signature_json(f) for f in control_factsets],
        "sealed_factsets": [signature_json(f) for f in sealed_factsets],
        "diagnostic_row_sha256": row_hashes,
        "train_query_counts": dict(Counter(int(p.rsplit("/", 1)[1].rstrip("=")) for p, _ in train_rows)),
        "train_answer_counts": dict(Counter(int(a) for _, a in train_rows)),
        "train_rows": [list(x) for x in train_rows],
        "diagnostic_rows": diagnostics,
        "order_sha256": {
            "CANONICAL_vs_ORDER_ONLY_same_query_same_answer": _order_audit(diagnostics, "ORDER_ONLY"),
            "QUERY_ONLY_stored_order": _query_only_audit(diagnostics),
        },
        "qualification_thresholds": QUALIFICATION_THRESHOLDS,
        "qualification_consecutive_evals": QUALIFICATION_CONSECUTIVE,
    }
    manifest["manifest_sha256"] = sha_json(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    return {
        "train_rows": train_rows,
        "train_factsets": train_factsets,
        "control_factsets": control_factsets,
        "sealed_factsets": sealed_factsets,
        "diagnostics": diagnostics,
        "manifest": manifest,
    }


def _order_audit(diagnostics, name: str) -> bool:
    """ORDER_ONLY must keep each canonical (query, answer) pair identical."""
    for split in ("BIND_CONTROL", "BIND_SEALED"):
        canon = {(r["query"], r["answer"]) for r in diagnostics["CANONICAL"][split]}
        other = {(r["query"], r["answer"]) for r in diagnostics[name][split]}
        if canon != other:
            return False
    return True


def _query_only_audit(diagnostics) -> bool:
    """QUERY_ONLY shares CANONICAL's fact order and answer universe (redundant)."""
    for split in ("BIND_CONTROL", "BIND_SEALED"):
        canon_pairs = {(tuple(map(tuple, r["facts"]))) for r in diagnostics["CANONICAL"][split]}
        other_pairs = {(tuple(map(tuple, r["facts"]))) for r in diagnostics["QUERY_ONLY"][split]}
        if canon_pairs != other_pairs:
            return False
    return True


def _assert_orthogonal(diagnostics) -> None:
    for split in ("BIND_CONTROL", "BIND_SEALED"):
        canon = diagnostics["CANONICAL"][split]
        order = diagnostics["ORDER_ONLY"][split]
        qonly = diagnostics["QUERY_ONLY"][split]
        qorder = diagnostics["QUERY_ORDER"][split]
        for a, b in zip(canon, order):
            if a["query"] != b["query"] or a["answer"] != b["answer"]:
                raise RuntimeError("ORDER_ONLY changed query or answer")
            if list(map(tuple, a["facts"])) == list(map(tuple, b["facts"])):
                raise RuntimeError("ORDER_ONLY did not change fact order")
        for a, b in zip(canon, qonly):
            if list(map(tuple, a["facts"])) != list(map(tuple, b["facts"])):
                raise RuntimeError("QUERY_ONLY changed fact order")
            if a["answer"] == b["answer"] and a["query"] == b["query"]:
                raise RuntimeError("QUERY_ONLY did not change the query")
        for a, b in zip(canon, qorder):
            if list(map(tuple, a["facts"])) == list(map(tuple, b["facts"])):
                raise RuntimeError("QUERY_ORDER did not change fact order")
            if a["query"] == b["query"]:
                raise RuntimeError("QUERY_ORDER did not change the query")


def _diagnostic_rows_grouped(task, split: str, diagnostic: str, facts=None) -> list[tuple[str, str]]:
    """(prompt, answer) pairs for one diagnostic of one split — the whole split
    or a single fact-set, in the same canonical row order as the manifest."""
    if facts is None:
        rows = task["diagnostics"][diagnostic][split]
    else:
        rows = _diagnostic_rows(tuple((int(k), int(v)) for k, v in facts), diagnostic)
    return [(row["prompt"], row["answer"]) for row in rows]


# ------------------------------------------------------------- augmentation

_MASK64 = (1 << 64) - 1
_PERM6 = tuple(itertools.permutations(range(3)))


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def augmentation_permutation_index(acq_seed: int, optimizer_step: int,
                                   batch_position: int, example_id: int) -> int:
    """Pure hash into the six permutations of three facts.

    Fixed function (frozen before execution): fold the four integer inputs
    through splitmix64 and reduce modulo 6. No RNG state is consumed, so the
    augmentation cannot perturb the matched torch streams.
    """
    h = 0
    for value in (int(acq_seed), int(optimizer_step), int(batch_position), int(example_id)):
        h = _splitmix64(h ^ (value & _MASK64))
    return h % len(_PERM6)


def augment_facts_with_index(facts, acq_seed: int, optimizer_step: int,
                             batch_position: int, example_id: int) -> tuple[tuple, int]:
    """Deterministic fact-order augmentation; returns (permuted facts, index).

    Single computation of the permutation index so callers that both apply and
    record the augmentation cannot observe two different derivations.
    """
    facts = tuple(facts)
    if len(facts) != 3:
        raise ValueError("augmentation is defined for three-fact fact-sets")
    index = augmentation_permutation_index(acq_seed, optimizer_step, batch_position, example_id)
    perm = _PERM6[index]
    return tuple(facts[perm[i]] for i in range(3)), index


def augment_facts(facts, acq_seed: int, optimizer_step: int,
                  batch_position: int, example_id: int) -> tuple:
    """Deterministic fact-order augmentation preserving query/answer semantics."""
    return augment_facts_with_index(facts, acq_seed, optimizer_step,
                                    batch_position, example_id)[0]


AUGMENTATION_SPEC = {
    "name": "splitmix64-fold-mod6",
    "inputs": ["acquisition_seed", "optimizer_step", "batch_position", "example_id"],
    "permutations_of_three": [list(p) for p in _PERM6],
    # Code identity is carried by the runner receipt's source_sha256 for this
    # module; no bytecode-derived hash is used (it would drift with the
    # interpreter, not with semantics).
    "answer_semantics": "reorders fact presentation only; query and answer unchanged",
}
