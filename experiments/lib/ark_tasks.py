"""Frozen task generators + canonical dataset manifests for Arkenstone arms.

Variance discipline (preregistered): the DATASET (membership + split) is frozen
once per task and hashed; training runs vary ONLY init_seed and order_seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def t1_pool() -> list[tuple[str, str]]:
    return [(f"{a} + {b} = ", f"{a + b}") for a in range(10) for b in range(10)]


def t2_rows(split: str, n: int, dataset_seed: int = 13) -> list[tuple[str, str]]:
    """Two-digit no-carry add, structural tens-band split (frozen dataset_seed)."""
    import random

    rng = random.Random(dataset_seed)  # dataset membership frozen, NOT the run seed
    tens_a = range(1, 6) if split == "train" else range(6, 8)
    rows, seen = [], set()
    guard = 0
    while len(rows) < n and guard < 2_000_000:
        guard += 1
        ta = rng.choice(list(tens_a))
        ua = rng.randrange(0, 10)
        tb = rng.randrange(1, 10 - ta)
        ub = rng.randrange(0, 10 - ua)
        a, b = ta * 10 + ua, tb * 10 + ub
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((f"{a} + {b} = ", f"{a + b}"))
    if len(rows) < n:
        raise ValueError(f"t2 {split} space exhausted: {len(rows)} < {n}")
    return rows


def build_task_manifest() -> dict:
    """Canonical T2 dataset manifest: membership frozen once, hashed, audited.

    ARK-002B split discipline: test pairs whose SORTED operand pair appears in
    training are EXCLUDED (commutation leakage: "34+62" trained must not be
    testable as "62+34"). ARK-001/002a's split contained 48 such commuted
    overlaps; see experiments/ARK-002/ERRATUM_002a.json.
    """

    train = t2_rows("train", 500)
    raw_test = t2_rows("test", 260)
    train_pairs = {tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0])))) for p, _ in train}
    test = []
    excluded = 0
    for prompt, answer in raw_test:
        a = int(prompt.split("+")[0])
        b = int(prompt.split("+")[1].split("=")[0])
        if tuple(sorted((a, b))) in train_pairs:
            excluded += 1
            continue
        test.append((prompt, answer))
        if len(test) == 200:
            break
    test_pairs = {tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0])))) for p, _ in test}
    overlap = train_pairs & test_pairs
    assert len(overlap) == 0, "commutation leakage survived the filter"
    ones_train = {(int(p[0]) % 10, int(p[1].split("=")[0].strip()) % 10) for p, _ in train}
    ones_test = {(int(p[0]) % 10, int(p[1].split("=")[0].strip()) % 10) for p, _ in test}
    manifest = {
        "schema": "arkenstone-task-manifest/v1",
        "task": "t2-no-carry-add",
        "semantics": {
            "operands": "a in [10..59], b in [10..89]; tens(a) in 1..5; no carry in either column",
            "train_tens_band": [1, 5],
            "test_tens_band": [6, 7],
            "band_overlap": 0,
            "ones_pair_overlap_note": "ones (ua, ub) pairs with ua+ub<=9 overlap between bands by construction; the OOD signal is the tens-column binding on unseen tens pairs",
        },
        "counts": {"train": len(train), "test": len(test)},
        "raw_test_drawn": len(raw_test),
        "commutation_excluded": excluded,
        "pair_overlap_train_test": len(overlap),
        "ones_pair_coverage": {"train": len(ones_train), "test": len(ones_test),
                               "test_not_in_train": len(ones_test - ones_train)},
        "train": [list(r) for r in train],
        "test": [list(r) for r in test],
    }
    manifest["split_sha256"] = _sha({"train": manifest["train"], "test": manifest["test"]})
    return manifest


def load_or_build_manifest(path: str) -> dict:
    """Load the frozen manifest; fail closed if it disagrees with a rebuild."""

    from pathlib import Path

    frozen = json.loads(Path(path).read_text("utf-8"))
    # verify the file content against its OWN recorded hash first (tamper check),
    # then against the canonical rebuild (generator drift check)
    content_sha = _sha({"train": frozen["train"], "test": frozen["test"]})
    if content_sha != frozen["split_sha256"]:
        raise ValueError("frozen manifest content does not match its recorded split hash (tampered)")
    rebuilt = build_task_manifest()
    if frozen["split_sha256"] != rebuilt["split_sha256"]:
        raise ValueError("frozen dataset manifest disagrees with canonical rebuild; refusing to train")
    return frozen
