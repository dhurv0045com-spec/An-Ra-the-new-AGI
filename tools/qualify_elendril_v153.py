"""Generate the V.153 anti-shortcut development suite and audit it on CPU.

This is dataset preparation, NOT GPU inference or a completed experiment.
The suite includes evaluator-side truth: never feed its JSON directly to an actor.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from anra_v5.elendril.v6_data import read_dataset
from anra_v5.elendril.v71_data import suite_from_record, suite_to_record
from anra_v5.elendril.v153_diagnostics import audit, build_suite


def prepare(destination, *, seed=153071, worlds=16, train_path=None):
    dest = Path(destination)
    if dest.is_symlink() or dest.exists():
        raise FileExistsError("refuse to overwrite an existing benchmark output")
    suite = build_suite(seed=seed, worlds=worlds)
    train = read_dataset(Path(train_path)) if train_path is not None else None
    report = audit(suite, seed=seed, train=train)
    portable = suite_to_record(suite)
    rebuilt = suite_from_record(portable)
    if audit(rebuilt, seed=seed, train=train)["suite_sha256"] != report["suite_sha256"]:
        raise ValueError("portable roundtrip differs")
    payload = (json.dumps(portable, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report = {
        **report,
        "portable_file": "V153_ROUTE_SUITE.json",
        "portable_file_sha256": sha256(payload).hexdigest(),
        "scientific_status": "NOT_RUN: CPU dataset qualification only",
        "actor_truth_firewall": "Portable dataset includes evaluation truth; only public cases are exposed to neural policies.",
    }
    parent = dest.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v153-prepare-", dir=parent) as tmp:
        temp = Path(tmp)
        (temp / "V153_ROUTE_SUITE.json").write_bytes(payload)
        (temp / "V153_ROUTE_AUDIT.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Commit the directory atomically where possible; no partial final reports.
        os.rename(temp, dest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=None,
                        help="Optional actual V7.1 train.json for exact state-disjointness audit")
    parser.add_argument("--seed", type=int, default=153071)
    parser.add_argument("--worlds", type=int, default=16)
    args = parser.parse_args(argv)
    report = prepare(args.output, seed=args.seed, worlds=args.worlds, train_path=args.train)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
