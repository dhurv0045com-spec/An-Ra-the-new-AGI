"""Measure a paired ELD-V3 difference using topology clusters as bootstrap units."""

import argparse
import json
import math
from pathlib import Path
import random


def _cluster_counts(metrics: dict) -> dict[str, tuple[int, int]]:
    counts = {}
    for cluster, (correct, total) in metrics["cluster_query_pairs"].items():
        if (type(correct) is not int or type(total) is not int
                or total <= 0 or not 0 <= correct <= total):
            raise ValueError(f"invalid pair counts in topology cluster {cluster!r}")
        counts[cluster] = (correct, total)
    actual = [sum(row[i] for row in counts.values()) for i in range(2)]
    if not counts or actual != metrics["query_swap_both_correct"]:
        raise ValueError("cluster counts do not cover the scored query pairs")
    return counts


def paired_cluster_interval(
    reference: dict[int, dict],
    candidate: dict[int, dict],
    *,
    draws: int = 4096,
    rng_seed: int = 106,
) -> dict:
    """Resample the same topology clusters across all matched seed pairs."""
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("a complete, matched seed set is required")
    if not 100 <= draws <= 100_000:
        raise ValueError("bootstrap draws must lie in [100, 100000]")

    first = _cluster_counts(reference[min(reference)])
    if len(first) < 2:
        raise ValueError("at least two topology clusters are required")
    order = sorted(first)
    differences = {key: 0 for key in order}
    denominators = {key: 0 for key in order}
    for seed in sorted(reference):
        a = _cluster_counts(reference[seed])
        b = _cluster_counts(candidate[seed])
        if a.keys() != first.keys() or b.keys() != first.keys():
            raise ValueError("receipts have different topology-cluster sets")
        for key in order:
            if a[key][1] != b[key][1] or a[key][1] != first[key][1]:
                raise ValueError(f"unmatched pair counts in topology cluster {key!r}")
            differences[key] += b[key][0] - a[key][0]
            denominators[key] += a[key][1]

    rng = random.Random(rng_seed)
    samples = []
    for _ in range(draws):
        clusters = rng.choices(order, k=len(order))
        samples.append(sum(differences[key] for key in clusters) /
                       sum(denominators[key] for key in clusters))
    samples.sort()
    count = draws - 1
    return {
        "method": "paired_topology_cluster_percentile_bootstrap",
        "interpretation": "descriptive for the observed development topology classes",
        "delta_candidate_minus_reference": sum(differences.values()) / sum(denominators.values()),
        "interval_95": [samples[math.floor(count * 0.025)], samples[math.ceil(count * 0.975)]],
        "independent_resampling_units": len(order),
        "matched_seeds": sorted(reference),
        "draws": draws,
        "rng_seed": rng_seed,
    }


def _receipt_group(paths: list[Path], panel: str) -> tuple[tuple, dict[int, dict]]:
    provenance = None
    scores = {}
    for path in paths:
        receipt = json.loads(path.read_bytes())
        if receipt["status"] != "completed" or not receipt["surface"].startswith("graph-v3"):
            raise ValueError(f"a completed V3 receipt is required: {path}")
        endpoint = receipt["evaluations"][-1] if panel == "base" else receipt["panels"][panel]
        fixture = endpoint.get("fixture_sha256", receipt["development_sha256"])
        current = (receipt["source_commit"], receipt["surface"], receipt["profile"],
                   receipt["development_sha256"], fixture, endpoint["update"])
        if provenance is not None and current != provenance:
            raise ValueError(f"unmatched source, fixture, profile or update: {path}")
        provenance = current
        seed = receipt["seed"]
        if type(seed) is not int or seed in scores:
            raise ValueError(f"duplicate or invalid seed: {path}")
        scores[seed] = endpoint["metrics"]
    return provenance, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--panel", default="base")
    args = parser.parse_args()
    reference_id, reference = _receipt_group(args.reference, args.panel)
    candidate_id, candidate = _receipt_group(args.candidate, args.panel)
    if reference_id != candidate_id:
        raise ValueError("candidate and reference must share source, fixture, profile and update")
    print(json.dumps({"panel": args.panel, **paired_cluster_interval(reference, candidate)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
