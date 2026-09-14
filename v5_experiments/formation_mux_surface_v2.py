"""Official FORMATION-MUX-001 Amendment-1 surface wrapper.

The historical CS-TRANSFER Amendment-2 contract used counts PER FAMILY:
train=600, development=80, sealed=120. The S1 FORMATION-MUX port accidentally
interpreted those numbers as whole-split totals. Official S2 restores the
per-family counts while reusing the deterministic generator.
"""

from __future__ import annotations

from typing import Any

from v5_experiments.formation_mux_data import FAMILIES, build_surface, assert_surface_clean

PER_FAMILY_COUNTS = {"training": 600, "development": 80, "sealed": 120}
TOTAL_COUNTS = {split: per * len(FAMILIES) for split, per in PER_FAMILY_COUNTS.items()}


def build_official_surface(*, seed: int, tokenizer: Any) -> dict[str, Any]:
    if tokenizer is None:
        raise RuntimeError("official surface requires the frozen production tokenizer")
    manifest = build_surface(seed=seed, tokenizer=tokenizer, select_counts=TOTAL_COUNTS)
    assert_surface_clean(manifest)
    validate_official_surface(manifest)
    return manifest


def validate_official_surface(manifest: dict[str, Any]) -> None:
    for split, per_family in PER_FAMILY_COUNTS.items():
        rows = list(manifest["splits"][split])
        if len(rows) != TOTAL_COUNTS[split]:
            raise RuntimeError(
                f"official surface count mismatch {split}: {len(rows)} != {TOTAL_COUNTS[split]}"
            )
        for family in FAMILIES:
            observed = sum(1 for row in rows if row["family"] == family)
            if observed != per_family:
                raise RuntimeError(
                    f"official surface family count mismatch {split}/{family}: "
                    f"{observed} != {per_family}"
                )
        if any("r0_prompt_ids" not in row or "r0_answer_ids" not in row for row in rows):
            raise RuntimeError(f"official surface missing production-BPE ids in {split}")
        latent_max = max(
            max([*row["prompt_ids"], *row["answer_ids"]]) for row in rows
        )
        if latent_max >= 4096:
            raise RuntimeError(
                f"official latent surface escaped shared 0..4095 region: {latent_max}"
            )
