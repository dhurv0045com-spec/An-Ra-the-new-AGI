"""Science-S5 surface: S4 custody with 60k unique training rows."""
from __future__ import annotations
from collections import Counter
from typing import Any
from v5_experiments import formation_mux_data as data
from v5_experiments import formation_mux_surface_v4 as s4

PUBLIC_SCHEMA = "anra.formation-mux-public-surface/v5"
PER_FAMILY_COUNTS = {"training": 10_000, "development": 80, "sealed": 120}
TOTAL_COUNTS = {k: v * len(data.FAMILIES) for k, v in PER_FAMILY_COUNTS.items()}


def _fast_shortcut_screens(rows):
    out = {}
    for family in data.FAMILIES:
        fam = [r for r in rows if r.family == family]
        counts = Counter(r.answer_ids for r in fam)
        mode = max(counts, key=lambda a: (counts[a], a))
        n = len(fam)
        out[family] = {
            "constant_full_answer": counts[mode] / n,
            "last_prompt_token_as_full_answer": sum(r.answer_ids == (r.prompt_ids[-1],) for r in fam) / n,
            "first_symbol_as_full_answer": sum(
                r.answer_ids == (next((x for x in r.prompt_ids if x in data.SYMBOLS), -1),)
                for r in fam
            ) / n,
        }
    return out


def _build_full(*, seed: int, tokenizer: Any):
    if tokenizer is None:
        raise RuntimeError("official S5 surface requires frozen production tokenizer")
    original = data.shortcut_screens
    try:
        data.shortcut_screens = _fast_shortcut_screens
        manifest = data.build_surface(seed=seed, tokenizer=tokenizer, select_counts=TOTAL_COUNTS)
    finally:
        data.shortcut_screens = original
    data.assert_surface_clean(manifest)
    for split, per_family in PER_FAMILY_COUNTS.items():
        rows = manifest["splits"][split]
        if len(rows) != TOTAL_COUNTS[split]:
            raise RuntimeError(f"S5 count mismatch {split}: {len(rows)} != {TOTAL_COUNTS[split]}")
        for family in data.FAMILIES:
            if sum(r["family"] == family for r in rows) != per_family:
                raise RuntimeError(f"S5 family count mismatch {split}/{family}")
        if any("r0_prompt_ids" not in r or "r0_answer_ids" not in r for r in rows):
            raise RuntimeError(f"S5 production-BPE ids missing in {split}")
    return manifest


def _bind() -> None:
    # S4 functions intentionally resolve these names at call time.
    s4.PUBLIC_SCHEMA = PUBLIC_SCHEMA
    s4.PER_FAMILY_COUNTS = PER_FAMILY_COUNTS
    s4.TOTAL_COUNTS = TOTAL_COUNTS
    s4.build_official_surface = _build_full


def build_public_surface(*, seed: int, tokenizer: Any):
    _bind()
    manifest = s4.build_public_surface(seed=seed, tokenizer=tokenizer)
    manifest["training_rows_unique_before_stream_wrap"] = TOTAL_COUNTS["training"]
    # Recompute S4 canonical hash after adding the S5 audit field.
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    manifest["sha256"] = s4._sha(body)
    validate_public_surface(manifest)
    return manifest


def validate_public_surface(manifest):
    _bind()
    s4.validate_public_surface(manifest)
    if manifest.get("training_rows_unique_before_stream_wrap") != 60_000:
        raise RuntimeError("S5 training-row audit field mismatch")


def load_public_surface(path):
    import json
    from pathlib import Path
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_public_surface(manifest)
    return manifest


def regenerate_sealed_rows(*, public_manifest, tokenizer, experiment: str):
    _bind()
    return s4.regenerate_sealed_rows(
        public_manifest=public_manifest, tokenizer=tokenizer, experiment=experiment
    )
