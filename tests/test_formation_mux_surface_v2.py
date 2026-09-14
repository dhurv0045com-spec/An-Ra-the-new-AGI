"""Surface-count regression test for FORMATION-MUX-001 Amendment-1."""

from v5_experiments.formation_mux_surface_v2 import (
    PER_FAMILY_COUNTS,
    TOTAL_COUNTS,
    build_official_surface,
)


class _Identity:
    vocabulary_size = 24576
    special_token_ids = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


class _Tokenizer:
    identity = _Identity()

    @staticmethod
    def encode(text: str) -> list[int]:
        # Deterministic in-range stand-in; official execution binds the real tokenizer.
        return [1000 + (sum(map(ord, word)) % 2000) for word in text.split()]


def test_official_surface_counts_are_per_family():
    manifest = build_official_surface(seed=73011, tokenizer=_Tokenizer())
    for split, total in TOTAL_COUNTS.items():
        assert len(manifest["splits"][split]) == total
        for family, expected in ((f, PER_FAMILY_COUNTS[split]) for f in manifest["families"]):
            assert sum(1 for row in manifest["splits"][split] if row["family"] == family) == expected
