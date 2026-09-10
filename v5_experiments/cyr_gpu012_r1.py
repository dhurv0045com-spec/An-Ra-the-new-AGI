"""CYR-GPU-012 / R1: representation-output-burden causal screen.

Primary question: with Cymek V5 geometry, ARK-002B rows, objective, optimizer,
active arithmetic token IDs, batch, model seed and example order held fixed,
does enlarging only the tied embedding/output vocabulary suppress structural
held-out acquisition in the early regime where CYR-GPU-011 showed a large
compact-vs-production divergence?

This is a development mechanism experiment. It cannot authorize PRE500M/500M
or general-language tokenizer changes by itself.
"""
from __future__ import annotations

from typing import Any, Mapping

from v5_experiments import cyr_gpu011 as base

CYR12_ID = "CYR-GPU-012-R1"
WALL_MINUTES = 175.0
PACKAGING_RESERVE_MINUTES = 5.0
BATCH_ROWS = 64
SCREEN_UPDATES = 8_000
SCREEN_ROW_PRESENTATIONS = SCREEN_UPDATES * BATCH_ROWS  # 512k
FULL_REFERENCE_UPDATES = 18_000
FULL_REFERENCE_ROWS = FULL_REFERENCE_UPDATES * BATCH_ROWS
PRIMARY_VOCABS = (19, 24_576)
OPTIONAL_VOCAB = 4_096
MODEL_SEEDS = (3511, 3512)
ORDER_SEEDS = (5801, 5802)
MIN_COMPACT_SIGNAL = 0.45
STRONG_GAP = 0.30
EQUIVALENT_GAP = 0.10
RUNTIME_SAFETY_FACTOR = 1.35
MIN_FINALIZE_SECONDS = 120.0


class PaddedCompactTokenizer:
    """Exact active CYR11 compact token IDs inside a larger declared vocabulary.

    IDs 0..18 are byte-for-byte the CYR11 compact alphabet. IDs >=19 are never
    emitted by the tokenizer and exist only as extra tied embedding/output
    classes. That makes 19-vs-N a targeted output/embedding-burden intervention,
    while prompt/answer segmentation and active token IDs remain unchanged.
    """

    pad_id, bos_id, eos_id = 0, 1, 2

    def __init__(self, vocabulary_size: int) -> None:
        vocabulary_size = int(vocabulary_size)
        if vocabulary_size < len(base.COMPACT_TOKENS):
            raise ValueError("vocabulary_size cannot be smaller than active compact alphabet")
        self.vocabulary_size = vocabulary_size
        self.table = {token: i for i, token in enumerate(base.COMPACT_TOKENS)}
        self.inverse = {i: token for i, token in enumerate(base.COMPACT_TOKENS)}

    def encode(self, text: str) -> list[int]:
        try:
            return [self.table[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"R1 compact tokenizer cannot encode {exc.args[0]!r}") from exc

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inverse.get(int(i), "") for i in ids
                       if int(i) not in (self.pad_id, self.bos_id, self.eos_id))

    @property
    def special(self) -> dict[str, int]:
        return {"pad_id": self.pad_id, "bos_id": self.bos_id, "eos_id": self.eos_id}


def tokenizer_for(vocab_size: int) -> PaddedCompactTokenizer:
    return PaddedCompactTokenizer(vocab_size)


def spec_for(vocab_size: int):
    return base.research_small_spec(int(vocab_size))


def arm_label(seed_index: int, vocab_size: int) -> str:
    return f"S{seed_index + 1}_CHAR_V{int(vocab_size)}"


def parameter_receipts() -> dict[str, int]:
    return {str(v): int(spec_for(v).parameter_receipt().total)
            for v in (19, OPTIONAL_VOCAB, 24_576)}


def _estimate_arm_seconds(cal: Mapping[str, Any]) -> float:
    ups = max(float(cal["training_updates_per_sec"]), 1e-9)
    eps = max(float(cal["generation_examples_per_sec"]), 1e-9)
    # At batch64 the inherited acquisition runner evaluates every 200 updates.
    # Besides controller/measurement/probe generation, it can run the full
    # structural battery at three fixed periodic points plus up to four
    # M99/G50 onset/confirmation events. Budget all seven conservatively.
    eval_count = SCREEN_UPDATES // 200
    basic_eval_examples = 64 + 85 + 100
    structural_examples = 85 + 85 + 96 + 64 + 48 + 48
    max_extra_structural_batteries = 7
    final_examples = structural_examples + 64
    raw = (SCREEN_UPDATES / ups
           + eval_count * basic_eval_examples / eps
           + max_extra_structural_batteries * structural_examples / eps
           + final_examples / eps)
    return raw * RUNTIME_SAFETY_FACTOR + MIN_FINALIZE_SECONDS


def resolve_from_calibrations(calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = [f"V{v}" for v in PRIMARY_VOCABS]
    for key in required:
        rec = calibrations.get(key)
        if not rec or rec.get("status") != "PASS" or int(rec.get("batch_rows", -1)) != BATCH_ROWS:
            raise ValueError(f"missing healthy fixed-batch calibration for {key}")
    estimates = {key: _estimate_arm_seconds(calibrations[key]) for key in calibrations
                 if calibrations[key].get("status") == "PASS"}
    science_seconds = (WALL_MINUTES - PACKAGING_RESERVE_MINUTES) * 60.0
    pair_seconds = estimates["V19"] + estimates["V24576"]
    if pair_seconds + 2 * MIN_FINALIZE_SECONDS > science_seconds:
        raise RuntimeError("R1 primary matched pair does not fit the 3-hour campaign wall")
    seeds = 2 if 2 * pair_seconds + 2 * MIN_FINALIZE_SECONDS <= science_seconds else 1
    optional = False
    if f"V{OPTIONAL_VOCAB}" in estimates:
        optional = seeds == 1 and pair_seconds + estimates[f"V{OPTIONAL_VOCAB}"] + 2 * MIN_FINALIZE_SECONDS <= science_seconds
    return {
        "schema": "anra-cyr-gpu012-r1-resolved/v1",
        "experiment": CYR12_ID,
        "wall_minutes": WALL_MINUTES,
        "packaging_reserve_minutes": PACKAGING_RESERVE_MINUTES,
        "batch_rows": BATCH_ROWS,
        "screen_updates": SCREEN_UPDATES,
        "screen_row_presentations": SCREEN_ROW_PRESENTATIONS,
        "primary_vocabs": list(PRIMARY_VOCABS),
        "seeds_to_run": seeds,
        "optional_v4096": optional,
        "estimated_arm_seconds": estimates,
        "runtime_safety_factor": RUNTIME_SAFETY_FACTOR,
        "claim_ceiling": "CONTROLLED_DEVELOPMENT_MECHANISM_ONLY",
        "pre500m_authorized": False,
        "training_500m_authorized": False,
    }


def score_at_rows(receipt: Mapping[str, Any], rows: int = SCREEN_ROW_PRESENTATIONS) -> float | None:
    for entry in receipt.get("trace", []):
        if int(entry.get("row_presentations", -1)) == int(rows):
            metric = entry.get("dev_measurement", {})
            if "complete_exact_with_valid_stop" in metric:
                return float(metric["complete_exact_with_valid_stop"])
    if int(receipt.get("row_presentations", -1)) == int(rows):
        return float(receipt.get("reasoning_battery_final", {}).get("STANDARD", {}).get(
            "complete_exact_with_valid_stop", 0.0))
    return None


def decision(arms: Mapping[str, Mapping[str, Any]], seeds_run: int) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for i in range(int(seeds_run)):
        c = arms.get(arm_label(i, 19))
        p = arms.get(arm_label(i, 24_576))
        if not c or not p:
            continue
        cs, ps = score_at_rows(c), score_at_rows(p)
        if cs is None or ps is None:
            continue
        per_seed.append({"seed_index": i + 1, "compact19": cs, "padded24576": ps, "gap": cs - ps})

    if not per_seed:
        verdict = "INCONCLUSIVE_NO_COMPLETE_PRIMARY_PAIR"
    else:
        supported = [x["compact19"] >= MIN_COMPACT_SIGNAL and x["gap"] >= STRONG_GAP for x in per_seed]
        equivalent = [x["compact19"] >= MIN_COMPACT_SIGNAL and abs(x["gap"]) <= EQUIVALENT_GAP for x in per_seed]
        if len(per_seed) >= 2 and all(supported):
            verdict = "REPLICATED_OUTPUT_VOCAB_BURDEN_SUPPORTED_AT_512K_ROWS"
        elif len(per_seed) >= 2 and all(equivalent):
            verdict = "REPLICATED_OUTPUT_VOCAB_BURDEN_NOT_PRIMARY_AT_512K_ROWS"
        elif len(per_seed) == 1 and supported[0]:
            verdict = "SINGLE_SEED_OUTPUT_VOCAB_BURDEN_SUPPORTED_AT_512K_ROWS"
        elif len(per_seed) == 1 and equivalent[0]:
            verdict = "SINGLE_SEED_OUTPUT_VOCAB_BURDEN_NOT_PRIMARY_AT_512K_ROWS"
        else:
            verdict = "MIXED_OR_INTERMEDIATE_REPRESENTATION_EFFECT"

    return {
        "schema": "anra-cyr-gpu012-r1-decision/v1",
        "verdict": verdict,
        "primary_row_presentations": SCREEN_ROW_PRESENTATIONS,
        "per_seed": per_seed,
        "thresholds": {"min_compact_signal": MIN_COMPACT_SIGNAL,
                       "strong_gap": STRONG_GAP, "equivalent_gap": EQUIVALENT_GAP},
        "interpretation": (
            "19-vs-24576 keeps active arithmetic token IDs, segmentation, model block geometry, "
            "objective, batch and semantic stream fixed. The manipulated factor is the size of the "
            "tied embedding/output class space; this does not separately identify embedding capacity "
            "from softmax competition because Cymek ties them."
        ),
        "historical_bpe_anchor": {
            "experiment": "CYR-GPU-011",
            "production_full_reference_rows": 1_152_000,
            "production_standard": 0.0,
            "production_sealed": "0/48",
            "compact_44p89pct_standard": 0.5647,
            "primary_verdict_uses_anchor": False,
        },
        "broad_reasoning_claim_authorized": False,
        "tokenizer_production_change_authorized": False,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
    }


__all__ = [
    "BATCH_ROWS", "CYR12_ID", "FULL_REFERENCE_ROWS", "MODEL_SEEDS", "OPTIONAL_VOCAB",
    "ORDER_SEEDS", "PACKAGING_RESERVE_MINUTES", "PRIMARY_VOCABS", "PaddedCompactTokenizer",
    "SCREEN_ROW_PRESENTATIONS", "SCREEN_UPDATES", "WALL_MINUTES", "arm_label", "decision",
    "parameter_receipts", "resolve_from_calibrations", "score_at_rows", "spec_for", "tokenizer_for",
]
