from __future__ import annotations

import pytest

from v5_experiments import cyr_gpu012_r1 as core


def _cal(vocab: int, ups: float, eps: float = 100.0) -> dict:
    return {
        "status": "PASS", "regime": f"R1_CHAR_V{vocab}", "batch_rows": 64,
        "training_updates_per_sec": ups, "generation_examples_per_sec": eps,
        "semantic_rows_per_sec": ups * 64,
    }


def test_padded_compact_keeps_active_ids_and_segmentation_exact() -> None:
    text = "74 + 15 = "
    ids = None
    for vocab in (19, 4096, 24576):
        tok = core.tokenizer_for(vocab)
        encoded = tok.encode(text)
        assert max(encoded) < 19
        assert tok.decode(encoded) == text
        assert tok.special == {"pad_id": 0, "bos_id": 1, "eos_id": 2}
        ids = encoded if ids is None else ids
        assert encoded == ids


def test_inactive_large_vocab_predictions_are_fail_visible_not_silently_dropped() -> None:
    tok = core.tokenizer_for(24_576)
    assert tok.decode([tok.table["7"], 20_000, tok.table["4"]]) == "7<inactive:20000>4"
    assert tok.decode([99_999]) == "<invalid:99999>"


def test_only_vocab_dependent_parameter_term_changes() -> None:
    counts = core.parameter_receipts()
    assert counts["19"] == 987_392
    assert counts["4096"] == 1_509_248
    assert counts["24576"] == 4_130_688
    assert counts["24576"] - counts["19"] == (24_576 - 19) * 128


def test_fixed_screen_is_exactly_512k_semantic_rows() -> None:
    assert core.BATCH_ROWS == 64
    assert core.SCREEN_UPDATES == 8_000
    assert core.SCREEN_ROW_PRESENTATIONS == 512_000


def test_resolver_prefers_two_replicated_primary_pairs_when_safe() -> None:
    r = core.resolve_from_calibrations({
        "V19": _cal(19, 8.0), "V4096": _cal(4096, 4.0), "V24576": _cal(24576, 3.0)
    })
    assert r["seeds_to_run"] == 2
    assert r["batch_rows"] == 64
    assert r["screen_row_presentations"] == 512_000


def test_resolver_falls_back_to_one_seed_not_lower_exposure() -> None:
    r = core.resolve_from_calibrations({
        "V19": _cal(19, 2.0), "V4096": _cal(4096, 1.0), "V24576": _cal(24576, 1.5)
    })
    assert r["seeds_to_run"] == 1
    assert r["screen_updates"] == 8_000


def test_resolver_fails_closed_if_one_primary_pair_cannot_fit() -> None:
    with pytest.raises(RuntimeError):
        core.resolve_from_calibrations({
            "V19": _cal(19, 0.5), "V4096": _cal(4096, 0.5), "V24576": _cal(24576, 0.5)
        })


def _receipt(score: float) -> dict:
    return {"trace": [{"row_presentations": 512_000,
                        "dev_measurement": {"complete_exact_with_valid_stop": score}}]}


def test_decision_requires_gap_and_compact_signal() -> None:
    arms = {
        core.arm_label(0, 19): _receipt(0.60), core.arm_label(0, 24_576): _receipt(0.05),
        core.arm_label(1, 19): _receipt(0.55), core.arm_label(1, 24_576): _receipt(0.10),
    }
    d = core.decision(arms, 2)
    assert d["verdict"] == "REPLICATED_OUTPUT_VOCAB_BURDEN_SUPPORTED_AT_512K_ROWS"
    assert d["pre500m_authorized"] is False


def test_decision_can_reject_output_burden_as_primary() -> None:
    arms = {
        core.arm_label(0, 19): _receipt(0.55), core.arm_label(0, 24_576): _receipt(0.51),
        core.arm_label(1, 19): _receipt(0.57), core.arm_label(1, 24_576): _receipt(0.54),
    }
    assert core.decision(arms, 2)["verdict"] == "REPLICATED_OUTPUT_VOCAB_BURDEN_NOT_PRIMARY_AT_512K_ROWS"


def test_matched_initializer_copies_shared_core_and_active_embedding() -> None:
    torch = pytest.importorskip("torch")
    from anra_v5 import cyr_gpu011_run as inherited
    from anra_v5.cyr_gpu012_r1_run import _screen_scope

    builds = []
    seed = core.MODEL_SEEDS[0]
    with _screen_scope(seed=seed, build_receipts=builds):
        small = inherited._build_model(core.spec_for(19), seed, torch=torch, device=torch.device("cpu"))
    with _screen_scope(seed=seed, build_receipts=builds):
        large = inherited._build_model(core.spec_for(24_576), seed, torch=torch, device=torch.device("cpu"))
    s = dict(small.named_parameters()); l = dict(large.named_parameters())
    for name in s:
        if name == "embedding.weight":
            assert torch.equal(s[name], l[name][:19])
        else:
            assert torch.equal(s[name], l[name])
