from __future__ import annotations

import time
from pathlib import Path

import pytest

from anra_v5 import cyr_gpu011_run as runner
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat
from v5_experiments import cyr_gpu011 as core


def _row(world_id: str, prompt: str, answer: str, **extra):
    body = {"world_id": world_id, "prompt": prompt, "answer": answer}
    body.update(extra)
    return body


def _tiny_battery():
    standard = [_row("s", "60 + 10 = ", "70", a=60, b=10)]
    commuted = [_row("c", "10 + 60 = ", "70", a=10, b=60)]
    locality = [
        _row("l/base", "60 + 10 = ", "70", pair_id="l", role="base", expected_delta=1, a=60, b=10),
        _row("l/cf", "61 + 10 = ", "71", pair_id="l", role="counterfactual", expected_delta=1, a=61, b=10),
    ]
    return {
        "STANDARD": standard,
        "COMMUTED": commuted,
        "LOCALITY": locality,
        "CARRY": [_row("carry", "68 + 15 = ", "83", a=68, b=15)],
        "TRIPLE_ADD": [_row("triple", "60 + 10 + 1 = ", "71", a=60, b=10, c=1)],
        "THREE_DIGIT": [_row("three", "600 + 100 = ", "700", a=600, b=100)],
        "VERBAL": [],
        "_manifest": [],
    }


def _smoke_data():
    train = [
        _row("t0", "12 + 13 = ", "25"),
        _row("t1", "21 + 14 = ", "35"),
        _row("t2", "30 + 15 = ", "45"),
        _row("t3", "22 + 16 = ", "38"),
    ]
    return {
        "source_split_sha256": "smoke",
        "train": train,
        "dev_controller": [_row("dc", "60 + 10 = ", "70")],
        "dev_measurement": [_row("dm", "61 + 10 = ", "71")],
        "sealed_reserved": [],
    }


def test_real_v5_compact_acquisition_executes_one_cpu_update(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(core, "CYR11_MAX_UPDATES", 1)
    monkeypatch.setattr(core, "CYR11_EVAL_EVERY_ROW_PRESENTATIONS", 1)

    data = _smoke_data()
    tokenizer = core.CompactCharTokenizer()
    spec = core.research_small_spec(tokenizer.vocabulary_size)

    with canonical_optimizer_compat():
        receipt = runner.run_acquisition(
            label="CPU_SMOKE",
            model_seed=5,
            order_seed=7,
            spec=spec,
            tokenizer=tokenizer,
            special=tokenizer.special,
            batch_rows=2,
            data=data,
            battery=_tiny_battery(),
            torch=torch,
            device=torch.device("cpu"),
            deadline=time.monotonic() + 120.0,
            out=tmp_path / "run",
            include_verbal=False,
        )

    assert receipt["updates"] == 1
    assert receipt["row_presentations"] == 2
    assert receipt["actual_real_tokens"] > 0
    assert receipt["semantic_stream_sha256"]
    assert receipt["status"] in {"MAX_UPDATES_NO_G90", "G90_CONFIRMED"}

    checkpoint = receipt["final_checkpoint"]
    assert checkpoint["schema"] == "anra-cyr-research-checkpoint/v1"
    assert len(checkpoint["model_sha256"]) == 64
    assert len(checkpoint["optimizer_sha256"]) == 64
    checkpoint_path = Path(checkpoint["path"])
    assert (checkpoint_path / "model.bin").exists()
    assert (checkpoint_path / "optimizer.bin").exists()
    assert (checkpoint_path / "receipt.json").exists()
    assert runner._checkpoint_path(checkpoint) == checkpoint_path

    final_battery = receipt["reasoning_battery_final"]
    assert final_battery["schema"] == "anra-cyr-gpu011-reasoning-battery/v2"
    assert final_battery["generation_max_new_tokens"] == 8
    standard_predictions = final_battery["candidate_free_predictions"]["STANDARD"]
    assert standard_predictions["count"] == 1
    assert len(standard_predictions["sha256"]) == 64
    assert standard_predictions["rows"][0]["world_id"] == "s"

    controller = receipt["dev_controller_final"]
    assert controller["prediction_receipt"]["count"] == 1
    assert len(controller["prediction_receipt"]["sha256"]) == 64
    assert controller["prediction_receipt"]["rows"][0]["world_id"] == "dc"


def test_controller_g90_does_not_stop_before_measurement_support(tmp_path, monkeypatch) -> None:
    """A small-controller hit cannot truncate the larger measurement qualification."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(core, "CYR11_MAX_UPDATES", 2)
    monkeypatch.setattr(core, "CYR11_EVAL_EVERY_ROW_PRESENTATIONS", 1)
    monkeypatch.setattr(core, "CYR11_CONFIRMATIONS", 1)

    measurement_calls = {"count": 0}

    def fake_rates(_model, _tokenizer, rows, **_kwargs):
        world = rows[0]["world_id"]
        if world == "dc":
            value = 0.95
        elif world == "dm":
            measurement_calls["count"] += 1
            value = 0.50 if measurement_calls["count"] == 1 else 0.95
        else:
            value = 0.0
        return {"complete_exact_with_valid_stop": value, "content_exact": value}

    def fake_battery(*_args, **_kwargs):
        return {
            "schema": "test-battery",
            "STANDARD": {"complete_exact_with_valid_stop": 0.95},
            "structural_flags": {},
        }

    monkeypatch.setattr(runner.legacy, "generate_rates_batched", fake_rates)
    monkeypatch.setattr(runner, "reasoning_battery", fake_battery)

    data = _smoke_data()
    tokenizer = core.CompactCharTokenizer()
    spec = core.research_small_spec(tokenizer.vocabulary_size)
    with canonical_optimizer_compat():
        receipt = runner.run_acquisition(
            label="QUALIFICATION_SMOKE",
            model_seed=11,
            order_seed=13,
            spec=spec,
            tokenizer=tokenizer,
            special=tokenizer.special,
            batch_rows=2,
            data=data,
            battery=_tiny_battery(),
            torch=torch,
            device=torch.device("cpu"),
            deadline=time.monotonic() + 120.0,
            out=tmp_path / "qualification",
            include_verbal=False,
        )

    assert receipt["updates"] == 2
    assert receipt["g90_controller_confirm_update"] == 1
    assert receipt["g90_confirm_update"] == 2
    assert receipt["status"] == "G90_CONFIRMED"
    assert "G90_QUALIFIED" in receipt["milestone_checkpoints"]
