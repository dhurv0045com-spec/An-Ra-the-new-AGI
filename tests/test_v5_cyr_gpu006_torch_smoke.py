"""Tiny CPU plumbing for CYR-GPU-006. Never scientific evidence."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from anra_v5 import cyr_gpu006_run as run
from v5_experiments import cyr_gpu006 as core


def test_tiny_shared_parent_four_fork_runtime(tmp_path: Path):
    tokenizer, identity = run.production_tokenizer(Path(__file__).resolve().parents[1])
    special = {"pad_id": identity["pad_id"], "bos_id": identity["bos_id"],
               "eos_id": identity["eos_id"]}
    registry = core.proxy_registry(vocab_size=identity["vocabulary_size"])
    spec = registry["TINY"]["spec"]
    rows = core.render_t2_worlds(worlds_per_split={
        "train": 48, "dev_controller": 16, "dev_measurement": 16,
        "sealed_reserved": 8})

    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    seed = 707
    model = initialize(spec, seed, torch_module=torch)
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=1e-3)
    backend = run._make_backend(model=model, optimizer=optimizer, special=special,
                                device=torch.device("cpu"), torch=torch, lr=1e-3)
    batch = rows["train"][:8]
    data_sha = "cyr006-smoke"
    update = run._one_update(backend=backend, tokenizer=tokenizer, rows=batch,
                             torch=torch, device=torch.device("cpu"), special=special,
                             cumulative=0, update=1, data_sha=data_sha)
    assert update["counted"]["real_tokens"] > 0
    assert update["backend_receipt"]["parameter_sha256_changed"]

    parent_path = tmp_path / "parent-707" / "parent-checkpoint"
    cp = run._save_checkpoint(parent_path, model=model, optimizer=optimizer,
                              torch=torch, counters={"seed": seed, "status": "G90_CONFIRMED"})
    parent = {"seed": seed, "parent_status": "G90_CONFIRMED",
              "parent_checkpoint": str(parent_path),
              "parent_model_sha256": cp["model_sha256"],
              "parent_optimizer_sha256": cp["optimizer_sha256"]}

    equivalence = run.verify_parent_equivalence(parent=parent, spec=spec,
                                                torch=torch, device=torch.device("cpu"))
    assert equivalence["identical"]

    stream = core.build_future_stream(seed=seed, world_count=len(rows["train"]),
                                      prefix_rows=32, tail_rows=128)
    arms = {}
    for arm in core.CYR6_ARMS:
        arms[arm] = run.continuation_arm(
            arm=arm, parent=parent, spec=spec, tokenizer=tokenizer,
            torch=torch, device=torch.device("cpu"), special=special,
            train_rows=rows["train"], controller_rows=rows["dev_controller"],
            measurement_rows=rows["dev_measurement"], stream=stream,
            target_actual_tokens=64, store_root=tmp_path,
            stage_deadline=time.monotonic() + 120.0, eval_interval_tokens=32)
        assert arms[arm]["status"] == "COMPLETE"
        assert arms[arm]["redteam_pass"]

    compared = min(len(receipt["consumed_batch_shas"]) for receipt in arms.values())
    tail = core.assert_future_tail_equality(
        {arm: receipt["consumed_batch_shas"][:compared] for arm, receipt in arms.items()})
    assert tail["identical"]

    parent_run = {"seed": seed, "parent_status": "G90_CONFIRMED",
                  "parent": parent, "arms": arms,
                  "parent_equivalence": equivalence, "future_tail": tail}
    decision = core.decide_campaign([parent_run])
    assert decision["verdict"] == "INCONCLUSIVE", "one plumbing parent must never become science"


def test_batched_candidate_free_generation_executes_on_real_v5(tmp_path: Path):
    tokenizer, identity = run.production_tokenizer(Path(__file__).resolve().parents[1])
    special = {"pad_id": identity["pad_id"], "bos_id": identity["bos_id"],
               "eos_id": identity["eos_id"]}
    spec = core.proxy_registry(vocab_size=identity["vocabulary_size"])["TINY"]["spec"]
    from v5_model.core import initialize
    model = initialize(spec, 9, torch_module=torch)
    rows = core.render_t2_worlds(worlds_per_split={
        "train": 16, "dev_controller": 8, "dev_measurement": 8,
        "sealed_reserved": 4})["dev_controller"]
    rates = run.generate_rates_batched(model, tokenizer, rows, torch=torch,
                                       device=torch.device("cpu"), special=special,
                                       batch_size=8, max_new_tokens=2)
    assert rates["total"] == len(rows)
    assert set(rates) >= {"content_exact", "complete_exact_with_valid_stop",
                          "eos_rate", "max_tokens_rate", "invalid_rate",
                          "prefix_correct_extra"}
