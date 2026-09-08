"""CYR-GPU-005 local plumbing: the ONE orchestrator executes every stage
on TINY with tiny budgets (section 46). Plumbing, not science: the G90
override is labeled PLUMBING_SMOKE_ONLY and full mode refuses it."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from anra_v5.cyr_gpu005_run import (  # noqa: E402
    BUNDLE_NAME,
    generate_rates,
    production_tokenizer,
    run_campaign,
)
from v5_experiments import cyr_gpu005 as core  # noqa: E402


@pytest.fixture(scope="module")
def smoke_campaign(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("cyr_gpu005_smoke")
    return run_campaign(out=out, mode="smoke")


def test_full_mode_fails_closed_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; CPU-refusal path not exercisable here")
    with pytest.raises(RuntimeError, match="requires Google Colab GPU"):
        run_campaign(out=Path(".tmp_full_refusal"), mode="full")


def test_campaign_completes_all_stages(smoke_campaign):
    campaign = smoke_campaign
    assert campaign["status"] == "COMPLETE"
    assert campaign["tokenizer_identity"]["vocabulary_size"] == 24576
    assert campaign["leak_audit"]["commutation_free"]
    assert campaign["resolved"]["proxy"] == "TINY"


def test_fork_contract_holds_end_to_end(smoke_campaign):
    campaign = smoke_campaign
    parent = campaign["parents"][0]
    assert parent["parent_status"] == "G90_CONFIRMED"
    assert parent["gate_override"] == "PLUMBING_SMOKE_ONLY"
    assert campaign["parent_equivalence"]["identical"]
    assert sorted(campaign["parent_equivalence"]["forks"]) == \
        sorted(core.CYR5_ARMS)
    assert campaign["future_tail"]["identical"]
    assert campaign["future_tail"]["batches_compared"] >= 2
    for arm in core.CYR5_ARMS:
        receipt = campaign["arms"][arm]
        assert receipt["status"] == "COMPLETE"
        assert receipt["actual_real_tokens"] == \
            receipt["target_actual_real_tokens"]
        assert receipt["shares_valid_parent"]
        assert receipt["redteam_pass"]


def test_fixed_time_arm_actually_switches_lr(smoke_campaign):
    receipt = smoke_campaign["arms"]["FIXED_TIME_HIGH_TO_LOW"]
    switch = core.fixed_time_switch_point(
        continuation_target_tokens=receipt["target_actual_real_tokens"])
    lrs = receipt["lr_trace"]
    highs = [value for value in lrs if value == core.CYR5_LRS["HIGH"]]
    lows = [value for value in lrs if value == core.CYR5_LRS["LOW"]]
    assert highs and lows, "fixed-time arm must spend time in BOTH regimes"
    first_low = lrs.index(core.CYR5_LRS["LOW"])
    assert all(value == core.CYR5_LRS["LOW"] for value in lrs[first_low:]), \
        "once switched, FIXED_TIME must stay LOW (preregistered policy)"
    assert switch > 0


def test_hysteresis_arm_runs_the_controller(smoke_campaign):
    snapshot = smoke_campaign["arms"]["HYSTERETIC_HIGH_LOW"]["controller_snapshot"]
    assert snapshot is not None
    assert snapshot["mode"] in ("plasticity", "retention")
    assert len(snapshot["decisions"]) >= 1


def test_transfer_stage_reports_honestly(smoke_campaign):
    transfer = smoke_campaign["transfer"]
    assert transfer is not None
    assert transfer["family"] == "registry"
    assert transfer["status"] in ("COMPLETE", "NOT_INFORMATIVE", "TIMEBOX")


def test_plumbing_verdict_is_not_a_science_claim(smoke_campaign):
    decision = smoke_campaign["decision"]
    assert decision["verdict"] in ("DECIDED", "TIED", "INCONCLUSIVE")
    assert smoke_campaign["mode"] == "smoke"


def test_evidence_bundle_complete_and_verifiable(smoke_campaign, tmp_path):
    bundle_path = Path(smoke_campaign["bundle"]["path"])
    assert bundle_path.name == BUNDLE_NAME
    import hashlib
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert digest == smoke_campaign["bundle"]["sha256"]
    with zipfile.ZipFile(bundle_path) as bundle:
        names = set(bundle.namelist())
    for required in ("SESSION_MANIFEST.json", "ENVIRONMENT.json",
                     "DECISION.json", "REDTEAM.json",
                     "NEGATIVE_CONTROL_TESTS.json",
                     "ACQUISITION/parents.json", "RETENTION/arms.json",
                     "CONTROLLER/hysteresis.json"):
        assert required in names, f"bundle missing {required}"
    controls = json.loads(zipfile.ZipFile(bundle_path).read(
        "NEGATIVE_CONTROL_TESTS.json"))
    assert controls["xla_oracle_passes"]
    assert controls["audit_catches_injected_leak"]
    assert controls["tail_equality_detects_divergence"]


def test_tokenizer_mini_end_to_end(tmp_path):
    """Section 47: REAL 24,576 tokenizer + real V5 proxy: encode, train,
    EOS supervision, generation, decode, checkpoint, reload (1-2 updates)."""

    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    from v5_training.production_backend import ProductionTrainingBackend
    from anra_v5.cyr_gpu005_run import (
        load_parent_checkpoint,
        render_batch,
        save_parent_checkpoint,
        state_fingerprint,
    )
    tokenizer, identity = production_tokenizer()
    special = {"pad_id": identity["pad_id"], "bos_id": identity["bos_id"],
               "eos_id": identity["eos_id"]}
    registry = core.proxy_registry(vocab_size=identity["vocabulary_size"])
    spec = registry["TINY"]["spec"]
    rows = core.render_t2_worlds()["train"][:8]
    tokens, segment_ids, eligible, counted = render_batch(
        tokenizer, rows, torch=torch, device=torch.device("cpu"),
        special=special)
    assert counted["supervised_tokens"] > 0
    assert int((tokens == identity["eos_id"]).sum()) == len(rows)
    model = initialize(spec, 11, torch_module=torch)
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=1e-3)
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=special["bos_id"],
        pad_id=special["pad_id"], device=torch.device("cpu"),
        schedule=lambda cumulative_tokens: 1e-3, bfloat16_autocast=False,
        torch_module=torch, activation_checkpointing=False)
    for _ in range(2):
        ctx = backend.begin_update(type("S", (), {"cumulative_tokens": 0})())
        ctx = backend.accumulate_microstep(
            ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
            tokens_by_source={"t2": counted["supervised_tokens"]},
            planned_total=counted["supervised_tokens"])
        backend.finish_update(type("S", (), {"cumulative_tokens": 0})(), ctx,
                              planned_total=counted["supervised_tokens"],
                              cursor=None)
    path = tmp_path / "mini_checkpoint"
    receipt = save_parent_checkpoint(path, model=model, optimizer=optimizer,
                                     torch=torch, counters={"updates": 2})
    reloaded_model = initialize(spec, 99, torch_module=torch)
    reloaded_optimizer = build_adamw_optimizer(reloaded_model, torch_module=torch)
    load_parent_checkpoint(path, model=reloaded_model,
                           optimizer=reloaded_optimizer, torch=torch)
    before = state_fingerprint(model, optimizer, torch=torch)
    after = state_fingerprint(reloaded_model, reloaded_optimizer, torch=torch)
    assert before == after == {"model_sha256": receipt["model_sha256"],
                               "optimizer_sha256": receipt["optimizer_sha256"]}
    rates = generate_rates(reloaded_model, tokenizer, rows[:4],
                           torch=torch, device=torch.device("cpu"),
                           special=special)
    assert rates["total"] == 4
    assert set(rates) >= {"content_exact", "complete_exact_with_valid_stop",
                          "eos_rate", "max_tokens_rate", "prefix_correct_extra"}
