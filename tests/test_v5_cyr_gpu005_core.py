"""CYR-GPU-005 pure core: registry, manifest, leak audit, fork stream,
policies, verdict rules, readiness gate, freeze contract."""

from __future__ import annotations

import pytest

from v5_experiments import cyr_gpu005 as core


def test_proxy_registry_identities_and_param_bands():
    registry = core.proxy_registry()
    assert set(registry) == {"TINY", "RESEARCH_SMALL", "MICRO", "MIDI", "P35"}
    assert registry["P35"]["parameters"] == 35_411_328
    assert registry["MIDI"]["parameters"] == 22_424_448
    assert registry["MICRO"]["parameters"] == 8_654_592
    assert registry["RESEARCH_SMALL"]["parameters"] == 4_130_688
    assert registry["TINY"]["parameters"] == 1_647_104
    for name, role in core.PROXY_ROLES.items():
        assert registry[name]["role"] == role


def test_registry_refuses_mislabeled_scale():
    registry = core.proxy_registry()
    with pytest.raises(ValueError, match="refusing mislabeled scale"):
        core.assert_proxy_in_registry("P35", 8_000_000, registry)
    core.assert_proxy_in_registry("P35", 35_411_328, registry)
    with pytest.raises(ValueError, match="not in the canonical registry"):
        core.assert_proxy_in_registry("P99", 1, registry)


def test_registry_matches_built_models():
    torch = pytest.importorskip("torch")
    from v5_model.core import initialize
    registry = core.proxy_registry()
    for name in ("TINY", "RESEARCH_SMALL", "MICRO"):
        model = initialize(registry[name]["spec"], 7, torch_module=torch)
        actual = sum(p.numel() for p in model.parameters())
        core.assert_proxy_in_registry(name, actual, registry)


def test_t2_worlds_holdout_axis_and_determinism():
    first = core.render_t2_worlds()
    second = core.render_t2_worlds()
    assert [row["prompt"] for row in first["train"]] == \
        [row["prompt"] for row in second["train"]]
    for split in ("dev_controller", "dev_measurement", "sealed_reserved"):
        assert all(6 <= row["tens_band"] <= 7 for row in first[split])
    assert all(1 <= row["tens_band"] <= 5 for row in first["train"])


def test_manifest_binds_actual_rows_and_fails_on_tamper():
    splits = core.render_t2_worlds()
    manifest = core.build_data_manifest(splits)
    core.assert_manifest_sha(manifest)
    tampered = dict(manifest)
    tampered["splits"] = dict(manifest["splits"])
    tampered["splits"]["train"] = manifest["splits"]["train"][:-1]
    with pytest.raises(ValueError, match="does not match its own rows"):
        core.assert_manifest_sha(tampered)


def test_leak_audit_passes_clean_data_and_catches_leak():
    splits = core.render_t2_worlds()
    clean = core.commutation_audit(splits)
    assert clean["commutation_free"], clean["findings"]
    assert clean["reported"]["canonical_reordering_crossing"][
        "crossed_canonical_pairs"] > 0  # declared, not hidden
    leaky = dict(splits)
    leaky["dev_controller"] = (leaky["dev_controller"][:8]
                               + [dict(leaky["train"][0])])
    caught = core.commutation_audit(leaky)
    assert not caught["commutation_free"]
    assert "row_crossing_or_duplicate" in caught["findings"]


def test_future_stream_and_tail_equality():
    stream = core.build_future_stream(seed=707, world_count=100,
                                      prefix_rows=40, tail_rows=32)
    assert stream["fork_boundary"] == 40
    shas = core.batch_shas(stream=stream, batch_size=4, count=4)
    receipt = core.assert_future_tail_equality(
        {"HIGH": shas, "LOW": shas, "FIXED": shas, "HYST": shas})
    assert receipt["identical"] and receipt["batches_compared"] == 4
    with pytest.raises(ValueError, match="fork contract is broken"):
        core.assert_future_tail_equality({"HIGH": shas, "LOW": list(reversed(shas))})


def test_fixed_time_switches_on_actual_tokens():
    assert core.fixed_time_switch_point(continuation_target_tokens=1_000_000) == 500_000
    assert core.lr_for_token("FIXED_TIME_HIGH_TO_LOW", 499_999,
                             switch_point=500_000) == core.CYR5_LRS["HIGH"]
    assert core.lr_for_token("FIXED_TIME_HIGH_TO_LOW", 500_000,
                             switch_point=500_000) == core.CYR5_LRS["LOW"]
    with pytest.raises(ValueError):
        core.fixed_time_switch_point(continuation_target_tokens=1000, fraction=1.0)


def test_hysteretic_lr_follows_controller_state():
    from v5_experiments.cyr_tournament import HysteresisController
    controller = HysteresisController(
        enter_retention=0.90, reenter_plasticity=0.50, confirmations=1)
    for metric in (0.95, 0.95):
        controller.observe(metric=metric, threshold_note="test",
                           token_position=0, lr_before=core.CYR5_LRS["HIGH"],
                           lr_plasticity=core.CYR5_LRS["HIGH"],
                           lr_retention=core.CYR5_LRS["LOW"])
    assert controller.snapshot()["mode"] == "retention"
    assert core.lr_for_token("HYSTERETIC_HIGH_LOW", 0, switch_point=0,
                             controller=controller) == core.CYR5_LRS["LOW"]


def test_verdict_requires_full_contract():
    good_arm = {"status": "COMPLETE", "shares_valid_parent": True,
                "redteam_pass": True, "retention_ret90": 0.9, "final_g": 0.9}
    base = dict(parent_equivalence={"identical": True},
                future_tail={"identical": True},
                leak_audit={"commutation_free": True},
                parent_status="G90_CONFIRMED")
    assert core.decide_verdict(
        arm_receipts={"HIGH_CONTINUE": good_arm, "LOW_CONTINUE": good_arm},
        **base)["verdict"] == "TIED"
    timeboxed = core.decide_verdict(
        arm_receipts={"HIGH_CONTINUE": good_arm,
                      "LOW_CONTINUE": {**good_arm, "status": "TIMEBOX"}},
        **base)
    assert timeboxed["verdict"] == "INCONCLUSIVE"
    assert any("TIMEBOX" in reason for reason in timeboxed["reasons"])
    assert core.decide_verdict(
        arm_receipts={"HIGH_CONTINUE": good_arm, "LOW_CONTINUE": good_arm},
        **{**base, "parent_status": "NO_G90"})["verdict"] == "INCONCLUSIVE"
    assert core.decide_verdict(
        arm_receipts={"HIGH_CONTINUE": good_arm, "LOW_CONTINUE": good_arm},
        **{**base, "parent_equivalence": {"identical": False}}
    )["verdict"] == "INCONCLUSIVE"
    decided = core.decide_verdict(
        arm_receipts={"HIGH_CONTINUE": good_arm,
                      "LOW_CONTINUE": {**good_arm, "retention_ret90": 0.5}},
        **base)
    assert decided["verdict"] == "DECIDED" and decided["winner"] == "HIGH_CONTINUE"


def test_run_readiness_gate_fails_closed():
    all_true = {name: True for name in core.READINESS_CONDITIONS}
    assert core.run_readiness_gate(all_true)["ready"] is True
    blocked = core.run_readiness_gate(
        {**all_true, "xla_accumulation_oracle_passes": False})
    assert blocked["ready"] is False
    assert blocked["blockers"] == ["xla_accumulation_oracle_passes"]
    with pytest.raises(ValueError, match="unknown readiness conditions"):
        core.run_readiness_gate({**all_true, "made_up": True})


def test_freeze_contract_fails_closed_on_any_mismatch():
    prereg = {"executable_sha256": "a" * 64,
              "executable_files": {"runner.py": "b" * 64}}
    ok = core.assert_freeze_contract(
        preregistration=prereg, executable_sha="a" * 64, head_sha="a" * 64,
        file_hashes={"runner.py": "b" * 64})
    assert ok["verified"]
    with pytest.raises(ValueError, match="different executable"):
        core.assert_freeze_contract(
            preregistration=prereg, executable_sha="c" * 64,
            head_sha="c" * 64, file_hashes={"runner.py": "b" * 64})
    with pytest.raises(ValueError, match="not the frozen executable"):
        core.assert_freeze_contract(
            preregistration=prereg, executable_sha="a" * 64,
            head_sha="c" * 64, file_hashes={"runner.py": "b" * 64})
    with pytest.raises(ValueError, match="hash mismatch"):
        core.assert_freeze_contract(
            preregistration=prereg, executable_sha="a" * 64,
            head_sha="a" * 64, file_hashes={"runner.py": "d" * 64})


def test_notebook_cell0_sequence_is_bound():
    sequence = core.notebook_cell0_sequence({"schema": "x"},
                                            executable_sha="a" * 64)
    assert sequence["order"][0] == "read_preregistration_from_commit_B_head"
    assert sequence["order"][2] == "checkout_executable_sha_A"
    assert sequence["order"][-1] == "run_with_external_preregistration"
