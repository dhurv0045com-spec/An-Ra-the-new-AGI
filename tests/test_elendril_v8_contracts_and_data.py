"""CPU unit tests for V8 contracts, protocol registry, data/holdouts, and typed model/policy mechanics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import pytest
import torch
import torch.nn.functional as F

from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import (
    ACTION_19_LABELS,
    EvaluationOutcome,
    PublicInput,
    build_v8_model_spec,
    choose_fcap_bottleneck,
)
from anra_v5.elendril.v8.data import (
    D0_EXPECTED_IDENTITY,
    build_d1_dev_suite,
    build_d1_training_dataset,
    load_d0_dataset,
    make_positive_canary_cohort,
)
from anra_v5.elendril.v8.evaluation import EpisodeRecord, summarize_suite_episodes
from anra_v5.elendril.v8.holdout import (
    audit_split_disjointness,
    build_confirmation_suite,
    render_confirmation_public,
)
from anra_v5.elendril.v8.models import (
    algebraic_null_from_joint,
    build_v8_model,
    compute_typed_nll,
    expected_active_parameter_names,
)
from anra_v5.elendril.v8.policies import make_policy
from anra_v5.elendril.v8.protocol import PHASES, PROTOCOL_SHA, jobs
from anra_v5.elendril.v8.verifier import verify_semantic_fixtures
from tools.elendril_v8_worker import prepare_verified_splits


def _tiny_base():
    return replace(
        MODEL_PROFILES["S20"],
        width=32,
        layers=1,
        query_heads=2,
        kv_heads=1,
        head_dimension=16,
        ffn_width=64,
        context_length=128,
    )


def test_v8_protocol_registry_and_fcap_budget() -> None:
    expected_counts = {
        "archive-diagnostic": 3,
        "data-control": 6,
        "head-screen": 6,
        "reader-screen": 6,
        "replication": 6,
    }
    for phase, count in expected_counts.items():
        assert phase in PHASES
        assert len(jobs(phase)) == count

    assert len(PROTOCOL_SHA) == 64
    r, fcap_added, rel_diff = choose_fcap_bottleneck(MODEL_PROFILES["S20"])
    assert r > 0
    assert rel_diff <= 0.01
    p_spec = build_v8_model_spec("P")
    assert abs(fcap_added - p_spec.added_parameters()) / p_spec.added_parameters() <= 0.01


def test_v8_verifier_fixtures_and_data_disjointness() -> None:
    sem = verify_semantic_fixtures()
    assert sem["status"] == "PASS" and sem["fixtures_verified"] == 9

    d0 = load_d0_dataset()
    assert d0.identity == D0_EXPECTED_IDENTITY
    assert len(d0.samples) == 9472

    d1 = build_d1_training_dataset(quota_per_family=16)
    assert len(d1.worlds) == 112
    assert d1.audit["teacher"]["answer_max_min_ratio"] <= 1.1

    dev_cohorts = build_d1_dev_suite(worlds_per_cohort=2, excluded_structure_hashes=d1.structural_hashes)
    dev_worlds = [w for rows in dev_cohorts.values() for w in rows]
    conf_cohorts = build_confirmation_suite(
        confirmation_seed=9191,
        worlds_per_cohort=2,
        excluded_structure_hashes=d1.structural_hashes | {str(w["semantic_hash"]) for w in dev_worlds},
    )
    conf_worlds = [w for rows in conf_cohorts.values() for w in rows]
    audit = audit_split_disjointness(list(d1.worlds) + dev_worlds + conf_worlds, min_distinct_classes_per_split=2)
    assert audit["status"] == "PASS"
    assert audit["leaked_components"] == 0
    assert audit["leaked_structures"] == 0

    canary = make_positive_canary_cohort(d1, worlds_target=16)
    assert 1 <= len(canary.worlds) <= 32
    assert len(canary.steps) >= 16

    cf_txt = render_confirmation_public(
        ordered_facts=(),
        goal_subject="A",
        hops=1,
        query_available=False,
        memory_available=False,
        template_id=3,
        alias_prefix="",
        order_seed=0,
        decision=0,
    )
    assert "Determine value of A." in cf_txt


def test_v8_models_backbone_parity_pointer_isolation_and_algebraic_null() -> None:
    base = _tiny_base()
    specs = {
        k: build_v8_model_spec(k, base_spec=base, pointer_dim=16, recurrent_dim=16, max_sequence_length=128)
        for k in ("B0", "J", "F", "P", "F-cap", "U")
    }
    specs["P_nocopy"] = build_v8_model_spec("P", base_spec=base, copy_enabled=False, max_sequence_length=128)
    models = {k: build_v8_model(s, 7, torch_module=torch) for k, s in specs.items()}

    b0_sd = dict(models["B0"].named_parameters())
    for k in ("J", "F", "P", "P_nocopy", "F-cap", "U"):
        for name, p in models[k].core.named_parameters():
            assert torch.equal(p, b0_sd[name])

    # Exact parity between F and P(copy_enabled=False) at both weight and forward level
    toks = torch.randint(4, 200, (3, 32), dtype=torch.long)
    segs = torch.zeros_like(toks)
    segs[:, 16:] = 1
    a_tok = UTF8.encode("A")[0]
    b_tok = UTF8.encode("B")[0]
    ans_tok = UTF8.special(SpecialToken.ANSWER)

    # Row 0: repeated symbol A in segment 0 prefix, symbol after action_pos, symbol in segment 1
    toks[0, 2] = a_tok
    toks[0, 4] = a_tok
    toks[0, 6] = ans_tok
    toks[0, 7] = b_tok
    toks[0, 20] = b_tok
    # Row 1: empty source symbols (pointer mask all False)
    toks[1, :] = 50
    toks[1, 5] = ans_tok
    segs[1, :] = 0
    # Row 2: single candidate B
    toks[2, 3] = b_tok
    toks[2, 8] = ans_tok

    action_pos = torch.tensor([6, 5, 8], dtype=torch.long)
    out_f = models["F"].forward_action(toks, segs, action_pos)
    out_pnc = models["P_nocopy"].forward_action(toks, segs, action_pos)
    out_p_off = models["P"].forward_action(toks, segs, action_pos, copy_enabled_override=False)
    assert torch.allclose(out_f.log_probs_19, out_pnc.log_probs_19, atol=1e-6)
    assert torch.allclose(out_pnc.log_probs_19, out_p_off.log_probs_19, atol=1e-6)

    # Empty source finiteness + suffix & packed segment isolation
    out_p = models["P"].forward_action(toks, segs, action_pos)
    assert torch.isfinite(out_p.log_probs_19).all()
    assert out_p.has_pointer_candidates.tolist() == [True, False, True]

    toks_seg1 = toks.clone()
    toks_seg1[0, 20] = a_tok + 3
    assert torch.allclose(out_p.log_probs_19[0], models["P"].forward_action(toks_seg1, segs, action_pos).log_probs_19[0], atol=1e-6)

    toks_suf = toks.clone()
    toks_suf[0, 7] = a_tok + 4
    assert torch.allclose(out_p.log_probs_19[0], models["P"].forward_action(toks_suf, segs, action_pos).log_probs_19[0], atol=1e-6)

    # Algebraic null equivalence (probabilities, argmax, and gradients)
    j1 = torch.randn(5, 19, requires_grad=True)
    j2 = j1.detach().clone().requires_grad_(True)
    tgts = torch.tensor([0, 4, 15, 16, 18])
    F.nll_loss(F.log_softmax(j1, dim=-1), tgts).backward()
    _, _, recombined = algebraic_null_from_joint(j2)
    assert torch.allclose(F.log_softmax(j2, dim=-1), recombined, atol=1e-6)
    assert torch.equal(F.log_softmax(j2, dim=-1).argmax(-1), recombined.argmax(-1))
    F.nll_loss(recombined, tgts).backward()
    assert torch.allclose(j1.grad, j2.grad, atol=1e-6)

    # F joint 19-way vs decomposed equivalence check on active params
    assert "val_head.weight" not in expected_active_parameter_names(models["F"], specs["F"], has_answer_targets=False)
    assert "val_head.weight" in expected_active_parameter_names(models["F"], specs["F"], has_answer_targets=True)
    nll_sum, _ = compute_typed_nll(out_f, torch.tensor([0, 2, 3]), torch.tensor([1, 0, 0]))
    assert torch.isfinite(nll_sum)


def test_v8_policy_truth_firewall_and_modes() -> None:
    base = _tiny_base()
    for arm in ("B0", "J", "F", "P", "F-cap"):
        spec = build_v8_model_spec(arm, base_spec=base, pointer_dim=16, max_sequence_length=128)
        model = build_v8_model(spec, 7, torch_module=torch)
        pub = PublicInput("Current value of A? Query=0 Memory=0\nA=B@1/s0\nAction:", False, False, 2)
        for mode in ("raw", "grammar", "availability-mask", "direct-answer-only"):
            pol = make_policy(model, spec, model_identity=arm, mode=mode)
            ex, raw = pol.propose(pub)
            assert ex.mode == mode
            assert raw.mode == "raw"
            with pytest.raises(TypeError):
                pol.propose({"text_utf8": "fake_world_dict"})
        assert len(ACTION_19_LABELS) == 19


def test_v8_r1_full_quota_split_firewall_and_isolated_breakdown(tmp_path: Path) -> None:
    split_root = tmp_path / "full_splits"
    splits = prepare_verified_splits(split_root, cpu_smoke=False)
    assert splits["cached"] is False
    assert splits["audit"]["audit_passed"] is True
    assert splits["audit"]["confirmation_built"] is False
    assert splits["audit"]["confirmation_seed"] is None
    assert splits["conf_cohorts"] == {}
    assert splits["conf_identity"] is None
    assert not (split_root / "splits" / "conf_cohorts.json").exists()
    assert splits["audit"]["counts"]["d1_worlds"] == 3584
    assert splits["audit"]["counts"]["d1_steps"] == 5888
    assert splits["audit"]["counts"]["dev_records"] == 3840
    assert splits["audit"]["counts"]["conf_records"] == 0
    assert len(splits["dev_cohorts"]) == 6
    assert splits["audit"]["disjointness_audit"]["status"] == "PASS"
    assert splits["audit"]["disjointness_audit"]["leaked_components"] == 0
    assert splits["audit"]["disjointness_audit"]["leaked_structures"] == 0
    assert (split_root / "splits" / "SPLIT_AUDIT.json").exists()

    t0 = time.perf_counter()
    reloaded = prepare_verified_splits(split_root, cpu_smoke=False)
    reload_s = time.perf_counter() - t0
    assert reloaded["cached"] is True
    assert reloaded["split_audit_sha256"] == splits["split_audit_sha256"]
    assert reloaded["audit"]["confirmation_built"] is False
    assert reloaded["conf_cohorts"] == {}
    assert reload_s < 10.0

    with_conf = prepare_verified_splits(split_root, cpu_smoke=False, confirmation_seed=777123)
    assert with_conf["audit"]["confirmation_built"] is True
    assert with_conf["audit"]["confirmation_seed"] == 777123
    assert with_conf["audit"]["audit_passed"] is True
    assert with_conf["audit"]["counts"]["d1_worlds"] == 3584
    assert with_conf["audit"]["counts"]["d1_steps"] == 5888
    assert with_conf["audit"]["counts"]["dev_records"] == 3840
    assert len(with_conf["conf_cohorts"]["R09"]) > 0
    assert sum(len(with_conf["conf_cohorts"][c]) for c in ("R09", "R10", "E13", "E14", "E15", "A19")) == 7680
    assert with_conf["audit"]["counts"]["conf_records"] == 7808
    assert {"R09", "R10", "E13", "E14", "E15", "A19"}.issubset(with_conf["conf_cohorts"])
    assert with_conf["audit"]["disjointness_audit"]["status"] == "PASS"
    assert with_conf["audit"]["disjointness_audit"]["leaked_components"] == 0
    assert with_conf["audit"]["disjointness_audit"]["leaked_structures"] == 0
    assert (split_root / "splits" / "conf_cohorts.json").exists()

    # Verify summarize_suite_episodes separates non-isolated from isolated probe accuracy
    def _make_ep(wid: str, *, correct: bool, isolated: bool | None) -> EpisodeRecord:
        return EpisodeRecord(
            world_id=wid,
            base_world_id=wid,
            cohort="R09",
            family="direct_one_hop",
            form="canonical",
            mode="raw",
            model_identity="F",
            steps=(),
            outcome=EvaluationOutcome(
                correct=correct,
                terminal_status="TERMINAL_ANSWER",
                charged_cost=0.0,
            ),
            first_observable_divergence=None,
            query_count=0,
            retrieve_count=0,
            model_calls=1,
            evaluated_positions=1,
            teacher_action_matches=1 if correct else 0,
            is_isolated_one_edge_probe=isolated,
        )

    eps = {
        "raw": [
            _make_ep("w-iso", correct=True, isolated=True),
            _make_ep("w-non1", correct=True, isolated=False),
            _make_ep("w-non2", correct=False, isolated=False),
        ]
    }
    summary = summarize_suite_episodes(eps)
    raw_sum = summary["raw"]
    assert raw_sum["isolated_probe_count"] == 1
    assert raw_sum["isolated_probe_exact_accuracy"] == pytest.approx(1.0)
    assert raw_sum["non_isolated_count"] == 2
    assert raw_sum["non_isolated_exact_accuracy"] == pytest.approx(0.5)
    assert raw_sum["exact_episode_accuracy"] == pytest.approx(2.0 / 3.0)

    # Also verify world_by_id fallback when is_isolated_one_edge_probe is None
    worlds = [
        {"world_id": "w-iso", "hops": 1, "initial_facts": [{"subject": "A"}], "hidden": {"query_payload": [], "memory_payload": []}},
        {"world_id": "w-non1", "hops": 1, "initial_facts": [{"subject": "A"}, {"subject": "B"}], "hidden": {"query_payload": [], "memory_payload": []}},
        {"world_id": "w-non2", "hops": 2, "initial_facts": [{"subject": "A"}], "hidden": {"query_payload": [], "memory_payload": []}},
    ]
    eps_none = {
        "raw": [
            _make_ep("w-iso", correct=True, isolated=None),
            _make_ep("w-non1", correct=True, isolated=None),
            _make_ep("w-non2", correct=False, isolated=None),
        ]
    }
    summary_inferred = summarize_suite_episodes(eps_none, worlds)
    assert summary_inferred["raw"]["isolated_probe_exact_accuracy"] == pytest.approx(1.0)
    assert summary_inferred["raw"]["non_isolated_exact_accuracy"] == pytest.approx(0.5)
