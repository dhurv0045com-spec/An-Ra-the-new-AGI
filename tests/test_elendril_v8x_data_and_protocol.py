"""Tests for V8-X1 contracts, D1-H/D2 datasets, 4 renderers, non-anticipating teacher, and 42-job protocol."""

from __future__ import annotations

from dataclasses import replace

from anra_v5.elendril.v8x.contracts import (
    X1_ARCHITECTURES,
    X1_FAMILIES,
    X1_MAX_SEQUENCE,
    build_x1_model_spec,
    build_x1_s20_base,
    solve_x1_capacity_controls,
)
from anra_v5.elendril.v8x.data import (
    PublicFactX1,
    build_x1_d1h_training_dataset,
    build_x1_d2_paired_dataset,
    build_x1_eval_splits,
    canonical_x1_graph_signature,
    enumerate_valid_x1_paths,
    parse_public_x1,
    render_public_x1,
    resolve_public_x1_belief,
    verify_x1_episode,
    x1_teacher_decide,
)
from anra_v5.elendril.v8x.protocol import (
    X1_MILESTONES,
    X1_TOTAL_UPDATES,
    core_x1_jobs,
    extension_x1_jobs,
    verify_against_design_registry,
)
from anra_v5.planthir.contracts import detect_truth_leaks
from anra_v5.planthir.design import validate_cohort_and_split_transitivity
from anra_v5.planthir.instrument import validate_nonanticipating_teacher


def test_x1_s20_specs_and_capacity_matching() -> None:
    s20 = build_x1_s20_base()
    assert s20.parameter_receipt().total == 19_871_616

    matched = solve_x1_capacity_controls(s20)
    assert matched["r_noedge_rel_diff"] <= 0.01
    assert matched["p_cap_r_rel_diff"] <= 0.01

    specs = {a: build_x1_model_spec(a) for a in X1_ARCHITECTURES}
    assert specs["B0"].added_parameters() == 0
    assert specs["J"].added_parameters() == 9_747
    assert specs["F"].added_parameters() == 10_260
    assert specs["P"].added_parameters() == 76_309
    assert specs["F-cap"].added_parameters() == 76_372
    assert abs(specs["R-noedge"].added_parameters() - specs["R"].added_parameters()) / specs["R"].added_parameters() <= 0.01
    assert abs(specs["P-capR"].added_parameters() - specs["R"].added_parameters()) / specs["R"].added_parameters() <= 0.01
    assert specs["U2-untied"].added_parameters() > specs["U-2"].added_parameters()

    # Tiny CPU spec also solves controls cleanly within 1%
    tiny = replace(s20, width=32, layers=1, query_heads=2, kv_heads=1, head_dimension=16, ffn_width=64, context_length=X1_MAX_SEQUENCE)
    r_tiny = build_x1_model_spec("R", base_spec=tiny, pointer_dim=8, slot_dim=16, num_slots=4, slot_edge_hidden=8)
    rn_tiny = build_x1_model_spec("R-noedge", base_spec=tiny, pointer_dim=8, slot_dim=16, num_slots=4, slot_edge_hidden=8)
    pr_tiny = build_x1_model_spec("P-capR", base_spec=tiny, pointer_dim=8, slot_dim=16, num_slots=4, slot_edge_hidden=8)
    assert abs(rn_tiny.added_parameters() - r_tiny.added_parameters()) / r_tiny.added_parameters() <= 0.01
    assert abs(pr_tiny.added_parameters() - r_tiny.added_parameters()) / r_tiny.added_parameters() <= 0.01


def test_x1_d1h_12_families_4_renderers_and_non_anticipating_teacher() -> None:
    ds = build_x1_d1h_training_dataset(quota_per_family=6)
    assert ds.manifest["prefix_collisions"] == 0
    assert set(ds.manifest["family_counts"]) == set(X1_FAMILIES)
    assert all(len(st.tokens) <= X1_MAX_SEQUENCE for st in ds.states)

    # Round-trip semantic parse across all 4 renderers
    sample_w = ds.worlds[0]
    for rid in range(4):
        txt = render_public_x1(
            ordered_facts=sample_w["initial_facts"],
            goal_subject=sample_w["goal_subject"],
            hops=sample_w["hops"],
            min_epoch=sample_w["min_epoch"],
            query_available=sample_w["allowed_public_tools"]["query_available"],
            memory_available=sample_w["allowed_public_tools"]["memory_available"],
            budget_remaining=4,
            renderer_id=rid,
        )
        assert detect_truth_leaks(txt) == []
        parsed = parse_public_x1(txt)
        assert parsed["goal_subject"] == sample_w["goal_subject"]
        assert parsed["hops"] == sample_w["hops"]
        assert len(parsed["facts"]) == len(sample_w["initial_facts"])

    # Teacher rollout reaches enumerated valid epistemic path & verifier catches lucky guesses
    for fam in X1_FAMILIES:
        w = next(x for x in ds.worlds if x["family"] == fam)
        valid_paths = enumerate_valid_x1_paths(w)
        assert valid_paths, f"expected valid reachable path for {w['world_id']}"
        # Follow teacher steps and confirm outcome is correct & epistemically supported
        facts = [PublicFactX1(**r) for r in w["initial_facts"]]
        qa = w["allowed_public_tools"]["query_available"]
        ma = w["allowed_public_tools"]["memory_available"]
        trace: list[tuple[str, str | None]] = []
        for _ in range(4):
            op, arg = x1_teacher_decide(facts, w["goal_subject"], w["hops"], w["min_epoch"], qa, ma)
            trace.append((op, arg))
            if op == "QUERY" and qa:
                qa = False
                facts.extend(PublicFactX1(**r) for r in w["hidden"]["query_payload"])
            elif op == "RETRIEVE" and ma:
                ma = False
                facts.extend(PublicFactX1(**r) for r in w["hidden"]["memory_payload"])
            else:
                break
        out = verify_x1_episode(w, trace)
        assert out.correct is True and out.epistemic_supported is True and out.lucky_unsupported_guess is False

    # Lucky unsupported guess check: answering expected code on X05 BEFORE calling QUERY
    w_x05 = next(x for x in ds.worlds if x["family"] == "X05")
    exp_sym = str(w_x05["hidden"]["expected"])
    lucky_out = verify_x1_episode(w_x05, [("ANSWER", exp_sym)])
    assert lucky_out.correct is False
    assert lucky_out.epistemic_supported is False
    assert lucky_out.lucky_unsupported_guess is True

    # Non-anticipating teacher check via Planthir audit on identical public inputs with altered hidden payload
    class _ShimX1Kernel:
        def initial_state(self, world):
            return {
                "facts": tuple(PublicFactX1(**r) for r in world["initial_facts"]),
                "goal": world["goal_subject"],
                "hops": world["hops"],
                "min_ep": world.get("min_epoch", 1),
                "qa": world["allowed_public_tools"]["query_available"],
                "ma": world["allowed_public_tools"]["memory_available"],
            }

        def observe(self, state):
            from anra_v5.planthir.contracts import UniversalObservation
            from anra_v5.elendril.codec import digest as d_fn

            txt = render_public_x1(
                ordered_facts=state["facts"],
                goal_subject=state["goal"],
                hops=state["hops"],
                min_epoch=state["min_ep"],
                query_available=state["qa"],
                memory_available=state["ma"],
                budget_remaining=4,
            )
            return UniversalObservation(
                obs_id=d_fn(txt)[:16],
                rendered_text=txt,
                available_tools={"query": state["qa"], "retrieve": state["ma"]},
                consumed_tools=frozenset(),
                decision_index=0,
                budget_remaining=4,
            )

        def ideal_action(self, state):
            return x1_teacher_decide(state["facts"], state["goal"], state["hops"], state["min_ep"], state["qa"], state["ma"])

    alt_w05 = {**w_x05, "hidden": {"expected": None, "query_payload": [], "memory_payload": []}}
    na = validate_nonanticipating_teacher(_ShimX1Kernel(), [(w_x05, alt_w05)])
    assert na["passed"] is True


def test_x1_d2_pairs_splits_transitivity_and_42_job_protocol() -> None:
    train_ds = build_x1_d1h_training_dataset(quota_per_family=4)
    d2_ds = build_x1_d2_paired_dataset(base_worlds=train_ds.worlds)
    assert len(d2_ds.groups) >= len(train_ds.worlds)
    assert d2_ds.manifest["eligible_eq_pairs"] > 0
    assert d2_ds.manifest["eligible_cf_pairs"] > 0

    splits = build_x1_eval_splits(
        train_worlds=train_ds.worlds,
        dev_per_family=2,
        conf_per_family=2,
        depth_per_hop=2,
        milestone_per_family=1,
    )
    assert {w["hops"] for w in splits["depth_stress_views"]} == {1, 2, 3, 4, 5}
    trans = validate_cohort_and_split_transitivity(
        {
            "train": train_ds.worlds,
            "dev": splits["dev_views"],
            "conf": splits["conf_views"],
            "depth": splits["depth_stress_views"],
        }
    )
    assert trans["passed"] is True
    tr_sigs = {canonical_x1_graph_signature(w) for w in train_ds.worlds}
    assert not ({canonical_x1_graph_signature(w) for w in splits["dev_bases"]} & tr_sigs)
    assert not ({canonical_x1_graph_signature(w) for w in splits["conf_bases"]} & tr_sigs)

    # Verify exact 1-for-1 correspondence with research/v8-x/X1_DESIGN_REGISTRY.json
    reg_check = verify_against_design_registry()
    assert reg_check["verified"] is True
    assert reg_check["core_jobs"] == 42
    assert X1_TOTAL_UPDATES == 2368
    assert X1_MILESTONES == (0, 592, 1184, 1472, 2368)
    assert len(core_x1_jobs()) == 42
    assert len(extension_x1_jobs()) == 12
    # Check non-X1-resolve sanity
    st, _ = resolve_public_x1_belief([PublicFactX1("n0", "A", "S0", 1)], "n0", 1, min_epoch=3)
    assert st == "missing"


def test_x1_development_gate_uses_one_observation_per_matched_seed():
    from anra_v5.elendril.v8x.analysis import evaluate_x1_development_selection
    pairs = [dict(init_seed=s, order_seed=s, ctrl_macro_12=0.5, trt_macro_12=0.6) for s in (7,19,41)]
    gate = evaluate_x1_development_selection(pairs)
    inference = gate['paired_seed_stats']['contrasts']['default']
    assert gate['qualified'] and inference['n_clusters'] == inference['n_pairs'] == 3
    assert not evaluate_x1_development_selection([pairs[0], pairs[0], pairs[1]])['qualified']
