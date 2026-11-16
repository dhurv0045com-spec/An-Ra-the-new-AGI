"""Stage 4 verification: typed design audit, transitive cohort leakage, paired-seed Holm-Bonferroni, and nonanticipating teacher checks."""

from __future__ import annotations

from dataclasses import replace

from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import build_v8_model_spec
from anra_v5.elendril.v8.data import build_d1_training_dataset
from anra_v5.elendril.v8.training import prepare_v8_training_data
from anra_v5.planthir import CartridgeArmSpec, V8EnvironmentKernel
from anra_v5.planthir.design import (
    CohortClusterSpec,
    TypedContrastSpec,
    TypedFactorSpec,
    analyze_paired_seed_runs,
    audit_typed_experimental_design,
    validate_cohort_and_split_transitivity,
)
from anra_v5.planthir.instrument import validate_nonanticipating_teacher


def _arm(arm_id: str, *, lr: float = 1e-4, order_seed: int = 7, precision: str = "fp32", arch: str = "J") -> CartridgeArmSpec:
    data = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=12)
    base = replace(
        MODEL_PROFILES["S20"],
        width=32,
        layers=1,
        query_heads=2,
        kv_heads=1,
        head_dimension=16,
        ffn_width=64,
        context_length=512,
    )
    return CartridgeArmSpec(
        arm_id=arm_id,
        architecture_family=arch,
        is_control=(arm_id == "CTRL"),
        v8_spec=build_v8_model_spec(arch, base_spec=base),
        recipe=Recipe(total_updates=2, batch_size=4, micro_batch_size=2, lr=lr, precision=precision),
        dataset_identity=data.data_identity,
        milestone_updates=(1, 2),
        order_seed=order_seed,
    )


def test_typed_design_audit_accepts_clean_and_rejects_nonvarying_missing_and_confounded() -> None:
    ctrl = _arm("CTRL", lr=1e-4)
    trt = _arm("TRT_LR", lr=2e-4)
    f_lr = TypedFactorSpec("lr", "lr", control_value=1e-4, levels=(1e-4, 2e-4))
    c_lr = TypedContrastSpec("C-LR", "lr", "CTRL", "TRT_LR", target_mechanism="step_size", expected_treatment_value=2e-4)

    ok = audit_typed_experimental_design(ctrl, (trt,), factors=(f_lr,), typed_contrasts=(c_lr,))
    assert ok["passed"] is True and ok["status"] == "VALID_DESIGN"

    # 1. Declared factor that does not actually vary across arms fails closed
    f_const = TypedFactorSpec("schedule", "schedule", control_value="cosine", levels=("cosine", "constant"))
    nonvar = audit_typed_experimental_design(ctrl, (trt,), factors=(f_lr, f_const), typed_contrasts=(c_lr,))
    assert nonvar["passed"] is False
    assert any(v["code"] == "NON_VARYING_DECLARED_FACTOR" for v in nonvar["violations"])

    # 2. Missing treatment arm fails closed
    missing_trt = audit_typed_experimental_design(ctrl, (), factors=(f_lr,))
    assert missing_trt["passed"] is False
    assert any(v["code"] == "MISSING_TREATMENT_ARMS" for v in missing_trt["violations"])

    # 3. Undeclared change (confounded order_seed or precision) fails closed
    confounded = _arm("TRT_LR", lr=2e-4, order_seed=99, precision="fp16")
    conf_res = audit_typed_experimental_design(ctrl, (confounded,), factors=(f_lr,), typed_contrasts=(c_lr,))
    assert conf_res["passed"] is False
    assert any(v["code"] == "UNDECLARED_CONFOUNDED_FACTOR" for v in conf_res["violations"])

    # 4. Contrast pointing to missing arm fails closed
    bad_con = TypedContrastSpec("C-BAD", "lr", "CTRL", "NO_SUCH_ARM", target_mechanism="step_size")
    miss_arm = audit_typed_experimental_design(ctrl, (trt,), typed_contrasts=(bad_con,))
    assert miss_arm["passed"] is False
    assert any(v["code"] == "MISSING_CONTRAST_ARM" for v in miss_arm["violations"])


def test_transitive_cohort_union_find_catches_multi_hop_and_allows_disjoint_splits() -> None:
    clean_splits = {
        "train": [
            {"world_id": "w_tr_1", "base_world_id": "b_tr_1", "group_id": "g_tr_1", "cohort_id": "c_tr"},
        ],
        "eval": [
            {"world_id": "w_ev_1", "base_world_id": "b_ev_1", "group_id": "g_ev_1", "cohort_id": "c_ev"},
        ],
        "probe_train": [
            {"world_id": "w_pr_1", "base_world_id": "b_pr_1", "group_id": "g_pr_1", "cohort_id": "c_pr"},
        ],
    }
    res_clean = validate_cohort_and_split_transitivity(clean_splits)
    assert res_clean["passed"] is True and res_clean["leak_count"] == 0

    # Transitive 3-hop chain: train(w_tr -> base b1) <-> multihop(b1 -> parent p1) <-> eval(p1 -> cohort c9) <-> probe(cohort c9)
    leaky_splits = {
        "train": [
            {"world_id": "w_tr_1", "base_world_id": "b1"},
        ],
        "multihop": [
            {"world_id": "w_mh_bridge", "base_world_id": "b1", "parent_world_id": "p1"},
        ],
        "eval": [
            {"world_id": "w_ev_1", "parent_world_id": "p1", "cohort_id": "c9"},
        ],
        "probe_train": [
            {"world_id": "w_pr_1", "cohort_id": "c9"},
        ],
    }
    res_leak = validate_cohort_and_split_transitivity(leaky_splits)
    assert res_leak["passed"] is False
    assert res_leak["status"] == "TRANSITIVE_COHORT_LEAKAGE"
    assert res_leak["leak_count"] >= 1
    leaked_splits = set(res_leak["leaks"][0]["splits"])
    assert {"train", "multihop", "eval", "probe_train"} <= leaked_splits

    # Explicit CohortClusterSpec tying train world and eval group triggers leak
    cohort_leak = validate_cohort_and_split_transitivity(
        clean_splits,
        cohorts=(CohortClusterSpec(cluster_id="explicit-bridge", world_ids=("w_tr_1",), group_ids=("g_ev_1",)),),
    )
    assert cohort_leak["passed"] is False


def test_paired_seed_analysis_intervals_holm_and_defect_rejection() -> None:
    obs = []
    # Contrast 1: strong consistent positive improvement across 6 independent (init, order) seed pairs
    diffs_strong = [0.22, 0.25, 0.24, 0.21, 0.26, 0.23]
    # Contrast 2: noise centered near zero across 6 independent pairs
    diffs_null = [0.02, -0.03, 0.01, -0.01, 0.00, 0.01]
    for i in range(6):
        obs.append(
            {
                "contrast_id": "C_STRONG",
                "pair_id": f"p{i}",
                "init_seed": 100 + i,
                "order_seed": 200 + i,
                "cluster_id": f"cl-{i}",
                "control_value": 0.50,
                "treatment_value": 0.50 + diffs_strong[i],
            }
        )
        obs.append(
            {
                "contrast_id": "C_NULL",
                "pair_id": f"n{i}",
                "init_seed": 300 + i,
                "order_seed": 400 + i,
                "cluster_id": f"cl-n{i}",
                "control_value": 0.50,
                "treatment_value": 0.50 + diffs_null[i],
            }
        )

    ok = analyze_paired_seed_runs(obs, alpha=0.05, n_bootstrap=500)
    assert ok["passed"] is True
    c1 = ok["contrasts"]["C_STRONG"]
    c2 = ok["contrasts"]["C_NULL"]
    assert c1["mean_paired_diff"] > 0.20
    assert c1["bootstrap_ci"][0] > 0.15 and c1["t_ci"][0] > 0.15
    assert c1["p_holm"] >= c1["p_raw"]
    assert c1["significant_holm"] is True
    assert c2["significant_holm"] is False
    assert c2["p_holm"] >= c2["p_raw"]

    # Reject single cluster (pseudoreplication)
    single_cluster = [
        {"contrast_id": "C1", "pair_id": f"s{i}", "init_seed": 1 + i, "order_seed": 2 + i, "cluster_id": "ONLY_ONE", "control_value": 0.4, "treatment_value": 0.6}
        for i in range(4)
    ]
    res_sc = analyze_paired_seed_runs(single_cluster)
    assert res_sc["passed"] is False
    assert "SINGLE_CLUSTER_OR_INSUFFICIENT_PAIRS" in res_sc["contrasts"]["C1"]["defects"]

    # Reject missing pair member & duplicate seed pair
    bad_rows = [
        {"contrast_id": "C1", "pair_id": "a", "init_seed": 7, "order_seed": 11, "control_value": 0.5, "treatment_value": None},
        {"contrast_id": "C1", "pair_id": "b", "init_seed": 7, "order_seed": 11, "control_value": 0.5, "treatment_value": 0.6},
    ]
    res_bad = analyze_paired_seed_runs(bad_rows)
    assert res_bad["passed"] is False
    defects_str = ",".join(res_bad["contrasts"]["C1"]["defects"])
    assert "MISSING_PAIR_MEMBER" in defects_str
    assert "DUPLICATE_SEED_PAIR" in defects_str

    # Reject unequal cluster sizes
    unequal = [
        {"contrast_id": "C1", "pair_id": "u1", "init_seed": 1, "order_seed": 1, "cluster_id": "A", "control_value": 0.4, "treatment_value": 0.6},
        {"contrast_id": "C1", "pair_id": "u2", "init_seed": 2, "order_seed": 2, "cluster_id": "A", "control_value": 0.4, "treatment_value": 0.6},
        {"contrast_id": "C1", "pair_id": "u3", "init_seed": 3, "order_seed": 3, "cluster_id": "B", "control_value": 0.4, "treatment_value": 0.6},
    ]
    res_un = analyze_paired_seed_runs(unequal)
    assert res_un["passed"] is False
    assert "UNEQUAL_CLUSTER_SIZES" in res_un["contrasts"]["C1"]["defects"]


def test_nonanticipating_teacher_validation_passes_v8_and_catches_leaky_teacher() -> None:
    kernel = V8EnvironmentKernel()
    worlds = build_d1_training_dataset(quota_per_family=16).worlds
    # Pick a world and create a counterfactual clone differing only in hidden future facts/answer after QUERY
    # while keeping initial visible public facts identical (or construct two worlds matching t=0 QUERY)
    probe_world = None
    for w in worlds:
        op, _ = kernel.ideal_action(kernel.initial_state(w))
        if op == "QUERY":
            probe_world = w
            break
    assert probe_world is not None

    from copy import deepcopy

    w_a = deepcopy(dict(probe_world))
    w_b = deepcopy(dict(probe_world))
    # Mutate hidden payload only while leaving visible t=0 facts and tools identical
    qp = list(w_b["hidden"].get("query_payload", []))
    if qp:
        qp[0] = {**qp[0], "value": ("Z" if qp[0].get("value") != "Z" else "Y")}
    else:
        qp = [{"subject": "A", "source": "live", "tick": 1, "value": "Z"}]
    w_b["hidden"]["query_payload"] = qp
    w_b["world_id"] = str(w_a["world_id"]) + "-hidden-clone"

    # Real V8 teacher only depends on visible public facts -> identical at t=0 -> nonanticipating passes!
    ok = validate_nonanticipating_teacher(kernel, [(w_a, w_b)])
    assert ok["passed"] is True and ok["status"] == "NONANTICIPATING_OK" and ok["checked_states"] >= 1

    # Leaky teacher peeking at hidden payload despite identical visible observation MUST fail closed!
    def leaky_teacher(state, world, obs):
        rows = world["hidden"].get("query_payload", [])
        val = rows[0].get("value") if rows else "NONE"
        return {f"ANSWER:{val}": 1.0}

    caught = validate_nonanticipating_teacher(kernel, [(w_a, w_b)], teacher_fn=leaky_teacher)
    assert caught["passed"] is False
    assert caught["status"] == "INSTRUMENT_DEFECT_DETECTED"
    assert caught["leaks"][0]["code"] == "ANTICIPATING_TEACHER_LEAK"

    # Non-comparable visible pairs must not vacuously pass
    different_worlds = [w for w in worlds if kernel.observe(kernel.initial_state(w)).rendered_text != kernel.observe(kernel.initial_state(w_a)).rendered_text]
    assert different_worlds
    vacuous = validate_nonanticipating_teacher(kernel, [(w_a, different_worlds[0])])
    assert vacuous["passed"] is False
