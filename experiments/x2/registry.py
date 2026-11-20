"""Elendril V8-X2 complete versioned design registry, test-first reader, first discriminating report & qualification."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8x.contracts import X1_FAMILIES
from anra_v5.elendril.v8x.protocol import X1_CORE_SESSIONS, core_x1_jobs, extension_x1_jobs
from anra_v5.palantiri import (
    analyze_paired_seed_runs,
    bootstrap_paired_families,
    build_analyst_first_page_report,
    build_paired_seed_observations,
    build_verification_matrix,
    render_campaign_results_markdown,
)

from .models import X2_ARCHITECTURES, X2_NEW_PAIRS, build_x2_model_spec
from .tasks import T13_DOSES, T14_STRATA, build_t13_family_bank

X2_REGISTRY_SCHEMA = "elendril-x2-design-registry-v2"
X1_LEGACY_ORIGINAL_SHA = "8e62ca0f2046c0f07910fdd20abee40508631227"
X1_LEGACY_COMPAT_SHA = "98b9e3318bfa3c02024685bd8668323065f62811"
X1_LEGACY_SOURCE_SHA = X1_LEGACY_COMPAT_SHA
X1_LEGACY_ACCEPTED_IMPORT_SHAS = (X1_LEGACY_ORIGINAL_SHA, X1_LEGACY_COMPAT_SHA)
X1_LEGACY_COMPAT_DISCLOSURE = {
    "original_sha": X1_LEGACY_ORIGINAL_SHA,
    "repaired_sha": X1_LEGACY_COMPAT_SHA,
    "module_path": "anra_v5/elendril/v8x/evaluation.py",
    "b0_model_spec_sha256": "2cee32519711622a17188bdb22b994dba60fdb980a184daad455d4368cf140b2",
    "change_summary": "Pass real V8ModelSpec(base_spec, vocabulary_size, max_sequence_length) to V8BytePolicy in X1S20ObservablePolicy; scientific settings unchanged.",
}
X2_DEV_SEEDS = ((7, 7), (19, 19), (41, 41))
X2_CONF_SEEDS = ((101, 101), (211, 211), (307, 307))
X2_MILESTONES_2368 = (0, 592, 1184, 1472, 2368)
X2_MILESTONES_4736 = (0, 592, 1184, 1472, 2368, 3552, 4736)
X2_BASE_JOB_EQUIVALENT_CEILING = 126

X2_PRIMARY_ENDPOINTS: dict[str, dict[str, Any]] = {
    "P2": {
        "control": "P2-dense",
        "primary_metric": "role_swapped_relevant_edit_both_correct",
        "threshold_gain": 0.05,
        "panel": "generator_b_role_swapped_interior_edit",
    },
    "R2": {
        "control": "R2-pool",
        "primary_metric": "relation_dependent_first_operation_accuracy",
        "threshold_gain": 0.05,
        "panel": "d1h_relation_dependent_routing",
    },
    "U-P": {
        "control": "U-P-par",
        "primary_metric": "interior_edge_both_correct_4_5_hop",
        "threshold_gain": 0.05,
        "panel": "generator_b_4_5_hop_interior_edit",
    },
    "DM": {
        "control": "DM-add",
        "primary_metric": "selective_edit_and_untouched_sibling_conjunction",
        "threshold_gain": 0.05,
        "panel": "d1h_selective_edit_128",
    },
    "PS": {
        "control": "PS-ctrl",
        "primary_metric": "t14_price_crossover_expected_utility_regret_reduction",
        "threshold_gain": 0.05,
        "panel": "t14_price_crossover_panel",
    },
}

EXECUTION_STATUSES = ("PENDING", "PARTIAL", "COMPLETE", "FAILED")
DELIVERY_STATUSES = ("PENDING", "PARTIAL", "VERIFIED", "FAILED")
SCIENTIFIC_STATUSES = ("UNRUN", "NEGATIVE_DELIVERED", "DEV_QUALIFIED", "CONFIRMED_REPLICATED")
CAUSAL_STATUSES = ("UNRESOLVED", "READOUT_CONTRIBUTES", "LOCAL_REVISION", "MECHANISM_SUPPORTED")


def verify_x1_core_inventory() -> dict[str, Any]:
    """Executable §0.1 inventory check against the pinned X1 core 42-job protocol."""
    jobs = core_x1_jobs()
    ids = [j["job_id"] for j in jobs]
    if len(set(ids)) != len(jobs) or len(jobs) != 42:
        raise ValueError(f"X1 core inventory must have 42 unique jobs, got {len(jobs)} ({len(set(ids))} unique)")
    counts = Counter(j["session"] for j in jobs)
    expected = {name: 6 for name in ("A0", "A1", "A2", "A3", "D0", "D1", "D2")}
    if dict(counts) != expected:
        raise ValueError(f"X1 core session counts mismatch: expected {expected}, got {dict(counts)}")
    return {
        "verified": True,
        "total_jobs": 42,
        "session_counts": dict(counts),
        "legacy_source_sha": X1_LEGACY_SOURCE_SHA,
    }


def _infer_execution_route(
    *,
    backend_id: str,
    block: str,
    objective: str,
    resource_class: str,
    parent_job_id: str | None,
    status: str,
) -> str:
    if status == "EVIDENCED_ACTION_REPLAYED" or resource_class == "evidenced_a0":
        return "evidenced_historical"
    if backend_id == "x1_s20_legacy":
        return "legacy_x1_train"
    if objective == "frozen_probe" or resource_class == "frozen_inference":
        return "x2_frozen_probe"
    if block == "V_HORIZON_4736":
        return "x2_horizon_continuation"
    if parent_job_id is not None:
        return "x2_task_adaptation"
    return "x2_scratch_train"


def is_paired_cartridge_row(row: Mapping[str, Any]) -> bool:
    """Return True iff `row` is the treatment half of a paired CartridgeRunner (`IIIb_NEW_MECHANISMS` or `CONF_SOURCE`) job."""
    route = str(
        row.get("execution_route")
        or _infer_execution_route(
            backend_id=str(row.get("backend_id", "")),
            block=str(row.get("block", "")),
            objective=str(row.get("objective", "")),
            resource_class=str(row.get("resource_class", "")),
            parent_job_id=row.get("parent_job_id"),
            status=str(row.get("status", "PENDING")),
        )
    )
    return bool(
        row.get("backend_id") == "elendril_x2"
        and route == "x2_scratch_train"
        and row.get("control_architecture")
        and row.get("block") in ("IIIb_NEW_MECHANISMS", "CONF_SOURCE")
    )


def _legacy_scientific_identity(
    *,
    job_id: str,
    architecture: str,
    dataset: str,
    objective: str,
    init_seed: int,
    order_seed: int,
    updates: int,
    milestones: Sequence[int],
    legacy_model_spec_sha256: str,
) -> str:
    return digest(
        {
            "job_id": str(job_id),
            "architecture": str(architecture),
            "dataset": str(dataset),
            "objective": str(objective),
            "init_seed": int(init_seed),
            "order_seed": int(order_seed),
            "updates": int(updates),
            "milestones": [int(m) for m in milestones],
            "legacy_model_spec_sha256": str(legacy_model_spec_sha256),
        }
    )


def _x2_row(
    *,
    job_id: str,
    block: str,
    session: str,
    architecture: str,
    control_architecture: str | None,
    source_commit: str,
    backend_id: str,
    dataset: str,
    objective: str,
    init_seed: int,
    order_seed: int,
    updates: int,
    milestones: Sequence[int],
    resource_class: str,
    counts_toward_126_base: bool,
    primary_endpoint: str,
    parent_job_id: str | None = None,
    schedule_note: str = "cosine_2368",
    status: str = "PENDING",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extra_dict = dict(extra or {})
    route = _infer_execution_route(
        backend_id=str(backend_id),
        block=str(block),
        objective=str(objective),
        resource_class=str(resource_class),
        parent_job_id=parent_job_id,
        status=str(status),
    )
    if backend_id == "x1_s20_legacy" and "legacy_scientific_identity_sha256" not in extra_dict:
        extra_dict["legacy_scientific_identity_sha256"] = _legacy_scientific_identity(
            job_id=str(job_id),
            architecture=str(architecture),
            dataset=str(dataset),
            objective=str(objective),
            init_seed=int(init_seed),
            order_seed=int(order_seed),
            updates=int(updates),
            milestones=milestones,
            legacy_model_spec_sha256=str(extra_dict.get("legacy_model_spec_sha256", "")),
        )
    est_hours = 0.40 if resource_class == "full_job" else 0.15
    if backend_id == "x1_s20_legacy" and "original_legacy_spec_sha256" not in extra_dict:
        # Exact Run 1 legacy schema (source_commit=8e62ca0f, no execution_route, Run 1 extra keys)
        strip_keys = {"legacy_scientific_identity_sha256", "original_legacy_spec_sha256"}
        if str(block) == "IIIa_ORIGINAL_U":
            strip_keys.add("legacy_model_spec_sha256")
        compat_extra = {k: v for k, v in extra_dict.items() if k not in strip_keys}
        compat_row = {
            "job_id": str(job_id),
            "block": str(block),
            "session": str(session),
            "architecture": str(architecture),
            "control_architecture": control_architecture,
            "source_commit": X1_LEGACY_ORIGINAL_SHA,
            "backend_id": str(backend_id),
            "dataset": str(dataset),
            "objective": str(objective),
            "init_seed": int(init_seed),
            "order_seed": int(order_seed),
            "updates": int(updates),
            "milestones": [int(m) for m in milestones],
            "resource_class": str(resource_class),
            "counts_toward_126_base": bool(counts_toward_126_base),
            "primary_endpoint": str(primary_endpoint),
            "parent_job_id": parent_job_id,
            "schedule_note": str(schedule_note),
            "estimated_worker_hours": est_hours,
            "extra": compat_extra,
        }
        extra_dict["original_legacy_spec_sha256"] = digest(compat_row)
    row = {
        "job_id": str(job_id),
        "block": str(block),
        "session": str(session),
        "architecture": str(architecture),
        "control_architecture": control_architecture,
        "source_commit": str(source_commit),
        "backend_id": str(backend_id),
        "execution_route": route,
        "dataset": str(dataset),
        "objective": str(objective),
        "init_seed": int(init_seed),
        "order_seed": int(order_seed),
        "updates": int(updates),
        "milestones": [int(m) for m in milestones],
        "resource_class": str(resource_class),
        "counts_toward_126_base": bool(counts_toward_126_base),
        "primary_endpoint": str(primary_endpoint),
        "parent_job_id": parent_job_id,
        "schedule_note": str(schedule_note),
        "status": str(status),
        "execution_status": "COMPLETE" if status == "EVIDENCED_ACTION_REPLAYED" else "PENDING",
        "delivery_status": "PARTIAL" if status == "EVIDENCED_ACTION_REPLAYED" else "PENDING",
        "scientific_status": "UNRUN",
        "causal_status": "UNRESOLVED",
        "estimated_worker_hours": est_hours,
        "extra": extra_dict,
    }
    row["spec_sha256"] = digest({k: v for k, v in row.items() if k not in ("status", "execution_status", "delivery_status", "scientific_status", "causal_status")})
    return row


def build_x2_design_registry(
    *,
    source_commit: str,
    legacy_x1_source_sha: str = X1_LEGACY_COMPAT_SHA,
    selected_candidate: str = "DM",
    selected_control: str = "DM-add",
    a0_evidenced: bool = True,
) -> dict[str, Any]:
    """Generate the complete versioned Elendril V8-X2 registry before the first job (§0.4, §3, §8)."""
    inv = verify_x1_core_inventory()
    t13_bank = build_t13_family_bank()

    rows: list[dict[str, Any]] = []

    # Block I: X1 Core (6 evidenced A0 + 36 remaining unreceipted jobs = 36 base equivalents)
    for j in core_x1_jobs():
        is_a0 = j["session"] == "A0"
        rows.append(
            _x2_row(
                job_id=j["job_id"],
                block="I_X1_CORE",
                session=j["session"],
                architecture=j["architecture"],
                control_architecture=None,
                source_commit=legacy_x1_source_sha,
                backend_id="x1_s20_legacy",
                dataset=j["dataset"],
                objective=j["objective"],
                init_seed=j["init_seed"],
                order_seed=j["order_seed"],
                updates=j["updates"],
                milestones=j["milestones"],
                resource_class="evidenced_a0" if (is_a0 and a0_evidenced) else "full_job",
                counts_toward_126_base=not (is_a0 and a0_evidenced),
                primary_endpoint="x1_core_supported_accuracy",
                status="EVIDENCED_ACTION_REPLAYED" if (is_a0 and a0_evidenced) else "PENDING",
                extra={"legacy_model_spec_sha256": j["model_spec_sha256"]},
            )
        )

    # Block II: Frozen A0 diagnosis (6 checkpoints x dev 2304 + depth 640 views; frozen inference class)
    for arch in ("P", "R"):
        for i_seed, o_seed in X2_DEV_SEEDS:
            rows.append(
                _x2_row(
                    job_id=f"X2-BII-FROZEN-A0-{arch}-s{i_seed}",
                    block="II_FROZEN_A0_DIAGNOSIS",
                    session="BII-FROZEN",
                    architecture=arch,
                    control_architecture="P" if arch == "R" else None,
                    source_commit=source_commit,
                    backend_id="elendril_x2",
                    dataset="x1-a0-frozen-dev2304-depth640-v1",
                    objective="frozen_probe",
                    init_seed=i_seed,
                    order_seed=o_seed,
                    updates=0,
                    milestones=X2_MILESTONES_2368,
                    resource_class="frozen_inference",
                    counts_toward_126_base=False,
                    primary_endpoint="frozen_lesion_and_renderer_factorial",
                    parent_job_id=f"X1-A0-S20-{arch}-standard-s{i_seed}",
                    schedule_note="frozen_checkpoint_inference",
                )
            )

    # Block IIIa: Original U extensions (12 full jobs: U0=U-1/U-2, U1=U-4/U2-untied x 3 seeds)
    for j in extension_x1_jobs():
        rows.append(
            _x2_row(
                job_id=j["job_id"],
                block="IIIa_ORIGINAL_U",
                session=j["session"],
                architecture=j["architecture"],
                control_architecture="U-1" if j["architecture"] != "U-1" else None,
                source_commit=legacy_x1_source_sha,
                backend_id="x1_s20_legacy",
                dataset=j["dataset"],
                objective=j["objective"],
                init_seed=j["init_seed"],
                order_seed=j["order_seed"],
                updates=j["updates"],
                milestones=j["milestones"],
                resource_class="full_job",
                counts_toward_126_base=True,
                primary_endpoint="u_recurrent_multihop_accuracy",
                extra={"legacy_model_spec_sha256": j["model_spec_sha256"]} if "model_spec_sha256" in j else None,
            )
        )

    # Block IIIb: Five isolated new mechanisms x 2 arms x 3 dev seeds = 30 full jobs
    for trt, ctrl in X2_NEW_PAIRS:
        ep = X2_PRIMARY_ENDPOINTS[trt]["primary_metric"]
        sess = f"IIIb-{trt}"
        obj = "T14_BELIEF" if trt == "PS" else "standard"
        ds = "x2-d1h-public-transitions-v1" if trt == "PS" else "x2-d1h-selective-edit-v1"
        for arch, c_arch in ((trt, ctrl), (ctrl, None)):
            mspec = build_x2_model_spec(arch, tiny_canary=False)
            for i_seed, o_seed in X2_DEV_SEEDS:
                rows.append(
                    _x2_row(
                        job_id=f"X2-{sess}-S20-{arch}-s{i_seed}",
                        block="IIIb_NEW_MECHANISMS",
                        session=sess,
                        architecture=arch,
                        control_architecture=c_arch,
                        source_commit=source_commit,
                        backend_id="elendril_x2",
                        dataset=ds,
                        objective=obj,
                        init_seed=i_seed,
                        order_seed=o_seed,
                        updates=2368,
                        milestones=X2_MILESTONES_2368,
                        resource_class="full_job",
                        counts_toward_126_base=True,
                        primary_endpoint=ep,
                        extra={"model_spec_sha256": mspec.sha256(), "pair_id": f"{trt}_vs_{ctrl}"},
                    )
                )

    # Block IV: Hard tasks (4 entrants x 2 task curricula T13/T14 x 3 dev seeds = 24 full jobs)
    entrants = ("P", "R", selected_candidate, selected_control)
    for task_name, obj_name, ds_name in (
        ("T13", "T13_ADAPT", "x2-t13-fst-train32-v1"),
        ("T14", "T14_BELIEF", "x2-t14-costly-6strata-v1"),
    ):
        for arch in entrants:
            parent_sess = "A0" if arch in ("P", "R") else f"IIIb-{selected_candidate}"
            parent_prefix = f"X1-A0-S20-{arch}-standard" if arch in ("P", "R") else f"X2-{parent_sess}-S20-{arch}"
            for i_seed, o_seed in X2_DEV_SEEDS:
                rows.append(
                    _x2_row(
                        job_id=f"X2-BIV-{task_name}-S20-{arch}-s{i_seed}",
                        block="IV_HARD_TASKS",
                        session=f"BIV-{task_name}",
                        architecture=arch,
                        control_architecture=selected_control if arch == selected_candidate else None,
                        source_commit=source_commit,
                        backend_id="elendril_x2",
                        dataset=ds_name,
                        objective=obj_name,
                        init_seed=i_seed,
                        order_seed=o_seed,
                        updates=2368,
                        milestones=X2_MILESTONES_2368,
                        resource_class="full_job",
                        counts_toward_126_base=True,
                        primary_endpoint="t13_16shot_paired_novel_comp" if task_name == "T13" else "t14_price_crossover_utility",
                        parent_job_id=f"{parent_prefix}-s{i_seed}",
                        schedule_note="cloned_lineage_adamw_reset_cosine_2368",
                    )
                )

    # Block V: Horizon 4,736 continuation (P + selected candidate x 3 dev seeds = 6 extensions)
    for arch in ("P", selected_candidate):
        parent_prefix = f"X1-A0-S20-P-standard" if arch == "P" else f"X2-IIIb-{selected_candidate}-S20-{arch}"
        for i_seed, o_seed in X2_DEV_SEEDS:
            rows.append(
                _x2_row(
                    job_id=f"X2-BV-H4736-S20-{arch}-s{i_seed}",
                    block="V_HORIZON_4736",
                    session="BV-H4736",
                    architecture=arch,
                    control_architecture="P" if arch != "P" else None,
                    source_commit=source_commit,
                    backend_id="elendril_x2",
                    dataset="x2-d1h-selective-edit-v1",
                    objective="standard",
                    init_seed=i_seed,
                    order_seed=o_seed,
                    updates=2368,
                    milestones=X2_MILESTONES_4736,
                    resource_class="full_job",
                    counts_toward_126_base=True,
                    primary_endpoint="horizon_4736_vs_2368_heldout_and_retention",
                    parent_job_id=f"{parent_prefix}-s{i_seed}",
                    schedule_note="continuation_second_cosine_peak_0.3x_term_0.1x",
                )
            )

    # Confirmation: winning mechanism + active control x 3 fresh seeds = 6 source + 12 task prep = 18 full jobs
    for arch, c_arch in ((selected_candidate, selected_control), (selected_control, None)):
        for i_seed, o_seed in X2_CONF_SEEDS:
            src_id = f"X2-CONF-SRC-S20-{arch}-s{i_seed}"
            rows.append(
                _x2_row(
                    job_id=src_id,
                    block="CONF_SOURCE",
                    session="CONF-SRC",
                    architecture=arch,
                    control_architecture=c_arch,
                    source_commit=source_commit,
                    backend_id="elendril_x2",
                    dataset="x2-d1h-confirmation-locked-v1",
                    objective="standard",
                    init_seed=i_seed,
                    order_seed=o_seed,
                    updates=2368,
                    milestones=X2_MILESTONES_2368,
                    resource_class="full_job",
                    counts_toward_126_base=True,
                    primary_endpoint=X2_PRIMARY_ENDPOINTS[selected_candidate]["primary_metric"],
                    schedule_note="locked_confirmation_cosine_2368",
                )
            )
            for task_name, obj_name, ds_name in (
                ("T13", "T13_ADAPT", "x2-t13-fst-confirmation12-locked-v1"),
                ("T14", "T14_BELIEF", "x2-t14-costly-confirmation-locked-v1"),
            ):
                rows.append(
                    _x2_row(
                        job_id=f"X2-CONF-{task_name}-S20-{arch}-s{i_seed}",
                        block=f"CONF_{task_name}",
                        session=f"CONF-{task_name}",
                        architecture=arch,
                        control_architecture=c_arch,
                        source_commit=source_commit,
                        backend_id="elendril_x2",
                        dataset=ds_name,
                        objective=obj_name,
                        init_seed=i_seed,
                        order_seed=o_seed,
                        updates=2368,
                        milestones=X2_MILESTONES_2368,
                        resource_class="full_job",
                        counts_toward_126_base=True,
                        primary_endpoint="t13_confirmation_16shot" if task_name == "T13" else "t14_confirmation_crossover",
                        parent_job_id=src_id,
                        schedule_note="locked_confirmation_cloned_lineage_2368",
                    )
                )

    # Separate explicit non-base ledgers (§0.4, §3, §6, §7, §8)
    t14_routing_prep_ledger = [
        _x2_row(
            job_id=f"X2-EXTRA-T14-ROUTE-S20-{arch}-s{i_seed}",
            block="EXTRA_T14_ROUTING_PREP",
            session="EXTRA-T14-PS",
            architecture=arch,
            control_architecture="PS-ctrl" if arch == "PS" else None,
            source_commit=source_commit,
            backend_id="elendril_x2",
            dataset="x2-t14-costly-6strata-v1",
            objective="T14_BELIEF",
            init_seed=i_seed,
            order_seed=o_seed,
            updates=2368,
            milestones=X2_MILESTONES_2368,
            resource_class="extra_t14_routing_prep",
            counts_toward_126_base=False,
            primary_endpoint=X2_PRIMARY_ENDPOINTS["PS"]["primary_metric"],
            parent_job_id=f"X2-IIIb-PS-S20-{arch}-s{i_seed}",
        )
        for arch in ("PS", "PS-ctrl")
        for i_seed, o_seed in X2_DEV_SEEDS
    ]

    t13_transfer_parents_ledger = [
        _x2_row(
            job_id=f"X2-EXTRA-T13-PARENT-{hist}-S20-{arch}-s{i_seed}",
            block="EXTRA_T13_TRANSFER_PARENTS",
            session=f"EXTRA-T13-{hist}",
            architecture=arch,
            control_architecture=selected_control if arch == selected_candidate else None,
            source_commit=source_commit,
            backend_id="elendril_x2",
            dataset=f"x2-t13-history-{hist.lower()}-v1",
            objective="T13_ADAPT",
            init_seed=i_seed,
            order_seed=o_seed,
            updates=2368,
            milestones=X2_MILESTONES_2368,
            resource_class="extra_transfer_parent",
            counts_toward_126_base=False,
            primary_endpoint="t13_forward_transfer_B_vs_Bprime",
        )
        for hist in ("HIST_A", "HIST_APRIME", "HIST_DISJOINT_PRETRAIN", "HIST_FRESH_CTRL")
        for arch in entrants
        for i_seed, o_seed in X2_DEV_SEEDS
    ]

    base_count = sum(1 for r in rows if r["counts_toward_126_base"])
    block_counts = dict(Counter(r["block"] for r in rows if r["counts_toward_126_base"]))
    reg = {
        "schema": X2_REGISTRY_SCHEMA,
        "source_commit": str(source_commit),
        "legacy_x1_source_sha": str(legacy_x1_source_sha),
        "legacy_x1_original_sha": X1_LEGACY_ORIGINAL_SHA,
        "legacy_compatibility_disclosure": dict(X1_LEGACY_COMPAT_DISCLOSURE),
        "x1_core_inventory": inv,
        "selected_candidate": selected_candidate,
        "selected_control": selected_control,
        "base_full_job_equivalents": int(base_count),
        "base_ceiling": X2_BASE_JOB_EQUIVALENT_CEILING,
        "base_counts_by_block": block_counts,
        "primary_endpoints": X2_PRIMARY_ENDPOINTS,
        "t13_split_hashes": t13_bank["split_hashes"],
        "t13_family_counts": t13_bank["counts"],
        "jobs": rows,
        "separate_ledgers": {
            "t14_routing_prep": {
                "count": len(t14_routing_prep_ledger),
                "max_allowed": 6,
                "jobs": t14_routing_prep_ledger,
            },
            "t13_transfer_parents": {
                "count": len(t13_transfer_parents_ledger),
                "max_allowed": 48,
                "jobs": t13_transfer_parents_ledger,
            },
        },
        "cost_by_class": {
            "full_job_base_count": int(base_count),
            "full_job_estimated_worker_hours": round(base_count * 0.40, 2),
            "evidenced_a0_count": sum(1 for r in rows if r["resource_class"] == "evidenced_a0"),
            "frozen_inference_count": sum(1 for r in rows if r["resource_class"] == "frozen_inference"),
            "extra_t14_routing_prep_count": len(t14_routing_prep_ledger),
            "extra_t13_transfer_parents_count": len(t13_transfer_parents_ledger),
        },
    }
    reg["registry_sha256"] = digest(
        {"jobs": [r["spec_sha256"] for r in rows], "source_commit": source_commit, "legacy": legacy_x1_source_sha}
    )
    return reg


def validate_x2_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validator catching duplicate IDs, missing seed pairs, bad archs, or count drift (§9)."""
    defects: list[str] = []
    if registry.get("schema") != X2_REGISTRY_SCHEMA:
        defects.append(f"INVALID_SCHEMA:{registry.get('schema')}")
    jobs = registry.get("jobs", [])
    seen_ids: set[str] = set()
    base_count = 0
    pair_seeds: dict[tuple[str, str], set[int]] = {}
    valid_archs = set(X2_ARCHITECTURES) | {"B0", "J", "F", "F-cap", "P-capR", "R-noedge", "U-1", "U-2", "U-4", "U2-untied"}

    for row in jobs:
        jid = str(row.get("job_id", ""))
        if not jid or jid in seen_ids:
            defects.append(f"DUPLICATE_OR_EMPTY_JOB_ID:{jid}")
        seen_ids.add(jid)
        arch = str(row.get("architecture", ""))
        if arch not in valid_archs:
            defects.append(f"INVALID_ARCHITECTURE:{jid}:{arch}")
        if row.get("counts_toward_126_base"):
            base_count += 1
        pair_key = (str(row.get("session", "")), arch)
        pair_seeds.setdefault(pair_key, set()).add(int(row.get("init_seed", -1)))

    if base_count != X2_BASE_JOB_EQUIVALENT_CEILING:
        defects.append(f"BASE_EQUIVALENT_COUNT_MISMATCH:expected={X2_BASE_JOB_EQUIVALENT_CEILING}:got={base_count}")

    for (sess, arch), seeds in sorted(pair_seeds.items()):
        if len(seeds) != 3:
            defects.append(f"INCOMPLETE_SEED_TRIPLET:{sess}:{arch}:seeds={sorted(seeds)}")

    t13_counts = registry.get("t13_family_counts", {})
    if t13_counts != {"train": 32, "dev": 8, "confirmation": 12}:
        defects.append(f"INVALID_T13_FAMILY_COUNTS:{t13_counts}")

    if defects:
        raise ValueError(f"X2 registry validation failed: {defects}")
    return {"verified": True, "total_rows": len(jobs), "base_full_job_equivalents": base_count, "defects": []}


# =============================================================================
# Test-First Reader, First Discriminating Report & Qualification (§8, §9, §11)
# =============================================================================

REQUIRED_TEST_CARD_SLUGS = (
    *(f"x1_family_{f}" for f in X1_FAMILIES),
    "x1_view_canonical",
    "x1_view_named",
    "x1_view_keyed",
    "x1_view_narrative",
    "x1_view_relevant_cf",
    "x1_view_irrelevant_cf",
    "renderer_2x2_canonical_facts_canonical_header",
    "renderer_2x2_canonical_facts_narrative_header",
    "renderer_2x2_narrative_facts_canonical_header",
    "renderer_2x2_narrative_facts_narrative_header",
    "depth_1_hop",
    "depth_2_hop",
    "depth_3_hop",
    "depth_4_hop",
    "depth_5_hop",
    "lesion_balanced_coverage",
    "generator_b_interior_edit_pair",
    *(f"t13_dose_k{k}" for k in T13_DOSES),
    "t13_retention_and_forward_transfer",
    *(f"t14_stratum_{s}" for s in T14_STRATA),
    "t14_forced_revision",
)


def build_x2_test_cards(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    split: str = "dev",
    generator: str = "x2_suite_v1",
) -> list[dict[str, Any]]:
    """Build §9 test-first reader cards preserving zeros, abstentions, ineligible rows, and both denominators."""
    by_slug: dict[str, list[Mapping[str, Any]]] = {slug: [] for slug in REQUIRED_TEST_CARD_SLUGS}
    for row in episode_rows:
        slug = str(row.get("test_slug", ""))
        if slug:
            by_slug.setdefault(slug, []).append(row)

    cards: list[dict[str, Any]] = []
    for slug in sorted(by_slug):
        rows = by_slug[slug]
        eligible = [r for r in rows if bool(r.get("eligible", True))]
        ineligible_n = len(rows) - len(eligible)
        abstentions = sum(1 for r in eligible if str(r.get("observed_op", "")) == "ABSTAIN")
        correct_n = sum(1 for r in eligible if bool(r.get("correct", False)))
        prior_ok = [r for r in eligible if bool(r.get("prior_canonical_correct", False))]
        cond_ok = sum(1 for r in prior_ok if bool(r.get("correct", False)))

        uncond_rate = float(correct_n / len(eligible)) if eligible else None
        cond_rate = float(cond_ok / len(prior_ok)) if prior_ok else None
        rep_success = next((r.get("world_id") for r in eligible if bool(r.get("correct", False))), None)
        rep_failure = next((r.get("world_id") for r in eligible if not bool(r.get("correct", False))), None)

        cards.append(
            {
                "test_slug": slug,
                "version": "v2",
                "source_commit": str(source_commit),
                "split": str(split),
                "generator": str(generator),
                "question": f"Paired causal evaluation for {slug}",
                "comparator": str(rows[0].get("comparator", "active_control")) if rows else "active_control",
                "intervention": str(rows[0].get("intervention", "intact")) if rows else "intact",
                "scoring_function": "exact_supported_accuracy",
                "total_rows": len(rows),
                "eligible_denominator": len(eligible),
                "ineligible_rows": ineligible_n,
                "abstention_count": abstentions,
                "unconditional_point_estimate": uncond_rate,
                "conditional_denominator": len(prior_ok),
                "conditional_point_estimate": cond_rate,
                "empty_stratum": len(eligible) == 0,
                "representative_success_id": rep_success,
                "representative_failure_id": rep_failure,
                "raw_ids_sample": [str(r.get("world_id", "")) for r in eligible[:8]],
            }
        )
    return cards


def build_x2_first_discriminating_report(
    *,
    seed: int,
    world_id: str,
    intact_trace: Mapping[str, Any],
    targeted_trace: Mapping[str, Any],
    sham_trace: Mapping[str, Any],
    cohort_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build §11 First Discriminating Report identifying earliest observed changed variable + full cohort counts."""
    ordered_fields = (
        "prefix_sha256",
        "write_read_vector",
        "operation_margin",
        "action",
        "observation_sha256",
        "terminal_supported",
        "untouched_sibling_correct",
    )
    earliest_changed: str | None = None
    diffs: dict[str, dict[str, Any]] = {}
    for field in ordered_fields:
        v_int = intact_trace.get(field)
        v_tgt = targeted_trace.get(field)
        v_shm = sham_trace.get(field)
        changed = v_int != v_tgt
        diffs[field] = {
            "intact": v_int,
            "targeted": v_tgt,
            "sham": v_shm,
            "targeted_changed_vs_intact": bool(changed),
            "sham_preserved_vs_intact": bool(v_int == v_shm),
        }
        if earliest_changed is None and changed:
            earliest_changed = field

    return {
        "schema": "elendril-x2-first-discriminating-report-v1",
        "seed": int(seed),
        "world_id": str(world_id),
        "prefix_identical_across_conditions": bool(
            intact_trace.get("prefix_sha256") == targeted_trace.get("prefix_sha256") == sham_trace.get("prefix_sha256")
        ),
        "earliest_observed_changed_variable": earliest_changed,
        "field_comparisons": diffs,
        "full_matched_cohort_counts": dict(cohort_counts),
    }


def holm_bonferroni_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Exact Holm-Bonferroni step-down adjustment across registered candidate primaries (§8)."""
    items = sorted(((k, float(v)) for k, v in p_values.items()), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for idx, (k, p) in enumerate(items):
        adj = min(1.0, (m - idx) * p)
        running_max = max(running_max, adj)
        adjusted[k] = round(float(running_max), 8)
    return adjusted


def _expected_artifacts_for_route(route: str, *, is_paired: bool) -> tuple[str, ...]:
    if route == "evidenced_historical":
        return ("Experiment Result/V8-X1/A0/METRICS.json",)
    if route == "legacy_x1_train":
        return ("job.json", "updates.jsonl", "episodes.jsonl", "summary.json")
    if route == "x2_frozen_probe":
        return ("FROZEN_PROBE_RECEIPT.json",)
    if route == "x2_task_adaptation":
        return ("PARENT_LINEAGE_RECEIPT.json", "FINAL_RECEIPT.json", "ARM_RESULTS.json")
    if route == "x2_horizon_continuation":
        return ("HORIZON_CONTINUATION_RECEIPT.json", "FINAL_RECEIPT.json", "ARM_RESULTS.json")
    if route == "x2_scratch_train" and is_paired:
        return ("FINAL_RECEIPT.json", "ARM_RESULTS.json", "ARM_TO_LOGICAL_ROW_MAP.json")
    return ("FINAL_RECEIPT.json", "ARM_RESULTS.json")


def build_x2_obligation_ledger(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a cross-session global obligation ledger linking all 192 logical rows, workers, parents & artifacts."""
    all_rows: list[Mapping[str, Any]] = [
        *registry.get("jobs", ()),
        *registry.get("separate_ledgers", {}).get("t14_routing_prep", {}).get("jobs", ()),
        *registry.get("separate_ledgers", {}).get("t13_transfer_parents", {}).get("jobs", ()),
    ]
    # Paired CartridgeRunner jobs (`IIIb_NEW_MECHANISMS` & `CONF_SOURCE`) run both arms in one worker unit
    paired_ctrl_to_worker: dict[tuple[str, int, str], str] = {}
    paired_workers: set[str] = set()
    for r in all_rows:
        if is_paired_cartridge_row(r):
            paired_ctrl_to_worker[(str(r["session"]), int(r["init_seed"]), str(r["control_architecture"]))] = str(r["job_id"])
            paired_workers.add(str(r["job_id"]))

    entries: list[dict[str, Any]] = []
    for r in all_rows:
        jid = str(r["job_id"])
        block = str(r["block"])
        route = str(r.get("execution_route") or _infer_execution_route(
            backend_id=str(r.get("backend_id", "")),
            block=block,
            objective=str(r.get("objective", "")),
            resource_class=str(r.get("resource_class", "")),
            parent_job_id=r.get("parent_job_id"),
            status=str(r.get("status", "PENDING")),
        ))
        ctrl_worker = paired_ctrl_to_worker.get((str(r["session"]), int(r["init_seed"]), str(r["architecture"])))
        if ctrl_worker and route == "x2_scratch_train" and block in ("IIIb_NEW_MECHANISMS", "CONF_SOURCE") and not r.get("control_architecture"):
            worker_id = ctrl_worker
            arm_role = "control"
            is_paired = True
        elif jid in paired_workers:
            worker_id = jid
            arm_role = "treatment"
            is_paired = True
        else:
            worker_id = jid
            arm_role = "frozen_probe" if route == "x2_frozen_probe" else ("standalone" if not r.get("control_architecture") else "treatment")
            is_paired = False

        is_conf = block.startswith("CONF_")
        initial_status = "COMPLETE" if route == "evidenced_historical" else "PENDING"
        entries.append(
            {
                "logical_id": jid,
                "block": block,
                "session": str(r["session"]),
                "architecture": str(r["architecture"]),
                "control_architecture": r.get("control_architecture"),
                "worker_unit_id": worker_id,
                "arm_role": arm_role,
                "is_paired_worker_half": is_paired,
                "execution_route": route,
                "parent_job_id": r.get("parent_job_id"),
                "dataset": str(r["dataset"]),
                "objective": str(r["objective"]),
                "updates": int(r["updates"]),
                "counts_toward_126_base": bool(r.get("counts_toward_126_base", False)),
                "mandatory": not is_conf,
                "obligation_scope": "conditional_on_trigger" if is_conf else "mandatory",
                "expected_artifacts": list(_expected_artifacts_for_route(route, is_paired=is_paired)),
                "status": initial_status,
            }
        )

    return {
        "schema": "elendril-x2-obligation-ledger-v1",
        "registry_sha256": registry.get("registry_sha256"),
        "total_logical_rows": len(entries),
        "mandatory_rows": sum(1 for e in entries if e["mandatory"]),
        "conditional_confirmation_rows": sum(1 for e in entries if not e["mandatory"]),
        "entries": entries,
    }


def evaluate_confirmation_trigger(
    candidate_screen_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate preregistered IIIb selection trigger without treating DM/DM-add as an automatic winner (§3, §8)."""
    if not candidate_screen_results:
        return {
            "schema": "elendril-x2-confirmation-trigger-v1",
            "triggered": False,
            "decision": "NOT_TRIGGERED_SIGNED_OFF",
            "selected_candidate": None,
            "selected_control": None,
            "reason": "NO_SCREEN_RESULTS_SUPPLIED",
            "failed_criteria_by_candidate": {},
        }

    p_map: dict[str, float] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for cand, info in sorted(candidate_screen_results.items()):
        scores = list(info.get("paired_seed_scores", ()))
        deltas = [float(r["treatment_score"]) - float(r["control_score"]) for r in scores]
        n_pos = sum(1 for d in deltas if d > 0.0)
        mean_gain = float(sum(deltas) / len(deltas)) if deltas else 0.0
        min_delta = min(deltas) if deltas else 0.0
        ret_delta = float(info.get("old_macro_retention_delta", 0.0))
        all_delivered = bool(info.get("complete_and_verified", True))
        # Exact paired sign-flip p-value bound (2**(-n) when all positive, else conservative >0.25)
        raw_p = float(info.get("p_value", (2.0 ** (-len(deltas)) if (deltas and n_pos == len(deltas)) else 0.5)))
        p_map[str(cand)] = raw_p
        parsed[str(cand)] = {
            "control": str(info.get("control", X2_PRIMARY_ENDPOINTS.get(str(cand), {}).get("control", "control"))),
            "mean_gain": round(mean_gain, 6),
            "positive_seeds": int(n_pos),
            "total_seeds": len(deltas),
            "min_seed_delta": round(min_delta, 6),
            "old_macro_retention_delta": round(ret_delta, 6),
            "complete_and_verified": all_delivered,
            "raw_p_value": raw_p,
        }

    holm = holm_bonferroni_adjust(p_map)
    failed: dict[str, list[str]] = {}
    passing: list[tuple[float, str]] = []
    for cand, stats in parsed.items():
        errs: list[str] = []
        if not stats["complete_and_verified"]:
            errs.append("INCOMPLETE_OR_UNVERIFIED_DELIVERY")
        if stats["total_seeds"] < 3:
            errs.append(f"INSUFFICIENT_SEEDS:{stats['total_seeds']}<3")
        if stats["mean_gain"] < 0.05:
            errs.append(f"MEAN_GAIN_BELOW_0.05:{stats['mean_gain']:+.4f}")
        if stats["positive_seeds"] < 2:
            errs.append(f"POSITIVE_SEEDS_BELOW_2:{stats['positive_seeds']}/{stats['total_seeds']}")
        if stats["min_seed_delta"] < -0.05:
            errs.append(f"SEED_REGRESSION_BELOW_-0.05:{stats['min_seed_delta']:+.4f}")
        if stats["old_macro_retention_delta"] < -0.03:
            errs.append(f"RETENTION_DROP_BELOW_-0.03:{stats['old_macro_retention_delta']:+.4f}")
        if errs:
            failed[cand] = errs
        else:
            passing.append((stats["mean_gain"], cand))

    if not passing:
        return {
            "schema": "elendril-x2-confirmation-trigger-v1",
            "triggered": False,
            "decision": "NOT_TRIGGERED_SIGNED_OFF",
            "selected_candidate": None,
            "selected_control": None,
            "reason": "NO_CANDIDATE_MET_PREREGISTERED_DEV_GATE",
            "holm_adjusted_p_values": holm,
            "candidate_summaries": parsed,
            "failed_criteria_by_candidate": failed,
        }

    passing.sort(reverse=True)
    winner = passing[0][1]
    return {
        "schema": "elendril-x2-confirmation-trigger-v1",
        "triggered": True,
        "decision": "TRIGGERED",
        "selected_candidate": winner,
        "selected_control": parsed[winner]["control"],
        "holm_adjusted_p_values": holm,
        "candidate_summaries": parsed,
        "failed_criteria_by_candidate": failed,
    }


def evaluate_x2_completion_audit(
    ledger: Mapping[str, Any],
    completed_receipts: Mapping[str, Mapping[str, Any]],
    *,
    confirmation_decision: Mapping[str, Any] | None = None,
    case_recomputation_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    session_queue_empty: bool | None = None,
) -> dict[str, Any]:
    """Global X2 completion audit separating session queue emptiness from 192-row protocol coverage (§2, §3)."""
    recomp = dict(case_recomputation_receipts or {})
    conf_untriggered = bool(
        confirmation_decision is not None and confirmation_decision.get("decision") == "NOT_TRIGGERED_SIGNED_OFF"
    )
    rows_out: list[dict[str, Any]] = []
    by_block: dict[str, dict[str, int]] = {}
    unresolved_mandatory: list[str] = []
    unresolved_conditional: list[str] = []
    missing_artifacts_list: list[dict[str, Any]] = []
    unverified_recomp_ids: list[str] = []

    for entry in ledger.get("entries", ()):
        lid = str(entry["logical_id"])
        block = str(entry["block"])
        b_counts = by_block.setdefault(
            block,
            {"eligible": 0, "completed": 0, "negative": 0, "invalid": 0, "pending": 0, "not_triggered": 0},
        )
        b_counts["eligible"] += 1

        rec = completed_receipts.get(lid)
        if entry["execution_route"] == "evidenced_historical":
            row_status = "COMPLETE"
            rec_present = rec or {"verified": True, "outcome": "complete"}
        elif not entry["mandatory"] and conf_untriggered and rec is None:
            row_status = "NOT_TRIGGERED_SIGNED_OFF"
            b_counts["not_triggered"] += 1
            rows_out.append({"logical_id": lid, "block": block, "status": row_status})
            continue
        elif rec is None:
            row_status = "PENDING"
        else:
            missing_art = list(rec.get("missing_artifacts", ()))
            expected = set(entry.get("expected_artifacts", ()))
            present = set(rec.get("present_artifacts", expected if not missing_art else ()))
            req_missing = sorted(set(missing_art) | (expected - present)) if "present_artifacts" in rec or missing_art else []
            if req_missing:
                missing_artifacts_list.append({"logical_id": lid, "missing": req_missing})
                row_status = "INVALID"
            elif rec.get("status") in ("INVALID", "FAILED"):
                row_status = "INVALID"
            elif rec.get("status") not in ("COMPLETE", "NEGATIVE_DELIVERED", "VERIFIED", None):
                row_status = "PENDING"
            else:
                if entry["execution_route"] != "evidenced_historical":
                    rc = recomp.get(lid)
                    if rc is None or not rc.get("verified", False):
                        unverified_recomp_ids.append(lid)
                        row_status = "INVALID"
                    elif rec.get("outcome") == "negative" or rec.get("status") == "NEGATIVE_DELIVERED":
                        row_status = "NEGATIVE_DELIVERED"
                    else:
                        row_status = "COMPLETE"
                else:
                    row_status = "COMPLETE"

        if row_status == "COMPLETE":
            b_counts["completed"] += 1
        elif row_status == "NEGATIVE_DELIVERED":
            b_counts["completed"] += 1
            b_counts["negative"] += 1
        elif row_status == "INVALID":
            b_counts["invalid"] += 1
        else:
            b_counts["pending"] += 1

        if row_status in ("PENDING", "INVALID"):
            if entry["mandatory"]:
                unresolved_mandatory.append(lid)
            else:
                unresolved_conditional.append(lid)

        rows_out.append({"logical_id": lid, "block": block, "worker_unit_id": entry["worker_unit_id"], "status": row_status})

    x2_complete = bool(
        not unresolved_mandatory
        and not unresolved_conditional
        and not missing_artifacts_list
        and not unverified_recomp_ids
    )
    totals = {
        "eligible": sum(v["eligible"] for v in by_block.values()),
        "completed": sum(v["completed"] for v in by_block.values()),
        "negative": sum(v["negative"] for v in by_block.values()),
        "invalid": sum(v["invalid"] for v in by_block.values()),
        "pending": sum(v["pending"] for v in by_block.values()),
        "not_triggered": sum(v["not_triggered"] for v in by_block.values()),
    }
    return {
        "schema": "elendril-x2-completion-audit-v1",
        "global_x2_status": "X2_COMPLETE" if x2_complete else "PARTIAL",
        "x2_complete": x2_complete,
        "x3_unlocked": x2_complete,
        "session_queue_empty": session_queue_empty,
        "totals": totals,
        "counts_by_block": by_block,
        "unresolved_mandatory_ids": unresolved_mandatory,
        "unresolved_conditional_ids": unresolved_conditional,
        "missing_artifacts": missing_artifacts_list,
        "unverified_case_recomputations": unverified_recomp_ids,
        "rows": rows_out,
    }


def evaluate_x2_qualification(
    *,
    campaign_id: str,
    candidate_name: str,
    control_name: str,
    paired_seed_scores: Sequence[Mapping[str, Any]],
    source_commit: str = "0" * 40,
    old_macro_retention_delta: float = 0.0,
    stage: str = "development",
    execution_status: str = "COMPLETE",
    delivery_status: str = "VERIFIED",
    causal_evidence_present: bool = False,
    missing_artifacts: Sequence[str] = (),
    all_candidate_pvalues: Mapping[str, float] | None = None,
    paired_world_or_family_scores: Mapping[str, tuple[float, float]] | None = None,
    verification_receipts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate X2 qualification reusing Planthir paired-seed inference & strict anti-drift verification (§8, §9)."""
    if execution_status not in EXECUTION_STATUSES:
        raise ValueError(f"invalid execution_status {execution_status!r}")
    if delivery_status not in DELIVERY_STATUSES:
        raise ValueError(f"invalid delivery_status {delivery_status!r}")

    arm_records: list[dict[str, Any]] = []
    for idx, r in enumerate(paired_seed_scores):
        seed_num = int(str(r.get("seed_id", idx)).lstrip("s") or idx)
        arm_records.append(
            {"architecture": control_name, "init_seed": seed_num, "order_seed": seed_num, "value": float(r["control_score"])}
        )
        arm_records.append(
            {"architecture": candidate_name, "init_seed": seed_num, "order_seed": seed_num, "value": float(r["treatment_score"])}
        )
    paired_bundle = build_paired_seed_observations(
        arm_records,
        contrast_id=f"{candidate_name}_vs_{control_name}",
        control_arm=control_name,
        treatment_arm=candidate_name,
        metric_key="value",
    )
    paired_report = analyze_paired_seed_runs(paired_bundle["observations"])
    deltas = [float(r["treatment_score"]) - float(r["control_score"]) for r in paired_seed_scores]
    n_pos = sum(1 for d in deltas if d > 0.0)
    mean_gain = float(sum(deltas) / max(1, len(deltas)))
    min_delta = min(deltas) if deltas else 0.0
    seed_variance = sum((d - mean_gain) ** 2 for d in deltas) / max(1, len(deltas) - 1) if len(deltas) > 1 else 0.0
    seed_uncertainty = {
        "n_seeds": len(deltas),
        "mean_gain": round(mean_gain, 6),
        "std_dev": round(seed_variance ** 0.5, 6),
        "min_seed_delta": round(min_delta, 6),
        "max_seed_delta": round(max(deltas) if deltas else 0.0, 6),
        "per_seed_deltas": {str(r.get("seed_id", i)): round(d, 6) for i, (r, d) in enumerate(zip(paired_seed_scores, deltas))},
    }

    # Bootstrap ONLY base worlds or rule families (NEVER training-seed scalars!)
    if paired_world_or_family_scores and len(paired_world_or_family_scores) >= 2:
        if isinstance(paired_world_or_family_scores, Mapping):
            ctrl_map = {str(k): float(v[0]) for k, v in paired_world_or_family_scores.items()}
            trt_map = {str(k): float(v[1]) for k, v in paired_world_or_family_scores.items()}
        else:
            ctrl_map = {
                str(r.get("family_id", r.get("world_id", i))): float(r["control_score"])
                for i, r in enumerate(paired_world_or_family_scores)
            }
            trt_map = {
                str(r.get("family_id", r.get("world_id", i))): float(r["treatment_score"])
                for i, r in enumerate(paired_world_or_family_scores)
            }
        fam_boot: dict[str, Any] = {
            **bootstrap_paired_families(ctrl_map, trt_map, resamples=1000, seed=17),
            "unit": "base_world_or_rule_family",
            "n_units": len(ctrl_map),
        }
    else:
        fam_boot = {"status": "UNAVAILABLE_NO_WORLD_OR_FAMILY_SCORES", "unit": "none", "n_units": 0}

    effective_delivery = "PARTIAL" if missing_artifacts else delivery_status
    if stage == "development":
        qualifies = bool(
            effective_delivery == "VERIFIED"
            and execution_status == "COMPLETE"
            and mean_gain >= 0.05
            and n_pos >= 2
            and min_delta >= -0.05
            and old_macro_retention_delta >= -0.03
        )
        sci_status = "DEV_QUALIFIED" if qualifies else ("NEGATIVE_DELIVERED" if execution_status == "COMPLETE" else "UNRUN")
    else:
        qualifies = bool(
            effective_delivery == "VERIFIED"
            and execution_status == "COMPLETE"
            and mean_gain >= 0.05
            and n_pos >= 3
            and old_macro_retention_delta >= -0.03
        )
        sci_status = "CONFIRMED_REPLICATED" if qualifies else ("NEGATIVE_DELIVERED" if execution_status == "COMPLETE" else "UNRUN")

    causal_status = "MECHANISM_SUPPORTED" if (qualifies and causal_evidence_present) else "UNRESOLVED"
    raw_p = float(paired_report.get("sign_flip_p_value", paired_report.get("exact_sign_flip_p_value", 1.0)))
    holm = holm_bonferroni_adjust(all_candidate_pvalues or {candidate_name: raw_p})

    vrec = dict(verification_receipts or {})

    def _receipt_label(key: str, *, fail_if_missing_artifact: bool = False) -> str:
        if fail_if_missing_artifact and missing_artifacts:
            return f"MISSING:{','.join(missing_artifacts)}"
        if key not in vrec or vrec[key] is None:
            return "UNKNOWN_MISSING_RECEIPT"
        return "PASS" if bool(vrec[key]) else "FAIL_VERIFICATION"

    first_page = build_analyst_first_page_report(
        title=f"Elendril V8-X2: {candidate_name} vs {control_name}",
        campaign_id=campaign_id,
        source_commit=source_commit,
        decision_summary=(
            f"Candidate={candidate_name} vs Control={control_name} | mean_gain={mean_gain:+.4f} | "
            f"positive_seeds={n_pos}/{len(deltas)} | execution={execution_status} | "
            f"delivery={effective_delivery} | scientific={sci_status} | causal={causal_status}"
        ),
        control_arm=control_name,
        treatment_arm=candidate_name,
        paired_seed_bundle={**paired_bundle, "analysis": paired_report},
        retrospective_findings=[
            f"Missing mandatory artifacts: {', '.join(missing_artifacts)}"
            if missing_artifacts
            else "All declared archives present; verify explicit replay/restore receipts."
        ],
    )
    ver_matrix = build_verification_matrix(
        byte_integrity=_receipt_label("byte_integrity_verified"),
        oracle_fixture_correctness=_receipt_label("oracle_fixture_verified"),
        recorded_action_replay=_receipt_label("recorded_action_replay_verified"),
        checkpoint_restore=_receipt_label("checkpoint_restore_verified", fail_if_missing_artifact=True),
        regenerated_per_case_agreement=_receipt_label("regenerated_per_case_verified", fail_if_missing_artifact=True),
        training_reproduction="COMPLETE" if (execution_status == "COMPLETE" and vrec.get("neural_replay_verified")) else (execution_status if vrec.get("neural_replay_verified") else "UNKNOWN_MISSING_RECEIPT"),
    )
    md = render_campaign_results_markdown(
        title=f"Elendril V8-X2 Qualification: {candidate_name} vs {control_name}",
        campaign_id=campaign_id,
        source_commit=source_commit,
        status=f"{execution_status} / {effective_delivery} / {sci_status} / {causal_status}",
        verification_matrix=ver_matrix,
        stage_summaries={
            "qualification": {
                "mean_gain": round(mean_gain, 6),
                "positive_seeds": int(n_pos),
                "seed_uncertainty": seed_uncertainty,
                "missing_artifacts": list(missing_artifacts),
            }
        },
        first_page_report=first_page,
    )
    return {
        "schema": "elendril-x2-qualification-v1",
        "campaign_id": campaign_id,
        "candidate": candidate_name,
        "control": control_name,
        "stage": stage,
        "mean_gain": round(mean_gain, 6),
        "positive_seeds": int(n_pos),
        "total_seeds": len(deltas),
        "min_seed_delta": round(min_delta, 6),
        "seed_uncertainty": seed_uncertainty,
        "old_macro_retention_delta": round(float(old_macro_retention_delta), 6),
        "execution_status": execution_status,
        "delivery_status": effective_delivery,
        "scientific_status": sci_status,
        "causal_status": causal_status,
        "missing_artifacts": list(missing_artifacts),
        "paired_analysis": paired_report,
        "family_bootstrap": fam_boot,
        "holm_adjusted_p_values": holm,
        "first_page_report": first_page,
        "verification_matrix": ver_matrix,
        "markdown_report": md,
    }


__all__ = [
    "CAUSAL_STATUSES",
    "DELIVERY_STATUSES",
    "EXECUTION_STATUSES",
    "REQUIRED_TEST_CARD_SLUGS",
    "SCIENTIFIC_STATUSES",
    "X1_LEGACY_ACCEPTED_IMPORT_SHAS",
    "X1_LEGACY_COMPAT_DISCLOSURE",
    "X1_LEGACY_COMPAT_SHA",
    "X1_LEGACY_ORIGINAL_SHA",
    "X1_LEGACY_SOURCE_SHA",
    "X2_BASE_JOB_EQUIVALENT_CEILING",
    "X2_CONF_SEEDS",
    "X2_DEV_SEEDS",
    "X2_PRIMARY_ENDPOINTS",
    "X2_REGISTRY_SCHEMA",
    "build_x2_design_registry",
    "build_x2_first_discriminating_report",
    "build_x2_obligation_ledger",
    "build_x2_test_cards",
    "evaluate_confirmation_trigger",
    "evaluate_x2_completion_audit",
    "evaluate_x2_qualification",
    "holm_bonferroni_adjust",
    "is_paired_cartridge_row",
    "validate_x2_registry",
    "verify_x1_core_inventory",
]
