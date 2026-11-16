"""Versioned V8 cartridge execution with admission, recovery and delivery checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
import re
import time
import tempfile
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from anra_v5.elendril.v8.contracts import SYMBOLS, V8ModelSpec
from anra_v5.elendril.v8.models import compute_typed_nll
from anra_v5.elendril.v8.training import V8TypedTrainer, batch_tensors, create_v8_trainer

from anra_v5.elendril.training import configure_device, runtime_identity
from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v717_training import IndependentOrderTrainer
from anra_v5.elendril.v8.holdout import canonical_graph_signature

from anra_v5.elendril.publication import quarantine_interrupted_publication
from .adapters.v8_adapter import V8EnvironmentKernel, V8ObservablePolicy
from .backend import (
    PlanthirBackend,
    build_provenance_identity,
    get_backend,
)
from .causal import (
    run_compute_depth_scaling_intervention,
    run_invariance_counterfactuals,
    run_oracle_information_intervention,
    run_representation_linear_probe,
    run_subgoal_hop_decomposition,
    run_surgical_mechanism_ablation,
)
from .checkpoint_intelligence import CheckpointCatalog
from .contracts import (
    CausalContrastSpec,
    EnvironmentKernel,
    ExperimentRegistration,
    HypothesisSpec,
)
from .evidence import (
    DEFAULT_RESULT_ZIP_MAX_BYTES,
    export_result_bundle,
    package_three_tier_evidence,
    probe_from_capsule,
    publish_bundle_generation,
    quarantine_partial_generations,
    verify_result_bundle,
    verify_tier2_capsule,
)
from .instrument import (
    validate_environment_kernel,
    validate_metric_aggregator,
    validate_model_and_gradients,
)
from .interpretation import synthesize_scientific_interpretation
from .observatory import (
    InternalTelemetryCollector,
    ProcessFunnelAnalyzer,
    analyze_checkpoint_trajectory,
)
from .registry import PlanthirRegistry
from .resources import (
    ResourceMonitor,
    ResourceQuota,
    ResourceQuotaExceededError,
    cleanup_unreferenced_staging,
)


@dataclass(frozen=True, slots=True)
class CartridgeArmSpec:
    """Declarative specification for one Control or Treatment arm in an ExperimentCartridge."""

    arm_id: str
    architecture_family: str
    is_control: bool
    v8_spec: V8ModelSpec | None
    recipe: Any
    dataset_identity: str
    order_seed: int = 7
    milestone_updates: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisSpec, ...] = ()
    contrasts: tuple[CausalContrastSpec, ...] = ()
    policy_mode: str = "raw"
    b0_job: Mapping[str, Any] | None = None
    trainer_factory: Callable[..., Any] | None = None
    backend_id: str = "v8"
    model_spec: Any = None


@dataclass(slots=True)
class ExperimentCartridge:
    """Self-contained experimental cartridge consumed by the 5-Gate Planthir CartridgeRunner."""

    campaign_id: str
    program: str
    source_commit: str
    kernel: EnvironmentKernel
    control_arm: CartridgeArmSpec
    treatment_arms: tuple[CartridgeArmSpec, ...]
    train_data: Any
    eval_worlds: tuple[Mapping[str, Any], ...]
    multihop_worlds: tuple[Mapping[str, Any], ...]
    probe_train_worlds: tuple[Mapping[str, Any], ...]
    parameter_tolerance_ratio: float = 2.0
    train_worlds: tuple[Mapping[str, Any], ...] = ()
    backend_id: str = "v8"
    backend_version: str | None = None


def audit_experimental_design(
    control: CartridgeArmSpec,
    treatments: Sequence[CartridgeArmSpec],
    *,
    max_param_ratio: float = 2.0,
) -> dict[str, Any]:
    """Gate 2 Confounder Firewall: reject confounded treatment-vs-control comparisons before training."""

    if not math.isfinite(max_param_ratio) or max_param_ratio < 1:
        raise ValueError("parameter tolerance ratio must be finite and >= 1")
    comparisons, violations = [], []
    aliases = {"arch": "architecture", "learning_rate": "lr", "mechanism": "architecture"}
    c_backend = get_backend(control.backend_id)
    c_factor_cfg = c_backend.canonical_factor_config(control)
    model_factors = set(c_factor_cfg.keys()) - {"architecture", "schema", "total_parameters", "base_spec"}
    configurable = set(asdict(control.recipe)) | model_factors | {"architecture", "dataset_identity", "order_seed", "model_size"}
    for arm in treatments:
        codes = []
        if arm.backend_id != control.backend_id:
            codes.append("BACKEND_MISMATCH")
        factors = {aliases.get(c.factor_name, c.factor_name) for c in arm.contrasts}
        if not arm.contrasts or any(c.control_arm != control.arm_id or c.treatment_arm != arm.arm_id for c in arm.contrasts):
            codes.append("INVALID_OR_MISSING_CONTRAST")
        if len(factors) != 1 or not factors <= configurable:
            codes.append("UNREGISTERED_OR_MULTIPLE_FACTORS")
        varied = next(iter(factors)) if len(factors) == 1 else None
        checks = {}
        c_recipe, t_recipe = asdict(control.recipe), asdict(arm.recipe)
        names = {"seed": "SEED", "total_updates": "UPDATE_BUDGET", "batch_size": "BATCH_SIZE", "lr": "LR", "schedule": "SCHEDULE"}
        for key in sorted(set(c_recipe) | set(t_recipe)):
            equal = c_recipe.get(key) == t_recipe.get(key)
            checks[key] = equal
            if not equal and varied != key:
                codes.append(names.get(key, key.upper()) + "_MISMATCH")
        for key, label in (("dataset_identity", "DATASET"), ("order_seed", "ORDER_SEED"), ("policy_mode", "POLICY_MODE")):
            equal = getattr(control, key) == getattr(arm, key)
            checks[key] = equal
            if not equal and varied != key:
                codes.append(label + "_MISMATCH")
        t_backend = get_backend(arm.backend_id)
        c_spec = c_backend.resolve_model_spec(control)
        t_spec = t_backend.resolve_model_spec(arm)
        cs = c_backend.canonical_model_spec(c_spec)
        ts = t_backend.canonical_model_spec(t_spec)
        cm = c_factor_cfg
        tm = t_backend.canonical_factor_config(arm)
        if varied in model_factors:
            if any(cm.get(k) != tm.get(k) for k in set(cm) | set(tm) if k != varied):
                codes.append("MODEL_SPEC_MISMATCH")
        elif cs != ts and varied not in {"architecture", "model_size"}:
            codes.append("MODEL_SPEC_MISMATCH")
        if varied == "model_size" and any(cm.get(k) != tm.get(k) for k in set(cm) | set(tm) if k != "base_spec"):
            codes.append("MODEL_SPEC_MISMATCH")
        for fixed in ("max_sequence_length", "vocabulary_size", "vocab_size"):
            if fixed in cm and fixed in tm and cm[fixed] != tm[fixed] and varied != fixed:
                label = "VOCABULARY_SIZE" if fixed == "vocab_size" else fixed.upper()
                codes.append(label + "_MISMATCH")
        if "base_spec" in cs and "base_spec" in ts and cs["base_spec"] != ts["base_spec"] and varied != "model_size":
            codes.append("BACKBONE_MISMATCH")
        if control.trainer_factory is not arm.trainer_factory:
            codes.append("TRAINER_IMPLEMENTATION_MISMATCH")
        cp = c_backend.model_parameter_count(c_spec)
        tp = t_backend.model_parameter_count(t_spec)
        ratio = tp / cp
        param_ok = 1 / max_param_ratio <= ratio <= max_param_ratio
        if not param_ok:
            codes.append("PARAM_SCALE_CONFOUND")
        rec = {"control_arm": control.arm_id, "treatment_arm": arm.arm_id,
               "registered_factor": varied, "field_equality": checks,
               "parameter_count_control": cp, "parameter_count_treatment": tp,
               "parameter_ratio": ratio, "parameter_tolerance_ratio": max_param_ratio,
               "param_ok": param_ok, "violations": codes}
        comparisons.append(rec)
        if codes:
            violations.append({"control_arm": control.arm_id, "treatment_arm": arm.arm_id, "codes": codes})
    return {"schema": "anra-planthir-gate2-confounder-firewall-v2",
            "gate": "gate2_confounder_firewall", "passed": not violations,
            "control_arm": control.arm_id, "arms_checked": len(treatments),
            "comparisons": comparisons, "violations": violations,
            "scope": "Declared recipe/spec equality and explicit factor, not proof of equal FLOPs or causal identification."}


def estimate_token_fuel_savings(
    *,
    arm_count: int,
    milestone_count: int,
    raw_summary_bytes: int,
    handoff_brief_bytes: int,
) -> dict[str, Any]:
    """Quantify AI authoring + diagnostic context compression ("rocket fuel savings")."""

    if min(arm_count, milestone_count, raw_summary_bytes, handoff_brief_bytes) < 0:
        raise ValueError("counts and byte sizes must be nonnegative")
    return {
        "schema": "anra-planthir-token-fuel-report-v2",
        "arms_evaluated": int(arm_count), "milestones_tracked": int(milestone_count),
        "baseline_from_scratch_tokens_est": None,
        "cartridge_workflow_tokens_est": None,
        "estimated_token_reduction_pct": None,
        "authoring_savings_status": "NOT_MEASURED",
        "raw_diagnostic_json_bytes": int(raw_summary_bytes),
        "handoff_brief_bytes": int(handoff_brief_bytes),
        "handoff_brief_tokens_est": max(1, int(handoff_brief_bytes) // 4),
        "token_estimate_method": "bytes/4 heuristic, not measured model tokens",
        "context_handoff_compression_ratio": (round(raw_summary_bytes / handoff_brief_bytes, 2) if handoff_brief_bytes else None),
    }


def build_ai_handoff_brief(
    *,
    campaign_id: str,
    source_commit: str,
    device: str,
    gates: Mapping[str, Any],
    arm_results: Mapping[str, Mapping[str, Any]],
    bundle_receipt: Mapping[str, Any] | None = None,
    token_fuel_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a compact (<8 KB) machine-readable handoff brief for follow-up AI experiment design."""

    arm_digests: list[dict[str, Any]] = []
    top_recs: list[dict[str, Any]] = []
    for arm_id, res in arm_results.items():
        funnel = res.get("process_funnel", {}).get("funnel", {})
        cd = res.get("causal_diagnostics", {})
        interp = res.get("interpretation", {})
        sa = cd.get("surgical_ablation", {})
        cds = cd.get("compute_depth_scaling", {})
        arm_digests.append({
            "arm": arm_id,
            "architecture": res.get("architecture"),
            "params": res.get("parameter_count"),
            "loss_init": res.get("loss_init"),
            "loss_final": res.get("loss_final"),
            "overall_acc": res.get("process_funnel", {}).get("overall_accuracy"),
            "stage1_info": funnel.get("stage1_info_seeking"),
            "stage3_1hop": funnel.get("stage3_single_hop_grounding"),
            "stage4_mhop": funnel.get("stage4_multi_hop_composition"),
            "oracle_delta": (cd.get("oracle_intervention") or {}).get("delta_oracle"),
            "probe_final": (cd.get("linear_probe") or {}).get("probe_final_symbol_accuracy"),
            "ablation_delta": sa.get("accuracy_delta"),
            "ornamental_bypass": sa.get("ornamental_mechanism_bypass"),
            "compute_depth_gain": cds.get("recurrence_gain_over_zero"),
            "confirmed_causes": [c["claim_id"] for c in interp.get("confirmed_causes", [])],
            "suspected_mechanisms": [c["claim_id"] for c in interp.get("suspected_mechanisms", [])],
            "checkpoint_sha256": res.get("checkpoint_sha256"),
        })
        for rec in interp.get("recommended_next_experiments", []):
            top_recs.append({"arm": arm_id, **rec})

    top_recs.sort(key=lambda r: float(r.get("expected_info_gain_per_gpu_hour", 0.0)), reverse=True)
    brief = {
        "schema": "anra-planthir-ai-handoff-brief-v1",
        "campaign_id": campaign_id,
        "source_commit": source_commit,
        "device": device,
        "all_5_gates_passed": bool(gates.get("all_5_gates_passed", False)),
        "gate_statuses": {
            k: bool(v.get("passed", False))
            for k, v in gates.items()
            if isinstance(v, Mapping)
        },
        "arms": arm_digests,
        "top_recommended_next_experiments": top_recs[:3],
        "bundle_summary": {
            "result_zip_bytes": (bundle_receipt or {}).get("result_zip_bytes"),
            "checkpoint_full_zip_bytes": (bundle_receipt or {}).get("checkpoint_full_zip_bytes"),
            "whole_checkpoint_embedded": (bundle_receipt or {}).get("whole_checkpoint_embedded"),
        },
        "token_fuel_report": dict(token_fuel_report or {}),
    }
    return brief


def _collect_capsule_arrays(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    eval_worlds: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    obs_list = []
    y_op = []
    y_hop1 = []
    y_final = []
    op_labels = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")

    for w in eval_worlds:
        vis_w = kernel.inject_oracle_information(w) if hasattr(kernel, "inject_oracle_information") else w
        st = kernel.initial_state(vis_w)
        obs = kernel.observe(st)
        obs_list.append(obs)
        ideal_op, _ = kernel.ideal_action(st)
        y_op.append(op_labels.index(ideal_op))
        exp = w.get("hidden", {}).get("expected")
        y_final.append(SYMBOLS.index(exp) if (isinstance(exp, str) and exp in SYMBOLS) else -1)
        sgs = {sg.stage_name: sg for sg in kernel.decompose_subgoals(vis_w)}
        h1_arg = sgs["hop_1"].expected_argument if "hop_1" in sgs else exp
        y_hop1.append(SYMBOLS.index(h1_arg) if (isinstance(h1_arg, str) and h1_arg in SYMBOLS) else -1)

    reps = policy.extract_representations(obs_list)
    arrays: dict[str, torch.Tensor] = {
        "decision_hidden": reps["decision_hidden"],
        "y_op": torch.tensor(y_op, dtype=torch.int64),
        "y_hop1": torch.tensor(y_hop1, dtype=torch.int64),
        "y_final": torch.tensor(y_final, dtype=torch.int64),
    }
    if "pointer_gate" in reps:
        arrays["pointer_gate"] = reps["pointer_gate"]
    return arrays


def _validate_cartridge(
    cartridge: ExperimentCartridge,
    device: str,
    cpu_canary: bool,
    *,
    checkpoint_interval: int = 64,
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
) -> tuple[dict[str, Any], PlanthirBackend]:
    """Validate backend contracts, splits, factors, and provenance before execution."""

    from .causal import _probe_split_keys

    def _assert_disjoint_probe_worlds(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> None:
        li, lg = _probe_split_keys(left)
        ri, rg = _probe_split_keys(right)
        if li & ri or lg & rg:
            raise ValueError("training/probe and evaluation worlds must be structurally disjoint")

    backend = get_backend(cartridge.backend_id, expected_version=cartridge.backend_version)
    backend.validate_kernel(cartridge.kernel)
    if not re.fullmatch(r"[0-9a-f]{40}", cartridge.source_commit):
        raise ValueError("source_commit must be a 40-character immutable Git SHA")
    if not cartridge.eval_worlds or not cartridge.probe_train_worlds:
        raise ValueError("nonempty evaluation and disjoint probe-training worlds are required")
    arms = (cartridge.control_arm, *cartridge.treatment_arms)
    ids = [a.arm_id for a in arms]
    if len(set(ids)) != len(ids) or not cartridge.control_arm.is_control or any(a.is_control for a in cartridge.treatment_arms):
        raise ValueError("unique arm IDs and exactly one designated control required")
    for name in [cartridge.campaign_id, *ids]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,70}", name):
            raise ValueError("campaign/arm IDs must be safe path components")
    for arm in arms:
        if arm.backend_id != cartridge.backend_id:
            raise ValueError(f"arm {arm.arm_id!r} backend {arm.backend_id!r} != cartridge backend {cartridge.backend_id!r}")
        backend.validate_arm(arm, cartridge.train_data, device=device, cpu_canary=cpu_canary)

    ds_info = backend.validate_dataset_and_worlds(cartridge.train_data, cartridge.train_worlds)
    train_worlds = ds_info["resolved_train_worlds"]
    for evaluation in (cartridge.eval_worlds, cartridge.multihop_worlds):
        if evaluation:
            _assert_disjoint_probe_worlds(train_worlds, evaluation)
            _assert_disjoint_probe_worlds(cartridge.probe_train_worlds, evaluation)

    gate2 = audit_experimental_design(
        cartridge.control_arm,
        cartridge.treatment_arms,
        max_param_ratio=cartridge.parameter_tolerance_ratio,
    )
    if not gate2["passed"]:
        raise ValueError(f"Gate 2 rejected design: {gate2['violations']}")

    dataset_digest = {k: v for k, v in ds_info.items() if k != "resolved_train_worlds"}
    configure_device(device)
    provenance = build_provenance_identity(
        cartridge=cartridge,
        backend=backend,
        train_worlds=train_worlds,
        resolved_dataset_digest=dataset_digest,
        device=device,
        checkpoint_interval=checkpoint_interval,
        max_result_zip_bytes=max_result_zip_bytes,
    )
    return provenance, backend


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("campaign deadline reached; committed checkpoints and partial receipts preserved")


class CartridgeRunner:
    """Backend-abstracted Planthir execution runner with admission, recovery, and delivery checks."""

    def run(
        self,
        cartridge: ExperimentCartridge,
        *,
        work_root: Path,
        bundle_out_dir: Path | None = None,
        device: str = "cpu",
        cpu_canary: bool = True,
        max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
        resume: bool = False,
        checkpoint_interval: int = 64,
        max_wall_seconds: float | None = None,
        resource_quota: ResourceQuota | None = None,
    ) -> dict[str, Any]:
        from .execution import _atomic_write, run_session

        if type(checkpoint_interval) is not int or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
            raise ValueError("max_wall_seconds must be positive and finite")
        provenance, backend = _validate_cartridge(
            cartridge,
            device,
            cpu_canary,
            checkpoint_interval=checkpoint_interval,
            max_result_zip_bytes=max_result_zip_bytes,
        )
        if torch.device(device).type == "cuda":
            repo = Path(__file__).resolve().parents[2]
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()
            if head != cartridge.source_commit or dirty:
                raise ValueError("production source must be a clean checkout of source_commit")
        deadline = None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
        with run_session(Path(work_root), provenance, resume=resume) as session:
            _atomic_write(
                Path(work_root) / "PROVENANCE_IDENTITY.json",
                (json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            )
            checkpoint_bytes = sum(
                backend.model_parameter_count(backend.resolve_model_spec(a))
                * 16
                * (len(set(a.milestone_updates) | {a.recipe.total_updates}) + 2)
                for a in (cartridge.control_arm, *cartridge.treatment_arms)
            )
            required_bytes = checkpoint_bytes * 3 + 512 * 1024 * 1024
            free_bytes = shutil.disk_usage(work_root).free
            session["disk_admission"] = {
                "required_additional_bytes": required_bytes,
                "free_bytes": free_bytes,
                "scope": "conservative serialization estimate, not a platform quota guarantee",
            }
            if free_bytes < required_bytes:
                raise OSError("insufficient free disk for checkpoints and verified staged exports")
            return self._run(
                cartridge,
                backend=backend,
                work_root=work_root,
                bundle_out_dir=bundle_out_dir,
                device=device,
                cpu_canary=cpu_canary,
                max_result_zip_bytes=max_result_zip_bytes,
                resume=resume,
                checkpoint_interval=checkpoint_interval,
                deadline=deadline,
                session=session,
                resource_quota=resource_quota,
            )

    def _run(
        self,
        cartridge: ExperimentCartridge,
        *,
        backend: PlanthirBackend | None = None,
        work_root: Path,
        bundle_out_dir: Path | None = None,
        device: str = "cpu",
        cpu_canary: bool = True,
        max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
        resume: bool = False, checkpoint_interval: int = 64,
        deadline: float | None = None, session: dict | None = None,
        resource_quota: ResourceQuota | None = None,
    ) -> dict[str, Any]:
        from .contracts import ObservablePolicy
        from .execution import append_update_log, read_update_log, _atomic_write, trigger_fault_point
        def atomic_write_json(path, data):
            _atomic_write(path, (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode())
        active_backend = backend or get_backend(cartridge.backend_id, expected_version=cartridge.backend_version)
        work = Path(work_root)
        work.mkdir(parents=True, exist_ok=True)
        monitor = ResourceMonitor(work_root=work, device=device, quota=resource_quota)
        run_resource_status = "FAILED"
        try:
            return self._run_monitored(
                cartridge,
                active_backend=active_backend,
                work=work,
                bundle_out_dir=bundle_out_dir,
                device=device,
                cpu_canary=cpu_canary,
                max_result_zip_bytes=max_result_zip_bytes,
                resume=resume,
                checkpoint_interval=checkpoint_interval,
                deadline=deadline,
                session=session if session is not None else {},
                monitor=monitor,
                atomic_write_json=atomic_write_json,
                append_update_log=append_update_log,
                read_update_log=read_update_log,
                trigger_fault_point=trigger_fault_point,
            )
        except ResourceQuotaExceededError:
            run_resource_status = "QUOTA_EXCEEDED"
            cleanup_unreferenced_staging(work)
            raise
        except OSError:
            cleanup_unreferenced_staging(work)
            raise
        finally:
            try:
                if (work / "FINAL_RECEIPT.json").is_file() and run_resource_status != "QUOTA_EXCEEDED":
                    run_resource_status = "COMPLETE"
            except Exception:
                pass
            monitor.write_receipt(status=run_resource_status)

    def _run_monitored(
        self,
        cartridge: ExperimentCartridge,
        *,
        active_backend: PlanthirBackend,
        work: Path,
        bundle_out_dir: Path | None,
        device: str,
        cpu_canary: bool,
        max_result_zip_bytes: int,
        resume: bool,
        checkpoint_interval: int,
        deadline: float | None,
        session: dict,
        monitor: ResourceMonitor,
        atomic_write_json: Any,
        append_update_log: Any,
        read_update_log: Any,
        trigger_fault_point: Any,
    ) -> dict[str, Any]:
        ckpt_root = work / "checkpoints"
        evidence_root = work / "evidence_bundles"
        bundle_dir = Path(bundle_out_dir) if bundle_out_dir is not None else (work / "result_bundle")

        all_arms: tuple[CartridgeArmSpec, ...] = (cartridge.control_arm, *cartridge.treatment_arms)
        registry = PlanthirRegistry()

        session["phase"] = "instrument"
        _check_deadline(deadline)
        pre_phase = monitor.enter_phase("gate1_and_gate2_pre_run")
        # Gate 1: Instrument Integrity
        kernel_check = validate_environment_kernel(
            cartridge.kernel,
            list(cartridge.eval_worlds[:8]),
            check_v8_evaluator_parity=isinstance(cartridge.kernel, V8EnvironmentKernel),
            v8_enforce_tool_consumption=True,
        )
        agg_check = validate_metric_aggregator(
            lambda items: {
                "final_epoch": max(items, key=lambda pair: int(pair[1]["epoch"]))[1]["epoch"],
                "final_accuracy": max(items, key=lambda pair: int(pair[1]["epoch"]))[1]["accuracy"],
            }
        )
        gate1 = {
            "gate": "gate1_instrument_integrity",
            "passed": bool(kernel_check["passed"] and agg_check["passed"]),
            "kernel_id": cartridge.kernel.kernel_id,
            "checks": {"kernel": kernel_check["checks"], "aggregator": agg_check},
            "defects": [*kernel_check["defects"], *agg_check["defects"]],
        }
        if not gate1["passed"]:
            raise RuntimeError(f"Gate 1 Instrument Integrity failed: {gate1['defects']}")

        # Gate 2: Confounder Firewall
        gate2 = audit_experimental_design(
            cartridge.control_arm,
            cartridge.treatment_arms,
            max_param_ratio=cartridge.parameter_tolerance_ratio,
        )
        if not gate2["passed"]:
            raise ValueError(f"Gate 2 Confounder Firewall rejected design: {gate2['violations']}")
        monitor.exit_phase(pre_phase)

        # Gate 3: Gradient Health & Bounded Training with Milestones
        train_phase = monitor.enter_phase("training_and_gate3")
        eval_worlds = list(cartridge.eval_worlds)
        multihop_worlds = list(cartridge.multihop_worlds)
        probe_train_worlds = list(cartridge.probe_train_worlds)
        parity_obs = [cartridge.kernel.observe(cartridge.kernel.initial_state(w)) for w in eval_worlds[:12]]

        arm_training_records: dict[str, dict[str, Any]] = {}
        gate3_arms: dict[str, Any] = {}
        dev_obj = torch.device(device)

        for arm in all_arms:
            session["phase"] = "train:" + arm.arm_id
            _check_deadline(deadline)
            arm_spec = active_backend.resolve_model_spec(arm)
            param_count = int(active_backend.model_parameter_count(arm_spec))
            exp_id = f"{cartridge.campaign_id}-{arm.arm_id}"
            lineage_id = f"{cartridge.campaign_id.lower()}-{arm.arm_id}-s{arm.order_seed}"
            reg = ExperimentRegistration(
                experiment_id=exp_id,
                program=cartridge.program,
                architecture_family=arm.architecture_family,
                parameter_count=param_count,
                dataset_identity=arm.dataset_identity,
                source_commit=cartridge.source_commit,
                hypotheses=arm.hypotheses,
                contrasts=arm.contrasts,
                budget={
                    "device": device,
                    "updates": int(arm.recipe.total_updates),
                    "batch_size": int(arm.recipe.batch_size),
                    "params": param_count,
                },
                status="RUNNING",
            )
            registry.register(reg)
            registry.save(work / "REGISTRY.json")
            arm_state_path = work / "arm_state" / (arm.arm_id + ".json")
            journal_path = work / "updates" / (arm.arm_id + ".jsonl")
            initial_record = json.loads(arm_state_path.read_text()) if arm_state_path.exists() else None
            lineage_dir = ckpt_root / lineage_id
            arm_resume = resume and (lineage_dir / "LATEST").is_file()
            if resume and lineage_dir.is_dir():
                for left_gen in sorted(p for p in lineage_dir.iterdir() if p.name.startswith(".staging-")):
                    try:
                        gen_num = int(left_gen.name.split("-", 1)[1])
                        quarantine_interrupted_publication(lineage_dir, gen_num)
                    except (ValueError, IndexError):
                        pass

            trainer = active_backend.create_trainer(
                arm,
                cartridge.train_data,
                source_commit=cartridge.source_commit,
                root=ckpt_root,
                lineage=lineage_id,
                device=device,
                cpu_canary=cpu_canary,
                resume=arm_resume,
            )
            if asdict(trainer.recipe) != asdict(arm.recipe):
                raise ValueError("actual trainer recipe differs from preregistration")
            step_logs = read_update_log(journal_path, trainer.state.global_update)
            if arm_resume and initial_record is None:
                raise ValueError("resume missing original gradient/initial telemetry receipt")

            # Pre-update gradient flow verification
            if initial_record is not None:
                grad_check = initial_record["grad_check"]
            else:
                loss_fn = active_backend.sample_gradient_loss_fn(trainer, arm, cartridge.train_data, dev_obj)
                grad_check = validate_model_and_gradients(trainer.model, loss_fn)

            if not grad_check["passed"]:
                raise RuntimeError(f"Gate 3 gradient check failed for {arm.arm_id}: {grad_check}")

            telemetry_probe_obs = [cartridge.kernel.observe(cartridge.kernel.initial_state(w)) for w in eval_worlds[:8]]
            telemetry_col = InternalTelemetryCollector()
            init_policy = active_backend.create_policy(
                trainer.model,
                arm_spec,
                model_identity=f"init-{lineage_id}",
                mode=arm.policy_mode,
            )
            snap_init = initial_record["snap_init"] if initial_record is not None else telemetry_col.measure_snapshot(
                trainer.model,
                step=0,
                lr=float(arm.recipe.lr),
                policy=init_policy,
                probe_observations=telemetry_probe_obs,
            )
            if initial_record is None:
                atomic_write_json(arm_state_path, {"grad_check": grad_check, "snap_init": snap_init})
            saved_milestones: list[dict[str, Any]] = []
            total_u = int(arm.recipe.total_updates)
            milestone_set = {int(u) for u in arm.milestone_updates if 1 <= int(u) <= total_u}
            milestone_set.add(total_u)

            latest_sha = trainer.store.latest_sha256() or ""
            # Repair the crash window between atomic checkpoint commit and milestone binding.
            if latest_sha and trainer.state.global_update in milestone_set:
                trainer.store.record_milestone(threshold_tokens=int(trainer.state.cumulative_tokens), checkpoint_sha256=latest_sha)
            for step_idx in range(trainer.state.global_update, total_u):
                if deadline is not None and time.monotonic() >= deadline:
                    if trainer.state.global_update > trainer.published_update:
                        trainer.save()
                    _check_deadline(deadline)
                step_info = trainer.step()
                record = {"update": int(step_info["update"]), "loss": float(step_info["loss"]),
                          "pre_clip": float(step_info["pre_clip"])}
                if record["update"] != step_idx + 1 or not all(math.isfinite(record[k]) for k in ("loss", "pre_clip")):
                    raise ValueError("nonfinite or out-of-sequence training update")
                trigger_fault_point(f"before_journal_append:{arm.arm_id}:u{record['update']}")
                append_update_log(journal_path, record)
                trigger_fault_point("after_journal_append")
                trigger_fault_point(f"after_journal_append:{arm.arm_id}:u{record['update']}")
                step_logs.append(record)
                monitor.check_quotas(phase=f"training:{arm.arm_id}:u{record['update']}")
                cur_u = step_idx + 1
                if cur_u in milestone_set or cur_u % checkpoint_interval == 0:
                    once_file = os.environ.get("PLANTHIR_FAULT_ONCE_FILE")
                    already_fired = bool(once_file and Path(once_file).exists())
                    fp_env = os.environ.get("PLANTHIR_FAULT_INJECTION_POINT", "")
                    inject_ckpt = None
                    if not already_fired:
                        inject_ckpt = os.environ.get("PLANTHIR_CHECKPOINT_INJECT_CRASH") or (
                            fp_env if fp_env.startswith("after_component:") else None
                        )
                    try:
                        latest_sha = trainer.save(inject_crash_at=inject_ckpt)
                    except BaseException:
                        if inject_ckpt and (fp_env == inject_ckpt or "PLANTHIR_FAULT_EXIT_CODE" in os.environ):
                            trigger_fault_point(inject_ckpt)
                            os._exit(int(os.environ.get("PLANTHIR_FAULT_EXIT_CODE", "99")))
                        raise
                    trigger_fault_point("after_checkpoint_commit")
                    trigger_fault_point(f"after_checkpoint_commit:{arm.arm_id}:u{cur_u}")
                    if cur_u in milestone_set:
                        trainer.store.record_milestone(threshold_tokens=int(trainer.state.cumulative_tokens), checkpoint_sha256=latest_sha)
                    trainer.store.prune(keep=set())
            for rec in CheckpointCatalog.discover_checkpoints([ckpt_root], include_all_generations=True).records:
                if rec.lineage_id == lineage_id and int(rec.global_update) in milestone_set:
                    saved_milestones.append({"global_update": int(rec.global_update),
                        "checkpoint_sha256": rec.checkpoint_sha256})
            saved_milestones.sort(key=lambda r: r["global_update"])
            if {r["global_update"] for r in saved_milestones} != milestone_set:
                raise ValueError("required milestone tensor checkpoint is missing")
            if len(step_logs) != total_u:
                raise ValueError("committed training journal does not cover full horizon")

            # Capture live trained model actions & hidden states for Gate 5 parity self-audit
            live_policy = active_backend.create_policy(
                trainer.model,
                arm_spec,
                model_identity=f"live-{lineage_id}",
                mode=arm.policy_mode,
            )
            snap_final = telemetry_col.measure_snapshot(
                trainer.model,
                step=total_u,
                lr=float(arm.recipe.lr),
                policy=live_policy,
                probe_observations=telemetry_probe_obs,
            )
            live_actions = [
                (a.operation, a.argument)
                for a in (live_policy.propose_with_telemetry(obs) for obs in parity_obs)
            ]
            live_hidden = live_policy.extract_representations(parity_obs)["decision_hidden"].cpu()

            arm_training_records[arm.arm_id] = {
                "exp_id": exp_id,
                "lineage_id": lineage_id,
                "latest_sha": latest_sha,
                "milestones": saved_milestones,
                "param_count": param_count,
                "reg": reg,
                "grad_check": grad_check,
                "step_logs": step_logs,
                "snap_init": snap_init or snap_final,
                "snap_final": snap_final,
                "live_actions": live_actions,
                "live_hidden": live_hidden,
            }
            gate3_arms[arm.arm_id] = {
                "passed": bool(grad_check["passed"]),
                "updates": total_u,
                "milestones_saved": len(saved_milestones),
                "loss_init": step_logs[0]["loss"],
                "loss_final": step_logs[-1]["loss"],
                "latest_checkpoint_sha256": latest_sha,
            }
            del init_policy, live_policy, trainer
            if dev_obj.type == "cuda":
                torch.cuda.empty_cache()

        gate3 = {
            "gate": "gate3_gradient_and_mechanism_health",
            "passed": all(v["passed"] for v in gate3_arms.values()),
            "arms": gate3_arms,
        }
        monitor.exit_phase(train_phase)

        # Gate 4: Diagnostic execution; this is not automatic causal attribution.
        eval_phase = monitor.enter_phase("evaluation_and_observatory")
        catalog = CheckpointCatalog.discover_checkpoints([ckpt_root], include_all_generations=True)
        funnel_analyzer = ProcessFunnelAnalyzer()
        arm_results: dict[str, Any] = {}
        arm_bundles: dict[str, dict[str, Any]] = {}
        latest_recs: dict[str, Any] = {}

        for arm in all_arms:
            tr_info = arm_training_records[arm.arm_id]
            exp_id = tr_info["exp_id"]
            lineage_id = tr_info["lineage_id"]
            latest_sha = tr_info["latest_sha"]
            reg = tr_info["reg"]
            arm_spec = active_backend.resolve_model_spec(arm)

            session["phase"] = "analyze:" + arm.arm_id
            _check_deadline(deadline)
            arm_recs = sorted(
                [r for r in catalog.records if r.lineage_id == lineage_id and int(r.global_update) in (set(arm.milestone_updates) | {arm.recipe.total_updates})],
                key=lambda r: (int(r.global_update), str(r.checkpoint_sha256)),
            )
            latest_matched = [r for r in arm_recs if r.checkpoint_sha256 == latest_sha]
            if len(latest_matched) != 1:
                raise RuntimeError(f"CheckpointCatalog failed to discover latest {arm.arm_id} checkpoint {latest_sha}")
            latest_rec = latest_matched[0]
            latest_recs[arm.arm_id] = latest_rec

            trajectory: dict[str, Any] | None = None
            if len(arm_recs) >= 2:
                trajectory = analyze_checkpoint_trajectory(
                    arm_recs,
                    load_fn=lambda rec, **kwargs: CheckpointCatalog.load_checkpoint_for_causal_probing(rec, device=device, **kwargs),
                    kernel=cartridge.kernel,
                    eval_worlds=eval_worlds,
                    probe_train_worlds=probe_train_worlds,
                    probe_fn=run_representation_linear_probe,
                )

            restored_raw = CheckpointCatalog.load_checkpoint_for_causal_probing(
                latest_rec, mode=arm.policy_mode, device=device
            )
            policy_raw = restored_raw["policy"]
            policy_masked = active_backend.create_policy(
                policy_raw.model,
                arm_spec,
                model_identity=latest_sha,
                mode="availability-mask",
            )

            funnel = funnel_analyzer.evaluate_suite(policy_raw, cartridge.kernel, eval_worlds, include_subgoals=True)
            oracle = (
                run_oracle_information_intervention(policy_raw, cartridge.kernel, eval_worlds)
                if hasattr(cartridge.kernel, "inject_oracle_information")
                else None
            )
            subgoal = run_subgoal_hop_decomposition(policy_raw, cartridge.kernel, multihop_worlds)
            inv = run_invariance_counterfactuals(
                policy_raw,
                cartridge.kernel,
                eval_worlds[:10],
                mode_policies={"raw": policy_raw, "availability-mask": policy_masked},
            )
            lin_probe = run_representation_linear_probe(
                policy_raw, cartridge.kernel, probe_train_worlds, eval_worlds, l2_reg=1.0
            )
            surgical = run_surgical_mechanism_ablation(policy_raw, cartridge.kernel, eval_worlds)
            depth_scaling = run_compute_depth_scaling_intervention(policy_raw, cartridge.kernel, eval_worlds)

            arm_inst = {
                "passed": bool(gate1["passed"] and tr_info["grad_check"]["passed"]),
                "status": "PASS",
                "kernel_id": cartridge.kernel.kernel_id,
                "checks": {**gate1["checks"], "grad_health": tr_info["grad_check"]},
                "defects": [],
            }
            falsifier_metrics = {
                "stage1_tool_recall_when_needed": funnel["funnel"]["stage1_tool_recall_when_needed"],
                "probe_final_symbol_accuracy": lin_probe.get("probe_final_symbol_accuracy"),
                "overall_accuracy": funnel["overall_accuracy"],
            }
            interp = synthesize_scientific_interpretation(
                experiment_id=exp_id,
                instrument_validation=arm_inst,
                process_funnel=funnel,
                oracle_intervention=oracle,
                subgoal_decomposition=subgoal,
                invariance_counterfactuals=inv,
                linear_probe=lin_probe,
                surgical_ablation=surgical,
                compute_depth_scaling=depth_scaling,
                learning_trajectory=trajectory,
                internal_telemetry=tr_info["snap_final"],
                hypotheses=reg.hypotheses,
                observed_metrics_for_falsifiers=falsifier_metrics,
            )

            probe_arrays = _collect_capsule_arrays(policy_raw, cartridge.kernel, eval_worlds)
            bundle = package_three_tier_evidence(
                evidence_root / arm.arm_id,
                registration=reg,
                instrument_validation=arm_inst,
                process_funnel=funnel,
                causal_diagnostics={
                    "oracle_intervention": oracle,
                    "subgoal_decomposition": subgoal,
                    "invariance_counterfactuals": inv,
                    "linear_probe": lin_probe,
                    "surgical_ablation": surgical,
                    "compute_depth_scaling": depth_scaling,
                    "learning_trajectory": trajectory,
                },
                interpretation=interp,
                probe_arrays=probe_arrays,
                learning_telemetry=[tr_info["snap_init"], tr_info["snap_final"]],
                checkpoint_root=ckpt_root,
                checkpoint_lineage=lineage_id,
            )
            verify_tier2_capsule(evidence_root / arm.arm_id / "tier2_capsule")
            offline_probe = probe_from_capsule(evidence_root / arm.arm_id / "tier2_capsule")
            registry.update_status(exp_id, interp.status, result_ref=latest_sha)
            registry.save(work / "REGISTRY.json")

            compact_funnel = {k: v for k, v in funnel.items() if k != "traces"}
            arm_results[arm.arm_id] = {
                "experiment_id": exp_id,
                "architecture": arm.architecture_family,
                "is_control": arm.is_control,
                "parameter_count": tr_info["param_count"],
                "checkpoint_sha256": latest_sha,
                "milestones": tr_info["milestones"],
                "loss_init": tr_info["step_logs"][0]["loss"],
                "loss_final": tr_info["step_logs"][-1]["loss"],
                "telemetry_init": tr_info["snap_init"],
                "telemetry_final": tr_info["snap_final"],
                "process_funnel": compact_funnel,
                "causal_diagnostics": {
                    "oracle_intervention": oracle,
                    "subgoal_decomposition": subgoal,
                    "invariance_counterfactuals": inv,
                    "linear_probe": lin_probe,
                    "surgical_ablation": surgical,
                    "compute_depth_scaling": depth_scaling,
                    "learning_trajectory": trajectory,
                },
                "offline_tier2_capsule_probe": offline_probe,
                "tier2_capsule_bytes": bundle["tier2_capsule"]["total_bytes"],
                "interpretation": interp.to_dict(),
            }
            arm_bundles[exp_id] = {
                **bundle,
                "lineage_id": lineage_id,
                "checkpoint_sha256": latest_sha,
            }
            del policy_raw, policy_masked, restored_raw
            if dev_obj.type == 'cuda':
                torch.cuda.empty_cache()

        checkpoint_diffs = {
            f"{arm.arm_id}_vs_{cartridge.control_arm.arm_id}": CheckpointCatalog.diff_checkpoints(
                latest_recs[cartridge.control_arm.arm_id],
                latest_recs[arm.arm_id],
            )
            for arm in cartridge.treatment_arms
        }
        gate4 = {
            "gate": "gate4_trajectory_and_causal_attribution",
            "passed": len(arm_results) == len(all_arms),
            "scope": "diagnostic execution and coverage, not scientific confirmation",
            "arms_analyzed": len(arm_results),
            "single_architecture_contrasts": {
                k: v["single_architecture_contrast"] for k, v in checkpoint_diffs.items()
            },
        }

        monitor.exit_phase(eval_phase)
        session["phase"] = "export"
        _check_deadline(deadline)
        pub_phase = monitor.enter_phase("bundle_publication")
        if resume:
            quarantine_partial_generations(bundle_dir)
        # Gate 5: Bundle Export + Live-vs-Restored Parity Self-Audit
        # If CURRENT.json already commits a COMPLETE generation, keep it isolated while staging Gate 5.
        existing_pointer = bundle_dir / "CURRENT.json"
        has_committed_complete = False
        if existing_pointer.is_file():
            try:
                has_committed_complete = json.loads(existing_pointer.read_text(encoding="utf-8")).get("status") == "COMPLETE"
            except Exception:
                has_committed_complete = False
        with tempfile.TemporaryDirectory(prefix="planthir-verify-") as restore_tmp:
            if has_committed_complete:
                gate5_stage_dir = Path(restore_tmp) / "gate5_stage"
                initial_receipt = export_result_bundle(
                    gate5_stage_dir,
                    campaign_id=cartridge.campaign_id,
                    source_commit=cartridge.source_commit,
                    status="PARTIAL",
                    experiment_bundles=arm_bundles,
                    checkpoint_root=ckpt_root,
                    max_result_zip_bytes=max_result_zip_bytes,
                )
            else:
                initial_receipt = publish_bundle_generation(
                    bundle_dir,
                    campaign_id=cartridge.campaign_id,
                    source_commit=cartridge.source_commit,
                    status="PARTIAL",
                    experiment_bundles=arm_bundles,
                    checkpoint_root=ckpt_root,
                    max_result_zip_bytes=max_result_zip_bytes,
                    export_fn=export_result_bundle,
                    verify_fn=None,
                    enable_fault_injection=False,
                )
            restored_store = CheckpointCatalog.restore_full_checkpoint_zip(
                initial_receipt["checkpoint_full_zip"],
                Path(restore_tmp) / "store",
                expected_source_commit=cartridge.source_commit,
            )

            parity_by_arm: dict[str, Any] = {}
            all_parity_ok = True
            for arm in all_arms:
                tr_info = arm_training_records[arm.arm_id]
                candidates = [r for r in restored_store["catalog"].records
                              if r.lineage_id == tr_info["lineage_id"] and r.checkpoint_sha256 == tr_info["latest_sha"]]
                if len(candidates) != 1:
                    raise ValueError("full export missing expected exact checkpoint")
                loaded_zip = CheckpointCatalog.load_checkpoint_for_causal_probing(
                    candidates[0], mode=arm.policy_mode, device=device)
                loaded_zip["source_archive"] = "checkpoint_full.zip (exact weights)"
                z_policy = loaded_zip["policy"]
                zip_actions = [
                    (a.operation, a.argument)
                    for a in (z_policy.propose_with_telemetry(obs) for obs in parity_obs)
                ]
                zip_hidden = z_policy.extract_representations(parity_obs)["decision_hidden"].cpu()

                live_actions = tr_info["live_actions"]
                live_hidden = tr_info["live_hidden"]
                matches = sum(1 for a, b in zip(live_actions, zip_actions) if a == b)
                parity_rate = float(matches / len(live_actions))
                max_diff = float((live_hidden - zip_hidden).abs().max().item())
                diff_tol = 0.0
                arm_ok = bool(parity_rate == 1.0 and max_diff <= diff_tol)
                if not arm_ok:
                    all_parity_ok = False
                parity_by_arm[arm.arm_id] = {
                    "passed": arm_ok,
                    "source_archive": loaded_zip["source_archive"],
                    "prediction_parity_rate": parity_rate,
                    "max_hidden_abs_diff": max_diff,
                    "worlds_checked": len(parity_obs),
                }
                # Do not retain one GPU model while allocating the next arm.
                del z_policy, loaded_zip
                if dev_obj.type == "cuda":
                    torch.cuda.empty_cache()

            gate5 = {
                "gate": "gate5_bundle_and_live_parity_self_audit",
                "passed": bool(all_parity_ok and restored_store["verified"]),
                "scope": "same-device exact full weights; inference parity is not optimizer resume qualification",
                "verified_full_lineages": len(restored_store["verified_lineages"]),
                "parity_by_arm": parity_by_arm,
            }
            if not gate5["passed"]:
                raise RuntimeError(f"Gate 5 Bundle/Parity Self-Audit failed: {gate5}")

        gates = {
            "gate1_instrument_integrity": gate1,
            "gate2_confounder_firewall": gate2,
            "gate3_gradient_and_mechanism_health": gate3,
            "gate4_trajectory_and_causal_attribution": gate4,
            "gate5_bundle_and_live_parity_self_audit": gate5,
            "all_5_gates_passed": bool(
                gate1["passed"] and gate2["passed"] and gate3["passed"] and gate4["passed"] and gate5["passed"]
            ),
        }

        raw_arm_bytes = len(json.dumps(arm_results, sort_keys=True, default=str).encode("utf-8"))
        # Build token-minimal AI handoff brief and embed both certificate & brief inside final ZIPs
        milestone_count = sum(len(v["milestones"]) for v in arm_results.values())
        prelim_brief = build_ai_handoff_brief(
            campaign_id=cartridge.campaign_id,
            source_commit=cartridge.source_commit,
            device=device,
            gates=gates,
            arm_results=arm_results,
            bundle_receipt=initial_receipt,
            token_fuel_report={},
        )
        prelim_len = len(json.dumps(prelim_brief, indent=2, sort_keys=True).encode("utf-8"))
        fuel_report = estimate_token_fuel_savings(
            arm_count=len(arm_results),
            milestone_count=milestone_count,
            raw_summary_bytes=raw_arm_bytes,
            handoff_brief_bytes=prelim_len,
        )
        final_brief = build_ai_handoff_brief(
            campaign_id=cartridge.campaign_id,
            source_commit=cartridge.source_commit,
            device=device,
            gates=gates,
            arm_results=arm_results,
            bundle_receipt=initial_receipt,
            token_fuel_report=fuel_report,
        )
        brief_bytes = (json.dumps(final_brief, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        # Avoid stale first-export size claims and self-inconsistent returned/embedded briefs.
        final_brief["bundle_summary"] = {"final_sizes_and_hashes": "external FINAL_RECEIPT.json", "exact_restore_source": "checkpoint_full.zip"}
        final_brief['total_arms'] = len(final_brief['arms'])
        while len(json.dumps(final_brief, indent=2).encode()) > 7000 and final_brief['top_recommended_next_experiments']:
            final_brief['top_recommended_next_experiments'].pop()
        while len(json.dumps(final_brief, indent=2).encode()) > 7000 and final_brief['arms']:
            final_brief['arms'].pop()
        final_brief['arms_omitted_from_brief'] = final_brief['total_arms'] - len(final_brief['arms'])
        final_brief['full_reports'] = 'per-experiment tier1 summaries in result.zip; ARM_RESULTS.json in work directory'
        for _ in range(20):
            brief_bytes = (json.dumps(final_brief, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            fuel_report = estimate_token_fuel_savings(arm_count=len(arm_results), milestone_count=milestone_count,
                raw_summary_bytes=raw_arm_bytes, handoff_brief_bytes=len(brief_bytes))
            if final_brief['token_fuel_report'] == fuel_report:
                break
            final_brief['token_fuel_report'] = fuel_report
        else:
            raise ValueError('handoff size accounting failed to converge')
        if len(brief_bytes) > 8192:
            raise ValueError('handoff brief exceeds bounded context limit')
        cert_bytes = (json.dumps(gates, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

        final_brief = json.loads(brief_bytes)
        # Verify final archives via generation staging. An unsuccessful verification leaves the PARTIAL bundle intact.
        final_receipt = publish_bundle_generation(
            bundle_dir,
            campaign_id=cartridge.campaign_id,
            source_commit=cartridge.source_commit,
            status="COMPLETE",
            experiment_bundles=arm_bundles,
            checkpoint_root=ckpt_root,
            max_result_zip_bytes=max_result_zip_bytes,
            extra_root_files={
                "AI_HANDOFF_BRIEF.json": brief_bytes,
                "LAUNCH_GATE_CERTIFICATE.json": cert_bytes,
            },
            export_fn=export_result_bundle,
            verify_fn=verify_result_bundle,
            enable_fault_injection=True,
        )
        monitor.exit_phase(pub_phase)
        resource_receipt = monitor.receipt(status="COMPLETE")
        atomic_write_json(work / "ARM_RESULTS.json", arm_results)
        atomic_write_json(work / "FINAL_RECEIPT.json", final_receipt)
        session["phase"] = "verified_complete"
        session["bundle_receipt"] = str(work / "FINAL_RECEIPT.json")

        return {
            "schema": "anra-planthir-cartridge-run-v2",
            "campaign_id": cartridge.campaign_id,
            "device": device,
            "gates": gates,
            "checkpoint_diffs": checkpoint_diffs,
            "arms": arm_results,
            "registry": registry.export_dict(),
            "bundle_receipt": final_receipt,
            "resource_receipt": resource_receipt,
            "ai_handoff_brief": final_brief,
            "token_fuel_report": fuel_report,
        }


__all__ = [
    "CartridgeArmSpec",
    "CartridgeRunner",
    "ExperimentCartridge",
    "audit_experimental_design",
    "build_ai_handoff_brief",
    "estimate_token_fuel_savings",
]
