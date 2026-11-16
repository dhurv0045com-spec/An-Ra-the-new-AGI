"""Read-only inspection, error explanation, and data-only declarative `anra-planthir-spec-v1` tools."""

from __future__ import annotations

from dataclasses import MISSING, asdict, fields, replace
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import V8ModelSpec, build_v8_model_spec
from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset
from anra_v5.elendril.v8.training import prepare_v8_training_data

from .adapters.v8_adapter import V8EnvironmentKernel
from .adapters.x1_rd2_adapter import (
    X1RD2EnvironmentKernel,
    X1RD2Recipe,
    X1RD2Spec,
    prepare_x1_rd2_training_data,
)
from .backend import get_backend, list_backends
from .cartridge import (
    CartridgeArmSpec,
    CartridgeRunner,
    ExperimentCartridge,
    audit_experimental_design,
)
from .causal import _probe_split_keys
from .contracts import CausalContrastSpec, FalsifierSpec, HypothesisSpec
from .design import (
    TypedContrastSpec,
    TypedFactorSpec,
    audit_typed_experimental_design,
    validate_cohort_and_split_transitivity,
)
from .evidence import DEFAULT_RESULT_ZIP_MAX_BYTES
from .resources import ResourceQuota

SPEC_SCHEMA_VERSION = "anra-planthir-spec-v1"
LEGACY_GPU_ERROR = "legacy_launcher_not_qualified"
LEGACY_STUDY_CLI_ERROR = "legacy_launcher_deprecated"

_ROOT_KEYS = {
    "schema",
    "campaign_id",
    "program",
    "source_commit",
    "backend_id",
    "backend_version",
    "dataset",
    "control_arm",
    "treatment_arms",
    "factors",
    "typed_contrasts",
    "hypotheses",
    "falsifiers",
    "execution",
}
_DATASET_KEYS = {
    "dataset_name",
    "d1_worlds_per_family",
    "num_groups",
    "sample_limit_for_cpu",
    "eval_world_count",
    "multihop_world_count",
    "probe_world_count",
    "seed_offset",
}
_ARM_KEYS = {
    "arm_id",
    "architecture",
    "model",
    "recipe",
    "milestone_updates",
    "policy_mode",
    "order_seed",
    "contrasts",
    "custom_factors",
}
_MODEL_KEYS = {
    "width",
    "hidden_dim",
    "layers",
    "num_layers",
    "query_heads",
    "num_heads",
    "kv_heads",
    "head_dimension",
    "ffn_width",
    "context_length",
    "max_sequence_length",
    "pointer_dim",
    "num_slots",
    "slot_dim",
    "slot_edge_hidden",
    "agg_iterations",
    "msg_iterations",
    "cap_hidden_dim",
}
_RECIPE_KEYS = {
    "seed",
    "total_updates",
    "batch_size",
    "micro_batch_size",
    "lr",
    "clip",
    "objective",
    "sampling",
    "schedule",
    "precision",
    "eq_weight",
    "cf_weight",
    "cf_margin",
}
_CONTRAST_KEYS = {
    "contrast_id",
    "factor",
    "factor_name",
    "control_arm",
    "control_arm_id",
    "treatment_arm",
    "treatment_arm_id",
    "held_constant",
    "target_mechanism",
    "expected_mechanism_signature",
}
_TYPED_FACTOR_KEYS = {"factor_name", "factor_type", "control_value", "levels"}
_TYPED_CONTRAST_KEYS = {
    "contrast_id",
    "factor_name",
    "control_arm_id",
    "treatment_arm_id",
    "target_mechanism",
    "held_constant",
    "expected_treatment_value",
    "primary_endpoint",
    "expected_direction",
}
_EXECUTION_KEYS = {
    "device",
    "cpu_canary",
    "checkpoint_interval",
    "max_result_zip_bytes",
    "parameter_tolerance_ratio",
    "max_wall_seconds",
    "max_total_disk_bytes",
    "min_free_disk_bytes",
    "max_cpu_rss_bytes",
    "max_cuda_allocated_bytes",
}

_FORBIDDEN_CODE_PATTERN = re.compile(
    r"(__import__|eval\s*\(|exec\s*\(|lambda\s|subprocess\.|os\.system|&&|\|\||\$\(|`)"
)
_DOTTED_IMPORT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+(:[A-Za-z_][A-Za-z0-9_]*)?$")


class InspectionError(Exception):
    def __init__(self, file: str, detail: str) -> None:
        super().__init__(detail)
        self.file = file
        self.detail = detail


def _parse_json_strict(path: Path) -> Any:
    if not path.is_file():
        raise InspectionError(path.name, f"missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-JSON constant {c}")))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InspectionError(path.name, str(exc)) from exc


def _fields(cls: type) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in fields(cls):
        default = f.default if f.default is not MISSING else (f.default_factory() if f.default_factory is not MISSING else None)
        rows.append(
            {
                "name": f.name,
                "required": f.default is MISSING and f.default_factory is MISSING,
                "default": repr(default) if default is not None and not isinstance(default, (str, int, float, bool, list, dict, tuple)) else default,
            }
        )
    return rows


def _repo_head_commit() -> str:
    try:
        repo = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "0" * 40


def _scan_safe_data(obj: Any, path: str, errors: list[dict[str, Any]]) -> None:
    if obj is None or isinstance(obj, (bool, int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            errors.append({"code": "NONFINITE_NUMBER", "field": path, "detail": "float values must be finite"})
        return
    if isinstance(obj, str):
        if _FORBIDDEN_CODE_PATTERN.search(obj) or ";" in obj:
            errors.append({"code": "FORBIDDEN_CODE_OR_SHELL_FRAGMENT", "field": path, "detail": f"unsafe string fragment rejected in {path}"})
        elif _DOTTED_IMPORT_PATTERN.match(obj):
            errors.append({"code": "FORBIDDEN_DOTTED_IMPORT", "field": path, "detail": f"dotted Python import reference rejected in data-only spec: {obj!r}"})
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _scan_safe_data(item, f"{path}[{i}]", errors)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                errors.append({"code": "INVALID_KEY_TYPE", "field": path, "detail": "mapping keys must be strings"})
                continue
            _scan_safe_data(v, f"{path}.{k}" if path else k, errors)
        return
    errors.append({"code": "UNSUPPORTED_VALUE_TYPE", "field": path, "detail": f"unsupported type {type(obj).__name__} in JSON spec"})


def _check_closed_keys(mapping: Any, allowed: set[str], path: str, errors: list[dict[str, Any]]) -> bool:
    if not isinstance(mapping, Mapping):
        errors.append({"code": "EXPECTED_OBJECT", "field": path, "detail": f"expected object at {path}"})
        return False
    unknown = sorted(set(str(k) for k in mapping) - allowed)
    if unknown:
        errors.append({"code": "UNKNOWN_FIELD", "field": f"{path}.{unknown[0]}" if path else unknown[0], "detail": f"unknown fields rejected: {unknown}"})
        return False
    return True


def _build_arm_components(
    arm_doc: Mapping[str, Any],
    *,
    backend_id: str,
    is_control: bool,
    dataset_identity: str,
    arm_path: str,
    errors: list[dict[str, Any]],
) -> CartridgeArmSpec | None:
    if not _check_closed_keys(arm_doc, _ARM_KEYS, arm_path, errors):
        return None
    arm_id = arm_doc.get("arm_id")
    arch = arm_doc.get("architecture")
    if not isinstance(arm_id, str) or not arm_id:
        errors.append({"code": "MISSING_FIELD", "field": f"{arm_path}.arm_id", "detail": "non-empty arm_id required"})
        return None
    if not isinstance(arch, str) or not arch:
        errors.append({"code": "MISSING_FIELD", "field": f"{arm_path}.architecture", "detail": "non-empty architecture required"})
        return None

    m_doc = arm_doc.get("model", {})
    r_doc = arm_doc.get("recipe", {})
    if not _check_closed_keys(m_doc, _MODEL_KEYS, f"{arm_path}.model", errors):
        return None
    if not _check_closed_keys(r_doc, _RECIPE_KEYS, f"{arm_path}.recipe", errors):
        return None

    contrasts_list: list[CausalContrastSpec] = []
    for idx, c_raw in enumerate(arm_doc.get("contrasts", [])):
        if not _check_closed_keys(c_raw, _CONTRAST_KEYS, f"{arm_path}.contrasts[{idx}]", errors):
            continue
        try:
            contrasts_list.append(
                CausalContrastSpec(
                    contrast_id=str(c_raw["contrast_id"]),
                    factor_name=str(c_raw.get("factor_name", c_raw.get("factor", ""))),
                    control_arm=str(c_raw.get("control_arm", c_raw.get("control_arm_id", ""))),
                    treatment_arm=str(c_raw.get("treatment_arm", c_raw.get("treatment_arm_id", ""))),
                    held_constant=tuple(str(x) for x in c_raw.get("held_constant", [])),
                    expected_mechanism_signature=str(
                        c_raw.get("expected_mechanism_signature", c_raw.get("target_mechanism", ""))
                    ),
                )
            )
        except Exception as exc:
            errors.append({"code": "INVALID_CONTRAST", "field": f"{arm_path}.contrasts[{idx}]", "detail": str(exc)})

    custom_factors = arm_doc.get("custom_factors")
    if custom_factors is not None and not isinstance(custom_factors, Mapping):
        errors.append({"code": "INVALID_CUSTOM_FACTORS", "field": f"{arm_path}.custom_factors", "detail": "must be an object"})
        return None

    try:
        milestones = tuple(int(u) for u in arm_doc.get("milestone_updates", []))
        policy_mode = str(arm_doc.get("policy_mode", "raw"))
        order_seed = int(arm_doc.get("order_seed", 7))

        if backend_id == "v8":
            width = int(m_doc.get("width", m_doc.get("hidden_dim", 32)))
            layers = int(m_doc.get("layers", m_doc.get("num_layers", 1)))
            qheads = int(m_doc.get("query_heads", m_doc.get("num_heads", 2)))
            kvheads = int(m_doc.get("kv_heads", 1))
            hdim = int(m_doc.get("head_dimension", max(1, width // max(qheads, 1))))
            ffn = int(m_doc.get("ffn_width", width * 2))
            ctx = int(m_doc.get("context_length", m_doc.get("max_sequence_length", 512)))
            base = replace(
                MODEL_PROFILES["S20"],
                width=width,
                layers=layers,
                query_heads=qheads,
                kv_heads=kvheads,
                head_dimension=hdim,
                ffn_width=ffn,
                context_length=ctx,
            )
            v8_spec = build_v8_model_spec(arch, base_spec=base)
            recipe = Recipe(
                seed=int(r_doc.get("seed", 7)),
                total_updates=int(r_doc.get("total_updates", 2)),
                batch_size=int(r_doc.get("batch_size", 4)),
                micro_batch_size=int(r_doc.get("micro_batch_size", 2)),
                lr=float(r_doc.get("lr", 1e-4)),
                clip=float(r_doc.get("clip", 1.0)),
                objective=str(r_doc.get("objective", "ce")),
                sampling=str(r_doc.get("sampling", "uniform")),
                schedule=str(r_doc.get("schedule", "cosine")),
                precision=str(r_doc.get("precision", "fp32")),
            )
            if not milestones:
                milestones = (recipe.total_updates,)
            return CartridgeArmSpec(
                arm_id=arm_id,
                architecture_family=arch,
                is_control=is_control,
                v8_spec=v8_spec,
                recipe=recipe,
                dataset_identity=dataset_identity,
                milestone_updates=milestones,
                policy_mode=policy_mode,
                order_seed=order_seed,
                contrasts=tuple(contrasts_list),
                backend_id="v8",
                model_spec=v8_spec,
            )

        if backend_id == "x1_rd2":
            x1_spec = X1RD2Spec(
                architecture=arch,
                hidden_dim=int(m_doc.get("hidden_dim", m_doc.get("width", 16))),
                num_layers=int(m_doc.get("num_layers", m_doc.get("layers", 1))),
                num_heads=int(m_doc.get("num_heads", m_doc.get("query_heads", 2))),
                pointer_dim=int(m_doc.get("pointer_dim", 8)),
                num_slots=int(m_doc.get("num_slots", 2)),
                slot_dim=int(m_doc.get("slot_dim", 8)),
                slot_edge_hidden=int(m_doc.get("slot_edge_hidden", 8)),
                agg_iterations=int(m_doc.get("agg_iterations", 2)),
                msg_iterations=int(m_doc.get("msg_iterations", 2)),
                cap_hidden_dim=int(m_doc.get("cap_hidden_dim", 16)),
                max_sequence_length=int(m_doc.get("max_sequence_length", m_doc.get("context_length", 512))),
            )
            x1_recipe = X1RD2Recipe(
                seed=int(r_doc.get("seed", 7)),
                total_updates=int(r_doc.get("total_updates", 2)),
                batch_size=int(r_doc.get("batch_size", 8)),
                micro_batch_size=int(r_doc.get("micro_batch_size", 4)),
                lr=float(r_doc.get("lr", 1e-3)),
                clip=float(r_doc.get("clip", 1.0)),
                objective=str(r_doc.get("objective", "pair_eq")),
                eq_weight=float(r_doc.get("eq_weight", 0.1)),
                cf_weight=float(r_doc.get("cf_weight", 0.1)),
                cf_margin=float(r_doc.get("cf_margin", 0.5)),
                sampling=str(r_doc.get("sampling", "uniform")),
                schedule=str(r_doc.get("schedule", "cosine")),
                precision=str(r_doc.get("precision", "fp32")),
            )
            if not milestones:
                milestones = (x1_recipe.total_updates,)
            return CartridgeArmSpec(
                arm_id=arm_id,
                architecture_family=arch,
                is_control=is_control,
                v8_spec=None,
                recipe=x1_recipe,
                dataset_identity=dataset_identity,
                milestone_updates=milestones,
                policy_mode=policy_mode,
                order_seed=order_seed,
                contrasts=tuple(contrasts_list),
                backend_id="x1_rd2",
                model_spec=x1_spec,
            )

        errors.append({"code": "UNSUPPORTED_BACKEND", "field": "backend_id", "detail": f"unsupported backend_id {backend_id!r}"})
        return None
    except Exception as exc:
        errors.append({"code": "INVALID_ARM_SPEC", "field": arm_path, "detail": str(exc)})
        return None


def planthir_spec_schema() -> dict[str, Any]:
    """Return machine-readable JSON schema & registered backend capabilities for agents."""

    backends_doc: dict[str, Any] = {}
    for bid in list_backends():
        be = get_backend(bid)
        caps = be.capabilities()
        backends_doc[bid] = {
            "backend_version": be.backend_version,
            "supported_architectures": list(caps.supported_architectures),
            "supported_objectives": list(caps.supported_objectives),
            "supported_samplings": list(caps.supported_samplings),
            "max_cpu_canary_params": caps.max_cpu_canary_params,
            "max_cpu_canary_samples": caps.max_cpu_canary_samples,
            "max_cpu_canary_updates": caps.max_cpu_canary_updates,
        }
    return {
        "schema": SPEC_SCHEMA_VERSION,
        "data_only_rules": {
            "unknown_fields": "REJECTED",
            "python_code_or_eval": "REJECTED",
            "dotted_import_strings": "REJECTED",
            "shell_fragments": "REJECTED",
        },
        "backends": backends_doc,
        "allowed_fields": {
            "root": sorted(_ROOT_KEYS),
            "dataset": sorted(_DATASET_KEYS),
            "arm": sorted(_ARM_KEYS),
            "model": sorted(_MODEL_KEYS),
            "recipe": sorted(_RECIPE_KEYS),
            "contrast": sorted(_CONTRAST_KEYS),
            "typed_factor": sorted(_TYPED_FACTOR_KEYS),
            "typed_contrast": sorted(_TYPED_CONTRAST_KEYS),
            "execution": sorted(_EXECUTION_KEYS),
        },
        "cli_subcommands": [
            "schema",
            "init",
            "validate",
            "run",
            "resume",
            "inspect-run",
            "verify-bundle",
            "replay-bundle",
            "explain-error",
            "agent-info",
            "calibrate",
            "historical-atlas",
        ],
    }


def init_spec_template(
    *,
    backend_id: str = "x1_rd2",
    canary: bool = True,
) -> dict[str, Any]:
    """Generate a valid starter `anra-planthir-spec-v1` JSON template ready for `validate` and `run`."""

    commit = _repo_head_commit()
    if backend_id == "v8":
        return {
            "schema": SPEC_SCHEMA_VERSION,
            "campaign_id": "V8-CANARY-TEMPLATE" if canary else "V8-CAMPAIGN",
            "program": "Elendril-V8",
            "source_commit": commit,
            "backend_id": "v8",
            "dataset": {
                "dataset_name": "D1",
                "d1_worlds_per_family": 16,
                "sample_limit_for_cpu": 8,
                "eval_world_count": 4,
                "multihop_world_count": 2,
                "probe_world_count": 4,
                "seed_offset": 7,
            },
            "control_arm": {
                "arm_id": "CTRL",
                "architecture": "J",
                "model": {"width": 32, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 16, "ffn_width": 64, "context_length": 512},
                "recipe": {"seed": 7, "total_updates": 2, "batch_size": 4, "micro_batch_size": 2, "lr": 1e-4, "objective": "ce", "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
                "milestone_updates": [1, 2],
                "order_seed": 7,
            },
            "treatment_arms": [
                {
                    "arm_id": "TRT_LR",
                    "architecture": "J",
                    "model": {"width": 32, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 16, "ffn_width": 64, "context_length": 512},
                    "recipe": {"seed": 7, "total_updates": 2, "batch_size": 4, "micro_batch_size": 2, "lr": 2e-4, "objective": "ce", "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
                    "milestone_updates": [1, 2],
                    "order_seed": 7,
                    "contrasts": [
                        {
                            "contrast_id": "C_LR",
                            "factor_name": "lr",
                            "control_arm": "CTRL",
                            "treatment_arm": "TRT_LR",
                            "held_constant": ["architecture", "dataset_identity", "precision", "order_seed"],
                            "expected_mechanism_signature": "learning-rate-sensitivity",
                        }
                    ],
                }
            ],
            "factors": [{"factor_name": "lr", "factor_type": "lr", "control_value": 1e-4, "levels": [1e-4, 2e-4]}],
            "typed_contrasts": [
                {
                    "contrast_id": "C_LR",
                    "factor_name": "lr",
                    "control_arm_id": "CTRL",
                    "treatment_arm_id": "TRT_LR",
                    "target_mechanism": "learning-rate-sensitivity",
                    "expected_treatment_value": 2e-4,
                }
            ],
            "execution": {
                "device": "cpu",
                "cpu_canary": bool(canary),
                "checkpoint_interval": 1,
                "parameter_tolerance_ratio": 1.05,
            },
        }

    if backend_id == "x1_rd2":
        return {
            "schema": SPEC_SCHEMA_VERSION,
            "campaign_id": "X1-RD2-CANARY-TEMPLATE" if canary else "X1-RD2-CAMPAIGN",
            "program": "X1-RD2",
            "source_commit": commit,
            "backend_id": "x1_rd2",
            "dataset": {
                "dataset_name": "X1-D2",
                "num_groups": 4,
                "eval_world_count": 4,
                "multihop_world_count": 2,
                "probe_world_count": 4,
                "seed_offset": 19,
            },
            "control_arm": {
                "arm_id": "X1P",
                "architecture": "X1-P",
                "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "pointer_dim": 8, "max_sequence_length": 512},
                "recipe": {"seed": 7, "total_updates": 2, "batch_size": 8, "micro_batch_size": 4, "lr": 1e-3, "objective": "pair_eq", "eq_weight": 0.1, "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
                "milestone_updates": [1, 2],
                "order_seed": 7,
            },
            "treatment_arms": [
                {
                    "arm_id": "X1R",
                    "architecture": "X1-R",
                    "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2, "pointer_dim": 8, "num_slots": 2, "slot_dim": 8, "slot_edge_hidden": 8, "max_sequence_length": 512},
                    "recipe": {"seed": 7, "total_updates": 2, "batch_size": 8, "micro_batch_size": 4, "lr": 1e-3, "objective": "pair_eq", "eq_weight": 0.1, "sampling": "uniform", "schedule": "cosine", "precision": "fp32"},
                    "milestone_updates": [1, 2],
                    "order_seed": 7,
                    "contrasts": [
                        {
                            "contrast_id": "C_R_VS_P",
                            "factor_name": "architecture",
                            "control_arm": "X1P",
                            "treatment_arm": "X1R",
                            "held_constant": ["lr", "objective", "precision", "order_seed"],
                            "expected_mechanism_signature": "x1_relational_slot_readout",
                        }
                    ],
                }
            ],
            "factors": [{"factor_name": "architecture", "factor_type": "architecture", "control_value": "X1-P", "levels": ["X1-P", "X1-R"]}],
            "typed_contrasts": [
                {
                    "contrast_id": "C_R_VS_P",
                    "factor_name": "architecture",
                    "control_arm_id": "X1P",
                    "treatment_arm_id": "X1R",
                    "target_mechanism": "x1_relational_slot_readout",
                    "expected_treatment_value": "X1-R",
                }
            ],
            "execution": {
                "device": "cpu",
                "cpu_canary": bool(canary),
                "checkpoint_interval": 1,
                "parameter_tolerance_ratio": 2.0,
            },
        }

    raise ValueError(f"unsupported template backend_id: {backend_id!r}")


def _validate_and_compile(
    doc: Mapping[str, Any],
    *,
    materialize_data: bool,
    canary_override: bool | None = None,
) -> tuple[dict[str, Any], ExperimentCartridge | None, dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(doc, Mapping):
        report = {"schema": "anra-planthir-spec-validation-v1", "valid": False, "status": "INVALID_SPEC", "errors": [{"code": "EXPECTED_OBJECT", "field": "", "detail": "spec root must be a JSON object"}]}
        return report, None, {}

    _scan_safe_data(dict(doc), "", errors)
    _check_closed_keys(doc, _ROOT_KEYS, "", errors)

    if doc.get("schema") != SPEC_SCHEMA_VERSION:
        errors.append({"code": "UNSUPPORTED_SCHEMA", "field": "schema", "detail": f"expected {SPEC_SCHEMA_VERSION!r}"})

    backend_id = str(doc.get("backend_id", ""))
    if backend_id not in list_backends():
        errors.append({"code": "UNSUPPORTED_BACKEND", "field": "backend_id", "detail": f"backend {backend_id!r} not in {list_backends()}"})

    ds_doc = doc.get("dataset", {})
    _check_closed_keys(ds_doc, _DATASET_KEYS, "dataset", errors)
    ex_doc = doc.get("execution", {})
    _check_closed_keys(ex_doc, _EXECUTION_KEYS, "execution", errors)

    for fn in ("campaign_id", "program", "source_commit"):
        v = doc.get(fn)
        if not isinstance(v, str) or not v.strip():
            errors.append({"code": "MISSING_FIELD", "field": fn, "detail": f"{fn} must be a non-empty string"})

    if errors:
        return {"schema": "anra-planthir-spec-validation-v1", "valid": False, "status": "INVALID_SPEC", "backend_id": backend_id, "errors": errors}, None, {}

    backend = get_backend(backend_id, expected_version=doc.get("backend_version"))
    caps = backend.capabilities()
    device = str(ex_doc.get("device", "cpu"))
    cpu_canary = bool(canary_override if canary_override is not None else ex_doc.get("cpu_canary", True))
    max_zip_bytes = int(ex_doc.get("max_result_zip_bytes", DEFAULT_RESULT_ZIP_MAX_BYTES))
    ckpt_interval = int(ex_doc.get("checkpoint_interval", 1))
    param_tol = float(ex_doc.get("parameter_tolerance_ratio", 1.05))
    max_wall = ex_doc.get("max_wall_seconds")

    # Materialize dataset identity (required for both full design check and execution)
    seed_offset = int(ds_doc.get("seed_offset", 7))
    eval_count = int(ds_doc.get("eval_world_count", 4))
    mh_count = int(ds_doc.get("multihop_world_count", 2))
    probe_count = int(ds_doc.get("probe_world_count", 4))
    d1_quota = int(ds_doc.get("d1_worlds_per_family", 16))
    num_groups = int(ds_doc.get("num_groups", 4))
    cpu_limit = int(ds_doc.get("sample_limit_for_cpu", 8))

    if backend_id == "v8":
        train_data = prepare_v8_training_data(
            str(ds_doc.get("dataset_name", "D1")),
            d1_worlds_per_family=d1_quota,
            sample_limit_for_cpu=cpu_limit,
        )
        train_worlds = build_d1_training_dataset(quota_per_family=max(16, d1_quota)).worlds
        kernel = V8EnvironmentKernel()
    else:
        train_data, train_worlds = prepare_x1_rd2_training_data(num_groups=num_groups)
        kernel = X1RD2EnvironmentKernel()

    ctrl_arm = _build_arm_components(
        doc.get("control_arm", {}),
        backend_id=backend_id,
        is_control=True,
        dataset_identity=train_data.data_identity,
        arm_path="control_arm",
        errors=errors,
    )
    trt_arms: list[CartridgeArmSpec] = []
    for idx, t_raw in enumerate(doc.get("treatment_arms", [])):
        arm_obj = _build_arm_components(
            t_raw,
            backend_id=backend_id,
            is_control=False,
            dataset_identity=train_data.data_identity,
            arm_path=f"treatment_arms[{idx}]",
            errors=errors,
        )
        if arm_obj is not None:
            trt_arms.append(arm_obj)

    if ctrl_arm is None or errors:
        return {"schema": "anra-planthir-spec-validation-v1", "valid": False, "status": "INVALID_SPEC", "backend_id": backend_id, "errors": errors}, None, {}

    # Check backend arm capability limits (architecture, objective, CPU canary envelope)
    for arm in (ctrl_arm, *trt_arms):
        try:
            backend.validate_arm(arm, train_data, device=device, cpu_canary=cpu_canary)
        except Exception as exc:
            errors.append({"code": "BACKEND_ADMISSION_FAILED", "field": f"arm:{arm.arm_id}", "detail": str(exc)})

    # Run Confounder Firewall (Gate 2) preflight audit before accepting spec
    design_audit = audit_experimental_design(ctrl_arm, tuple(trt_arms), max_param_ratio=param_tol)
    if not design_audit["passed"]:
        for v in design_audit["violations"]:
            codes = v.get("codes", ["CONFOUNDER_FIREWALL_FAILED"])
            errors.append({"code": str(codes[0]), "field": f"arm:{v.get('arm_id', 'treatment')}", "detail": str(v), "all_codes": codes})

    # Optional strict typed factors & typed contrasts audit
    fspecs: list[TypedFactorSpec] = []
    for idx, f_raw in enumerate(doc.get("factors", [])):
        if not _check_closed_keys(f_raw, _TYPED_FACTOR_KEYS, f"factors[{idx}]", errors):
            continue
        try:
            fspecs.append(
                TypedFactorSpec(
                    factor_name=str(f_raw["factor_name"]),
                    factor_type=str(f_raw["factor_type"]),
                    control_value=f_raw["control_value"],
                    levels=tuple(f_raw["levels"]),
                )
            )
        except Exception as exc:
            errors.append({"code": "INVALID_TYPED_FACTOR", "field": f"factors[{idx}]", "detail": str(exc)})

    tcons: list[TypedContrastSpec] = []
    for idx, tc_raw in enumerate(doc.get("typed_contrasts", [])):
        if not _check_closed_keys(tc_raw, _TYPED_CONTRAST_KEYS, f"typed_contrasts[{idx}]", errors):
            continue
        try:
            tcons.append(
                TypedContrastSpec(
                    contrast_id=str(tc_raw["contrast_id"]),
                    factor_name=str(tc_raw["factor_name"]),
                    control_arm_id=str(tc_raw["control_arm_id"]),
                    treatment_arm_id=str(tc_raw["treatment_arm_id"]),
                    target_mechanism=str(tc_raw["target_mechanism"]),
                    held_constant=tuple(str(x) for x in tc_raw.get("held_constant", [])),
                    expected_treatment_value=tc_raw.get("expected_treatment_value"),
                    primary_endpoint=str(tc_raw.get("primary_endpoint", "accuracy")),
                    expected_direction=str(tc_raw.get("expected_direction", "greater")),
                )
            )
        except Exception as exc:
            errors.append({"code": "INVALID_TYPED_CONTRAST", "field": f"typed_contrasts[{idx}]", "detail": str(exc)})

    if (fspecs or tcons) and not errors:
        typed_audit = audit_typed_experimental_design(ctrl_arm, tuple(trt_arms), factors=tuple(fspecs), typed_contrasts=tuple(tcons))
        if not typed_audit["passed"]:
            for v in typed_audit["violations"]:
                errors.append({"code": str(v["code"]), "field": "typed_design", "detail": str(v)})

    quota = ResourceQuota(
        max_wall_seconds=float(max_wall) if max_wall is not None else None,
        max_total_disk_bytes=ex_doc.get("max_total_disk_bytes"),
        min_free_disk_bytes=ex_doc.get("min_free_disk_bytes"),
        max_cpu_rss_bytes=ex_doc.get("max_cpu_rss_bytes"),
        max_cuda_allocated_bytes=ex_doc.get("max_cuda_allocated_bytes"),
    )

    seen_ids, seen_graphs = _probe_split_keys(train_worlds)
    pool1 = build_d1_training_dataset(seed=D1_TRAIN_SEED + seed_offset, quota_per_family=max(16, d1_quota)).worlds
    pool2 = build_d1_training_dataset(seed=D1_TRAIN_SEED + seed_offset + 53, quota_per_family=16).worlds

    eval_list: list[dict[str, Any]] = []
    mh_list: list[dict[str, Any]] = []
    probe_list: list[dict[str, Any]] = []
    for w in (*pool1, *pool2):
        if int(w["renderer_template"]) != 0:
            continue
        wi, wg = _probe_split_keys([w])
        if (wi & seen_ids) or (wg & seen_graphs):
            continue
        if len(eval_list) < eval_count:
            eval_list.append(w)
        elif len(mh_list) < mh_count:
            mh_list.append(w)
        elif len(probe_list) < probe_count:
            probe_list.append(w)
        else:
            break
        seen_ids |= wi
        seen_graphs |= wg

    if len(eval_list) < eval_count or len(mh_list) < mh_count or len(probe_list) < probe_count:
        errors.append(
            {
                "code": "INSUFFICIENT_DISJOINT_WORLDS",
                "field": "dataset",
                "detail": (
                    f"disjoint split selection short: eval={len(eval_list)}/{eval_count}, "
                    f"multihop={len(mh_list)}/{mh_count}, probe={len(probe_list)}/{probe_count}"
                ),
            }
        )

    if errors:
        return {
            "schema": "anra-planthir-spec-validation-v1",
            "valid": False,
            "status": "INVALID_SPEC",
            "backend_id": backend_id,
            "errors": errors,
            "design_audit": design_audit,
        }, None, {}

    eval_w = tuple(eval_list)
    mh_w = tuple(mh_list)
    probe_w = tuple(probe_list)

    transitivity = validate_cohort_and_split_transitivity(
        {
            "train": train_worlds,
            "eval": eval_w,
            "multihop": mh_w,
            "probe_train": probe_w,
        }
    )
    if not transitivity["passed"]:
        return {
            "schema": "anra-planthir-spec-validation-v1",
            "valid": False,
            "status": "INVALID_SPEC",
            "backend_id": backend_id,
            "errors": [{"code": "TRANSITIVE_COHORT_LEAKAGE", "field": "dataset", "detail": str(transitivity["leaks"])}],
        }, None, {}

    cartridge = None
    if materialize_data:
        cartridge = ExperimentCartridge(
            campaign_id=str(doc["campaign_id"]),
            program=str(doc["program"]),
            source_commit=str(doc["source_commit"]),
            kernel=kernel,
            control_arm=ctrl_arm,
            treatment_arms=tuple(trt_arms),
            train_data=train_data,
            eval_worlds=eval_w,
            multihop_worlds=mh_w,
            probe_train_worlds=probe_w,
            parameter_tolerance_ratio=param_tol,
            train_worlds=tuple(train_worlds),
            backend_id=backend_id,
            backend_version=doc.get("backend_version"),
        )

    run_kwargs = {
        "device": device,
        "cpu_canary": cpu_canary,
        "checkpoint_interval": ckpt_interval,
        "max_result_zip_bytes": max_zip_bytes,
        "max_wall_seconds": float(max_wall) if max_wall is not None else None,
        "resource_quota": quota,
    }
    return {
        "schema": "anra-planthir-spec-validation-v1",
        "valid": True,
        "status": "VALID",
        "backend_id": backend_id,
        "backend_version": backend.backend_version,
        "errors": [],
        "design_audit": design_audit,
        "cohort_transitivity": transitivity,
        "capabilities": asdict(caps),
    }, cartridge, run_kwargs


def validate_spec_document(doc: Mapping[str, Any], *, canary_override: bool | None = None) -> dict[str, Any]:
    """Validate a data-only `anra-planthir-spec-v1` document without starting training."""

    report, _cart, _kw = _validate_and_compile(doc, materialize_data=False, canary_override=canary_override)
    return report


def build_cartridge_from_spec(
    doc: Mapping[str, Any],
    *,
    canary_override: bool | None = None,
) -> tuple[ExperimentCartridge, dict[str, Any]]:
    """Compile a validated `anra-planthir-spec-v1` spec into `(ExperimentCartridge, runner_kwargs)`."""

    report, cartridge, run_kwargs = _validate_and_compile(doc, materialize_data=True, canary_override=canary_override)
    if not report["valid"] or cartridge is None:
        first = report["errors"][0] if report["errors"] else {"code": "INVALID_SPEC", "detail": "unknown spec error"}
        raise ValueError(f"{first['code']}: {first['detail']}")
    return cartridge, run_kwargs


_ERROR_CATALOG: dict[str, dict[str, Any]] = {
    "PARAM_SCALE_CONFOUND": {
        "meaning": "Treatment arm parameter count differs from control arm beyond parameter_tolerance_ratio for a non-capacity contrast.",
        "remediation": [
            "Match model dimensions (width/hidden_dim, layers, heads, slots) between control and treatment arms, or raise parameter_tolerance_ratio when intentionally comparing architectural heads (e.g., X1-R vs X1-P).",
            "Re-run `python -m anra_v5.planthir validate --spec <spec.json>` before training.",
        ],
    },
    "PRECISION_MISMATCH": {
        "meaning": "Control and treatment recipes differ in numeric precision without an explicit precision contrast.",
        "remediation": ["Set identical `recipe.precision` across control and treatment arms unless studying precision directly."],
    },
    "SEED_CONFOUND": {
        "meaning": "Treatment arm uses a different `recipe.seed` or `order_seed` than control without declaring a seed contrast.",
        "remediation": ["Match `recipe.seed` and `order_seed` across compared arms within each seed pair."],
    },
    "UNDECLARED_CONFOUNDED_FACTOR": {
        "meaning": "Multiple factors vary between control and treatment arms beyond the declared contrast factor.",
        "remediation": ["Hold all non-target factors constant (`lr`, `objective`, `precision`, `order_seed`, architecture) or split into separate single-factor contrasts."],
    },
    "NON_VARYING_DECLARED_FACTOR": {
        "meaning": "A factor listed in `factors` has identical values across all arms.",
        "remediation": ["Vary the factor across at least one treatment arm or remove it from `factors`."],
    },
    "UNKNOWN_FIELD": {
        "meaning": "Data-only spec contains an unrecognized key.",
        "remediation": ["Inspect `python -m anra_v5.planthir schema` for allowed keys and remove/rename the extra key."],
    },
    "FORBIDDEN_DOTTED_IMPORT": {
        "meaning": "Spec contains a dotted Python module/attribute import string, rejected for safety.",
        "remediation": ["Use declarative `backend_id` (`v8` or `x1_rd2`) and built-in scalar parameters only."],
    },
    "TRANSITIVE_COHORT_LEAKAGE": {
        "meaning": "Union-find transitive closure across world_id/base_world_id/parent_world_id/group_id/cohort_id spans more than one split.",
        "remediation": ["Remove overlapping cohort/base ancestors across train, eval, multihop, and probe_train splits."],
    },
    "RESUME_REQUIRED": {
        "meaning": "Target `work_root` already contains session files or checkpoints.",
        "remediation": ["Run `python -m anra_v5.planthir resume --spec <spec.json> --work-root <dir>` or choose a clean `work_root`."],
    },
    "SIDECAR_SHA256_MISMATCH": {
        "meaning": "Exported `result.zip` or `checkpoint_full.zip` bytes no longer match its `.sha256` custody sidecar.",
        "remediation": ["Treat the bundle directory as corrupted/tampered and re-export or restore from an uncorrupted generation under `generations/`."],
    },
}


def explain_error_code(code: str) -> dict[str, Any]:
    """Return structured diagnosis and concrete CLI remediation steps for a Planthir error code."""

    norm = str(code).strip()
    if norm in _ERROR_CATALOG:
        return {"schema": "anra-planthir-error-explanation-v1", "known": True, "code": norm, **_ERROR_CATALOG[norm]}
    for k, doc in _ERROR_CATALOG.items():
        if k in norm.upper():
            return {"schema": "anra-planthir-error-explanation-v1", "known": True, "code": k, **doc}
    return {
        "schema": "anra-planthir-error-explanation-v1",
        "known": False,
        "code": norm,
        "meaning": "Unindexed error code or message.",
        "remediation": [
            "Run `python -m anra_v5.planthir schema` and `validate --spec <spec.json>` for exact schema checks.",
            "Inspect `<work_root>/RUN_STATUS.json` and `<work_root>/RESOURCE_RECEIPT.json` via `inspect-run`.",
        ],
    }


def agent_info() -> dict[str, Any]:
    """Return read-only machine-readable contract derived from dataclass definitions and registered backends."""

    return {
        "schema": "anra-planthir-agent-info-v1",
        "inputs": {
            "ExperimentCartridge": _fields(ExperimentCartridge),
            "CartridgeArmSpec": _fields(CartridgeArmSpec),
            "CausalContrastSpec": _fields(CausalContrastSpec),
            "HypothesisSpec": _fields(HypothesisSpec),
            "FalsifierSpec": _fields(FalsifierSpec),
            "Recipe": _fields(Recipe),
            "X1RD2Recipe": _fields(X1RD2Recipe),
        },
        "outputs": {
            "CartridgeRunner.run": ["schema", "campaign_id", "device", "gates", "checkpoint_diffs", "arms", "registry", "bundle_receipt", "resource_receipt", "ai_handoff_brief", "token_fuel_report"],
            "result_bundle": ["FINAL_STATUS.json", "CHECKPOINT_INDEX.json", "RESULT_OVERVIEW.md", "CURRENT.json", "generations/", "AI_HANDOFF_BRIEF.json", "LAUNCH_GATE_CERTIFICATE.json", "result.zip", "result.zip.sha256", "checkpoint_full.zip", "checkpoint_full.zip.sha256"],
        },
        "declarative_spec": planthir_spec_schema(),
        "cli": {
            "agent-info": {"mutates_files": False, "trains": False},
            "schema": {"mutates_files": False, "trains": False},
            "init": {"mutates_files": "only with --out", "trains": False},
            "validate": {"mutates_files": False, "trains": False},
            "run": {"mutates_files": True, "trains": True},
            "resume": {"mutates_files": True, "trains": True},
            "inspect-run": {"mutates_files": False, "trains": False},
            "verify-bundle": {"mutates_files": False, "trains": False},
            "replay-bundle": {"mutates_files": "only with --output", "trains": False},
            "explain-error": {"mutates_files": False, "trains": False},
        },
        "legacy_commands": {
            "run-gpu-validation": "NOT_QUALIFIED: requires an interactive owner GPU session before launching cloud workloads",
            "run-validation-study": "DEPRECATED_CLI: migrate to `schema` / `init` / `validate` / `run` workflow",
        },
    }


def inspect_run(work_root: str | Path) -> dict[str, Any]:
    """Read-only inspection of a Planthir run directory, localizing corrupt/missing files."""

    root = Path(work_root)
    identity = _parse_json_strict(root / "RUN_IDENTITY.json")
    status = _parse_json_strict(root / "RUN_STATUS.json")
    attempts = []
    attempts_path = root / "ATTEMPTS.jsonl"
    if attempts_path.is_file():
        try:
            for line in attempts_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    attempts.append(json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c))))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise InspectionError(attempts_path.name, str(exc)) from exc
    final_receipt = None
    if (root / "FINAL_RECEIPT.json").exists():
        final_receipt = _parse_json_strict(root / "FINAL_RECEIPT.json")
    resource_receipt = None
    if (root / "RESOURCE_RECEIPT.json").exists():
        resource_receipt = _parse_json_strict(root / "RESOURCE_RECEIPT.json")
    return {
        "work_root": str(root),
        "identity": identity,
        "status": status,
        "attempts": attempts,
        "final_receipt_present": final_receipt is not None,
        "final_receipt": final_receipt,
        "resource_receipt_present": resource_receipt is not None,
        "resource_receipt": resource_receipt,
        "provenance_identity_present": (root / "PROVENANCE_IDENTITY.json").is_file(),
        "bundle_pointer_present": (root / "result_bundle" / "CURRENT.json").is_file(),
        "verification": "NOT_CHECKED",
    }


__all__ = [
    "InspectionError",
    "LEGACY_GPU_ERROR",
    "LEGACY_STUDY_CLI_ERROR",
    "SPEC_SCHEMA_VERSION",
    "agent_info",
    "build_cartridge_from_spec",
    "explain_error_code",
    "init_spec_template",
    "inspect_run",
    "planthir_spec_schema",
    "validate_spec_document",
]
