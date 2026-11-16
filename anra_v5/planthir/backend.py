"""Versioned backend contract and content-addressed provenance identity for Planthir."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import batch_tensors, configure_device, runtime_identity
from anra_v5.elendril.v717_training import IndependentOrderTrainer
from anra_v5.elendril.v8.contracts import ARCHITECTURES as V8_ARCHITECTURES, V8ModelSpec
from anra_v5.elendril.v8.data import build_d1_training_dataset
from anra_v5.elendril.v8.models import compute_typed_nll
from anra_v5.elendril.v8.training import V8TypedTrainer, prepare_v8_training_data
from v5_contracts.model_spec import ModelSpec
from v5_training.checkpoint import CheckpointStore

from .adapters.v8_adapter import V8EnvironmentKernel, V8ObservablePolicy
from .contracts import EnvironmentKernel, ObservablePolicy

BACKEND_CONTRACT_VERSION = "anra-planthir-backend-v1"
PROVENANCE_IDENTITY_SCHEMA = "anra-planthir-provenance-identity-v1"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared execution capabilities of a Planthir training/evaluation backend."""

    backend_id: str
    backend_version: str
    contract_version: str
    supported_architectures: tuple[str, ...]
    supported_objectives: tuple[str, ...]
    supported_samplings: tuple[str, ...]
    supported_devices: tuple[str, ...]
    supports_grouped_loss: bool
    supports_resume: bool
    supports_fp16: bool
    max_cpu_canary_params: int = 100_000
    max_cpu_canary_samples: int = 32
    max_cpu_canary_updates: int = 8

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class PlanthirBackend(Protocol):
    """Versioned typed contract decoupling Planthir execution from any single architecture."""

    backend_id: str
    backend_version: str
    contract_version: str

    def capabilities(self) -> BackendCapabilities: ...

    def resolve_model_spec(self, arm: Any) -> Any: ...

    def canonical_model_spec(self, model_spec: Any) -> dict[str, Any]: ...

    def model_parameter_count(self, model_spec: Any) -> int: ...

    def canonical_factor_config(self, arm: Any) -> dict[str, Any]: ...

    def validate_kernel(self, kernel: EnvironmentKernel) -> None: ...

    def validate_arm(
        self,
        arm: Any,
        train_data: Any,
        *,
        device: str,
        cpu_canary: bool,
    ) -> dict[str, Any]: ...

    def validate_dataset_and_worlds(
        self,
        train_data: Any,
        train_worlds: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]: ...

    def create_trainer(
        self,
        arm: Any,
        train_data: Any,
        *,
        source_commit: str,
        root: Path,
        lineage: str,
        device: str,
        cpu_canary: bool,
        resume: bool,
    ) -> Any: ...

    def sample_gradient_loss_fn(
        self,
        trainer: Any,
        arm: Any,
        train_data: Any,
        device: torch.device,
    ) -> Callable[[torch.nn.Module], torch.Tensor]: ...

    def create_policy(
        self,
        model: Any,
        model_spec: Any,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> ObservablePolicy: ...


def _safe_git_output(repo_root: Path, args: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return True, out
    except Exception:
        return False, ""


def compute_source_tree_fingerprint(repo_root: Path | str) -> dict[str, Any]:
    """Hash source commit, tracked tree/dirty state, and pyproject lock metadata."""

    root = Path(repo_root).resolve()
    ok_head, head_sha = _safe_git_output(root, ["rev-parse", "HEAD"])
    ok_tree, tree_sha = _safe_git_output(root, ["rev-parse", "HEAD^{tree}"])
    ok_status, status_porcelain = _safe_git_output(
        root, ["status", "--porcelain", "--untracked-files=no"]
    )
    dirty_hashes: dict[str, str] = {}
    if ok_status and status_porcelain:
        for line in status_porcelain.splitlines():
            rel = line[3:].strip().strip('"')
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1].strip()
            fpath = root / rel
            if fpath.is_file():
                dirty_hashes[rel] = sha256(fpath.read_bytes()).hexdigest()
    pyproj = root / "pyproject.toml"
    pyproject_sha = sha256(pyproj.read_bytes()).hexdigest() if pyproj.is_file() else None
    payload = {
        "git_available": ok_head,
        "head_commit": head_sha if ok_head else None,
        "tree_sha": tree_sha if ok_tree else None,
        "dirty": bool(status_porcelain),
        "dirty_file_hashes": dirty_hashes,
        "pyproject_sha256": pyproject_sha,
    }
    return {**payload, "source_fingerprint_sha256": digest(payload)}


def environment_fingerprint() -> dict[str, Any]:
    """Capture normalized Python/PyTorch/OS identity without fragile pointer hashes."""

    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    return {**info, "environment_sha256": digest(info)}


def build_provenance_identity(
    *,
    cartridge: Any,
    backend: PlanthirBackend,
    train_worlds: Sequence[Mapping[str, Any]],
    resolved_dataset_digest: dict[str, Any],
    device: str,
    checkpoint_interval: int,
    max_result_zip_bytes: int,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a complete, collision-resistant content-addressed provenance identity."""

    root = repo_root or Path(__file__).resolve().parents[2]
    arms = (cartridge.control_arm, *cartridge.treatment_arms)
    normalized_arms = []
    for arm in arms:
        spec = backend.resolve_model_spec(arm)
        normalized_arms.append({
            "arm_id": arm.arm_id,
            "architecture_family": arm.architecture_family,
            "is_control": arm.is_control,
            "backend_id": backend.backend_id,
            "backend_version": backend.backend_version,
            "model_spec": backend.canonical_model_spec(spec),
            "model_spec_sha256": digest(backend.canonical_model_spec(spec)),
            "factor_config": backend.canonical_factor_config(arm),
            "recipe": asdict(arm.recipe),
            "dataset_identity": arm.dataset_identity,
            "order_seed": arm.order_seed,
            "milestone_updates": list(arm.milestone_updates),
            "policy_mode": arm.policy_mode,
        })

    ordered_splits = {
        "train_world_ids": [str(w.get("world_id", i)) for i, w in enumerate(train_worlds)],
        "eval_world_ids": [str(w.get("world_id", i)) for i, w in enumerate(cartridge.eval_worlds)],
        "multihop_world_ids": [str(w.get("world_id", i)) for i, w in enumerate(cartridge.multihop_worlds)],
        "probe_train_world_ids": [str(w.get("world_id", i)) for i, w in enumerate(cartridge.probe_train_worlds)],
        "train_worlds_ordered_sha256": digest(list(train_worlds)),
        "eval_worlds_ordered_sha256": digest(list(cartridge.eval_worlds)),
        "multihop_worlds_ordered_sha256": digest(list(cartridge.multihop_worlds)),
        "probe_train_worlds_ordered_sha256": digest(list(cartridge.probe_train_worlds)),
    }

    source_fp = compute_source_tree_fingerprint(root)
    env_fp = environment_fingerprint()
    manifest = {
        "schema": PROVENANCE_IDENTITY_SCHEMA,
        "campaign_id": cartridge.campaign_id,
        "program": cartridge.program,
        "source_commit": cartridge.source_commit,
        "source_tree": source_fp,
        "environment": env_fp,
        "backend": backend.capabilities().canonical(),
        "kernel": {
            "kernel_id": cartridge.kernel.kernel_id,
            "config": getattr(cartridge.kernel, "canonical_config", lambda: {"kernel_id": cartridge.kernel.kernel_id})(),
        },
        "dataset": resolved_dataset_digest,
        "tokenizer": UTF8.manifest.record(),
        "ordered_splits": ordered_splits,
        "arms": normalized_arms,
        "parameter_tolerance_ratio": float(cartridge.parameter_tolerance_ratio),
        "execution_device": str(torch.device(device)),
        "execution_runtime": runtime_identity(torch.device(device)),
        "checkpoint_interval": int(checkpoint_interval),
        "max_result_zip_bytes": int(max_result_zip_bytes),
    }
    return {
        **manifest,
        "provenance_sha256": digest(manifest),
    }


def verify_provenance_identity(
    expected_manifest: Mapping[str, Any],
    observed_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two provenance manifests and report every localized divergence."""

    mismatches: list[str] = []
    if expected_manifest.get("schema") != PROVENANCE_IDENTITY_SCHEMA or observed_manifest.get("schema") != PROVENANCE_IDENTITY_SCHEMA:
        mismatches.append("PROVENANCE_SCHEMA_MISMATCH")
    field_codes = (
        ("source_commit", "SOURCE_COMMIT_CHANGED"),
        ("source_tree", "SOURCE_TREE_OR_DIRTY_STATE_CHANGED"),
        ("environment", "ENVIRONMENT_OR_LOCK_CHANGED"),
        ("backend", "BACKEND_CONTRACT_OR_VERSION_CHANGED"),
        ("kernel", "KERNEL_CONFIGURATION_CHANGED"),
        ("dataset", "DATASET_BYTES_OR_IDENTITY_CHANGED"),
        ("tokenizer", "TOKENIZER_IDENTITY_CHANGED"),
        ("ordered_splits", "ORDERED_SPLITS_OR_MEMBERSHIP_CHANGED"),
        ("arms", "NORMALIZED_CONFIG_OR_SEED_CHANGED"),
        ("parameter_tolerance_ratio", "PARAMETER_TOLERANCE_CHANGED"),
        ("execution_device", "EXECUTION_DEVICE_CHANGED"),
        ("execution_runtime", "RUNTIME_DEVICE_CHANGED"),
        ("checkpoint_interval", "CHECKPOINT_INTERVAL_CHANGED"),
        ("max_result_zip_bytes", "RESULT_CAP_CHANGED"),
    )
    for key, code in field_codes:
        if expected_manifest.get(key) != observed_manifest.get(key):
            mismatches.append(code)
    if (
        expected_manifest.get("provenance_sha256") != observed_manifest.get("provenance_sha256")
        and not mismatches
    ):
        mismatches.append("PROVENANCE_DIGEST_MISMATCH")
    return {
        "schema": "anra-planthir-provenance-check-v1",
        "matched": len(mismatches) == 0,
        "expected_sha256": expected_manifest.get("provenance_sha256"),
        "observed_sha256": observed_manifest.get("provenance_sha256"),
        "mismatches": mismatches,
    }


class V8PlanthirBackend:
    """Qualified V8/Elendril implementation of PlanthirBackend."""

    backend_id = "v8"
    backend_version = "elendril-v8-1.0"
    contract_version = BACKEND_CONTRACT_VERSION

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            contract_version=self.contract_version,
            supported_architectures=V8_ARCHITECTURES,
            supported_objectives=("ce",),
            supported_samplings=("uniform",),
            supported_devices=("cpu", "cuda"),
            supports_grouped_loss=False,
            supports_resume=True,
            supports_fp16=True,
        )

    def resolve_model_spec(self, arm: Any) -> V8ModelSpec:
        spec = getattr(arm, "v8_spec", None) or getattr(arm, "model_spec", None)
        if not isinstance(spec, V8ModelSpec):
            raise ValueError("V8 backend requires arm.v8_spec to be a V8ModelSpec")
        spec.assert_valid()
        return spec

    def canonical_model_spec(self, model_spec: Any) -> dict[str, Any]:
        if not isinstance(model_spec, V8ModelSpec):
            raise ValueError("expected V8ModelSpec")
        return model_spec.canonical()

    def model_parameter_count(self, model_spec: Any) -> int:
        if not isinstance(model_spec, V8ModelSpec):
            raise ValueError("expected V8ModelSpec")
        return int(model_spec.total_parameters())

    def canonical_factor_config(self, arm: Any) -> dict[str, Any]:
        spec = self.resolve_model_spec(arm)
        data = spec.canonical()
        return {
            "architecture": data["architecture"],
            "pointer_dim": data["pointer_dim"],
            "bottleneck_width": data["bottleneck_width"],
            "recurrent_dim": data["recurrent_dim"],
            "recurrent_steps": data["recurrent_steps"],
            "copy_enabled": data["copy_enabled"],
            "max_sequence_length": data["max_sequence_length"],
            "vocabulary_size": data["vocabulary_size"],
            "base_spec": data["base_spec"],
        }

    def validate_kernel(self, kernel: EnvironmentKernel) -> None:
        if not isinstance(kernel, V8EnvironmentKernel):
            raise ValueError("V8 backend requires V8EnvironmentKernel; use the matching backend for non-V8 kernels")

    def validate_arm(
        self,
        arm: Any,
        train_data: Any,
        *,
        device: str,
        cpu_canary: bool,
    ) -> dict[str, Any]:
        caps = self.capabilities()
        if getattr(arm, "trainer_factory", None) is not None:
            raise ValueError("unqualified custom trainer backend: register a PlanthirBackend implementation instead")
        if getattr(arm, "b0_job", None) is not None:
            raise ValueError("legacy b0_job is ambiguous with recipe; omit b0_job")
        spec = self.resolve_model_spec(arm)
        if arm.architecture_family != spec.architecture:
            raise ValueError("architecture label differs from V8ModelSpec.architecture")
        if spec.architecture not in caps.supported_architectures:
            raise ValueError(f"unsupported V8 architecture: {spec.architecture!r}")
        arm.recipe.validate()
        if arm.recipe.sampling not in caps.supported_samplings or arm.recipe.objective not in caps.supported_objectives:
            raise ValueError("V8 backend qualifies uniform CE only; unsupported recipe must not run")
        if arm.dataset_identity != train_data.data_identity:
            raise ValueError("arm dataset identity differs from actual prepared data")
        if any(type(u) is not int or not 1 <= u <= arm.recipe.total_updates for u in arm.milestone_updates):
            raise ValueError("milestone updates must lie within the registered training horizon")
        samples = train_data.byte_samples if arm.architecture_family == "B0" else train_data.typed_samples
        if not samples:
            raise ValueError("missing training samples for V8 architecture")
        dev_type = torch.device(device).type
        if dev_type not in caps.supported_devices:
            raise ValueError(f"unsupported device type: {dev_type!r}")
        if dev_type == "cpu" and (
            not cpu_canary
            or len(samples) > caps.max_cpu_canary_samples
            or spec.total_parameters() > caps.max_cpu_canary_params
            or arm.recipe.total_updates > caps.max_cpu_canary_updates
        ):
            raise ValueError("CPU use is restricted to tiny mechanical fixtures")
        return {
            "backend_id": self.backend_id,
            "architecture": spec.architecture,
            "parameter_count": spec.total_parameters(),
            "sample_count": len(samples),
        }

    def validate_dataset_and_worlds(
        self,
        train_data: Any,
        train_worlds: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        resolved_worlds = list(train_worlds)
        if getattr(train_data, "dataset_name", None) == "D1":
            quota = int(train_data.manifest["world_count"]) // 7
            expected = prepare_v8_training_data(
                "D1",
                d1_worlds_per_family=quota,
                sample_limit_for_cpu=train_data.manifest.get("slice_limit"),
            )
            if expected != train_data:
                raise ValueError("prepared D1 corpus content/identity does not match its claimed provenance")
            wanted = {s.world_id for s in train_data.byte_samples}
            resolved_worlds = [
                w for w in build_d1_training_dataset(quota_per_family=quota).worlds if w["world_id"] in wanted
            ]
        if not resolved_worlds:
            raise ValueError("training worlds required to verify structural split separation")
        raw_digest = digest({
            "dataset_name": getattr(train_data, "dataset_name", "unknown"),
            "data_identity": getattr(train_data, "data_identity", ""),
            "manifest": getattr(train_data, "manifest", {}),
            "typed_samples": [asdict(s) for s in getattr(train_data, "typed_samples", ())],
            "byte_samples": [asdict(s) for s in getattr(train_data, "byte_samples", ())],
        })
        return {
            "dataset_name": getattr(train_data, "dataset_name", "unknown"),
            "data_identity": getattr(train_data, "data_identity", ""),
            "content_sha256": raw_digest,
            "world_count": len(resolved_worlds),
            "resolved_train_worlds": tuple(resolved_worlds),
        }

    def create_trainer(
        self,
        arm: Any,
        train_data: Any,
        *,
        source_commit: str,
        root: Path,
        lineage: str,
        device: str,
        cpu_canary: bool,
        resume: bool,
    ) -> Any:
        spec = self.resolve_model_spec(arm)
        arm_resume = resume and (root / lineage / "LATEST").is_file()
        if arm.architecture_family == "B0":
            trainer = IndependentOrderTrainer(
                train_data.byte_samples,
                spec=spec.base_spec,
                recipe=arm.recipe,
                order_seed=arm.order_seed,
                source_commit=source_commit,
                root=root,
                lineage=lineage,
                data_provenance={
                    **train_data.manifest,
                    "v8_job": {
                        "schema": "planthir-b0-recipe-v1",
                        "arm": "B0",
                        "max_sequence_length": spec.max_sequence_length,
                        "recipe": asdict(arm.recipe),
                    },
                },
                tokenizer_manifest=UTF8.manifest.record(),
                device=device,
                method="uniform",
                resume=arm_resume,
            )
            trainer.v8_spec = spec
            return trainer
        return V8TypedTrainer(
            train_data.typed_samples,
            v8_spec=spec,
            recipe=arm.recipe,
            order_seed=arm.order_seed,
            source_commit=source_commit,
            root=root,
            lineage=lineage,
            data_provenance=train_data.manifest,
            device=device,
            cpu_canary=(torch.device(device).type == "cpu" and cpu_canary),
            resume=arm_resume,
        )

    def sample_gradient_loss_fn(
        self,
        trainer: Any,
        arm: Any,
        train_data: Any,
        device: torch.device,
    ) -> Callable[[torch.nn.Module], torch.Tensor]:
        if arm.architecture_family == "B0":
            chunk = train_data.byte_samples[:4]
            toks, segs, eligible = batch_tensors([(s.tokens, s.eligible) for s in chunk], device)

            def _byte_loss(m: Any) -> torch.Tensor:
                logits = trainer.backend.logits(toks, segs)
                loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    toks[:, 1:].reshape(-1),
                    reduction="none",
                ).reshape_as(toks[:, 1:])
                return (loss * eligible[:, 1:]).sum()

            return _byte_loss

        sample_chunk = train_data.typed_samples[:4]
        toks, segs, _ = batch_tensors([(s.tokens, s.eligible) for s in sample_chunk], device)
        apos = torch.tensor([s.action_position for s in sample_chunk], device=device, dtype=torch.int64)
        g_ops = torch.tensor([s.gold_op for s in sample_chunk], device=device, dtype=torch.int64)
        g_args = torch.tensor([s.gold_arg for s in sample_chunk], device=device, dtype=torch.int64)

        def _typed_loss(m: Any) -> torch.Tensor:
            out = m.forward_action(toks, segs, apos)
            nll, _ = compute_typed_nll(out, g_ops, g_args)
            return nll

        return _typed_loss

    def create_policy(
        self,
        model: Any,
        model_spec: Any,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> ObservablePolicy:
        if not isinstance(model_spec, V8ModelSpec):
            raise ValueError("V8 backend policy requires V8ModelSpec")
        return V8ObservablePolicy(
            model,
            model_spec,
            model_identity=model_identity,
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )


_BACKENDS: dict[str, PlanthirBackend] = {
    "v8": V8PlanthirBackend(),
}


def register_backend(backend: PlanthirBackend, *, contract_version: str = BACKEND_CONTRACT_VERSION) -> None:
    """Register a validated PlanthirBackend after enforcing contract version compatibility."""

    if contract_version != BACKEND_CONTRACT_VERSION or getattr(backend, "contract_version", None) != BACKEND_CONTRACT_VERSION:
        raise ValueError(
            f"incompatible backend contract_version {getattr(backend, 'contract_version', None)!r}; "
            f"expected {BACKEND_CONTRACT_VERSION!r}"
        )
    if not isinstance(backend, PlanthirBackend):
        raise TypeError("backend does not implement PlanthirBackend protocol")
    _BACKENDS[backend.backend_id] = backend


def get_backend(backend_id: str, *, expected_version: str | None = None) -> PlanthirBackend:
    """Resolve a registered backend by ID and optional version check."""

    if not _BACKENDS_LOADED_X1:
        _ensure_builtin_backends()
    if backend_id not in _BACKENDS:
        raise ValueError(f"unknown or unsupported backend: {backend_id!r} (available={sorted(_BACKENDS)})")
    backend = _BACKENDS[backend_id]
    if expected_version is not None and backend.backend_version != expected_version:
        raise ValueError(
            f"backend {backend_id!r} version mismatch: got {backend.backend_version!r}, expected {expected_version!r}"
        )
    return backend


def list_backends() -> dict[str, dict[str, Any]]:
    if not _BACKENDS_LOADED_X1:
        _ensure_builtin_backends()
    return {k: b.capabilities().canonical() for k, b in sorted(_BACKENDS.items())}


_BACKENDS_LOADED_X1 = False


def _ensure_builtin_backends() -> None:
    global _BACKENDS_LOADED_X1
    if _BACKENDS_LOADED_X1:
        return
    _BACKENDS_LOADED_X1 = True
    try:
        from .adapters.x1_rd2_adapter import X1RD2Backend

        register_backend(X1RD2Backend())
    except ImportError:
        pass


__all__ = [
    "BACKEND_CONTRACT_VERSION",
    "BackendCapabilities",
    "PROVENANCE_IDENTITY_SCHEMA",
    "PlanthirBackend",
    "V8PlanthirBackend",
    "build_provenance_identity",
    "compute_source_tree_fingerprint",
    "environment_fingerprint",
    "get_backend",
    "list_backends",
    "register_backend",
    "verify_provenance_identity",
]
