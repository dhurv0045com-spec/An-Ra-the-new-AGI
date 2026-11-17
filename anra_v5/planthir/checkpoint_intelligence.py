"""Checkpoint Intelligence: discover, query, diff, and restore checkpoints for causal probing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import torch

from anra_v5.elendril.training import _load
from anra_v5.elendril.v8.contracts import ARCHITECTURES, V8ModelSpec, build_v8_model_spec
from anra_v5.elendril.v8.models import build_v8_model
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore, MILESTONES_FILENAME
from v5_training.state import TrainingState

from .adapters.v8_adapter import V8ObservablePolicy
from .evidence import verify_single_planthir_zip

_HEX64 = set("0123456789abcdef")


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX64 for c in value)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Indexed metadata summary of a committed Tier-3 checkpoint without loading weights."""

    checkpoint_sha256: str
    lineage_id: str
    checkpoint_root: str
    generation: int
    global_update: int
    cumulative_tokens: int
    curriculum_phase: str
    source_commit: str
    model_spec_sha256: str
    data_manifest_sha256: str
    program_family: str
    architecture: str
    parameter_count: int | None
    total_byte_size: int
    is_latest: bool
    is_milestone: bool
    parent_checkpoint_sha256: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_store_for_lineage_dir(lineage_dir: Path, sample_sha: str | None = None) -> CheckpointStore | None:
    """Resolve the (root, lineage_id) pair matching an on-disk CheckpointStore lineage directory."""

    lineage_id = lineage_dir.name
    candidates = [lineage_dir.parent, lineage_dir.parent.parent]
    resolved_dir = lineage_dir.resolve()
    for cand_root in candidates:
        try:
            store = CheckpointStore(cand_root, lineage_id)
        except ValueError:
            continue
        if store.lineage_root.resolve() != resolved_dir:
            continue
        if sample_sha is None:
            if store.objects.is_dir() or (lineage_dir / "LATEST").is_file():
                return store
        else:
            if (store.objects / sample_sha / "manifest.json").is_file():
                return store
    return None


def _reconstruct_model_spec(raw: Mapping[str, Any]) -> ModelSpec:
    valid_keys = {f.name for f in fields(ModelSpec)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return ModelSpec(**filtered)


def _reconstruct_v8_spec(raw: Mapping[str, Any]) -> V8ModelSpec:
    data = dict(raw)
    if isinstance(data.get("base_spec"), Mapping):
        data["base_spec"] = _reconstruct_model_spec(data["base_spec"])
    if isinstance(data.get("output_mapping"), list):
        data["output_mapping"] = tuple(str(x) for x in data["output_mapping"])
    valid_keys = {f.name for f in fields(V8ModelSpec)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return V8ModelSpec(**filtered)


def _reconstruct_x1_s20_spec(raw: Mapping[str, Any]):
    from anra_v5.elendril.v8x.contracts import X1ModelSpec

    data = dict(raw)
    if isinstance(data.get("base_spec"), Mapping):
        data["base_spec"] = _reconstruct_model_spec(data["base_spec"])
    if isinstance(data.get("output_mapping"), list):
        data["output_mapping"] = tuple(str(x) for x in data["output_mapping"])
    valid_keys = {f.name for f in fields(X1ModelSpec)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return X1ModelSpec(**filtered)


def _reconstruct_x1_spec(raw: Mapping[str, Any]):
    from .adapters.x1_rd2_adapter import X1RD2Spec

    valid_keys = {f.name for f in fields(X1RD2Spec)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return X1RD2Spec(**filtered)


def _extract_arch_and_params(
    scheduler_payload: Mapping[str, Any],
) -> tuple[str, str, int | None, Any, ModelSpec | None]:
    meta = scheduler_payload.get("metadata", scheduler_payload)
    if not isinstance(meta, Mapping):
        meta = {}

    if isinstance(meta.get("x1_model_spec"), Mapping):
        raw_x1 = meta["x1_model_spec"]
        if meta.get("schema") == "elendril-x1-s20-training-v1" or isinstance(raw_x1.get("base_spec"), Mapping):
            x1_s20_spec = _reconstruct_x1_s20_spec(raw_x1)
            prog = "X1-byte" if x1_s20_spec.architecture == "B0" else "X1-S20"
            return prog, x1_s20_spec.architecture, int(x1_s20_spec.total_parameters()), x1_s20_spec, x1_s20_spec.base_spec
        x1_spec = _reconstruct_x1_spec(raw_x1)
        return "X1-RD2", x1_spec.architecture, int(x1_spec.total_parameters()), x1_spec, None

    if isinstance(meta.get("v8_model_spec"), Mapping):
        v8_spec = _reconstruct_v8_spec(meta["v8_model_spec"])
        try:
            param_count = int(v8_spec.total_parameters())
        except Exception:
            param_count = None
        prog = "V8-byte" if v8_spec.architecture == "B0" else "V8-typed"
        return prog, v8_spec.architecture, param_count, v8_spec, v8_spec.base_spec

    for key in ("model_spec", "spec", "base_spec", "model"):
        if isinstance(meta.get(key), Mapping):
            spec = _reconstruct_model_spec(meta[key])
            prov = meta.get("provenance")
            if isinstance(prov, Mapping) and isinstance(prov.get("v8_job"), Mapping):
                v8_job = prov["v8_job"]
                if str(v8_job.get("arm", "")) == "B0":
                    max_seq = int(v8_job.get("max_sequence_length", spec.context_length))
                    v8_spec = build_v8_model_spec(
                        "B0", base_spec=spec, max_sequence_length=max_seq
                    )
                    return (
                        "V8-byte",
                        "B0",
                        int(v8_spec.total_parameters()),
                        v8_spec,
                        spec,
                    )
            try:
                param_count = int(spec.parameter_receipt().total)
            except Exception:
                param_count = None
            arch = str(meta.get("architecture", meta.get("method", spec.family)))
            prog = str(meta.get("schema", "Elendril-V5/V7"))
            return prog, arch, param_count, None, spec

    arch = str(meta.get("architecture", meta.get("method", "unknown")))
    return "unknown", arch, None, None, None


def _read_nearby_metrics(lineage_dir: Path) -> dict[str, Any]:
    for candidate_dir in (lineage_dir, lineage_dir.parent, lineage_dir.parent.parent):
        for fname in ("SCIENTIFIC_SUMMARY.json", "METRICS.json"):
            path = candidate_dir / fname
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, Mapping):
                        return dict(data)
                except Exception:
                    continue
    return {}


def _build_record_from_object_dir(
    store: CheckpointStore,
    obj_dir: Path,
    *,
    latest_sha: str | None,
    milestone_shas: set[str],
    nearby_metrics: Mapping[str, Any],
) -> CheckpointRecord | None:
    sha = obj_dir.name
    state_path = obj_dir / "training_state.json"
    sched_path = obj_dir / "scheduler.json"
    manifest_path = obj_dir / "manifest.json"
    if not (state_path.is_file() and sched_path.is_file() and manifest_path.is_file()):
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        sched = json.loads(sched_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if state.get("lineage_id") != store.lineage_id:
        return None

    ident = state.get("identities", {})
    prog, arch, pcount, _, _ = _extract_arch_and_params(sched)
    total_bytes = sum(int(comp.get("byte_size", 0)) for comp in manifest.get("components", []))
    return CheckpointRecord(
        checkpoint_sha256=sha,
        lineage_id=store.lineage_id,
        checkpoint_root=str(store.root),
        generation=int(state.get("generation", 0)),
        global_update=int(state.get("global_update", 0)),
        cumulative_tokens=int(state.get("cumulative_tokens", 0)),
        curriculum_phase=str(state.get("curriculum_phase", "unknown")),
        source_commit=str(ident.get("source_commit", "")),
        model_spec_sha256=str(ident.get("model_spec_sha256", "")),
        data_manifest_sha256=str(ident.get("data_manifest_sha256", "")),
        program_family=prog,
        architecture=arch,
        parameter_count=pcount,
        total_byte_size=total_bytes,
        is_latest=(sha == latest_sha),
        is_milestone=(sha in milestone_shas),
        parent_checkpoint_sha256=state.get("parent_checkpoint_sha256"),
        metrics=dict(nearby_metrics) if sha == latest_sha else {},
        metadata={
            "schema": manifest.get("schema"),
            "run_spec_sha256": ident.get("run_spec_sha256"),
            "object_dir": str(obj_dir),
        },
    )


def _safe_extract_members(
    zf: zipfile.ZipFile,
    members: Sequence[str],
    dest_dir: Path,
) -> None:
    """Extract specific ZIP members into dest_dir while rejecting path traversal."""

    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    for member in members:
        if member.endswith("/"):
            continue
        if "\\" in member:
            raise ValueError(f"unsafe archive member backslash in path: {member!r}")
        p = PurePosixPath(member)
        if p.is_absolute() or ".." in p.parts or any(part == "" for part in p.parts):
            raise ValueError(f"unsafe archive member path: {member!r}")
        target = (dest_root / Path(*p.parts)).resolve()
        if target != dest_root and dest_root not in target.parents:
            raise ValueError(f"archive member escapes destination directory: {member!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member, "r") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _instantiate_checkpoint_bundle(
    *,
    sha: str,
    state: TrainingState,
    sched: Mapping[str, Any],
    weights: Mapping[str, Any],
    payload_keys: Sequence[str],
    mode: str,
    copy_enabled_override: bool | None,
    device: str,
) -> dict[str, Any]:
    """Build model + ObservablePolicy and load weights for frozen causal probing."""

    from anra_v5.elendril.v8x.contracts import X1ModelSpec
    from anra_v5.elendril.v8x.evaluation import X1S20ObservablePolicy
    from anra_v5.elendril.v8x.models import build_x1_s20_model
    from .adapters.x1_rd2_adapter import X1ObservablePolicy, X1RD2Spec, build_x1_model

    prog, arch, pcount, backend_spec, base_spec = _extract_arch_and_params(sched)
    if isinstance(backend_spec, X1ModelSpec):
        v8_spec = None
        model = build_x1_s20_model(backend_spec, 0, torch_module=torch).to(torch.device(device))
        model.load_state_dict(weights, strict=True)
        model.eval()
        policy = X1S20ObservablePolicy(
            model,
            backend_spec,
            model_identity=f"{state.lineage_id}:{sha[:12]}",
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )
    elif isinstance(backend_spec, X1RD2Spec):
        model = build_x1_model(backend_spec, 0).to(torch.device(device))
        model.load_state_dict(weights, strict=True)
        model.eval()
        policy: Any = X1ObservablePolicy(
            model,
            backend_spec,
            model_identity=f"{state.lineage_id}:{sha[:12]}",
            mode=mode,
        )
        v8_spec = None
    elif isinstance(backend_spec, V8ModelSpec) and backend_spec.architecture in ARCHITECTURES:
        v8_spec = backend_spec
        model = build_v8_model(v8_spec, 0, torch_module=torch).to(torch.device(device))
        model.load_state_dict(weights, strict=True)
        model.eval()
        policy = V8ObservablePolicy(
            model,
            v8_spec,
            model_identity=f"{state.lineage_id}:{sha[:12]}",
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )
    elif base_spec is not None:
        v8_spec = None
        model = initialize(base_spec, 0).to(torch.device(device))
        model.load_state_dict(weights, strict=True)
        model.eval()
        policy = None
    else:
        raise ValueError(f"cannot reconstruct model spec from scheduler.json for {sha}")

    return {
        "checkpoint_sha256": sha,
        "lineage_id": state.lineage_id,
        "global_update": state.global_update,
        "cumulative_tokens": state.cumulative_tokens,
        "curriculum_phase": state.curriculum_phase,
        "program_family": prog,
        "architecture": arch,
        "parameter_count": pcount,
        "state": state,
        "model": model,
        "policy": policy,
        "v8_spec": v8_spec,
        "x1_spec": backend_spec if isinstance(backend_spec, X1RD2Spec) else None,
        "model_spec": base_spec or (backend_spec if isinstance(backend_spec, X1RD2Spec) else None),
        "scheduler": dict(sched),
        "payload_keys": sorted(payload_keys),
    }


def _restore_compact_fp16_object(
    obj_dir: Path,
    *,
    expected_lineage: str,
    expected_sha: str,
    mode: str,
    copy_enabled_override: bool | None,
    device: str,
) -> dict[str, Any]:
    """Verify manifest/state digests and load model_fp16.bin into FP32 model weights."""

    required_compact = (
        "training_state.json",
        "scheduler.json",
        "cursor.json",
        "ledger.json",
        "rng.bin",
    )
    manifest_path = obj_dir / "manifest.json"
    fp16_path = obj_dir / "model_fp16.bin"
    if not (manifest_path.is_file() and fp16_path.is_file()):
        raise ValueError(f"compact checkpoint object missing manifest.json or model_fp16.bin in {obj_dir}")

    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes).hexdigest() != expected_sha:
        raise ValueError(f"compact checkpoint manifest SHA-256 mismatch for {expected_sha}")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("lineage_id") != expected_lineage:
        raise ValueError("compact checkpoint lineage mismatch in manifest.json")

    comp_map = {str(item["name"]): item for item in manifest.get("components", [])}
    for req_name in required_compact:
        if req_name not in comp_map:
            raise ValueError(f"compact checkpoint manifest missing component: {req_name}")

    for comp_name, item in comp_map.items():
        comp_path = obj_dir / comp_name
        if comp_name in ("model.bin", "optimizer.bin") and not comp_path.exists():
            continue
        if not comp_path.is_file():
            raise ValueError(f"compact checkpoint component missing: {comp_name}")
        raw = comp_path.read_bytes()
        if len(raw) != int(item["byte_size"]) or sha256(raw).hexdigest() != str(item["sha256"]):
            raise ValueError(f"compact checkpoint component corrupted: {comp_name}")

    state = TrainingState.from_dict(manifest["state"])
    if state.lineage_id != expected_lineage:
        raise ValueError("compact checkpoint TrainingState lineage mismatch")
    if state.sha256() != manifest.get("state_sha256"):
        raise ValueError("compact checkpoint TrainingState SHA-256 mismatch")

    sched = json.loads((obj_dir / "scheduler.json").read_text(encoding="utf-8"))
    fp16_weights = _load(fp16_path.read_bytes())
    if not isinstance(fp16_weights, Mapping):
        raise ValueError("model_fp16.bin did not deserialize to a state_dict mapping")
    weights = {
        key: (val.float() if isinstance(val, torch.Tensor) and torch.is_floating_point(val) else val)
        for key, val in fp16_weights.items()
    }
    payload_keys = [p.name for p in obj_dir.iterdir() if p.is_file()]
    return _instantiate_checkpoint_bundle(
        sha=expected_sha,
        state=state,
        sched=sched,
        weights=weights,
        payload_keys=payload_keys,
        mode=mode,
        copy_enabled_override=copy_enabled_override,
        device=device,
    )


class CheckpointCatalog:
    """Discover, filter, diff, and restore historical checkpoints for counterfactual probing."""

    def __init__(self, records: Sequence[CheckpointRecord] = ()) -> None:
        self._records: list[CheckpointRecord] = list(records)

    @property
    def records(self) -> tuple[CheckpointRecord, ...]:
        return tuple(self._records)

    @classmethod
    def discover_checkpoints(
        cls,
        roots: Sequence[Path | str],
        *,
        include_all_generations: bool = False,
    ) -> "CheckpointCatalog":
        """Scan root directories for CheckpointStore lineages and index committed checkpoints."""

        discovered: list[CheckpointRecord] = []
        seen_keys: set[tuple[str, str]] = set()
        lineage_dirs_seen: set[Path] = set()

        for raw_root in roots:
            root = Path(raw_root)
            if not root.exists():
                continue
            for latest_path in sorted(root.rglob("LATEST")):
                if latest_path.is_file():
                    lineage_dirs_seen.add(latest_path.parent.resolve())
            for ms_path in sorted(root.rglob(MILESTONES_FILENAME)):
                if ms_path.is_file():
                    lineage_dirs_seen.add(ms_path.parent.resolve())

        for lineage_dir in sorted(lineage_dirs_seen):
            latest_file = lineage_dir / "LATEST"
            latest_sha: str | None = None
            if latest_file.is_file():
                cand = latest_file.read_text(encoding="ascii", errors="ignore").strip()
                if _is_sha256(cand):
                    latest_sha = cand

            milestone_shas: set[str] = set()
            ms_file = lineage_dir / MILESTONES_FILENAME
            if ms_file.is_file():
                try:
                    ms_doc = json.loads(ms_file.read_text(encoding="utf-8"))
                    milestone_shas = {
                        str(v) for v in ms_doc.get("milestones", {}).values() if _is_sha256(str(v))
                    }
                except Exception:
                    milestone_shas = set()

            probe_sha = latest_sha or (next(iter(milestone_shas)) if milestone_shas else None)
            store = _resolve_store_for_lineage_dir(lineage_dir, sample_sha=probe_sha)
            if store is None:
                continue

            nearby_metrics = _read_nearby_metrics(lineage_dir)
            target_shas: list[str]
            if include_all_generations and store.objects.is_dir():
                target_shas = [
                    child.name
                    for child in sorted(store.objects.iterdir())
                    if child.is_dir() and _is_sha256(child.name)
                ]
            else:
                target_shas = sorted(({latest_sha} if latest_sha else set()) | milestone_shas)

            for sha in target_shas:
                key = (str(store.lineage_root.resolve()), sha)
                if key in seen_keys:
                    continue
                obj_dir = store.objects / sha
                if not obj_dir.is_dir():
                    continue
                rec = _build_record_from_object_dir(
                    store,
                    obj_dir,
                    latest_sha=latest_sha,
                    milestone_shas=milestone_shas,
                    nearby_metrics=nearby_metrics,
                )
                if rec is not None:
                    seen_keys.add(key)
                    discovered.append(rec)

        return cls(discovered)

    def bind_metrics(self, checkpoint_sha256: str, metrics: Mapping[str, Any]) -> None:
        updated: list[CheckpointRecord] = []
        for rec in self._records:
            if rec.checkpoint_sha256 == checkpoint_sha256:
                merged = dict(rec.metrics)
                merged.update(dict(metrics))
                updated.append(
                    CheckpointRecord(
                        **{**rec.to_dict(), "metrics": merged}
                    )
                )
            else:
                updated.append(rec)
        self._records = updated

    def query_compatible_checkpoints(
        self,
        *,
        architecture: str | None = None,
        program_family: str | None = None,
        curriculum_phase: str | None = None,
        min_global_update: int | None = None,
        max_parameter_count: int | None = None,
        only_latest: bool = False,
        only_milestones: bool = False,
    ) -> list[CheckpointRecord]:
        out: list[CheckpointRecord] = []
        for rec in self._records:
            if architecture is not None and rec.architecture != architecture:
                continue
            if program_family is not None and rec.program_family != program_family:
                continue
            if curriculum_phase is not None and rec.curriculum_phase != curriculum_phase:
                continue
            if min_global_update is not None and rec.global_update < min_global_update:
                continue
            if max_parameter_count is not None and rec.parameter_count is not None:
                if rec.parameter_count > max_parameter_count:
                    continue
            if only_latest and not rec.is_latest:
                continue
            if only_milestones and not rec.is_milestone:
                continue
            out.append(rec)
        return out

    @staticmethod
    def diff_checkpoints(
        rec_a: CheckpointRecord,
        rec_b: CheckpointRecord,
    ) -> dict[str, Any]:
        """Compare two checkpoints to expose controlled vs confounded experimental factors."""

        fields_to_compare = (
            "program_family",
            "architecture",
            "parameter_count",
            "curriculum_phase",
            "source_commit",
            "model_spec_sha256",
            "data_manifest_sha256",
            "global_update",
            "cumulative_tokens",
        )
        same: dict[str, Any] = {}
        changed: dict[str, tuple[Any, Any]] = {}
        for fname in fields_to_compare:
            va = getattr(rec_a, fname)
            vb = getattr(rec_b, fname)
            if va == vb:
                same[fname] = va
            else:
                changed[fname] = (va, vb)
        return {
            "checkpoint_a": rec_a.checkpoint_sha256,
            "checkpoint_b": rec_b.checkpoint_sha256,
            "held_constant": same,
            "changed_factors": changed,
            "single_architecture_contrast": (
                "architecture" in changed
                and "data_manifest_sha256" in same
                and "source_commit" in same
            ),
        }

    @staticmethod
    def load_checkpoint_for_causal_probing(
        target: CheckpointRecord | Path | str,
        *,
        lineage_id: str | None = None,
        checkpoint_sha256: str | None = None,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Verify SHA-256 inventory and restore a frozen checkpoint wrapped for causal diagnostics."""

        if isinstance(target, CheckpointRecord):
            store = CheckpointStore(Path(target.checkpoint_root), target.lineage_id)
            sha = checkpoint_sha256 or target.checkpoint_sha256
        else:
            root_path = Path(target)
            if lineage_id is None:
                # Caller may pass the lineage_root directory itself containing LATEST
                if (root_path / "LATEST").is_file() or (root_path / MILESTONES_FILENAME).is_file():
                    resolved = _resolve_store_for_lineage_dir(root_path, sample_sha=checkpoint_sha256)
                    if resolved is None:
                        raise ValueError(f"cannot resolve CheckpointStore from lineage directory: {root_path}")
                    store = resolved
                else:
                    raise ValueError("lineage_id is required when target is a checkpoint store root")
            else:
                store = CheckpointStore(root_path, lineage_id)
            sha = checkpoint_sha256 or store.latest_sha256()
            if sha is None:
                raise ValueError(f"no LATEST checkpoint committed for lineage {store.lineage_id}")

        state, payloads = store.restore(sha)
        sched = json.loads(payloads["scheduler.json"])
        weights = _load(payloads["model.bin"])
        return _instantiate_checkpoint_bundle(
            sha=sha,
            state=state,
            sched=sched,
            weights=weights,
            payload_keys=list(payloads.keys()),
            mode=mode,
            copy_enabled_override=copy_enabled_override,
            device=device,
        )

    @classmethod
    def load_from_result_zip(
        cls,
        result_zip: Path | str,
        *,
        experiment_id: str | None = None,
        lineage_id: str | None = None,
        expected_source_commit: str | None = None,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Load a frozen model + policy directly from a verified result.zip (full_embedded or compact_fp16)."""

        verified = verify_single_planthir_zip(
            result_zip, expected_source_commit=expected_source_commit
        )
        index_doc = verified["checkpoint_index"]
        entries = list(index_doc.get("entries", []))
        if index_doc.get("checkpoint_mode_in_result_zip") == "sidecar_only":
            raise ValueError("result.zip contains evidence only; restore checkpoint_full.zip with restore_full_checkpoint_zip")

        candidates = [
            entry
            for entry in entries
            if entry.get("checkpoint_prefix") and entry.get("latest_checkpoint_sha256")
        ]
        if experiment_id is not None:
            candidates = [e for e in candidates if e.get("experiment_id") == experiment_id]
        if lineage_id is not None:
            candidates = [e for e in candidates if e.get("lineage_id") == lineage_id]

        if not candidates:
            raise ValueError(
                f"no matching checkpoint entry in {result_zip!r} "
                f"(experiment_id={experiment_id!r}, lineage_id={lineage_id!r})"
            )

        unique_targets = {
            (str(e["lineage_id"]), str(e["latest_checkpoint_sha256"])) for e in candidates
        }
        if len(unique_targets) > 1:
            raise ValueError(
                f"ambiguous checkpoint selection ({len(unique_targets)} targets); "
                "specify experiment_id or lineage_id"
            )
        selected = candidates[0]
        checkpoint_prefix = str(selected["checkpoint_prefix"])
        sel_lineage = str(selected["lineage_id"])
        sel_sha = str(selected["latest_checkpoint_sha256"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            with zipfile.ZipFile(result_zip, "r") as zf:
                ckpt_members = [
                    m
                    for m in zf.namelist()
                    if m.startswith(checkpoint_prefix) and not m.endswith("/")
                ]
                if not ckpt_members:
                    raise ValueError(
                        f"checkpoint files missing inside {result_zip!r} for prefix {checkpoint_prefix!r}"
                    )
                _safe_extract_members(zf, ckpt_members, tmp_root)

            obj_dir = tmp_root / "checkpoints" / sel_lineage / "objects" / sel_sha
            if not obj_dir.is_dir():
                raise ValueError(f"missing extracted checkpoint object directory for {sel_lineage}:{sel_sha}")

            if (obj_dir / "model.bin").is_file() and (obj_dir / "optimizer.bin").is_file():
                sub_catalog = cls.discover_checkpoints(
                    [tmp_root / "checkpoints"], include_all_generations=True
                )
                matched = [
                    rec
                    for rec in sub_catalog.records
                    if rec.lineage_id == sel_lineage and rec.checkpoint_sha256 == sel_sha
                ]
                if len(matched) != 1:
                    raise ValueError(
                        f"failed to discover extracted full checkpoint {sel_lineage}:{sel_sha}"
                    )
                bundle = cls.load_checkpoint_for_causal_probing(
                    matched[0],
                    mode=mode,
                    copy_enabled_override=copy_enabled_override,
                    device=device,
                )
                source_archive = "result.zip (full_embedded)"
            else:
                bundle = _restore_compact_fp16_object(
                    obj_dir,
                    expected_lineage=sel_lineage,
                    expected_sha=sel_sha,
                    mode=mode,
                    copy_enabled_override=copy_enabled_override,
                    device=device,
                )
                source_archive = "result.zip (compact_fp16)"

        bundle["source_archive"] = source_archive
        bundle["weights_lossy"] = source_archive.endswith("(compact_fp16)")
        bundle["resumable"] = source_archive.endswith("(full_embedded)")
        bundle["whole_checkpoint_embedded"] = bool(index_doc.get("whole_checkpoint_embedded", False))
        bundle["bundle_status"] = verified["final_status"].get("status")
        bundle["experiment_id"] = selected.get("experiment_id")
        return bundle

    @staticmethod
    def _validate_full_restore(dest: Path, verified: Mapping[str, Any]) -> None:
        checkpoints_root = dest / "checkpoints"
        expected = {
            (str(e["lineage_id"]), str(e["latest_checkpoint_sha256"]))
            for e in verified["checkpoint_index"].get("entries", [])
            if e.get("lineage_id") and e.get("latest_checkpoint_sha256")
        }
        for lineage, sha in expected:
            if not _is_sha256(sha) or Path(lineage).name != lineage or lineage in (".", ".."):
                raise ValueError("invalid checkpoint index target")
            store = CheckpointStore(checkpoints_root, lineage)
            if store.latest_sha256() != sha:
                raise ValueError(f"restored lineage LATEST mismatch: {lineage}")
            store.restore(sha)
        if checkpoints_root.is_dir():
            for ldir in checkpoints_root.iterdir():
                if ldir.is_dir():
                    store = CheckpointStore(checkpoints_root, ldir.name)
                    if store.latest_sha256() is None:
                        raise ValueError(f"restored lineage missing LATEST commit: {ldir.name}")
                    store.restore(store.latest_sha256())
                    if store.objects.is_dir():
                        for obj in store.objects.iterdir():
                            if obj.is_dir():
                                store.restore(obj.name)

    @classmethod
    def restore_full_checkpoint_zip(
        cls,
        full_zip: Path | str,
        dest_dir: Path | str,
        *,
        expected_source_commit: str | None = None,
    ) -> dict[str, Any]:
        """Verify and extract checkpoint_full.zip, validating full CheckpointStore restoration."""

        verified = verify_single_planthir_zip(
            full_zip, expected_source_commit=expected_source_commit
        )
        if verified["manifest"].get("kind") != "checkpoint_full":
            raise ValueError("archive is not a full checkpoint archive")
        dest = Path(dest_dir)
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            raise ValueError(f"restore destination must be absent or empty: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            staged = Path(tmp) / "restore"
            with zipfile.ZipFile(full_zip, "r") as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                _safe_extract_members(zf, members, staged)
            cls._validate_full_restore(staged, verified)
            if dest.exists():
                dest.rmdir()
            staged.rename(dest)

        checkpoints_root = dest / "checkpoints"
        verified_lineages: list[dict[str, Any]] = []
        if checkpoints_root.is_dir():
            for ldir in sorted(p for p in checkpoints_root.iterdir() if p.is_dir()):
                if not ((ldir / "LATEST").is_file() or (ldir / MILESTONES_FILENAME).is_file()):
                    continue
                store = CheckpointStore(checkpoints_root, ldir.name)
                latest_sha = store.latest_sha256()
                if latest_sha is None:
                    raise ValueError(f"restored lineage missing LATEST commit: {ldir.name}")
                latest_state, _ = store.restore(latest_sha)
                object_count = 0
                if store.objects.is_dir():
                    for obj in sorted(store.objects.iterdir()):
                        if obj.is_dir():
                            store.restore(obj.name)
                            object_count += 1
                verified_lineages.append({
                    "lineage_id": ldir.name,
                    "latest_checkpoint_sha256": latest_sha,
                    "global_update": int(latest_state.global_update),
                    "object_count": object_count,
                    "restored": True,
                })

        catalog = (
            cls.discover_checkpoints([checkpoints_root], include_all_generations=True)
            if checkpoints_root.is_dir()
            else cls(())
        )
        return {
            "verified": True,
            "dest_dir": str(dest),
            "checkpoints_root": str(checkpoints_root),
            "catalog": catalog,
            "verified_lineages": verified_lineages,
            "checkpoint_count": len(verified_lineages),
            "manifest_sha256": verified["manifest"]["manifest_sha256"],
        }


__all__ = ["CheckpointCatalog", "CheckpointRecord"]
