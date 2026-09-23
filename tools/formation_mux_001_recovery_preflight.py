"""Fail-closed exact-recovery preflight for the 2026-09-23 FORMATION-MUX run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from v5_experiments import formation_mux_protocol_v5 as s5_protocol
from v5_experiments import formation_mux_surface_v5 as public_surface
from v5_experiments import tie_role_protocol_v1 as frontier_protocol

SCHEMA = "anra.formation-mux-recovery-preflight/v2"
SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
OPERATOR_COMMIT = "4ee05f6e386f15d34f9dfa7bd7f3300a496b9896"
OPERATOR_BLOB = "e9e1f701b0d4edc509194da55fe1ba37ed62ef86"
PUBLIC_SURFACE_SHA256 = "f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c"
TOKENIZER_SHA256 = "97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc"
FRONTIER_EXTENSION = "TIE-ROLE-FRONTIER-001"
SEED_LABELS = tuple(f"S{index}" for index in range(1, 5))
S5_ARMS = {
    s5_protocol.EXPERIMENT_A: tuple(s5_protocol.ARMS_A),
    s5_protocol.EXPERIMENT_B: tuple(s5_protocol.ARMS_B),
}
FRONTIER_ARMS = {
    frontier_protocol.EXPERIMENT_A: tuple(frontier_protocol.ARMS_A),
    frontier_protocol.EXPERIMENT_B: tuple(frontier_protocol.ARMS_B),
}
REQUIRED_FRONTIER_COMPLETE = {
    (frontier_protocol.EXPERIMENT_A, "T0_CANONICAL", "S1"),
    (frontier_protocol.EXPERIMENT_A, "T0_CANONICAL", "S2"),
}
CHECKPOINT_STATE_FIELDS = (
    "updates",
    "processed_tokens",
    "supervised_tokens",
    "stream_cursor",
    "timing",
    "last_eval_axis",
    "last_checkpoint_axis",
    "clip_events",
)
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
NONCE = re.compile(r"^[0-9a-f]{64}$")


class RecoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfficialSlot:
    experiment: str
    arm: str
    seed_label: str
    seed_bundle: int
    protocol_sha256: str
    exposure_axis: str
    exposure_target: int
    eligible_from: int
    progress_schema: str
    campaign: str

    @property
    def key(self) -> str:
        return f"{self.experiment}/{self.arm}/{self.seed_label}"

    @property
    def relative_directory(self) -> str:
        return f"{self.experiment}/{self.arm}/{self.seed_label}"

    def endpoint_reached(self, updates: int, processed_tokens: int) -> bool:
        if self.exposure_axis == "updates":
            return updates == self.exposure_target
        return processed_tokens >= self.exposure_target


def _slots(
    arms_by_experiment: Mapping[str, tuple[str, ...]],
    seed_bundles: tuple[int, ...],
    protocol_sha: Callable[[str], str],
    a_experiment: str,
    a_updates: int,
    a_eligible: int,
    b_tokens: int,
    b_eligible: int,
    progress_a_schema: str,
    progress_b_schema: str,
    campaign: str,
) -> tuple[OfficialSlot, ...]:
    rows = []
    for experiment, arms in arms_by_experiment.items():
        for arm in arms:
            for seed_label, seed_bundle in zip(SEED_LABELS, seed_bundles, strict=True):
                is_a = experiment == a_experiment
                rows.append(OfficialSlot(
                    experiment=experiment,
                    arm=arm,
                    seed_label=seed_label,
                    seed_bundle=seed_bundle,
                    protocol_sha256=protocol_sha(experiment),
                    exposure_axis="updates" if is_a else "processed_nonpadding_tokens",
                    exposure_target=a_updates if is_a else b_tokens,
                    eligible_from=a_eligible if is_a else b_eligible,
                    progress_schema=progress_a_schema if is_a else progress_b_schema,
                    campaign=campaign,
                ))
    return tuple(rows)


S5_SLOTS = _slots(
    S5_ARMS,
    tuple(s5_protocol.SEED_BUNDLES),
    s5_protocol.protocol_sha,
    s5_protocol.EXPERIMENT_A,
    s5_protocol.A_UPDATES,
    s5_protocol.A_ELIGIBLE_FROM_UPDATE,
    s5_protocol.B_PROCESSED_TOKEN_BUDGET,
    s5_protocol.B_ELIGIBLE_FROM_TOKENS,
    "anra.formation-mux-progress/v5",
    "anra.formation-mux-progress/v5",
    "FORMATION-MUX-001",
)
FRONTIER_SLOTS = _slots(
    FRONTIER_ARMS,
    tuple(frontier_protocol.SEED_BUNDLES),
    frontier_protocol.protocol_sha,
    frontier_protocol.EXPERIMENT_A,
    frontier_protocol.A_UPDATES,
    frontier_protocol.A_ELIGIBLE_FROM_UPDATE,
    frontier_protocol.B_PROCESSED_TOKEN_BUDGET,
    frontier_protocol.B_ELIGIBLE_FROM_TOKENS,
    "anra.tie-role-progress/v1",
    "anra.tie-role-progress/v1",
    FRONTIER_EXTENSION,
)
OFFICIAL_SLOTS = S5_SLOTS + FRONTIER_SLOTS
SLOT_BY_KEY = {slot.key: slot for slot in OFFICIAL_SLOTS}
S5_KEYS = frozenset(slot.key for slot in S5_SLOTS)
FRONTIER_KEYS = frozenset(slot.key for slot in FRONTIER_SLOTS)
REQUIRED_ARCHIVE_MEMBERS = frozenset(
    f"{slot.relative_directory}/resume.pt" for slot in OFFICIAL_SLOTS
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecoveryError(f"required recovery metadata missing: {path}") from exc
    except (OSError, ValueError) as exc:
        raise RecoveryError(f"invalid recovery metadata: {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise RecoveryError(f"recovery metadata is not an object: {path}")
    return body


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_integer(value: Any, label: str, path: Path, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecoveryError(f"{label} is not an integer >= {minimum}: {path}: {value!r}")
    return int(value)


def _require_mapping(value: Any, label: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryError(f"{label} is not a mapping: {path}")
    return value


def _require_identity(body: Mapping[str, Any], slot: OfficialSlot, path: Path) -> None:
    expected = {
        "experiment": slot.experiment,
        "arm": slot.arm,
        "seed_bundle": slot.seed_bundle,
        "protocol_sha256": slot.protocol_sha256,
        "data_manifest_sha256": PUBLIC_SURFACE_SHA256,
        "engineering_only": False,
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise RecoveryError(
                f"{key} identity mismatch for {slot.key}: {path}: "
                f"{body.get(key)!r} != {value!r}"
            )


def _safe_member_name(name: str) -> PurePosixPath:
    if "\\" in name:
        raise RecoveryError(f"ZIP member uses a Windows path separator: {name}")
    member = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        member.is_absolute()
        or windows.drive
        or windows.root
        or ".." in member.parts
        or not member.parts
        or any(":" in part for part in member.parts)
    ):
        raise RecoveryError(f"unsafe ZIP member path: {name}")
    return member


def _archive_info(archive: Path) -> dict[str, Any] | None:
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RecoveryError(f"invalid recovery ZIP: {archive}: {exc}") from exc
    with handle:
        names = []
        for info in handle.infolist():
            _safe_member_name(info.filename)
            names.append(info.filename)
        folded = [name.casefold() for name in names]
        if len(folded) != len(set(folded)):
            raise RecoveryError("recovery ZIP contains case-ambiguous member names")
        suffix = "FORMATION_MUX_001/CAMPAIGN_STATE.json"
        states = [name for name in names if name == suffix or name.endswith("/" + suffix)]
        if not states:
            return None
        if len(states) != 1:
            raise RecoveryError(
                f"recovery ZIP must contain exactly one {suffix}; found {len(states)}"
            )
        prefix = states[0][:-len("CAMPAIGN_STATE.json")]
        relative = {
            name[len(prefix):]
            for name in names
            if name.startswith(prefix)
        }
        official_count = len(REQUIRED_ARCHIVE_MEMBERS.intersection(relative))
        return {
            "prefix": prefix,
            "official_checkpoint_count": official_count,
            "evidence_only": official_count == 0,
        }


def _extract_archive(archive: Path, destination: Path) -> int:
    info = _archive_info(archive)
    if info is None:
        raise RecoveryError(f"recovery archive has no FORMATION-MUX campaign state: {archive}")
    if info["evidence_only"]:
        raise RecoveryError(
            "EVIDENCE_ONLY_NOT_RESUMABLE: the ZIP has campaign JSON but no official "
            "resume.pt checkpoint payloads; download the complete saved Kaggle Output tree"
        )
    if destination.exists():
        raise RecoveryError(f"refusing to overwrite recovery staging path: {destination}")
    destination.mkdir(parents=True)
    prefix = str(info["prefix"])
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                name = _safe_member_name(member.filename)
                if not member.filename.startswith(prefix):
                    continue
                relative_name = member.filename[len(prefix):]
                if not relative_name:
                    continue
                relative = _safe_member_name(relative_name)
                folded = relative_name.casefold()
                if folded in seen:
                    raise RecoveryError(
                        f"recovery ZIP contains duplicate extracted member: {relative_name}"
                    )
                seen.add(folded)
                target = destination.joinpath(*relative.parts)
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise RecoveryError(
                        f"recovery ZIP contains a symlink: {member.filename}"
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
    except Exception as exc:
        raise RecoveryError(
            "recovery archive extraction failed; preserved staging path for inspection: "
            f"{destination}: {exc}"
        ) from exc
    return int(info["official_checkpoint_count"])


def _reject_symlink_ancestors(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise RecoveryError(f"recovery path contains a symlink component: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _validate_filesystem_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RecoveryError(f"recovery campaign root is not a regular directory: {root}")
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            if path.is_symlink():
                raise RecoveryError(f"recovery tree contains a symlink: {path}")
            if not path.is_dir() and not path.is_file():
                raise RecoveryError(f"recovery tree contains a non-regular entry: {path}")


def _locate(input_path: Path) -> dict[str, Any]:
    _reject_symlink_ancestors(input_path)
    if input_path.is_file():
        if not zipfile.is_zipfile(input_path):
            raise RecoveryError(f"recovery input is not a ZIP archive: {input_path}")
        info = _archive_info(input_path)
        if info is None:
            raise RecoveryError(f"recovery ZIP has no FORMATION-MUX campaign state: {input_path}")
        if info["evidence_only"]:
            raise RecoveryError(
                "EVIDENCE_ONLY_NOT_RESUMABLE: the ZIP has campaign JSON but no official "
                "resume.pt checkpoint payloads; download the complete saved Kaggle Output tree"
            )
        return {
            "kind": "archive",
            "source": input_path,
            "prefix": info["prefix"],
            "archive_checkpoint_count": info["official_checkpoint_count"],
        }
    if not input_path.exists():
        raise RecoveryError(f"recovery input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise RecoveryError(f"recovery input is neither a directory nor a ZIP: {input_path}")
    if (input_path / "CAMPAIGN_STATE.json").is_file():
        return {"kind": "directory", "source": input_path}
    candidates = sorted(
        path.parent
        for path in input_path.rglob("FORMATION_MUX_001/CAMPAIGN_STATE.json")
        if path.is_file()
    )
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise RecoveryError(f"ambiguous recovery input; preserve each run separately: {names}")
    archives = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".zip"
    )
    resumable = []
    evidence = []
    for archive in archives:
        try:
            info = _archive_info(archive)
        except RecoveryError:
            if "FORMATION_MUX_001" in archive.name.upper():
                raise
            continue
        if info is None:
            continue
        if info["evidence_only"]:
            evidence.append(archive)
        else:
            resumable.append((archive, info))
    if len(candidates) + len(resumable) > 1:
        names = ", ".join(
            [*(str(path) for path in candidates), *(str(item[0]) for item in resumable)]
        )
        raise RecoveryError(
            f"ambiguous checkpoint-bearing recovery input; attach exactly one saved Output: {names}"
        )
    if len(candidates) == 1:
        return {"kind": "directory", "source": candidates[0]}
    if len(resumable) == 1:
        archive, info = resumable[0]
        return {
            "kind": "archive",
            "source": archive,
            "prefix": info["prefix"],
            "archive_checkpoint_count": info["official_checkpoint_count"],
        }
    if evidence:
        names = ", ".join(str(path) for path in evidence)
        raise RecoveryError(
            "EVIDENCE_ONLY_NOT_RESUMABLE: only result ZIP(s) were found, not a saved "
            f"Output tree with official resume.pt files: {names}"
        )
    raise RecoveryError(
        f"no FORMATION_MUX_001 campaign tree or checkpoint-bearing ZIP found under: "
        f"{input_path}"
    )


def _validate_layout(root: Path) -> None:
    for experiment, arms in {**S5_ARMS, **FRONTIER_ARMS}.items():
        experiment_root = root / experiment
        if not experiment_root.exists():
            continue
        if not experiment_root.is_dir():
            raise RecoveryError(f"official experiment path is not a directory: {experiment_root}")
        observed_arms = {path.name for path in experiment_root.iterdir() if path.is_dir()}
        unexpected_arms = observed_arms - set(arms)
        if unexpected_arms:
            raise RecoveryError(
                f"unregistered official experiment arms under {experiment}: "
                + ", ".join(sorted(unexpected_arms))
            )
        for arm in arms:
            arm_root = experiment_root / arm
            if not arm_root.exists():
                continue
            if not arm_root.is_dir():
                raise RecoveryError(f"official arm path is not a directory: {arm_root}")
            observed_seeds = {path.name for path in arm_root.iterdir() if path.is_dir()}
            unexpected_seeds = observed_seeds - set(SEED_LABELS)
            if unexpected_seeds:
                raise RecoveryError(
                    f"unregistered seed directories under {arm_root}: "
                    + ", ".join(sorted(unexpected_seeds))
                )


def _validate_public_manifest(manifest: Mapping[str, Any]) -> None:
    try:
        public_surface.validate_public_surface(manifest)
    except Exception as exc:
        raise RecoveryError(f"pinned public-surface validation failed: {exc}") from exc
    if manifest.get("sha256") != PUBLIC_SURFACE_SHA256:
        raise RecoveryError(
            f"public surface identity mismatch: {manifest.get('sha256')} != "
            f"{PUBLIC_SURFACE_SHA256}"
        )
    if manifest.get("tokenizer_artifact_sha256") != TOKENIZER_SHA256:
        raise RecoveryError(
            "public surface tokenizer identity mismatch: "
            f"{manifest.get('tokenizer_artifact_sha256')}"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"training", "development"}:
        raise RecoveryError("public surface split set is not exactly training+development")
    if manifest.get("sealed_rows_persisted") is not False:
        raise RecoveryError("public surface must declare sealed_rows_persisted=false")


def _formation_summary(trace: list[Mapping[str, Any]], eligible_from: int) -> dict[str, Any]:
    points = [
        (int(row["axis"]), float(row["identity_exact_valid_eos"]))
        for row in trace
        if int(row["axis"]) >= int(eligible_from)
    ]
    if not points:
        return {
            "formation_auc": 0.0,
            "endpoint": 0.0,
            "first_acquisition_axis": None,
            "points": 0,
        }
    if len(points) == 1:
        area_score = points[0][1]
    else:
        area = 0.0
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            area += (x1 - x0) * (y0 + y1) / 2.0
        span = max(points[-1][0] - points[0][0], 1)
        area_score = area / span
    first = next((axis for axis, score in points if score >= 0.5), None)
    return {
        "formation_auc": round(float(area_score), 6),
        "endpoint": round(float(points[-1][1]), 6),
        "first_acquisition_axis": first,
        "points": len(points),
    }


def _safe_checkpoint_load(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RecoveryError(
            "PyTorch is required for safe checkpoint inspection on Kaggle"
        ) from exc
    try:
        body = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RecoveryError(
            f"safe checkpoint load failed with weights_only=True: {path}: {exc}"
        ) from exc
    if not isinstance(body, Mapping):
        raise RecoveryError(f"checkpoint payload is not a mapping: {path}")
    return body


def _require_tensor(
    value: Any,
    label: str,
    path: Path,
    *,
    dimensions: tuple[int, ...],
    dtype_name: str | None = None,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for checkpoint tensor inspection") from exc
    if not isinstance(value, torch.Tensor):
        raise RecoveryError(f"{label} is not a torch.Tensor: {path}")
    if value.ndim not in dimensions:
        raise RecoveryError(
            f"{label} has {value.ndim} dimensions, expected one of {dimensions}: {path}"
        )
    if dtype_name is not None and str(value.dtype) != dtype_name:
        raise RecoveryError(
            f"{label} dtype is {value.dtype}, expected {dtype_name}: {path}"
        )
    return value


def _validate_checkpoint(
    root: Path,
    slot: OfficialSlot,
    result: dict[str, Any] | None,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any] | None:
    directory = root.joinpath(*slot.relative_directory.split("/"))
    checkpoint = directory / "resume.pt"
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    progress_path = directory / "LATEST_PROGRESS.json"
    if not checkpoint.exists():
        evidence = []
        if result is not None:
            evidence.append("ARM_RESULT.json")
        if receipt_path.exists():
            evidence.append("CHECKPOINT_RECEIPT.json")
        if progress_path.exists():
            evidence.append("LATEST_PROGRESS.json")
        if (directory / "progress").exists():
            evidence.append("progress/")
        if directory.exists():
            evidence.extend(path.name for path in directory.iterdir() if path.name.endswith(".tmp"))
        if evidence:
            raise RecoveryError(
                f"checkpoint missing for {slot.key} with persisted evidence: "
                + ", ".join(sorted(evidence))
            )
        return None
    if not directory.is_dir():
        raise RecoveryError(f"official checkpoint parent is not a directory: {directory}")
    if not receipt_path.is_file():
        raise RecoveryError(f"resume.pt has no CHECKPOINT_RECEIPT.json: {directory}")
    if not progress_path.is_file():
        raise RecoveryError(f"resume.pt has no LATEST_PROGRESS.json: {directory}")
    body = _require_mapping(
        checkpoint_loader(checkpoint), f"checkpoint payload for {slot.key}", checkpoint
    )
    _require_identity(body, slot, checkpoint)
    for key in ("model", "main_optimizer", "cpu_rng_state", "cuda_rng_state", "trace", "diagnostics"):
        if key not in body:
            raise RecoveryError(f"checkpoint payload missing {key}: {checkpoint}")
    model_state = _require_mapping(body["model"], "checkpoint model state", checkpoint)
    if not model_state or any(not str(key) for key in model_state):
        raise RecoveryError(f"checkpoint model state is empty or invalid: {checkpoint}")
    for key, value in model_state.items():
        _require_tensor(
            value,
            f"checkpoint model tensor {key}",
            checkpoint,
            dimensions=(0, 1, 2),
        )
    main_optimizer = _require_mapping(
        body["main_optimizer"], "checkpoint main optimizer state", checkpoint
    )
    if not isinstance(main_optimizer.get("state"), Mapping) or not isinstance(
        main_optimizer.get("param_groups"), list
    ):
        raise RecoveryError(f"checkpoint main optimizer topology is invalid: {checkpoint}")
    if "row_optimizer" not in body:
        raise RecoveryError(f"checkpoint row optimizer state is missing: {checkpoint}")
    expects_row_optimizer = slot.experiment == s5_protocol.EXPERIMENT_A
    row_optimizer = body["row_optimizer"]
    if expects_row_optimizer:
        if not isinstance(row_optimizer, Mapping) or not {
            "step_count",
            "lr",
            "exp_avg",
            "exp_avg_sq",
        }.issubset(row_optimizer):
            raise RecoveryError(
                f"S5 mechanism checkpoint lacks row optimizer state: {checkpoint}"
            )
        for key in ("exp_avg", "exp_avg_sq"):
            _require_tensor(
                row_optimizer[key],
                f"checkpoint row optimizer {key}",
                checkpoint,
                dimensions=(2,),
            )
    elif row_optimizer is not None:
        raise RecoveryError(f"checkpoint has unexpected row optimizer state: {checkpoint}")
    _require_tensor(
        body["cpu_rng_state"],
        "checkpoint CPU RNG state",
        checkpoint,
        dimensions=(1,),
        dtype_name="torch.uint8",
    )
    if not isinstance(body["cuda_rng_state"], list) or not body["cuda_rng_state"]:
        raise RecoveryError(f"checkpoint CUDA RNG state is missing: {checkpoint}")
    for index, value in enumerate(body["cuda_rng_state"]):
        _require_tensor(
            value,
            f"checkpoint CUDA RNG state[{index}]",
            checkpoint,
            dimensions=(1,),
            dtype_name="torch.uint8",
        )
    if not isinstance(body["trace"], list) or not isinstance(body["diagnostics"], list):
        raise RecoveryError(f"checkpoint trace/diagnostics state is invalid: {checkpoint}")
    if not isinstance(body["timing"], Mapping):
        raise RecoveryError(f"checkpoint timing state is invalid: {checkpoint}")
    initial_sha = body.get("initial_model_sha256")
    if not isinstance(initial_sha, str) or HEX_64.fullmatch(initial_sha) is None:
        raise RecoveryError(f"checkpoint initial model identity is malformed: {checkpoint}")
    for key in CHECKPOINT_STATE_FIELDS:
        if key not in body:
            raise RecoveryError(f"checkpoint payload missing state field {key}: {checkpoint}")
    updates = _require_integer(body["updates"], "checkpoint updates", checkpoint)
    processed_tokens = _require_integer(
        body["processed_tokens"], "checkpoint processed_tokens", checkpoint
    )
    _require_integer(body["supervised_tokens"], "checkpoint supervised_tokens", checkpoint)
    _require_integer(body["stream_cursor"], "checkpoint stream_cursor", checkpoint)
    _require_integer(body["last_eval_axis"], "checkpoint last_eval_axis", checkpoint)
    _require_integer(
        body["last_checkpoint_axis"], "checkpoint last_checkpoint_axis", checkpoint
    )
    _require_integer(body["clip_events"], "checkpoint clip_events", checkpoint)
    if updates == 0 or processed_tokens == 0:
        raise RecoveryError(f"official checkpoint has no positive training exposure: {checkpoint}")

    receipt = _read_json(receipt_path)
    actual_sha = _sha256_file(checkpoint)
    if receipt.get("sha256") != actual_sha:
        raise RecoveryError(
            f"checkpoint hash mismatch: {checkpoint}: receipt={receipt.get('sha256')} "
            f"actual={actual_sha}"
        )
    _require_identity(receipt, slot, receipt_path)
    if receipt.get("initial_model_sha256") != initial_sha:
        raise RecoveryError(f"checkpoint receipt initial model identity mismatch: {receipt_path}")
    for key in CHECKPOINT_STATE_FIELDS:
        if key not in receipt or receipt[key] != body[key]:
            raise RecoveryError(
                f"checkpoint receipt state mismatch for {key}: {slot.key}: "
                f"receipt={receipt.get(key)!r} checkpoint={body[key]!r}"
            )

    progress = _read_json(progress_path)
    if progress.get("schema") != slot.progress_schema:
        raise RecoveryError(
            f"checkpoint progress schema mismatch for {slot.key}: {progress.get('schema')}"
        )
    if "engineering_only" in progress and progress["engineering_only"] is not False:
        raise RecoveryError(f"checkpoint progress is marked engineering-only: {progress_path}")
    _require_identity(
        {**progress, "engineering_only": False},
        slot,
        progress_path,
    )
    if progress.get("checkpoint_sha256") != actual_sha:
        raise RecoveryError(f"checkpoint progress hash mismatch: {progress_path}")
    if progress.get("checkpoint_file") != "resume.pt":
        raise RecoveryError(f"checkpoint progress file identity mismatch: {progress_path}")
    if slot.campaign == FRONTIER_EXTENSION:
        if progress.get("extension") != FRONTIER_EXTENSION:
            raise RecoveryError(f"frontier progress extension mismatch: {progress_path}")
        if progress.get("science_s5_unchanged") is not True:
            raise RecoveryError(f"frontier progress S5 invariant missing: {progress_path}")
    for key in (
        "updates",
        "processed_tokens",
        "supervised_tokens",
        "timing",
        "clip_events",
    ):
        if key not in progress or progress[key] != body[key]:
            raise RecoveryError(
                f"checkpoint progress state mismatch for {key}: {slot.key}"
            )
    try:
        formation = _formation_summary(body["trace"], slot.eligible_from)
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryError(
            f"checkpoint development trace is malformed for {slot.key}: {exc}"
        ) from exc
    if progress.get("formation_so_far") != formation:
        raise RecoveryError(f"checkpoint progress formation mismatch: {progress_path}")
    latest = body["trace"][-1] if body["trace"] else None
    if progress.get("latest_development") != latest:
        raise RecoveryError(f"checkpoint progress latest development mismatch: {progress_path}")

    if result is not None:
        if result.get("schema") != "anra.formation-mux-arm-result/v2":
            raise RecoveryError(f"arm result schema mismatch: {directory / 'ARM_RESULT.json'}")
        _require_identity(result, slot, directory / "ARM_RESULT.json")
        for key in (
            "updates",
            "processed_tokens",
            "supervised_tokens",
            "stream_cursor",
        ):
            if key not in result or result[key] != body[key]:
                raise RecoveryError(
                    f"completed result/checkpoint state mismatch for {key}: {slot.key}"
                )
        if result.get("status") != "COMPLETE":
            raise RecoveryError(f"persisted ARM_RESULT.json is not COMPLETE: {slot.key}")
        if result.get("exposure_target") != slot.exposure_target:
            raise RecoveryError(f"completed result exposure target mismatch: {slot.key}")
        if result.get("exposure_axis") != slot.exposure_axis:
            raise RecoveryError(f"completed result exposure axis mismatch: {slot.key}")
        if not slot.endpoint_reached(updates, processed_tokens):
            raise RecoveryError(
                f"completed result is backed by a non-endpoint checkpoint: {slot.key}: "
                f"updates={updates} processed_tokens={processed_tokens}"
            )
        if result.get("trace") != body["trace"] or result.get("diagnostics") != body["diagnostics"]:
            raise RecoveryError(f"completed result trace/diagnostics mismatch: {slot.key}")
        result_formation = _formation_summary(body["trace"], slot.eligible_from)
        if result_formation["points"] == 0:
            raise RecoveryError(f"completed result has no eligible development trace: {slot.key}")
        if result.get("formation") != result_formation:
            raise RecoveryError(f"completed result formation mismatch: {slot.key}")
        expected_clip_fraction = body["clip_events"] / max(updates, 1)
        if not math.isclose(
            float(result.get("clip_fraction", -1.0)),
            expected_clip_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RecoveryError(f"completed result clip fraction mismatch: {slot.key}")
    return {
        "path": f"{slot.relative_directory}/resume.pt",
        "slot": slot.key,
        "sha256": actual_sha,
        "byte_size": checkpoint.stat().st_size,
        "updates": updates,
        "processed_tokens": processed_tokens,
        "supervised_tokens": body["supervised_tokens"],
        "result_complete": result is not None,
        "endpoint_reached": slot.endpoint_reached(updates, processed_tokens),
    }


def _validate_state_maps(
    campaign: Mapping[str, Any],
    frontier_state: Mapping[str, Any],
    completed: set[str],
) -> None:
    if campaign.get("schema") != "anra.formation-mux-state/v4":
        raise RecoveryError(f"S5 campaign state schema mismatch: {campaign.get('schema')}")
    if campaign.get("science_commit") != SCIENCE_COMMIT:
        raise RecoveryError(
            f"science identity mismatch: {campaign.get('science_commit')} != {SCIENCE_COMMIT}"
        )
    if campaign.get("global_failure") is not None:
        raise RecoveryError(
            f"S5 latched global failure blocks exact continuation: {campaign.get('global_failure')}"
        )
    if campaign.get("status") != "ARMS_COMPLETE":
        raise RecoveryError("S5 is not ARMS_COMPLETE in the recovered Output tree")
    complete_arms = _require_integer(
        campaign.get("complete_arms"),
        "S5 state complete_arms",
        Path("CAMPAIGN_STATE.json"),
    )
    required_arms = _require_integer(
        campaign.get("required_arms"),
        "S5 state required_arms",
        Path("CAMPAIGN_STATE.json"),
    )
    if complete_arms != 24 or required_arms != 24:
        raise RecoveryError("S5 state does not record an exact 24/24 official inventory")
    campaign_arms = _require_mapping(
        campaign.get("arms"), "S5 state arms", Path("CAMPAIGN_STATE.json")
    )
    if set(campaign_arms) != S5_KEYS:
        raise RecoveryError("S5 state arms map is not exactly the 24 verified official slots")
    if any(value != "COMPLETE" for value in campaign_arms.values()):
        raise RecoveryError("S5 state contains a non-COMPLETE official arm entry")
    completed_seed_bundles = campaign.get("completed_seed_bundles")
    if not isinstance(completed_seed_bundles, list) or set(completed_seed_bundles) != set(
        s5_protocol.SEED_BUNDLES
    ):
        raise RecoveryError("S5 completed seed bundle state mismatch")
    if campaign.get("pending_seed_bundles") not in ([], None):
        raise RecoveryError("S5 state still reports pending seed bundles")

    if frontier_state.get("schema") != "anra.tie-role-frontier-state/v1":
        raise RecoveryError(
            f"frontier state schema mismatch: {frontier_state.get('schema')}"
        )
    if frontier_state.get("extension") != FRONTIER_EXTENSION:
        raise RecoveryError(f"frontier extension mismatch: {frontier_state.get('extension')}")
    if frontier_state.get("global_failure") is not None:
        raise RecoveryError(
            "frontier latched global failure blocks v12 continuation: "
            f"{frontier_state.get('global_failure')}"
        )
    observed_frontier = completed.intersection(FRONTIER_KEYS)
    recorded_frontier = _require_integer(
        frontier_state.get("complete_arms"),
        "frontier state complete_arms",
        Path("TIE_ROLE_FRONTIER_STATE.json"),
    )
    required_frontier = _require_integer(
        frontier_state.get("required_arms"),
        "frontier state required_arms",
        Path("TIE_ROLE_FRONTIER_STATE.json"),
    )
    if recorded_frontier != len(observed_frontier):
        raise RecoveryError(
            f"frontier state count mismatch: recorded={recorded_frontier} "
            f"observed={len(observed_frontier)}"
        )
    if required_frontier != 24:
        raise RecoveryError("frontier state required_arms is not 24")
    expected_status = "ARMS_COMPLETE" if len(observed_frontier) == 24 else "PARTIAL_SESSION"
    if frontier_state.get("status") != expected_status:
        raise RecoveryError(
            f"frontier status/count mismatch: status={frontier_state.get('status')} "
            f"expected={expected_status}"
        )
    frontier_arms = _require_mapping(
        frontier_state.get("arms"), "frontier state arms", Path("TIE_ROLE_FRONTIER_STATE.json")
    )
    if not set(frontier_arms).issubset(FRONTIER_KEYS):
        raise RecoveryError("frontier state contains unregistered arm keys")
    for key, value in frontier_arms.items():
        if value not in {"COMPLETE", "ENGINEERING_FAILURE"}:
            raise RecoveryError(f"frontier state has invalid arm status {key}={value!r}")
    if {key for key, value in frontier_arms.items() if value == "COMPLETE"} != observed_frontier:
        raise RecoveryError("frontier state COMPLETE map disagrees with verified checkpoint inventory")
    complete_bundles = []
    pending_bundles = []
    for seed_label, seed_bundle in zip(SEED_LABELS, frontier_protocol.SEED_BUNDLES, strict=True):
        bundle_slots = {
            f"{experiment}/{arm}/{seed_label}"
            for experiment, arms in FRONTIER_ARMS.items()
            for arm in arms
        }
        (complete_bundles if bundle_slots.issubset(observed_frontier) else pending_bundles).append(
            seed_bundle
        )
    if frontier_state.get("completed_seed_bundles") != complete_bundles:
        raise RecoveryError("frontier completed seed bundle list mismatch")
    if frontier_state.get("pending_seed_bundles") != pending_bundles:
        raise RecoveryError("frontier pending seed bundle list mismatch")


def _validate_final_result(
    experiment: str,
    final: Mapping[str, Any],
    public_manifest: Mapping[str, Any],
) -> None:
    if experiment in S5_ARMS:
        expected = {
            "schema": "anra.formation-mux-final-result/v5",
            "science_commit": SCIENCE_COMMIT,
            "experiment": experiment,
            "sealed_commitment_sha256": public_manifest["sealed_commitments"][experiment],
            "raw_sealed_rows_persisted": False,
            "sealed_consumed": True,
        }
    else:
        expected = {
            "schema": "anra.tie-role-final-result/v1",
            "extension": FRONTIER_EXTENSION,
            "experiment": experiment,
            "sealed_source_commitment_experiment": s5_protocol.EXPERIMENT_A,
            "sealed_source_commitment_sha256": public_manifest["sealed_commitments"][
                s5_protocol.EXPERIMENT_A
            ],
            "raw_sealed_rows_persisted": False,
            "sealed_consumed": True,
        }
    for key, value in expected.items():
        if final.get(key) != value:
            raise RecoveryError(
                f"sealed final identity mismatch for {experiment}: {key}: "
                f"{final.get(key)!r} != {value!r}"
            )
    if not isinstance(final.get("contrast_results"), list) or not final["contrast_results"]:
        raise RecoveryError(f"sealed final contrast results missing for {experiment}")


def _validate_sealed_markers(
    root: Path,
    public_manifest: Mapping[str, Any],
    campaign: Mapping[str, Any],
    frontier_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    receipts = []
    for experiment in (*S5_ARMS, *FRONTIER_ARMS):
        marker_name = (
            "TIE_ROLE_SEALED_MARKER.json"
            if experiment in FRONTIER_ARMS
            else "SEALED_MARKER.json"
        )
        marker_path = root / experiment / marker_name
        final_path = root / experiment / "FINAL_RESULT.json"
        marker: dict[str, Any] = {"state": "NOT_CONSUMED"}
        if marker_path.exists():
            marker = _read_json(marker_path)
            if marker.get("experiment") != experiment:
                raise RecoveryError(f"sealed marker experiment mismatch: {marker_path}")
            if experiment in FRONTIER_ARMS and marker.get("schema") != "anra.tie-role-sealed-marker/v1":
                raise RecoveryError(f"frontier sealed marker schema mismatch: {marker_path}")
        state = marker.get("state")
        if state not in {"NOT_CONSUMED", "STARTED", "COMPLETE"}:
            raise RecoveryError(f"unknown sealed marker state for {experiment}: {state!r}")
        if state == "STARTED":
            raise RecoveryError(
                f"sealed custody is STARTED and requires manual transaction review: {marker_path}"
            )
        if state == "COMPLETE" and not final_path.exists():
            raise RecoveryError(f"sealed marker COMPLETE but final result missing: {marker_path}")
        if state != "COMPLETE" and final_path.exists():
            raise RecoveryError(f"sealed final exists without COMPLETE marker: {final_path}")
        if state == "COMPLETE":
            _validate_final_result(experiment, _read_json(final_path), public_manifest)
        state_map = campaign if experiment in S5_ARMS else frontier_state
        sealed_map = _require_mapping(state_map.get("sealed"), f"sealed state map for {experiment}", root)
        if set(sealed_map) != set(S5_ARMS if experiment in S5_ARMS else FRONTIER_ARMS):
            raise RecoveryError(f"sealed state map has the wrong experiment set for {experiment}")
        if sealed_map.get(experiment) != state:
            raise RecoveryError(
                f"sealed state/marker mismatch for {experiment}: "
                f"state={sealed_map.get(experiment)!r} marker={state!r}"
            )
        receipts.append({
            "experiment": experiment,
            "marker": f"{experiment}/{marker_name}",
            "state": state,
            "final_result_present": final_path.exists(),
        })
    return receipts


def _validate_tokenizer_receipt(
    receipt: Mapping[str, Any],
    public_manifest: Mapping[str, Any],
) -> None:
    expected = {
        "artifact_sha256": TOKENIZER_SHA256,
        "vocabulary_size": 24_576,
        "training_rows": 60_000,
        "raw_sealed_rows_persisted": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RecoveryError(
                f"tokenizer receipt mismatch for {key}: {receipt.get(key)!r} != {value!r}"
            )
    commitments = receipt.get("sealed_commitments")
    expected_commitments = public_manifest.get("sealed_commitments")
    if commitments != expected_commitments or not isinstance(commitments, Mapping):
        raise RecoveryError("tokenizer receipt sealed commitments disagree with public manifest")


def validate_source(
    source: Path,
    *,
    located: Mapping[str, Any],
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    expected_archive_checkpoint_count: int | None = None,
) -> dict[str, Any]:
    _validate_filesystem_tree(source)
    _validate_layout(source)
    campaign = _read_json(source / "CAMPAIGN_STATE.json")
    public_manifest = _read_json(source / "PUBLIC_SURFACE_MANIFEST.json")
    _validate_public_manifest(public_manifest)
    tokenizer = _read_json(source / "TOKENIZER_RECEIPT.json")
    _validate_tokenizer_receipt(tokenizer, public_manifest)
    if (source / "SURFACE_MANIFEST.json").exists():
        raise RecoveryError("legacy raw sealed-containing SURFACE_MANIFEST.json detected")
    frontier_state = _read_json(source / "TIE_ROLE_FRONTIER_STATE.json")
    loader = checkpoint_loader or _safe_checkpoint_load
    completed: set[str] = set()
    checkpoints = []
    for slot in OFFICIAL_SLOTS:
        directory = source.joinpath(*slot.relative_directory.split("/"))
        result_path = directory / "ARM_RESULT.json"
        result = _read_json(result_path) if result_path.exists() else None
        if result is not None and result.get("status") != "COMPLETE":
            raise RecoveryError(f"persisted ARM_RESULT.json is not COMPLETE: {result_path}")
        checkpoint = _validate_checkpoint(source, slot, result, loader)
        if checkpoint is not None:
            checkpoints.append(checkpoint)
        if result is not None:
            completed.add(slot.key)
    s5_completed = completed.intersection(S5_KEYS)
    frontier_completed = completed.intersection(FRONTIER_KEYS)
    if len(s5_completed) != 24:
        raise RecoveryError(f"S5 verified completed slots {len(s5_completed)} != 24")
    missing_controls = REQUIRED_FRONTIER_COMPLETE - frontier_completed
    if missing_controls:
        raise RecoveryError(
            "required completed frontier controls missing: "
            + ", ".join("/".join(item) for item in sorted(missing_controls))
        )
    if len(checkpoints) < 26:
        raise RecoveryError(
            f"verified official checkpoint count {len(checkpoints)} is below 26"
        )
    _validate_state_maps(campaign, frontier_state, completed)
    sealed = _validate_sealed_markers(
        source,
        public_manifest,
        campaign,
        frontier_state,
    )
    if expected_archive_checkpoint_count is not None and expected_archive_checkpoint_count != len(checkpoints):
        raise RecoveryError(
            "archive official checkpoint count disagrees with extracted verified inventory: "
            f"archive={expected_archive_checkpoint_count} extracted={len(checkpoints)}"
        )
    inventory_digest = _canonical_sha256([
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "updates": row["updates"],
            "processed_tokens": row["processed_tokens"],
        }
        for row in checkpoints
    ])
    nonce = os.environ.get("FORMATION_MUX_RECOVERY_NONCE")
    if nonce is not None and NONCE.fullmatch(nonce) is None:
        raise RecoveryError("FORMATION_MUX_RECOVERY_NONCE must be 64 lowercase hex characters")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "recovery_only": True,
        "recovery_nonce": nonce,
        "preflight_source_sha256": _sha256_file(Path(__file__)),
        "science_commit": SCIENCE_COMMIT,
        "operator_commit": OPERATOR_COMMIT,
        "operator_blob": OPERATOR_BLOB,
        "operator_identity_enforcement": (
            "pinned recovery launcher; recovered ENVIRONMENT.json is not trusted because "
            "the partial frontier path historically reported an older inherited operator"
        ),
        "source_kind": located["kind"],
        "archive_checkpoint_count": located.get("archive_checkpoint_count"),
        "public_surface_sha256": public_manifest["sha256"],
        "tokenizer_artifact_sha256": tokenizer["artifact_sha256"],
        "public_surface_validation": "pinned formation_mux_surface_v5 canonical validator",
        "checkpoint_validation": "torch.load weights_only=True plus identity/state/result/receipt/progress checks",
        "recovery_runtime_contract": "official Kaggle NVIDIA Tesla T4 x2 worker checkpoints with CUDA RNG state",
        "s5_status": campaign["status"],
        "s5_completed_arms": len(s5_completed),
        "s5_required_arms": 24,
        "frontier_status": frontier_state["status"],
        "frontier_completed_arms": len(frontier_completed),
        "frontier_required_arms": 24,
        "official_checkpoint_count": len(checkpoints),
        "checkpoint_inventory_sha256": inventory_digest,
        "completed_arm_keys": sorted(completed),
        "pending_official_slots": sorted(set(SLOT_BY_KEY).difference(completed)),
        "checkpoints": checkpoints,
        "sealed_markers": sealed,
        "raw_sealed_rows_read": False,
        "claim_ceiling": "custody preflight only; no scientific verdict",
    }


def _paths_overlap(source: Path, output: Path) -> bool:
    try:
        output.relative_to(source)
        return True
    except ValueError:
        return False


def _staging_path(output: Path) -> Path:
    staging = output.with_name(output.name + ".recovery.tmp")
    if staging.exists():
        raise RecoveryError(f"stale recovery staging path exists: {staging}")
    return staging


def _rename_noreplace(source: Path, destination: Path) -> None:
    if sys.platform == "win32":
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(str(destination))
        os.rename(source, destination)
        return
    if sys.platform != "linux":
        raise RecoveryError(
            f"atomic no-replace directory install is unsupported on {sys.platform}"
        )
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RecoveryError("Linux atomic no-replace install requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _install_validated_staging(
    staging: Path,
    output: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    _atomic_json(staging / "RECOVERY_PREFLIGHT.json", receipt)
    try:
        _rename_noreplace(staging, output)
    except Exception as exc:
        raise RecoveryError(
            f"validated recovery staging install failed; preserved staging path: "
            f"{staging}: {exc}"
        ) from exc
    return receipt


def run_preflight(
    *,
    input_path: Path,
    output_path: Path,
    install: bool,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_input = input_path.absolute()
    raw_output = output_path.absolute()
    _reject_symlink_ancestors(raw_input)
    _reject_symlink_ancestors(raw_output)
    input_resolved = raw_input.resolve()
    output = raw_output.resolve()
    if install and output.exists():
        raise RecoveryError(f"refusing to overwrite existing working output: {output}")
    located = _locate(input_resolved)
    if not install:
        return validate_source(
            located["source"],
            located=located,
            checkpoint_loader=checkpoint_loader,
        )
    if located["kind"] == "directory" and _paths_overlap(located["source"], output):
        raise RecoveryError(
            f"output path must not be inside the recovery source tree: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(output)
    if located["kind"] == "archive":
        archive_count = _extract_archive(located["source"], staging)
        receipt = validate_source(
            staging,
            located={
                "kind": "installed_archive",
                "source": staging,
                "archive_checkpoint_count": archive_count,
            },
            checkpoint_loader=checkpoint_loader,
            expected_archive_checkpoint_count=archive_count,
        )
    else:
        _validate_filesystem_tree(located["source"])
        shutil.copytree(located["source"], staging)
        receipt = validate_source(
            staging,
            located={"kind": "installed_directory", "source": staging},
            checkpoint_loader=checkpoint_loader,
        )
    return _install_validated_staging(staging, output, receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--out", type=Path, default=Path("/kaggle/working/FORMATION_MUX_001")
    )
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = run_preflight(
            input_path=args.input,
            output_path=args.out,
            install=args.install,
        )
    except RecoveryError as exc:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL",
            "error": str(exc),
            "recovery_nonce": os.environ.get("FORMATION_MUX_RECOVERY_NONCE"),
            "preflight_source_sha256": _sha256_file(Path(__file__)),
            "science_commit": SCIENCE_COMMIT,
            "operator_commit": OPERATOR_COMMIT,
            "operator_blob": OPERATOR_BLOB,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
