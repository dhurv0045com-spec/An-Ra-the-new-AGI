"""Kaggle campaign environment contracts (hard architecture).

All Kaggle/session path, identity, device and subprocess rules live here as
tested library code. The owner notebook is a thin caller: minimal source
bootstrap (pre-import) then these contracts. No notebook-inline re-implementation,
no CWD-dependent hacks, no `locals()`/`globals()`, no bare `except: pass`,
no hardcoded user paths.

Environment overrides (explicit, validated):
- BRAMASTRA_WORKING: working root (default /kaggle/working)
- BRAMASTRA_INPUT: inputs root (default /kaggle/input)
- BRAMASTRA_REPO: pinned source tree (discovery fallback otherwise)
- BRAMASTRA_BUNDLE_DIR: pinned K8 bundle (discovery fallback otherwise)
- BRAMASTRA_RUN_ID: campaign identity (persisted per working root otherwise)

Hardware contract: exactly two visible CUDA devices (Kaggle GPU T4 x2).
Kaggle offers T4 x2 (2 GPUs) or P100 (1 GPU); there is no 3-GPU option and
API-submitted runs default to P100, which this campaign refuses by contract.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

WORKING_DEFAULT = Path("/kaggle/working")
INPUT_DEFAULT = Path("/kaggle/input")
BUNDLE_MANIFEST_SCHEMA = "bramastra-k8-data/v1"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")
INSTANCE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{4,64}")
GIT_REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
DEVICE_PATTERN = re.compile(r"cuda:\d+")
SAFE_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class K8EnvironmentError(RuntimeError):
    """Kaggle campaign environment violated its contract."""


def resolve_working(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Working root, created. Explicit arg > BRAMASTRA_WORKING > default."""
    raw = explicit if explicit is not None else os.environ.get("BRAMASTRA_WORKING", str(WORKING_DEFAULT))
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise K8EnvironmentError(f"cannot create working root {path}: {exc}") from exc
    return path


def resolve_input(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Inputs root (read-only; never created or written)."""
    raw = explicit if explicit is not None else os.environ.get("BRAMASTRA_INPUT", str(INPUT_DEFAULT))
    return Path(raw).expanduser()


def resolve_run_id(working: str | os.PathLike[str], explicit: str | None = None) -> str:
    """Campaign identity: explicit > BRAMASTRA_RUN_ID > persisted > generated."""
    working_path = Path(working)
    requested = explicit if explicit is not None else os.environ.get("BRAMASTRA_RUN_ID")
    if requested:
        run_id = requested.strip()
    else:
        run_id_file = working_path / "bramastra-k8-run-id.txt"
        if run_id_file.is_file():
            try:
                run_id = run_id_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise K8EnvironmentError(f"cannot read run-id file {run_id_file}: {exc}") from exc
        else:
            run_id = f"k8-{uuid.uuid4().hex[:12]}"
            try:
                run_id_file.write_text(run_id + "\n", encoding="utf-8")
            except OSError as exc:
                raise K8EnvironmentError(f"cannot persist run id to {run_id_file}: {exc}") from exc
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise K8EnvironmentError("BRAMASTRA_RUN_ID must be 3-80 safe filename characters [A-Za-z0-9._-]")
    return run_id


def resolve_instance_id(run_root: str | os.PathLike[str]) -> str:
    """Stable per-campaign instance token, persisted under run_root."""
    root = Path(run_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise K8EnvironmentError(f"cannot create run root {root}: {exc}") from exc
    stamp = root / "instance_id.txt"
    if stamp.is_file():
        try:
            token = stamp.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise K8EnvironmentError(f"cannot read instance stamp {stamp}: {exc}") from exc
        if INSTANCE_ID_PATTERN.fullmatch(token):
            return token
    token = uuid.uuid4().hex[:12]
    try:
        stamp.write_text(token + "\n", encoding="utf-8")
    except OSError as exc:
        raise K8EnvironmentError(f"cannot persist instance stamp {stamp}: {exc}") from exc
    return token


@dataclass(frozen=True)
class RunPaths:
    """Bound campaign filesystem identity. All directories exist except the
    build-report/export targets, which their commands create."""

    working: Path
    input_root: Path
    run_id: str
    instance_id: str
    run_root: Path
    run_dir: Path
    build_report_dir: Path
    export_dir: Path

    def build_report_file(self) -> Path:
        return self.build_report_dir / "build_verification.json"


def build_run_paths(*, working: str | os.PathLike[str] | None = None,
                    input_root: str | os.PathLike[str] | None = None,
                    run_id: str | None = None) -> RunPaths:
    """Bind working/input/run identity with zero implicit CWD dependence."""
    work = resolve_working(working)
    inputs = resolve_input(input_root)
    rid = resolve_run_id(work, run_id)
    root = work / "bramastra-k8" / rid
    token = resolve_instance_id(root)
    return RunPaths(
        working=work,
        input_root=inputs,
        run_id=rid,
        instance_id=token,
        run_root=root,
        run_dir=root / "campaign",
        build_report_dir=root / f"build-verification-{token}",
        export_dir=root / f"export-{token}",
    )


def is_source_tree(path: str | os.PathLike[str]) -> bool:
    """A usable BRAMASTRA source tree ships pyproject + bramastra_lab."""
    candidate = Path(path)
    return (candidate / "pyproject.toml").is_file() and (candidate / "bramastra_lab").is_dir()


def _children(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    try:
        return sorted(path.iterdir(), key=lambda item: item.name)
    except OSError:
        return []


def discover_source(*, configured: str | os.PathLike[str] | None = None,
                    cwd: str | os.PathLike[str] | None = None,
                    working: str | os.PathLike[str] | None = None,
                    input_root: str | os.PathLike[str] | None = None) -> tuple[Path | None, list[str]]:
    """Locate the source tree without cloning. Returns (path, diagnostics)."""
    checked: list[str] = []
    candidates: list[Path] = []
    env_repo = os.environ.get("BRAMASTRA_REPO")
    if configured is not None:
        candidates.append(Path(configured))
    elif env_repo:
        candidates.append(Path(env_repo))
    candidates.append(Path(cwd) if cwd is not None else Path.cwd())
    work = Path(working) if working is not None else resolve_working()
    inputs = Path(input_root) if input_root is not None else resolve_input()
    candidates.append(work / "An-Ra-the-new-AGI")
    for parent in (inputs, work):
        for child in _children(parent):
            candidates.append(child)
            candidates.extend(_children(child)[:50])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        checked.append(str(candidate))
        if is_source_tree(resolved):
            return resolved, checked
    return None, checked


def is_k8_bundle(path: str | os.PathLike[str]) -> bool:
    """A valid K8 bundle root carries manifest.json with the K8 schema."""
    manifest = Path(path) / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("schema") == BUNDLE_MANIFEST_SCHEMA
    except (OSError, ValueError):
        return False


def discover_bundle(*, configured: str | os.PathLike[str] | None = None,
                    input_root: str | os.PathLike[str] | None = None) -> tuple[Path | None, list[str]]:
    """Locate an attached valid bundle. Returns (path, diagnostics)."""
    checked: list[str] = []
    candidates: list[Path] = []
    env_bundle = os.environ.get("BRAMASTRA_BUNDLE_DIR")
    if configured is not None:
        candidates.append(Path(configured))
    elif env_bundle:
        candidates.append(Path(env_bundle))
    inputs = Path(input_root) if input_root is not None else resolve_input()
    for child in _children(inputs):
        candidates.append(child)
        candidates.extend(_children(child)[:50])
    for candidate in candidates:
        checked.append(str(candidate))
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if is_k8_bundle(resolved):
            return resolved, checked
    return None, checked


def find_build_report(run_root: str | os.PathLike[str],
                      preferred: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the build report for run/summarize cells.

    Preference: existing preferred report > latest verified report under
    run_root > preferred path itself (caller then produces it). Never raises
    on absence; the campaign preflight owns the final gate.
    """
    root = Path(run_root)
    if preferred is not None:
        direct = Path(preferred)
        if direct.is_file():
            return direct
        candidate_dir = direct if direct.suffix != ".json" else direct.parent
        report_in_dir = candidate_dir / "build_verification.json"
        if report_in_dir.is_file():
            return report_in_dir
    reports = sorted(root.glob("build-verification-*/build_verification.json"))
    if reports:
        return reports[-1]
    if preferred is not None:
        return Path(preferred)
    return root / "build-verification" / "build_verification.json"


def normalize_devices(raw: str | Sequence[str]) -> tuple[str, str]:
    """Strict two-device contract shared by CLI, runner and notebook.

    Accepts "cuda:0,cuda:1" (whitespace tolerated) or a 2-sequence.
    Returns the stripped pair. Raises K8EnvironmentError otherwise.
    """
    if isinstance(raw, str):
        parts = [tok.strip() for tok in raw.split(",")]
    else:
        parts = [str(tok).strip() for tok in list(raw)]
    parts = [tok for tok in parts if tok]
    if len(parts) != 2 or len(set(parts)) != 2:
        raise K8EnvironmentError(
            f"K8 requires exactly two distinct devices (got {list(raw)!r}); "
            "use cuda:0,cuda:1 on a Kaggle GPU T4 x2 session. "
            "Kaggle offers T4 x2 (2 GPUs) or P100 (1 GPU); there is no 3-GPU option.")
    for tok in parts:
        if not DEVICE_PATTERN.fullmatch(tok):
            raise K8EnvironmentError(
                f"invalid device {tok!r}; expected cuda:N pairs such as cuda:0,cuda:1")
    first, second = parts
    return first, second


def resolve_build_report_arg(raw: str | None) -> str | None:
    """Normalize --build-report to a report FILE path (file or dir accepted)."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        raise K8EnvironmentError("--build-report must not be blank")
    if text.endswith(".json"):
        return text
    return os.path.join(text, "build_verification.json")


def check_git_ref(ref: str) -> str:
    """Validate a git branch ref before any clone (injection guard)."""
    if not GIT_REF_PATTERN.fullmatch(ref):
        raise K8EnvironmentError("BRAMASTRA_GIT_REF contains unsafe characters")
    return ref


def gpu_contract(*, required: int = 2) -> dict[str, object]:
    """Verify the visible-GPU contract. Imports torch lazily; never at module load."""
    try:
        import torch
    except ImportError as exc:
        raise K8EnvironmentError(
            "torch is unavailable; select a Kaggle GPU image with torch>=2.6") from exc
    try:
        count = int(torch.cuda.device_count())
    except Exception as exc:
        raise K8EnvironmentError(f"GPU query failed: {exc}") from exc
    names: list[str] = []
    for index in range(count):
        try:
            names.append(str(torch.cuda.get_device_name(index)))
        except Exception as exc:
            names.append(f"<name query failed: {exc}>")
    try:
        version = str(torch.__version__)
    except Exception:
        version = "unknown"
    report: dict[str, object] = {"count": count, "names": names, "torch": version}
    if count != required:
        raise K8EnvironmentError(
            f"K8 requires exactly {required} visible GPUs (found {count}: {names}). "
            "In Kaggle Settings > Accelerator select GPU T4 x2 and restart the session. "
            "Kaggle offers T4 x2 (2 GPUs) or P100 (1 GPU); there is no 3-GPU option, "
            "and API-submitted runs default to P100.")
    return report


def disk_contract(path: str | os.PathLike[str], *, minimum_gib: float = 2.0) -> dict[str, object]:
    """Free-space contract for the working root."""
    try:
        usage = shutil.disk_usage(str(path))
    except OSError as exc:
        raise K8EnvironmentError(f"disk query failed for {path}: {exc}") from exc
    free_gib = usage.free / (1024 ** 3)
    report: dict[str, object] = {"path": str(path), "free_gib": round(free_gib, 2)}
    if free_gib < minimum_gib:
        raise K8EnvironmentError(
            f"only {free_gib:.1f} GiB free under {path} (need >= {minimum_gib:.0f} GiB); "
            "clean /kaggle/working or start a fresh session")
    return report


@dataclass(frozen=True)
class StreamResult:
    label: str
    argv: tuple[str, ...]
    returncode: int
    log_path: Path


@dataclass(frozen=True)
class ArtifactArchive:
    archive: Path
    receipt: Path
    sha256: str
    bytes: int


def package_artifacts(run_root: str | os.PathLike[str], working: str | os.PathLike[str],
                      run_id: str, instance_id: str, reason: str) -> ArtifactArchive | None:
    """Create one verified ZIP of the run root (contracts, not notebook inline).

    Returns None when the run root does not exist yet. Verifies the ZIP with
    testzip, hashes it, atomically promotes the partial file, and writes a
    JSON receipt beside the archive.
    """
    import hashlib
    import shutil
    import zipfile

    root = Path(run_root)
    work = Path(working)
    if not root.exists():
        return None
    safe_run = SAFE_LABEL_PATTERN.sub("-", run_id).strip("-") or "run"
    archive = work / f"{safe_run}-artifacts-{instance_id}.zip"
    if archive.exists():
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        return ArtifactArchive(archive=archive, receipt=archive.with_suffix(".json"),
                               sha256=digest, bytes=archive.stat().st_size)
    partial_base = work / f".{safe_run}-artifacts-{instance_id}.partial"
    partial_archive = Path(shutil.make_archive(
        str(partial_base), "zip", root_dir=root.parent, base_dir=root.name))
    with zipfile.ZipFile(partial_archive) as bundle:
        bad_member = bundle.testzip()
    if bad_member is not None:
        try:
            partial_archive.unlink(missing_ok=True)
        except OSError:
            pass
        raise K8EnvironmentError(f"artifact archive verification failed at {bad_member}")
    digest = hashlib.sha256(partial_archive.read_bytes()).hexdigest()
    try:
        partial_archive.replace(archive)
    except OSError as exc:
        raise K8EnvironmentError(f"cannot promote artifact archive {archive}: {exc}") from exc
    receipt = archive.with_suffix(".json")
    try:
        receipt.write_text(json.dumps({
            "schema": "bramastra-k8-artifact-archive/v1",
            "reason": reason, "run_id": run_id,
            "archive": archive.name, "sha256": digest,
            "bytes": archive.stat().st_size,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise K8EnvironmentError(f"cannot write artifact receipt {receipt}: {exc}") from exc
    return ArtifactArchive(archive=archive, receipt=receipt,
                           sha256=digest, bytes=archive.stat().st_size)


def stream_command(label: str, argv: Sequence[str], *, cwd: str | os.PathLike[str],
                   log_dir: str | os.PathLike[str],
                   extra_env: dict[str, str] | None = None) -> StreamResult:
    """Run a child with live line streaming + durable log (no output buffering hacks).

    stdout/stderr merge to the log and to the caller console. Raises
    K8EnvironmentError (not RuntimeError) on spawn failure; non-zero exit is
    reported in the result for the caller to gate on.
    """
    safe = SAFE_LABEL_PATTERN.sub("-", label).strip("-") or "command"
    out_dir = Path(log_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise K8EnvironmentError(f"cannot create log dir {out_dir}: {exc}") from exc
    log_path = out_dir / f"k8-{safe}.log"
    environment = os.environ.copy()
    if extra_env:
        environment.update(extra_env)
    try:
        handle = log_path.open("a", encoding="utf-8")
    except OSError as exc:
        raise K8EnvironmentError(f"cannot open log {log_path}: {exc}") from exc
    with handle:
        handle.write(f"\n=== {label}: {' '.join(argv)} ===\n")
        handle.flush()
        try:
            proc = subprocess.Popen(
                list(argv), cwd=str(cwd), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except OSError as exc:
            raise K8EnvironmentError(f"cannot spawn {' '.join(argv)}: {exc}") from exc
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
        proc.wait()
        handle.write(f"=== exit {proc.returncode} ===\n")
        return StreamResult(label=label, argv=tuple(argv),
                            returncode=int(proc.returncode), log_path=log_path)
