"""Read-only checkpoint inspection and format-aware continuation qualification.

No command advances a training cursor, optimizer, pointer or promotion. Legacy
inspection proves byte custody and safe tensor decoding; exact continuation
requires the original trainer. Foundation action dry-runs reconstruct that
trainer from the public data recipe and validate its committed head.
"""

from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path

from .codec import digest


DEFAULT_MAX_BYTES = 512 * 1024**2


def _current_git_head():
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # Installed wheels need not include a Git executable/tree.


def _source_tree_clean():
    import subprocess

    try:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", "*.py"],
            cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL, text=True)
        return not bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _existing_store(root: Path, lineage: str):
    from v5_training.checkpoint import CheckpointStore, _validate_lineage_id

    _validate_lineage_id(lineage)
    root = Path(root).resolve(strict=True)
    folder = root / lineage
    objects = folder / "objects"
    if (not root.is_dir() or not objects.is_dir() or folder.is_symlink() or objects.is_symlink()
            or not folder.resolve().is_relative_to(root) or not objects.resolve().is_relative_to(root)):
        raise ValueError("inspection requires an existing contained checkpoint objects directory")
    return CheckpointStore(root, lineage)


def _read(root, lineage, checkpoint_sha256, *, max_bytes):
    from v5_training.checkpoint import _validate_checkpoint_sha, REQUIRED_COMPONENTS, DISTRIBUTED_REQUIRED_COMPONENTS

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("payload read limit must be a positive integer")
    store = _existing_store(root, lineage)
    identity = store.latest_sha256() if checkpoint_sha256 is None else checkpoint_sha256
    if identity is None:
        raise ValueError("lineage has no committed checkpoint")
    _validate_checkpoint_sha(identity)
    path = store.objects / identity / "manifest.json"
    if not path.is_file() or path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 1024**2:
        raise ValueError("checkpoint manifest missing, linked or oversized")
    manifest_bytes = path.read_bytes()
    if sha256(manifest_bytes).hexdigest() != identity:
        raise ValueError("checkpoint manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("components"), list):
        raise ValueError("invalid checkpoint component inventory")
    allowed = REQUIRED_COMPONENTS | DISTRIBUTED_REQUIRED_COMPONENTS
    if any(not isinstance(entry, dict) or set(entry) != {"name", "byte_size", "sha256"}
           or entry["name"] not in allowed for entry in manifest["components"]):
        raise ValueError("invalid checkpoint component inventory")
    sizes = [entry["byte_size"] for entry in manifest["components"]]
    if any(type(size) is not int or size < 0 for size in sizes) or sum(sizes) > max_bytes:
        raise ValueError("checkpoint exceeds explicit payload read limit")
    actual_sizes = []
    for entry in manifest["components"]:
        component = path.parent / entry["name"]
        if component.is_symlink() or not component.is_file():
            raise ValueError("checkpoint component missing or linked")
        actual_sizes.append(component.stat().st_size)
    if sum(actual_sizes) > max_bytes:
        raise ValueError("checkpoint exceeds explicit payload read limit")
    state, payloads = store.restore(identity)
    return store, identity, state, payloads, manifest


def _safe_load(payload):
    import torch

    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        # No fallback to unrestricted pickle loading, even for old files.
        raise ValueError("safe weights-only tensor decoding failed") from exc


def _tensor_rows(value, prefix=""):
    import torch

    rows = {}
    if isinstance(value, torch.Tensor):
        rows[prefix] = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "elements": value.numel(), "finite": bool(torch.isfinite(value).all())}
    elif isinstance(value, dict):
        for key, item in value.items():
            rows.update(_tensor_rows(item, prefix + ("/" if prefix else "") + str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.update(_tensor_rows(item, prefix + "/" + str(index)))
    return rows


def _decoded(payloads):
    import torch

    weights = _safe_load(payloads["model.bin"])
    if not isinstance(weights, dict) or not weights or any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
                                                          for name, value in weights.items()):
        raise ValueError("model component must be a nonempty named tensor state_dict")
    optimizer = _safe_load(payloads["optimizer.bin"])
    if not isinstance(optimizer, dict):
        raise ValueError("optimizer component must decode to a state mapping")
    return weights, optimizer


def inspect_checkpoint(root: Path, lineage: str, checkpoint_sha256: str | None = None,
                       *, max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    _, identity, state, payloads, manifest = _read(root, lineage, checkpoint_sha256, max_bytes=max_bytes)
    weights, optimizer = _decoded(payloads)
    scheduler = json.loads(payloads["scheduler.json"])
    metadata = scheduler.get("metadata", {}) if isinstance(scheduler, dict) else {}
    foundation = isinstance(metadata, dict) and metadata.get("schema") == "elendril-foundation-training-v1"
    steps = {}
    for key, slot in optimizer.get("state", {}).items():
        value = slot.get("step") if isinstance(slot, dict) else None
        if value is not None and hasattr(value, "numel") and value.numel() == 1:
            # Keep nonfinite step values JSON safe and visible as failures.
            row = _tensor_rows(value)[""]
            steps[str(key)] = value.item() if row["finite"] else "NONFINITE"
    return {"schema": "elendril-checkpoint-inspection-v1", "checkpoint_sha256": identity,
            "lineage": lineage, "byte_verification": "PASS", "state": state.canonical(),
            "components": manifest["components"], "model_tensors": _tensor_rows(weights),
            "optimizer_tensors": _tensor_rows(optimizer), "optimizer_steps": steps,
            "training_format": "foundation-v1" if foundation else "legacy-or-external",
            "source_binding": "recorded-not-currently-verified", "current_git_head": _current_git_head(),
            "current_python_tree_clean": _source_tree_clean(),
            "semantic_validation": "NOT_RUN; use verify-checkpoint and format-aware resume-dry-run",
            "persistence_scope": "local bytes verified; no fresh cloud session qualified"}


def _foundation_metadata_checks(state, payloads, weights, optimizer):
    """Non-allocating checks; trainer restoration provides the full continuation gate."""
    from v5_contracts.model_spec import ModelSpec
    from .foundation_training import Recipe
    from .representation import tokenizer_from_manifest

    scheduler = json.loads(payloads["scheduler.json"])
    metadata = scheduler.get("metadata", {}) if isinstance(scheduler, dict) else {}
    if not isinstance(metadata, dict) or metadata.get("schema") != "elendril-foundation-training-v1":
        return "FORMAT_SPECIFIC_NOT_RUN; legacy trainer required for exact continuation"
    spec = ModelSpec(**metadata["model"])
    recipe = Recipe(**metadata["recipe"])
    spec.assert_valid()
    recipe.validate()
    tokenizer = tokenizer_from_manifest(metadata["tokenizer"])
    tokenizer.require_binding(state.identities.tokenizer_sha256, spec.vocabulary_size)
    if spec.sha256() != state.identities.model_spec_sha256 or digest(metadata) != state.identities.run_spec_sha256:
        raise ValueError("checkpoint architecture or run metadata identity mismatch")
    inventory = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in weights.items()}
    if inventory != metadata["tensors"] or sum(value.numel() for value in weights.values()) != spec.parameter_receipt().total:
        raise ValueError("checkpoint tensor inventory disagrees with model manifest")
    slots = optimizer.get("state", {})
    if not isinstance(slots, dict) or not slots or any(not isinstance(slot, dict) or "step" not in slot or not hasattr(slot["step"], "numel") or slot["step"].numel() != 1
                        or slot["step"].item() != state.global_update for slot in slots.values()):
        raise ValueError("checkpoint optimizer counters disagree with completed update")
    return "MODEL_TOKENIZER_RECIPE_COUNTERS_PASS; exact continuation requires resume-dry-run"


def verify_checkpoint(root: Path, lineage: str, checkpoint_sha256: str | None = None,
                      *, max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    report = inspect_checkpoint(root, lineage, checkpoint_sha256, max_bytes=max_bytes)
    if any(not row["finite"] for kind in ("model_tensors", "optimizer_tensors") for row in report[kind].values()):
        raise ValueError("checkpoint contains nonfinite model or optimizer tensors")
    _, _, state, payloads, _ = _read(root, lineage, report["checkpoint_sha256"], max_bytes=max_bytes)
    weights, optimizer = _decoded(payloads)
    report["semantic_validation"] = _foundation_metadata_checks(state, payloads, weights, optimizer)
    report["verification"] = "PASS_WITH_DECLARED_FORMAT_SCOPE"
    return report


def compare_checkpoints(root: Path, lineage: str, checkpoint_sha256: str | None,
                        other_sha256: str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    import torch

    _, left_sha, left, left_payloads, _ = _read(root, lineage, checkpoint_sha256, max_bytes=max_bytes)
    _, right_sha, right, right_payloads, _ = _read(root, lineage, other_sha256, max_bytes=max_bytes)
    if (left.identities.model_spec_sha256 != right.identities.model_spec_sha256
            or left.identities.tokenizer_sha256 != right.identities.tokenizer_sha256):
        raise ValueError("comparison rejected incompatible model/tokenizer identities")
    a, _ = _decoded(left_payloads)
    b, _ = _decoded(right_payloads)
    if a.keys() != b.keys() or any(a[key].shape != b[key].shape or a[key].dtype != b[key].dtype for key in a):
        raise ValueError("comparison rejected incompatible tensor names/shapes/dtypes")
    if any(not torch.isfinite(tensor).all() for values in (a, b) for tensor in values.values()):
        raise ValueError("comparison rejected nonfinite tensors")
    differences = {name: float((a[name].to(torch.float64) - b[name].to(torch.float64)).abs().max())
                   if a[name].numel() else 0. for name in a}
    return {"schema": "elendril-checkpoint-comparison-v1", "left_sha256": left_sha, "right_sha256": right_sha,
            "left_state": left.canonical(), "right_state": right.canonical(),
            "max_abs_parameter_difference": max(differences.values(), default=0.),
            "max_abs_difference_by_tensor": differences,
            "interpretation": "compatible tensor difference only; no capability or exact-resume claim"}


def resume_dry_run(root: Path, lineage: str, checkpoint_sha256: str | None = None,
                   *, device: str = "cpu", max_bytes: int = DEFAULT_MAX_BYTES,
                   allow_source_mismatch: bool = False) -> dict:
    from v5_contracts.model_spec import ModelSpec
    from .action_learning import VERSION, action_fixture
    from .foundation_training import FoundationTrainer, Recipe, StageTransition, TokenSample
    from .training import configure_device

    store, identity, state, payloads, _ = _read(root, lineage, checkpoint_sha256, max_bytes=max_bytes)
    scheduler = json.loads(payloads["scheduler.json"])
    metadata = scheduler.get("metadata", {}) if isinstance(scheduler, dict) else {}
    if not isinstance(metadata, dict) or metadata.get("schema") != "elendril-foundation-training-v1":
        raise ValueError("exact dry-run unsupported for this format; use its original trainer/source")
    provenance = metadata["provenance"]
    if provenance.get("schema") != VERSION or provenance.get("split") != "train":
        raise ValueError("exact dry-run requires a reconstructible action-training data manifest")
    if store.latest_sha256() != identity:
        raise ValueError("exact dry-run supports committed LATEST only; inspect historical SHA without changing pointers")
    current_source = _current_git_head()
    clean_source = _source_tree_clean()
    same_source = current_source == state.identities.source_commit and clean_source is True
    if not same_source and not allow_source_mismatch:
        raise ValueError("dry-run source mismatch, dirty Python tree or unverified Git source; pin clean recorded commit or explicitly allow diagnostic source mismatch")
    spec = ModelSpec(**metadata["model"])
    # Allocation remains bounded by default. Large owner-run models require a
    # proportionate operator instead of surprising this local inspection tool.
    if spec.parameter_receipt().total > 25_000_000:
        raise ValueError("dry-run model exceeds 25M-parameter local allocation bound")
    data = action_fixture("train", worlds=provenance["episodes"], seed=provenance["seed"])
    if data.manifest != provenance:
        raise ValueError("reconstructed action fixture does not match checkpoint provenance")
    samples = tuple(TokenSample(row.task_id, row.tokens, row.eligible, row.world_id, row.kind) for row in data.examples)
    if device not in ("cpu", "cuda:0", "cuda:1"):
        raise ValueError("dry-run requires an explicit CPU or single CUDA device")
    configure_device(device)
    transition = StageTransition(**metadata["stage_transition"]) if "stage_transition" in metadata else None
    trainer = FoundationTrainer(samples, spec=spec, recipe=Recipe(**metadata["recipe"]),
        source_commit=state.identities.source_commit, root=store.root, lineage=lineage,
        data_provenance=data.manifest, tokenizer_manifest=metadata["tokenizer"], device=device,
        resume=True, transition=transition)
    if trainer.parent != identity:
        raise ValueError("checkpoint head changed during dry-run; rerun against a stable snapshot")
    index = trainer.state.global_update
    return {"schema": "elendril-resume-dry-run-v1", "checkpoint_sha256": identity,
            "validation": "PASS" if same_source else "DIAGNOSTIC_PASS_SOURCE_UNVERIFIED", "device": device, "optimizer_updates_executed": 0,
            "checkpoint_writes": 0, "completed_updates": index, "complete": trainer.state.complete,
            "next_update": None if trainer.state.complete else index + 1,
            "next_indices": [] if trainer.state.complete else list(trainer.plan[index]),
            "next_supervised_tokens": 0 if trainer.state.complete else trainer.counts[index],
            "identities": asdict(trainer.state.identities),
            "source_binding": "matching-git-head" if same_source else "recorded-not-currently-verified",
            "current_git_head": current_source,
            "current_python_tree_clean": clean_source,
            "qualification": "restore validated in this process/runtime; no next update or cloud session claimed"}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect-checkpoint", "verify-checkpoint", "compare-checkpoints", "resume-dry-run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--sha")
    parser.add_argument("--other-sha")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-source-mismatch", action="store_true",
                        help="diagnostic restore only; cannot qualify exact continuation")
    parser.add_argument("--max-payload-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if not 1 <= args.max_payload_mib <= 32768:
        parser.error("max-payload-mib must be 1..32768")
    kwargs = {"max_bytes": args.max_payload_mib * 1024**2}
    if args.command == "compare-checkpoints":
        if args.other_sha is None:
            parser.error("comparison requires --other-sha")
        report = compare_checkpoints(args.root, args.lineage, args.sha, args.other_sha, **kwargs)
    elif args.command == "resume-dry-run":
        report = resume_dry_run(args.root, args.lineage, args.sha, device=args.device,
                                allow_source_mismatch=args.allow_source_mismatch, **kwargs)
    else:
        operation = inspect_checkpoint if args.command == "inspect-checkpoint" else verify_checkpoint
        report = operation(args.root, args.lineage, args.sha, **kwargs)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
