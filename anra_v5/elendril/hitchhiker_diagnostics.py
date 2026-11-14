"""Stage 0 frozen-checkpoint admission, three-decoder suite, and logit diagnostics."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import torch

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from .backend import BackendIdentity, NeuralBackend, tensor_sha256
from .codec import canonical, digest
from .cognition import CognitiveState, OperationCandidate
from .hitchhiker_data import (
    ExternalSlotCanonicalizer,
    SYMBOLS,
    render_hitchhiker_public,
)
from .hitchhiker_protocol import HISTORICAL_S20_LOWLR_MANIFESTS, resolve_scale_profile
from .representation import SpecialToken, UTF8, tokenizer_from_manifest
from .training import _load
from .v6_data import action_text, public_action
from .v6_evaluation import cases_identity, parse_action


ANSWER_TOKEN_IDS = tuple(UTF8.encode(s)[0] for s in SYMBOLS)
TOKEN_R = UTF8.encode("R")[0]
TOKEN_Q = UTF8.encode("Q")[0]
TOKEN_ABSTAIN = UTF8.encode("!")[0]
TOKEN_DELIBERATE = UTF8.encode("~")[0]


def _token_label(token_id: int) -> str:
    specials = {
        UTF8.pad_id: "<PAD>",
        UTF8.bos_id: "<BOS>",
        UTF8.eos_id: "<EOS>",
        UTF8.special(SpecialToken.ANSWER): "<ANSWER>",
    }
    if token_id in specials:
        return specials[token_id]
    b = token_id - 4
    if 32 <= b < 127:
        return chr(b)
    return f"byte:{b}"


def _load_verified_checkpoint_backend(
    store: CheckpointStore,
    checkpoint_sha256: str,
    *,
    expected_spec: ModelSpec | None = None,
    expected_update: int | None = None,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any], ModelSpec, dict[str, str], NeuralBackend]:
    """Restore a checkpoint and validate spec, update, tokenizer, and finite tensors."""
    state, payloads = store.restore(checkpoint_sha256)
    try:
        scheduler = json.loads(payloads["scheduler.json"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint scheduler payload is missing or invalid JSON") from exc
    metadata = scheduler.get("metadata") if isinstance(scheduler, dict) else None
    if not isinstance(metadata, dict) or "model" not in metadata or "tokenizer" not in metadata:
        raise ValueError("checkpoint metadata missing model or tokenizer receipt")

    spec = ModelSpec(**metadata["model"])
    spec.assert_valid()
    if expected_spec is not None and spec.sha256() != expected_spec.sha256():
        raise ValueError("checkpoint model spec disagrees with expected specification")
    if state.identities.model_spec_sha256 != spec.sha256():
        raise ValueError("checkpoint state identity disagrees with model specification")
    if expected_update is not None and state.global_update != expected_update:
        raise ValueError("checkpoint update counter disagrees with expected update")

    tokenizer = tokenizer_from_manifest(metadata["tokenizer"])
    tokenizer.require_binding(UTF8.identity, spec.vocabulary_size)
    if state.identities.tokenizer_sha256 != UTF8.identity:
        raise ValueError("checkpoint state identity disagrees with UTF8 tokenizer")

    weights = _load(payloads["model.bin"])
    model = initialize(spec, seed=0)
    expected_state = model.state_dict()
    if set(weights) != set(expected_state):
        raise ValueError("checkpoint model parameter inventory mismatch")

    tensor_hashes: dict[str, str] = {}
    for name, tensor in weights.items():
        expected_tensor = expected_state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected_tensor.shape
            or tensor.dtype != expected_tensor.dtype
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"checkpoint tensor validation failed for '{name}'")
        tensor_hashes[name] = tensor_sha256(tensor)

    model.load_state_dict(weights, strict=True)
    model = model.to(torch.device(device))
    model.eval()
    backend = NeuralBackend(
        model,
        BackendIdentity(
            spec.sha256(),
            UTF8.identity,
            spec.vocabulary_size,
            spec.context_length,
            checkpoint_sha256=checkpoint_sha256,
        ),
        pad_id=UTF8.pad_id,
        bos_id=UTF8.bos_id,
        eos_id=UTF8.eos_id,
    )
    return state, metadata, spec, tensor_hashes, backend


def admit_frozen_checkpoint(
    checkpoint_root: Path | str | None,
    lineage: str,
    *,
    expected_spec: ModelSpec | None = None,
    expected_update: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Verify and admit a generic local checkpoint or return CHECKPOINT_AUDIT_PENDING when absent."""
    if checkpoint_root is None:
        return {
            "status": "CHECKPOINT_AUDIT_PENDING",
            "reason": "PARENT_WEIGHTS_UNAVAILABLE",
            "fabricate_permitted": False,
            "checkpoint_root": None,
            "lineage": lineage,
            "admission_scope": "generic_local",
        }
    root_path = Path(checkpoint_root)
    latest_path = root_path / lineage / "LATEST"
    if not root_path.exists() or not latest_path.is_file():
        return {
            "status": "CHECKPOINT_AUDIT_PENDING",
            "reason": "PARENT_WEIGHTS_UNAVAILABLE",
            "fabricate_permitted": False,
            "checkpoint_root": str(root_path),
            "lineage": lineage,
            "admission_scope": "generic_local",
        }

    store = CheckpointStore(root_path, lineage)
    latest_sha = store.latest_sha256()
    if latest_sha is None:
        return {
            "status": "CHECKPOINT_AUDIT_PENDING",
            "reason": "PARENT_WEIGHTS_UNAVAILABLE",
            "fabricate_permitted": False,
            "checkpoint_root": str(root_path),
            "lineage": lineage,
            "admission_scope": "generic_local",
        }

    state, metadata, spec, tensor_hashes, backend = _load_verified_checkpoint_backend(
        store,
        latest_sha,
        expected_spec=expected_spec,
        expected_update=expected_update,
        device=device,
    )
    return {
        "status": "ADMITTED",
        "fabricate_permitted": False,
        "checkpoint_sha256": latest_sha,
        "lineage": lineage,
        "global_update": state.global_update,
        "model_spec_sha256": spec.sha256(),
        "tokenizer_sha256": UTF8.identity,
        "parameters": spec.parameter_receipt().total,
        "tensor_sha256s": tensor_hashes,
        "metadata": metadata,
        "backend": backend,
        "admission_scope": "generic_local",
    }


def _pending_historical_s20_receipt(manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "status": "CHECKPOINT_AUDIT_PENDING",
        "reason": "PARENT_WEIGHTS_UNAVAILABLE",
        "fabricate_permitted": False,
        "seed": seed,
        "lineage": manifest["lineage"],
        "expected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "admission_scope": "historical_s20_low_lr",
    }


def _validate_historical_s20_metadata(
    metadata: dict[str, Any],
    state: Any,
    manifest: dict[str, Any],
    seed: int,
    *,
    expected_data_sha: str | None,
) -> None:
    """Validate V7.17 S20-low-lr state, recipe, and provenance invariants."""
    if state.identities.source_commit != manifest["source_commit"]:
        raise ValueError("historical S20 checkpoint source_commit mismatch")
    if state.global_update != manifest["expected_update"]:
        raise ValueError("historical S20 checkpoint update counter mismatch")
    if state.complete is not True:
        raise ValueError("historical S20 checkpoint is not complete")

    recipe = metadata.get("recipe")
    provenance = metadata.get("provenance")
    if not isinstance(recipe, dict) or not isinstance(provenance, dict):
        raise ValueError("historical S20 checkpoint missing recipe or provenance metadata")

    v717_job = provenance.get("v717_job")
    if v717_job is not None and not isinstance(v717_job, dict):
        raise ValueError("historical S20 checkpoint v717_job metadata must be a mapping")

    if recipe.get("lr") != manifest["lr"]:
        raise ValueError("historical S20 checkpoint lr mismatch")
    if recipe.get("batch_size") != manifest["batch_size"]:
        raise ValueError("historical S20 checkpoint batch_size mismatch")
    if recipe.get("micro_batch_size") != manifest["micro_batch_size"]:
        raise ValueError("historical S20 checkpoint micro_batch_size mismatch")
    if recipe.get("seed") != seed:
        raise ValueError("historical S20 checkpoint seed mismatch")
    if recipe.get("objective") != manifest["mechanism"]:
        raise ValueError("historical S20 checkpoint objective mismatch")
    if "total_updates" in recipe and recipe.get("total_updates") != manifest["expected_update"]:
        raise ValueError("historical S20 checkpoint recipe total_updates mismatch")

    epochs_values = [
        val
        for val in (
            recipe.get("epochs"),
            v717_job.get("epochs") if isinstance(v717_job, dict) else None,
        )
        if val is not None
    ]
    if not epochs_values or any(val != manifest["epochs"] for val in epochs_values):
        raise ValueError("historical S20 checkpoint epochs mismatch")

    order_seed_values = [
        val
        for val in (
            recipe.get("order_seed"),
            provenance.get("order_seed"),
            v717_job.get("order_seed") if isinstance(v717_job, dict) else None,
        )
        if val is not None
    ]
    if not order_seed_values or any(val != seed for val in order_seed_values):
        raise ValueError("historical S20 checkpoint order_seed mismatch")

    sampling = recipe.get("sampling")
    v7_method = provenance.get("v7_sampling_method")
    job_method = v717_job.get("method") if isinstance(v717_job, dict) else None
    if v7_method is not None and v7_method != manifest["method"]:
        raise ValueError("historical S20 checkpoint v7_sampling_method mismatch")
    if job_method is not None and job_method != manifest["method"]:
        raise ValueError("historical S20 checkpoint job method mismatch")
    has_blocked_method = v7_method == manifest["method"] or job_method == manifest["method"]
    if sampling == manifest["method"]:
        pass
    elif sampling == "uniform" and has_blocked_method:
        pass
    else:
        raise ValueError("historical S20 checkpoint sampling method must be blocked")

    if isinstance(v717_job, dict):
        expected_job_id = manifest.get("job_id", manifest["lineage"])
        if v717_job.get("job_id") != expected_job_id or v717_job.get("profile") != manifest["profile"]:
            raise ValueError("historical S20 checkpoint v717_job identity mismatch")
        if v717_job.get("lr") != manifest["lr"] or v717_job.get("seed") != seed:
            raise ValueError("historical S20 checkpoint v717_job lr/seed mismatch")
        if v717_job.get("mechanism") != manifest["mechanism"]:
            raise ValueError("historical S20 checkpoint v717_job mechanism mismatch")

    proto_values = [
        val
        for val in (provenance.get("v717_protocol_sha256"), provenance.get("protocol_sha256"))
        if val is not None
    ]
    if not proto_values or any(val != manifest["protocol_sha256"] for val in proto_values):
        raise ValueError("historical S20 checkpoint protocol_sha256 mismatch")

    if "split" in provenance and provenance.get("split") != "train":
        raise ValueError("historical S20 checkpoint provenance split must be train")

    train_seed = provenance.get("seed") if "seed" in provenance else provenance.get("train_seed")
    if train_seed != manifest["train_seed"]:
        raise ValueError("historical S20 checkpoint train seed mismatch")

    train_bytes = provenance.get("target_bytes") if "target_bytes" in provenance else provenance.get("train_bytes")
    if train_bytes != manifest["train_bytes"]:
        raise ValueError("historical S20 checkpoint train bytes mismatch")

    if expected_data_sha is not None:
        actual_data_sha = provenance.get("v717_data_identity") or provenance.get("data_sha256")
        if actual_data_sha is not None:
            if actual_data_sha != expected_data_sha:
                raise ValueError("historical S20 checkpoint data identity disagrees with expected_data_sha")
        elif state.identities.data_manifest_sha256 != expected_data_sha:
            raise ValueError("historical S20 checkpoint data manifest disagrees with expected_data_sha")


def admit_historical_s20_checkpoint(
    checkpoint_root: Path | str | None,
    seed: int,
    *,
    expected_data_sha: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Strictly verify and admit a historical V7.17 S20-low-lr checkpoint against its manifest."""
    if type(seed) is not int or seed not in HISTORICAL_S20_LOWLR_MANIFESTS:
        raise ValueError(f"invalid historical S20 low-lr seed: {seed!r}")
    manifest = HISTORICAL_S20_LOWLR_MANIFESTS[seed]
    if checkpoint_root is None:
        return _pending_historical_s20_receipt(manifest, seed)

    root_path = Path(checkpoint_root)
    lineage = manifest["lineage"]
    lineage_dir = root_path / lineage
    latest_path = lineage_dir / "LATEST"
    if not root_path.exists() or not latest_path.is_file():
        return _pending_historical_s20_receipt(manifest, seed)

    latest_sha = latest_path.read_text(encoding="ascii").strip()
    if len(latest_sha) != 64 or any(c not in "0123456789abcdef" for c in latest_sha):
        raise ValueError("LATEST pointer is corrupt")
    if latest_sha != manifest["checkpoint_sha256"]:
        raise ValueError("historical S20 checkpoint SHA disagrees with manifest")

    expected_object_dir = lineage_dir / "objects" / manifest["checkpoint_sha256"]
    if not expected_object_dir.is_dir():
        return _pending_historical_s20_receipt(manifest, seed)

    store = CheckpointStore(root_path, lineage)
    expected_spec = resolve_scale_profile(manifest["profile"])
    state, metadata, spec, tensor_hashes, backend = _load_verified_checkpoint_backend(
        store,
        latest_sha,
        expected_spec=expected_spec,
        expected_update=manifest["expected_update"],
        device=device,
    )
    _validate_historical_s20_metadata(
        metadata,
        state,
        manifest,
        seed,
        expected_data_sha=expected_data_sha,
    )
    return {
        "status": "ADMITTED",
        "fabricate_permitted": False,
        "seed": seed,
        "job_id": manifest.get("job_id", manifest["lineage"]),
        "checkpoint_sha256": latest_sha,
        "expected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "lineage": lineage,
        "global_update": state.global_update,
        "model_spec_sha256": spec.sha256(),
        "tokenizer_sha256": UTF8.identity,
        "parameters": spec.parameter_receipt().total,
        "tensor_sha256s": tensor_hashes,
        "metadata": metadata,
        "backend": backend,
        "admission_scope": "historical_s20_low_lr",
    }


def _first_byte_logits(backend: NeuralBackend, ids: list[int]) -> torch.Tensor:
    tokens = torch.tensor([ids], device=backend.device, dtype=torch.int64)
    modes = [(module, module.training) for module in backend.model.modules()]
    backend.model.eval()
    try:
        with torch.inference_mode():
            return backend.logits(tokens, torch.zeros_like(tokens))[0, -1].float().cpu()
    finally:
        for module, mode in modes:
            module.training = mode


def _apply_availability_mask(
    logits: torch.Tensor,
    *,
    memory_available: bool,
    query_available: bool,
) -> torch.Tensor:
    masked = logits.clone()
    if not memory_available:
        masked[TOKEN_R] = float("-inf")
    if not query_available:
        masked[TOKEN_Q] = float("-inf")
    return masked


class RawHitchhikerPolicy:
    """Unmodified greedy byte generation and strict raw action parsing."""

    origin = "model"
    representation = "utf8-raw"
    decoder_schema = "utf8-raw"

    def __init__(self, backend: NeuralBackend, surface: Any = None, *, memory_available: bool = False, max_new_tokens: int = 4) -> None:
        if type(max_new_tokens) is not int or not 2 <= max_new_tokens <= 8:
            raise ValueError("bounded action token limit must be 2..8")
        if backend.binding.tokenizer_sha256 != UTF8.identity or backend.binding.vocabulary_size != UTF8.vocabulary_size:
            raise ValueError("Hitchhiker policy requires qualified UTF8 tokenizer binding")
        self.backend = backend
        self.surface = surface
        self.memory_available = memory_available
        self.max_new_tokens = max_new_tokens
        self.model_identity = backend.binding.checkpoint_sha256 or backend.binding.identity
        self.prompts: list[str] = []

    def for_case(self, case: Any, *, memory_available: bool | None = None) -> "RawHitchhikerPolicy":
        return RawHitchhikerPolicy(
            self.backend,
            case.surface,
            memory_available=(case.memory is not None if memory_available is None else memory_available),
            max_new_tokens=self.max_new_tokens,
        )

    def propose(self, state: CognitiveState, legal: frozenset[str], channels: tuple[str, ...], positions: int) -> OperationCandidate | None:
        if self.surface is None:
            raise ValueError("bind a public case surface before decoding")
        prompt = render_hitchhiker_public(state, channels, self.surface, self.memory_available and "RETRIEVE" in legal)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.backend.binding.context_length or len(ids) > positions:
            return None
        raw, probabilities, used = [], [], 0
        for _ in range(self.max_new_tokens):
            if len(ids) > self.backend.binding.context_length or used + len(ids) > positions:
                break
            distribution = self.backend.next_distribution(ids)
            token = int(distribution.argmax())
            probabilities.append(float(distribution[token]))
            used += len(ids)
            raw.append(token)
            if token in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)):
                break
            ids.append(token)
        self.prompts.append(prompt)
        operation, argument = parse_action(raw, channels)
        return OperationCandidate(operation, argument, math.prod(probabilities), used, len(raw), tuple(raw))


class AvailabilityOnlyMaskPolicy(RawHitchhikerPolicy):
    """Availability-only first-byte mask: blocks only unavailable R/Q at step 0."""

    decoder_schema = "utf8-availability-only"

    def for_case(self, case: Any, *, memory_available: bool | None = None) -> "AvailabilityOnlyMaskPolicy":
        return AvailabilityOnlyMaskPolicy(
            self.backend,
            case.surface,
            memory_available=(case.memory is not None if memory_available is None else memory_available),
            max_new_tokens=self.max_new_tokens,
        )

    def propose(self, state: CognitiveState, legal: frozenset[str], channels: tuple[str, ...], positions: int) -> OperationCandidate | None:
        if self.surface is None:
            raise ValueError("bind a public case surface before decoding")
        mem_on = self.memory_available and "RETRIEVE" in legal
        qry_on = bool(channels) and "QUERY" in legal
        prompt = render_hitchhiker_public(state, channels, self.surface, mem_on)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.backend.binding.context_length or len(ids) > positions:
            return None
        raw, probabilities, used = [], [], 0
        for step in range(self.max_new_tokens):
            if len(ids) > self.backend.binding.context_length or used + len(ids) > positions:
                break
            if step == 0:
                logits = _first_byte_logits(self.backend, ids)
                masked_logits = _apply_availability_mask(logits, memory_available=mem_on, query_available=qry_on)
                distribution = masked_logits.softmax(-1)
            else:
                distribution = self.backend.next_distribution(ids)
            token = int(distribution.argmax())
            probabilities.append(float(distribution[token]))
            used += len(ids)
            raw.append(token)
            if token in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)):
                break
            ids.append(token)
        self.prompts.append(prompt)
        operation, argument = parse_action(raw, channels)
        return OperationCandidate(operation, argument, math.prod(probabilities), used, len(raw), tuple(raw))


class GrammarControlHitchhikerPolicy(RawHitchhikerPolicy):
    """Explicit GrammarPolicy reproduction supporting HitchhikerSurface."""

    decoder_schema = "utf8-grammar-control"

    def for_case(self, case: Any, *, memory_available: bool | None = None) -> "GrammarControlHitchhikerPolicy":
        return GrammarControlHitchhikerPolicy(
            self.backend,
            case.surface,
            memory_available=(case.memory is not None if memory_available is None else memory_available),
            max_new_tokens=self.max_new_tokens,
        )

    def propose(self, state: CognitiveState, legal: frozenset[str], channels: tuple[str, ...], positions: int) -> OperationCandidate | None:
        if self.surface is None:
            raise ValueError("bind a public case surface before decoding")
        mem_on = self.memory_available and "RETRIEVE" in legal
        prompt = render_hitchhiker_public(state, channels, self.surface, mem_on)
        self.prompts.append(prompt)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        choices: dict[str, tuple[str, str | None]] = {}
        if "ANSWER" in legal:
            choices.update({x: ("ANSWER", x) for x in SYMBOLS})
        if "ABSTAIN" in legal:
            choices["!"] = ("ABSTAIN", None)
        if "QUERY" in legal and len(channels) == 1:
            choices["Q"] = ("QUERY", channels[0])
        if "RETRIEVE" in legal and self.memory_available:
            choices["R"] = ("RETRIEVE", None)
        if "DELIBERATE" in legal:
            choices["~"] = ("DELIBERATE", None)
        if not choices:
            return None
        tokens, probabilities, used = [], [], 0
        op = arg = None
        for stage in range(3):
            if len(ids) > self.backend.binding.context_length or used + len(ids) > positions:
                return None
            distribution = self.backend.next_distribution(ids)
            used += len(ids)
            allowed = (
                [UTF8.encode(x)[0] for x in choices]
                if stage == 0
                else [UTF8.encode(x)[0] for x in SYMBOLS]
                if op == "DELIBERATE" and stage == 1
                else [UTF8.eos_id]
            )
            token = max(allowed, key=lambda x: (float(distribution[x]), -x))
            tokens.append(token)
            probabilities.append(float(distribution[token]))
            ids.append(token)
            if stage == 0:
                op, arg = choices[UTF8.decode([token])]
            elif op == "DELIBERATE" and stage == 1:
                arg = UTF8.decode([token])
            else:
                break
        return OperationCandidate(op, arg, math.prod(probabilities), used, len(tokens), tuple(tokens))


class ExternalSlotCanonicalizerPolicy:
    """Engineered public-only canonicalization wrapper; attributed separately from model."""

    origin = "engineered_canonicalizer"
    representation = "utf8-raw"
    decoder_schema = "utf8-raw"

    def __init__(
        self,
        backend: NeuralBackend,
        surface: Any = None,
        *,
        memory_available: bool = False,
        assignment: str = "first-mention",
        slot_seed: int = 17,
        strip_aliases: bool = True,
        max_new_tokens: int = 4,
    ) -> None:
        self.backend = backend
        self.surface = surface
        self.memory_available = memory_available
        self.assignment = assignment
        self.slot_seed = slot_seed
        self.strip_aliases = strip_aliases
        self.max_new_tokens = max_new_tokens
        self.model_identity = backend.binding.checkpoint_sha256 or backend.binding.identity
        self.canonicalizer = ExternalSlotCanonicalizer(
            assignment=assignment, slot_seed=slot_seed, strip_aliases=strip_aliases
        )
        self._mapping: dict[str, str] | None = None
        self._inverse: dict[str, str] | None = None
        self.prompts: list[str] = []
        self.overhead_receipts: list[dict[str, int]] = []

    def for_case(self, case: Any, *, memory_available: bool | None = None) -> "ExternalSlotCanonicalizerPolicy":
        return ExternalSlotCanonicalizerPolicy(
            self.backend,
            case.surface,
            memory_available=(case.memory is not None if memory_available is None else memory_available),
            assignment=self.assignment,
            slot_seed=self.slot_seed,
            strip_aliases=self.strip_aliases,
            max_new_tokens=self.max_new_tokens,
        )

    def propose(self, state: CognitiveState, legal: frozenset[str], channels: tuple[str, ...], positions: int) -> OperationCandidate | None:
        if self.surface is None:
            raise ValueError("bind a public case surface before decoding")
        if self._mapping is None:
            self._mapping, self._inverse = self.canonicalizer.build_mapping(state, self.surface)
        else:
            self._mapping, self._inverse = self.canonicalizer.extend_mapping(self._mapping, self._inverse, state)
        mapped_state, mapped_surface, overhead = self.canonicalizer.canonicalize_public(
            state, self.surface, self._mapping
        )
        self.overhead_receipts.append(overhead)

        mem_on = self.memory_available and "RETRIEVE" in legal
        prompt = render_hitchhiker_public(mapped_state, channels, mapped_surface, mem_on)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.backend.binding.context_length or len(ids) > positions:
            return None
        raw, probabilities, used = [], [], 0
        for _ in range(self.max_new_tokens):
            if len(ids) > self.backend.binding.context_length or used + len(ids) > positions:
                break
            distribution = self.backend.next_distribution(ids)
            token = int(distribution.argmax())
            probabilities.append(float(distribution[token]))
            used += len(ids)
            raw.append(token)
            if token in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)):
                break
            ids.append(token)
        self.prompts.append(prompt)
        raw_op, raw_arg = parse_action(raw, channels)
        op, arg = self.canonicalizer.dereference_output(raw_op, raw_arg, self._inverse)
        return OperationCandidate(op, arg, math.prod(probabilities), used, len(raw), tuple(raw))


def inspect_first_byte_distribution(
    backend: NeuralBackend,
    case: Any,
    *,
    state: CognitiveState | None = None,
    channels: tuple[str, ...] | None = None,
    memory_available: bool | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Capture complete 260-way first-byte logits, availability-masked probs, and gold ranks."""
    resolved_state = CognitiveState(case.goal).observe(case.environment.initial) if state is None else state
    resolved_channels = tuple(case.environment.channels) if channels is None else tuple(channels)
    resolved_memory = (case.memory is not None) if memory_available is None else bool(memory_available)
    query_available = bool(resolved_channels)

    prompt = render_hitchhiker_public(resolved_state, resolved_channels, case.surface, resolved_memory)
    ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
    if len(ids) > backend.binding.context_length:
        raise ValueError("prompt exceeds model context_length during first-byte inspection")

    raw_logits_t = _first_byte_logits(backend, ids)
    raw_probs_t = raw_logits_t.softmax(-1)
    masked_logits_t = _apply_availability_mask(
        raw_logits_t, memory_available=resolved_memory, query_available=query_available
    )
    masked_probs_t = masked_logits_t.softmax(-1)

    raw_logits = [float(x) for x in raw_logits_t.tolist()]
    raw_probs = [float(x) for x in raw_probs_t.tolist()]
    masked_probs = [float(x) for x in masked_probs_t.tolist()]

    gold_op, gold_arg = public_action(resolved_state, resolved_channels, case.surface, resolved_memory)
    gold_text = action_text(gold_op, gold_arg)
    gold_token_id = UTF8.encode(gold_text)[0]

    def top_entries(probs: list[float]) -> list[dict[str, Any]]:
        ordered = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:top_k]
        return [
            {
                "token_id": i,
                "label": _token_label(i),
                "logit": raw_logits[i],
                "probability": probs[i],
            }
            for i in ordered
        ]

    rank_raw = 1 + sum(1 for p in raw_probs if p > raw_probs[gold_token_id])
    rank_masked = 1 + sum(1 for p in masked_probs if p > masked_probs[gold_token_id])
    rank_answers = (
        1 + sum(1 for tid in ANSWER_TOKEN_IDS if raw_probs[tid] > raw_probs[gold_token_id])
        if gold_op == "ANSWER"
        else None
    )

    illegal_mass = (0.0 if resolved_memory else raw_probs[TOKEN_R]) + (0.0 if query_available else raw_probs[TOKEN_Q])
    max_answer_logit = max(raw_logits[tid] for tid in ANSWER_TOKEN_IDS)
    best_ans_idx = max(range(len(SYMBOLS)), key=lambda idx: raw_probs[ANSWER_TOKEN_IDS[idx]])
    best_ans_symbol = SYMBOLS[best_ans_idx]

    # Separate complete-action greedy rollout including EOS continuation.
    comp_tokens, comp_probs = [], []
    comp_ids = list(ids)
    for _ in range(4):
        if len(comp_ids) > backend.binding.context_length:
            break
        dist = backend.next_distribution(comp_ids)
        tok = int(dist.argmax())
        comp_tokens.append(tok)
        comp_probs.append(float(dist[tok]))
        if tok in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)):
            break
        comp_ids.append(tok)
    comp_op, comp_arg = parse_action(comp_tokens, resolved_channels)

    return {
        "case_id": case.case_id,
        "world_id": case.world_id,
        "panel": case.panel,
        "memory_available": resolved_memory,
        "query_available": query_available,
        "gold_action": {"operation": gold_op, "argument": gold_arg, "text": gold_text, "token_id": gold_token_id},
        "raw_logits_260": raw_logits,
        "raw_probs_260": raw_probs,
        "availability_masked_probs_260": masked_probs,
        "top_k_raw": top_entries(raw_probs),
        "top_k_masked": top_entries(masked_probs),
        "gold_rank_overall_raw": rank_raw,
        "gold_rank_overall_masked": rank_masked,
        "gold_rank_among_answers": rank_answers,
        "illegal_action_mass": illegal_mass,
        "logit_gap_R_minus_max_answer": raw_logits[TOKEN_R] - max_answer_logit,
        "logit_gap_Q_minus_max_answer": raw_logits[TOKEN_Q] - max_answer_logit,
        "total_answer_letter_mass_raw": math.fsum(raw_probs[tid] for tid in ANSWER_TOKEN_IDS),
        "total_answer_letter_mass_masked": math.fsum(masked_probs[tid] for tid in ANSWER_TOKEN_IDS),
        "abstain_mass_raw": raw_probs[TOKEN_ABSTAIN],
        "abstain_mass_masked": masked_probs[TOKEN_ABSTAIN],
        "answer_only_forced_choice": {
            "predicted_symbol": best_ans_symbol,
            "probability": raw_probs[ANSWER_TOKEN_IDS[best_ans_idx]],
            "correct": best_ans_symbol == case.truth.expected,
            "label": "FORCED_CHOICE_DIAGNOSTIC_NOT_AUTONOMOUS_POLICY",
        },
        "complete_action_score": {
            "raw_tokens": comp_tokens,
            "operation": comp_op,
            "argument": comp_arg,
            "joint_probability": math.prod(comp_probs) if comp_probs else 0.0,
            "ended_eos": bool(comp_tokens) and comp_tokens[-1] == UTF8.eos_id,
            "label": "COMPLETE_ACTION_SCORE_SEPARATE_FROM_FIRST_BYTE",
        },
    }


def score_entity_equivariance(
    backend: NeuralBackend,
    paired_cases: tuple[Any, ...],
    permutations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Measure argmax equivariance, joint accuracy, and inverse-mapped distribution TV distance."""
    grouped: dict[str, list[Any]] = {}
    for case in paired_cases:
        grouped.setdefault(case.pair_id, []).append(case)

    complete = 0
    argmax_equiv = 0
    both_correct_and_equiv = 0
    tv_distances: list[float] = []

    for pid, pair in sorted(grouped.items()):
        if len(pair) != 2 or pid not in permutations:
            continue
        meta = permutations[pid]
        mapping = meta["mapping"]
        inverse = meta["inverse_mapping"]
        by_id = {c.case_id: c for c in pair}
        orig = by_id[meta["original_case_id"]]
        perm = by_id[meta["permuted_case_id"]]

        insp_o = inspect_first_byte_distribution(backend, orig)
        insp_p = inspect_first_byte_distribution(backend, perm)

        op_o = insp_o["complete_action_score"]["operation"]
        arg_o = insp_o["complete_action_score"]["argument"]
        op_p = insp_p["complete_action_score"]["operation"]
        arg_p = insp_p["complete_action_score"]["argument"]

        if op_o == op_p:
            if op_o == "ANSWER":
                equiv = inverse.get(arg_p) == arg_o
            elif op_o == "DELIBERATE":
                equiv = inverse.get(arg_p) == arg_o
            else:
                equiv = arg_o == arg_p
        else:
            equiv = False

        correct_o = op_o == "ANSWER" and arg_o == orig.truth.expected
        correct_p = op_p == "ANSWER" and arg_p == perm.truth.expected
        complete += 1
        argmax_equiv += int(equiv)
        both_correct_and_equiv += int(equiv and correct_o and correct_p)

        # Normalized answer-letter distributions over A..P, aligned by inverse map.
        p_o = [insp_o["raw_probs_260"][tid] for tid in ANSWER_TOKEN_IDS]
        p_p_raw = {SYMBOLS[i]: insp_p["raw_probs_260"][ANSWER_TOKEN_IDS[i]] for i in range(len(SYMBOLS))}
        sum_o = sum(p_o) or 1.0
        sum_p = sum(p_p_raw.values()) or 1.0
        tv = 0.5 * sum(abs(p_o[i] / sum_o - p_p_raw[mapping[SYMBOLS[i]]] / sum_p) for i in range(len(SYMBOLS)))
        tv_distances.append(tv)

    return {
        "model_identity": backend.binding.checkpoint_sha256 or backend.binding.identity,
        "cases_sha256": cases_identity(paired_cases),
        "complete_pairs": complete,
        "argmax_equivariance_rate": argmax_equiv / complete if complete else None,
        "equivariant_and_both_correct_rate": both_correct_and_equiv / complete if complete else None,
        "mean_answer_distribution_tv": math.fsum(tv_distances) / len(tv_distances) if tv_distances else None,
    }


def classify_diagnostic_outcomes(
    raw_metrics: dict[str, Any],
    masked_metrics: dict[str, Any],
    *,
    factor_metrics: dict[str, dict[str, Any]] | None = None,
    equivariance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify Stage 0/counterfactual outcomes per Section 3 interpretation rules."""
    raw_acc = raw_metrics.get("accuracy") or 0.0
    masked_acc = masked_metrics.get("accuracy") or 0.0
    raw_inv = raw_metrics.get("invalid_rate") or 0.0
    masked_inv = masked_metrics.get("invalid_rate") or 0.0

    inv_drop = raw_inv - masked_inv
    acc_gain = masked_acc - raw_acc
    findings: list[str] = []

    if inv_drop >= 0.10 and acc_gain < 0.05:
        findings.append("CONTROLLER_LEGALITY_ONLY_MAPPING_UNRESOLVED")
    if acc_gain >= 0.05:
        findings.append("ACTION_SELECTION_CONTRIBUTED_UNDER_MASK")

    if factor_metrics:
        surface_accs = [
            factor_metrics[k]["accuracy"]
            for k in ("factor-wording", "factor-timestamps", "factor-row-order", "factor-alias-clause")
            if k in factor_metrics and factor_metrics[k].get("accuracy") is not None
        ]
        if len(surface_accs) >= 2 and (max(surface_accs) - min(surface_accs)) >= 0.10:
            findings.append("SURFACE_DISTRIBUTION_MISMATCH_PRIORITIZE_DATA")
        dist = factor_metrics.get("factor-distractors", {})
        if dist.get("accuracy_gap") is not None and abs(dist["accuracy_gap"]) >= 0.05:
            findings.append("INVARIANCE_FAILURE")
        ans_chg = factor_metrics.get("factor-answer-edge-swap", {})
        if ans_chg.get("both_correct_rate") is not None and ans_chg["both_correct_rate"] >= 0.80:
            findings.append("BEHAVIORAL_GROUNDING_SUPPORTED")

    if equivariance and equivariance.get("argmax_equivariance_rate") is not None:
        if equivariance["argmax_equivariance_rate"] < 0.90 and "INVARIANCE_FAILURE" not in findings:
            findings.append("INVARIANCE_FAILURE")

    return {
        "raw_accuracy": raw_acc,
        "masked_accuracy": masked_acc,
        "invalid_rate_drop": inv_drop,
        "accuracy_gain_under_mask": acc_gain,
        "findings": findings,
    }
