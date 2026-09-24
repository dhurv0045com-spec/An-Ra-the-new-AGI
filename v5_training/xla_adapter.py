"""Experimental XLA topology and collective primitives, not a production trainer.

This module can inspect PJRT topology, reduce gradients, and serialize
rank-local CPU/XLA generator state. The production entry remains fail-closed
for ``execution="xla"``; its only XLA route is an explicitly unqualified
one/two-update ``execution="xla-development"`` path. That path does not prove
exact continuation of the complete training state or qualify the production
trainer. Hardware qualification must never be inferred from a successful
import or a synthetic canary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from typing import Any, Mapping

ADAPTER_SCHEMA = "anra-v5-xla-adapter/v1"
PENDING_STATUS = "IMPLEMENTED_PENDING_PRE500M_TPU"
EVIDENCE_REQUIRED = "TPU_EVIDENCE_REQUIRED"
XLA_RNG_STATE_SCHEMA = "anra-v5-xla-rng-state/v1"
XLA_STATUS_VOTE_SCHEMA = "anra-v5-xla-status-vote/v2"
_XLA_STATUS_STAGES = frozenset({
    "forward", "backward", "local_update", "pre_collective",
})


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _load_xla() -> Any:
    try:
        import torch_xla.runtime as xruntime  # type: ignore
        import torch_xla.core.xla_model as xm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch_xla is not installed: XLA execution needs PRE500M "
            f"certification ({EVIDENCE_REQUIRED})") from exc
    return xruntime, xm


def xla_status() -> dict[str, object]:
    """Discover the live XLA topology without executing anything."""

    try:
        xruntime, xm = _load_xla()
    except RuntimeError as exc:
        return {"schema": ADAPTER_SCHEMA, "status": EVIDENCE_REQUIRED,
                "reason": str(exc)}
    try:
        device_type = str(xruntime.device_type())
        world_size = int(xruntime.world_size())
        ordinal = int(xruntime.global_ordinal())
        versions = {"torch_xla": getattr(__import__("torch_xla"), "__version__", "unknown")}
    except Exception as exc:
        return {"schema": ADAPTER_SCHEMA, "status": EVIDENCE_REQUIRED,
                "reason": f"XLA discovery failed: {exc}"}
    _ = xm
    return {"schema": ADAPTER_SCHEMA, "status": PENDING_STATUS,
            "device_type": device_type, "world_size": world_size,
            "ordinal": ordinal, "versions": versions}


def require_frozen_topology(world_size: int, *, replicas: int) -> dict[str, object]:
    """Require the live world to equal the frozen replica count."""

    if replicas <= 0:
        raise ValueError("frozen replica count must be positive")
    if world_size != replicas:
        raise ValueError(
            f"XLA world size {world_size} does not equal the frozen "
            f"{replicas} replicas: refusing production mode")
    return {"schema": ADAPTER_SCHEMA, "status": PENDING_STATUS,
            "world_size": world_size, "replicas": replicas}


def capture_xla_rng_state(
    *, device: str, torch_module: Any | None = None,
) -> bytes:
    """Capture canonical process-CPU and rank-local XLA generator state."""

    torch = torch_module
    if torch is None:
        import torch as torch
    if not isinstance(device, str) or not device:
        raise ValueError("XLA RNG capture needs the live rank device name")
    _, xm = _load_xla()
    getter = getattr(xm, "get_rng_state", None)
    if not callable(getter):
        raise RuntimeError("XLA runtime lacks get_rng_state for rank-safe continuation")
    xla_state = getter(device=device)
    if type(xla_state) is not int or not 0 <= xla_state < 2**64:
        raise RuntimeError("XLA RNG state is not a supported unsigned 64-bit integer")
    cpu_state = torch.get_rng_state()
    if str(getattr(cpu_state, "device", "cpu")) != "cpu":
        raise RuntimeError("CPU RNG capture returned a non-CPU state")
    document = {
        "schema": XLA_RNG_STATE_SCHEMA,
        "cpu_state_base64": base64.b64encode(
            cpu_state.detach().cpu().numpy().tobytes()
        ).decode("ascii"),
        "xla_state": xla_state,
        "xla_device": device,
    }
    return _canonical_json(document)


def restore_xla_rng_state(
    payload: bytes, *, device: str, torch_module: Any | None = None,
) -> None:
    """Restore CPU and rank-local XLA RNG transactionally when possible.

    If the target XLA setter mutates its state and then fails, the adapter
    restores the prior XLA and CPU states. Callers must discard the worker if
    it reports an incomplete rollback.
    """

    torch = torch_module
    if torch is None:
        import torch as torch
    try:
        document = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("XLA rank RNG checkpoint is not valid JSON") from exc
    if (not isinstance(document, dict)
            or set(document) != {
                "schema", "cpu_state_base64", "xla_state", "xla_device",
            }
            or document["schema"] != XLA_RNG_STATE_SCHEMA
            or type(document["xla_state"]) is not int
            or not 0 <= document["xla_state"] < 2**64):
        raise ValueError("XLA rank RNG checkpoint fields do not match schema")
    try:
        cpu_bytes = base64.b64decode(document["cpu_state_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("XLA rank RNG CPU state is not valid base64") from exc
    if payload != _canonical_json(document) or not cpu_bytes:
        raise ValueError("XLA rank RNG checkpoint must be canonical and nonempty")
    if document["xla_device"] != device:
        raise ValueError("XLA RNG checkpoint device differs from the live rank device")
    _, xm = _load_xla()
    setter = getattr(xm, "set_rng_state", None)
    getter = getattr(xm, "get_rng_state", None)
    if not callable(setter) or not callable(getter):
        raise RuntimeError("XLA runtime lacks get/set_rng_state for rank-safe continuation")
    cpu_tensor = torch.tensor(list(cpu_bytes), dtype=torch.uint8, device="cpu")
    # Validate the CPU state using an isolated generator before touching
    # process-global or XLA state.
    torch.Generator(device="cpu").set_state(cpu_tensor)
    previous_cpu_state = torch.get_rng_state().clone()
    previous_xla_state = getter(device=device)
    if type(previous_xla_state) is not int or not 0 <= previous_xla_state < 2**64:
        raise RuntimeError("live XLA RNG state is not a supported unsigned 64-bit integer")
    try:
        setter(document["xla_state"], device=device)
        torch.set_rng_state(cpu_tensor)
    except Exception as exc:
        rollback_errors = []
        try:
            setter(previous_xla_state, device=device)
        except Exception as rollback_exc:  # pragma: no cover - runtime-specific failure
            rollback_errors.append(f"XLA rollback failed: {rollback_exc}")
        try:
            torch.set_rng_state(previous_cpu_state)
        except Exception as rollback_exc:  # pragma: no cover - runtime-specific failure
            rollback_errors.append(f"CPU rollback failed: {rollback_exc}")
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise RuntimeError(
                "XLA rank RNG restore failed; rollback incomplete: "
                f"{detail}. Discard this worker process."
            ) from exc
        raise RuntimeError(
            "XLA rank RNG restore failed; previous CPU and XLA states were restored"
        ) from exc


def xla_rng_state_sha256(
    *, device: str, torch_module: Any | None = None,
) -> str:
    """Hash the exact canonical payload consumed by the restore function."""

    return hashlib.sha256(capture_xla_rng_state(
        device=device, torch_module=torch_module,
    )).hexdigest()


class XLAReplicatedBackend:
    """Replica helper for unqualified development, not a production trainer.

    Production execution remains gated. Its reduction methods and bounded
    development use do not qualify rank-safe TPU optimizer or checkpoint
    behavior.
    """

    def __init__(self, *, replica_backend: Any, replicas: int,
                 world_size: int | None = None,
                 torch_module: Any | None = None) -> None:
        self.replica_backend = replica_backend
        self.replicas = int(replicas)
        if self.replicas <= 0:
            raise ValueError("replica count must be positive")
        if world_size is not None:
            require_frozen_topology(int(world_size), replicas=self.replicas)
        self.torch = torch_module
        self.last_receipt: dict[str, object] | None = None
        self.last_collective_receipt_sha256: str | None = None
        self._status_vote_sequence = 0

    def vote_status(
        self, *, stage: str, error: BaseException | str | None = None,
        evidence_sha256: str | None = None,
    ) -> dict[str, object]:
        """Gather one fail-symmetric local status claim from every replica.

        Call this on every rank after forward/backward work, passing any local
        exception as ``error``. A failing claim makes every rank raise the
        same deterministic error before the caller can enter a later
        collective. Tags use a per-backend monotonic sequence; stage is
        validated from the gathered claims so a stage mismatch is shared by
        all participants.
        """

        _, xm = _load_xla()
        return self._vote_status_with_xm(
            xm, stage=stage, error=error, evidence_sha256=evidence_sha256,
        )

    def run_local_stage(self, *, stage: str, callback: Any) -> Any:
        """Run rank-local work, then make every rank agree before proceeding.

        A rank that raises locally still enters the same vote as healthy
        ranks. If any claim reports an error, all participants raise from the
        shared vote before a later collective can begin.
        """

        if stage not in {"forward", "backward", "local_update"}:
            raise ValueError("rank-local status stage must describe local compute")
        if not callable(callback):
            raise TypeError("rank-local stage callback must be callable")
        result = None
        local_error: Exception | None = None
        try:
            result = callback()
        except Exception as exc:
            local_error = exc
        # XLA is lazy: Python forward/backward can enqueue invalid work without
        # executing it yet. Flush and wait before the out-of-graph status vote
        # so a device-side failure is included in the shared claim instead of
        # surfacing after healthy ranks enter the next collective.
        try:
            _, xm = _load_xla()
            mark_step = getattr(xm, "mark_step", None)
            wait_device_ops = getattr(xm, "wait_device_ops", None)
            if not callable(mark_step) or not callable(wait_device_ops):
                raise RuntimeError(
                    "XLA runtime lacks mark_step/wait_device_ops for local-stage voting"
                )
            mark_step()
            wait_device_ops()
        except Exception as exc:
            if local_error is None:
                local_error = exc
        self.vote_status(stage=stage, error=local_error)
        if local_error is not None:
            # A correctly shared vote must reject this rank's own failure.
            raise RuntimeError(
                f"XLA {stage} status vote returned success after a local failure"
            ) from local_error
        return result

    def _vote_status_with_xm(
        self, xm: Any, *, stage: str, error: BaseException | str | None,
        evidence_sha256: str | None = None,
    ) -> dict[str, object]:
        mesh_reduce = getattr(xm, "mesh_reduce", None)
        if not callable(mesh_reduce):
            raise RuntimeError(
                "XLA runtime lacks mesh_reduce for rank-symmetric status voting"
            )
        try:
            rank = self.ordinal()
        except Exception as exc:
            # A rank that cannot identify itself cannot safely participate.
            raise RuntimeError(f"cannot determine XLA rank for status vote: {exc}") from exc
        sequence = getattr(self, "_status_vote_sequence", 0)
        self._status_vote_sequence = sequence + 1
        if error is None:
            error_claim = None
        elif isinstance(error, BaseException):
            error_claim = {
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
            }
        else:
            error_claim = {"type": "builtins.str", "message": str(error)}
        claim = {
            "schema": XLA_STATUS_VOTE_SCHEMA,
            "sequence": sequence,
            "stage": stage,
            "rank": rank,
            "world_size": self.replicas,
            "error": error_claim,
            "evidence_sha256": evidence_sha256,
        }
        tag = f"{XLA_STATUS_VOTE_SCHEMA}:{sequence:020d}"
        try:
            claims = mesh_reduce(tag, claim, lambda gathered: gathered)
        except Exception as exc:
            raise RuntimeError(
                f"XLA {stage!r} status vote sequence {sequence} failed: {exc}"
            ) from exc
        ordered = self._validate_status_claims(
            claims, sequence=sequence, expected_stage=stage,
        )
        failures = [item for item in ordered if item["error"] is not None]
        receipt: dict[str, object] = {
            "schema": XLA_STATUS_VOTE_SCHEMA,
            "sequence": sequence,
            "stage": stage,
            "world_size": self.replicas,
            "status": "failed" if failures else "passed",
            "claims": ordered,
        }
        if failures:
            detail = "; ".join(
                f"rank {item['rank']}: {item['error']['type']}: "
                f"{item['error']['message']}" for item in failures
            )
            raise RuntimeError(
                f"XLA {stage} status vote failed on {len(failures)} "
                f"of {self.replicas} ranks ({detail})"
            )
        if stage == "pre_collective":
            layout_hashes = {item["evidence_sha256"] for item in ordered}
            if len(layout_hashes) != 1:
                raise RuntimeError(
                    "XLA pre_collective gradient layout differs across ranks"
                )
        return receipt

    def _validate_status_claims(
        self, claims: Any, *, sequence: int, expected_stage: str,
    ) -> list[dict[str, object]]:
        """Validate and rank-order the complete shared host-side vote."""

        if not isinstance(claims, (list, tuple)):
            raise RuntimeError("XLA status vote returned a non-sequence of claims")
        if len(claims) != self.replicas:
            raise RuntimeError(
                f"XLA status vote expected {self.replicas} claims, received {len(claims)}"
            )
        ranks: dict[int, dict[str, object]] = {}
        required = {
            "schema", "sequence", "stage", "rank", "world_size", "error",
            "evidence_sha256",
        }
        for claim in claims:
            if not isinstance(claim, Mapping) or set(claim) != required:
                raise RuntimeError("XLA status vote contains a malformed claim")
            rank = claim["rank"]
            if type(rank) is not int or not 0 <= rank < self.replicas:
                raise RuntimeError(f"XLA status vote contains invalid rank {rank!r}")
            if rank in ranks:
                raise RuntimeError(f"XLA status vote contains duplicate rank {rank}")
            if claim["schema"] != XLA_STATUS_VOTE_SCHEMA:
                raise RuntimeError(f"XLA status vote rank {rank} has a mismatched schema")
            if type(claim["sequence"]) is not int or claim["sequence"] != sequence:
                raise RuntimeError(f"XLA status vote rank {rank} has a mismatched sequence")
            if type(claim["world_size"]) is not int or claim["world_size"] != self.replicas:
                raise RuntimeError(f"XLA status vote rank {rank} has a mismatched world size")
            if not isinstance(claim["stage"], str) or claim["stage"] not in _XLA_STATUS_STAGES:
                raise RuntimeError(f"XLA status vote rank {rank} has an invalid stage")
            if claim["stage"] != expected_stage:
                raise RuntimeError(
                    f"XLA status vote stage mismatch: expected {expected_stage!r}, "
                    f"rank {rank} claimed {claim['stage']!r}"
                )
            failure = claim["error"]
            if failure is not None and (
                    not isinstance(failure, Mapping)
                    or set(failure) != {"type", "message"}
                    or not isinstance(failure["type"], str)
                    or not isinstance(failure["message"], str)):
                raise RuntimeError(f"XLA status vote rank {rank} has a malformed error")
            evidence_sha256 = claim["evidence_sha256"]
            if evidence_sha256 is not None and (
                    not isinstance(evidence_sha256, str)
                    or len(evidence_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in evidence_sha256)):
                raise RuntimeError(
                    f"XLA status vote rank {rank} has a malformed evidence SHA-256"
                )
            if (claim["stage"] == "pre_collective" and failure is None
                    and evidence_sha256 is None):
                raise RuntimeError(
                    f"XLA status vote rank {rank} omitted gradient layout evidence"
                )
            ranks[rank] = {
                "rank": rank,
                "stage": claim["stage"],
                "error": None if failure is None else {
                    "type": failure["type"], "message": failure["message"],
                },
                "evidence_sha256": evidence_sha256,
            }
        if set(ranks) != set(range(self.replicas)):
            missing = sorted(set(range(self.replicas)) - set(ranks))
            raise RuntimeError(f"XLA status vote is missing ranks {missing}")
        return [ranks[rank] for rank in range(self.replicas)]

    def status(self) -> dict[str, object]:
        """Adapter status: pending until PRE500M measures it on TPU."""

        discovered = xla_status()
        if discovered.get("status") != PENDING_STATUS:
            return discovered
        if int(discovered["world_size"]) != self.replicas:
            return {"schema": ADAPTER_SCHEMA, "status": EVIDENCE_REQUIRED,
                    "reason": "live XLA world does not match the frozen replicas"}
        return {"schema": ADAPTER_SCHEMA, "status": PENDING_STATUS,
                "replicas": self.replicas,
                "world_size": discovered["world_size"],
                "device_type": discovered.get("device_type")}

    def all_reduce_sum_gradients(self, model: Any, *, scale: float | None = None) -> None:
        """SUM replica gradients into every rank, optionally scaling the sum."""

        _, xm = _load_xla()
        local_error: BaseException | None = None
        layout_sha256: str | None = None
        try:
            if scale is not None and (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
            ):
                raise ValueError("gradient-reduction scale must be finite")
            parameters = list(model.parameters())
            named_parameters = list(model.named_parameters())
            if len(parameters) != len(named_parameters):
                raise ValueError("model parameter names do not align with parameter order")
            layout = [{
                "index": index,
                "name": named_parameters[index][0],
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "has_gradient": parameter.grad is not None,
            } for index, parameter in enumerate(parameters)]
            layout_sha256 = hashlib.sha256(_canonical_json({
                "gradient_layout": layout,
                "reduction_scale": None if scale is None else float(scale),
            })).hexdigest()
            gradients = [parameter.grad for parameter in parameters
                         if parameter.grad is not None]
            if not gradients:
                raise ValueError("no gradients to reduce")
        except Exception as exc:
            gradients = []
            local_error = exc
        self._vote_status_with_xm(
            xm, stage="pre_collective", error=local_error,
            evidence_sha256=layout_sha256,
        )
        reduced = (
            xm.all_reduce(xm.REDUCE_SUM, gradients)
            if scale is None
            else xm.all_reduce(xm.REDUCE_SUM, gradients, scale=float(scale))
        )
        if len(reduced) != len(gradients):
            raise RuntimeError("XLA gradient reduction returned an incomplete gradient list")
        for parameter, value in zip(
                [parameter for parameter in model.parameters()
                 if parameter.grad is not None], reduced):
            parameter.grad = value
        self.last_collective_receipt_sha256 = hashlib.sha256(_canonical_json({
            "schema": "anra-v5-xla-gradient-sum-receipt/v1",
            "rank": self.ordinal(),
            "world_size": self.replicas,
            "gradient_count": len(reduced),
            "gradient_shapes": [list(value.shape) for value in reduced],
            "reduction_scale": None if scale is None else float(scale),
            "operation": "xm.all_reduce(xm.REDUCE_SUM)",
        })).hexdigest()

    def capture_rng_state(self, *, torch_module: Any | None = None) -> bytes:
        """Capture CPU and XLA generator states in a versioned rank payload.

        PyTorch/XLA exposes its generator state as an integer seed/state value;
        this adapter stores that exact value together with the CPU generator
        bytes used by Python-side PyTorch operations. Exact continuation still
        requires target-side replay qualification for the Kaggle runtime.
        """

        parameter = next(self.replica_backend.model.parameters())
        device = str(parameter.device)
        return capture_xla_rng_state(
            device=device, torch_module=torch_module or self.torch,
        )

    def restore_rng_state(
        self, payload: bytes, *, torch_module: Any | None = None,
    ) -> None:
        """Restore CPU state then the XLA rank-local generator state."""

        device = str(next(self.replica_backend.model.parameters()).device)
        restore_xla_rng_state(
            payload,
            device=device,
            torch_module=torch_module or self.torch,
        )

    def rng_state_sha256(self, *, torch_module: Any | None = None) -> str:
        """Hash the exact versioned CPU-plus-XLA payload used by v2 restore."""

        return hashlib.sha256(self.capture_rng_state(torch_module=torch_module)).hexdigest()

    def synchronized_step(self, *, model: Any, optimizer: Any) -> None:
        """One synchronized optimizer step across replicas (rank-symmetric)."""

        self.all_reduce_sum_gradients(model)
        optimizer.step()

    def rank_owns_checkpoints(self) -> bool:
        """Checkpoint writer ownership: rank 0 only."""

        return self.ordinal() == 0

    def checkpoint_coordinator(self, *, initial_rank_cumulative_tokens: int = 0):
        """Create the rank-zero publish/broadcast coordinator for this world.

        The coordinator uses PyTorch/XLA's out-of-graph ``mesh_reduce`` to
        gather state claims, publish on rank zero only, and return a shared
        commit result. The production entry still fails closed before this
        path until TPU backend and resume qualification pass.
        """

        _, xm = _load_xla()
        mesh_reduce = getattr(xm, "mesh_reduce", None)
        if not callable(mesh_reduce):
            raise RuntimeError(
                "XLA runtime lacks mesh_reduce for rank-safe checkpoint publication"
            )
        from v5_training.distributed import RankZeroCheckpointCoordinator

        return RankZeroCheckpointCoordinator(
            rank=self.ordinal(), world_size=self.replicas, mesh_reduce=mesh_reduce,
            initial_rank_cumulative_tokens=initial_rank_cumulative_tokens,
        )

    def ordinal(self) -> int:
        """This rank's global ordinal (fails closed without XLA)."""

        xruntime, _ = _load_xla()
        try:
            return int(xruntime.global_ordinal())
        except Exception as exc:
            raise RuntimeError(f"cannot determine XLA rank: {exc}") from exc


__all__ = ["ADAPTER_SCHEMA", "EVIDENCE_REQUIRED", "PENDING_STATUS",
           "XLA_RNG_STATE_SCHEMA", "XLA_STATUS_VOTE_SCHEMA",
           "XLAReplicatedBackend", "capture_xla_rng_state", "restore_xla_rng_state",
           "xla_rng_state_sha256", "require_frozen_topology", "xla_status"]
