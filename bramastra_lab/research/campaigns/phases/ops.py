"""Model-operations interface for phase executors (D4).

Production ops perform real training/generation on GPU. Test doubles implement
the same interface, record calls (init, treatments, streams, checkpoints,
budgets, generations) and return deterministic counts WITHOUT optimizer
updates. Executors never branch on ops type; they call the interface.
"""
from __future__ import annotations

from typing import Any, Protocol


class ModelOps(Protocol):
    """Expensive model boundary behind every phase executor."""

    def init_model(self, *, seed: int, profile: str, device: str) -> Any:
        """Random initialization for one seed/profile/device; returns handle."""
        ...

    def snapshot_state(self, handle: Any) -> Any:
        """Capture exact starting state for paired-equality checks."""
        ...

    def states_equal(self, left: Any, right: Any) -> bool:
        ...

    def training_update(self, handle: Any, *, batch: Any, window: Any,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """One explicit accumulate+finalize update; returns actual counters."""
        ...

    def save_checkpoint(self, handle: Any, *, path: str,
                        fraction: float) -> str:
        """Publish checkpoint; returns checkpoint identity."""
        ...

    def free_generation(self, handle: Any, *, prompt: Any,
                        max_new_tokens: int) -> dict[str, Any]:
        """Free-generation evaluation with stop evidence."""
        ...

    def optimizer_updates(self, handle: Any) -> int:
        ...


class ProductionOps:
    """Real GPU ops (never executed locally per policy; owner launch only)."""

    def __init__(self, *, precision: str = "fp16_autocast") -> None:
        self.precision = precision

    def init_model(self, *, seed: int, profile: str, device: str):
        from bramastra_lab.research.config import BuildConfig, seed_everything
        from bramastra_lab.research.models import IntegratedModel
        from bramastra_lab.research.learning.k8_trainer import (
            AllocationContext, K8Trainer)
        import time

        seed_everything(seed)
        config = BuildConfig.from_dict({"model": {"profile": profile}})
        model = IntegratedModel(config).to(device)
        trainer = K8Trainer(
            config, model, device=device,
            precision="fp32" if not str(device).startswith("cuda") else self.precision,
            require_allocation=False)
        return {"config": config, "model": model, "trainer": trainer,
                "seed": seed, "profile": profile, "device": device,
                "started_updates": 0}

    def snapshot_state(self, handle):
        import copy

        return copy.deepcopy(handle["model"].state_dict())

    def states_equal(self, left, right) -> bool:
        try:
            if set(left) != set(right):
                return False
            import torch

            return all(bool((left[k].detach().cpu() == right[k].detach().cpu()).all())
                       for k in left)
        except Exception:
            return False

    def training_update(self, handle, *, batch, window, extra=None):
        trainer = handle["trainer"]
        trainer.accumulate_full_window(
            batch, window_builder=lambda _: window,
            extra_terms_fn=(lambda: extra) if extra else None)
        report = trainer.finalize_update()
        return {"committed": 1, "attempted": 1,
                "exposure": batch.target_count,
                "optimizer_update": report.optimizer_update}

    def save_checkpoint(self, handle, *, path, fraction):
        import os

        from bramastra_lab.research.runtime.checkpoint import publish_checkpoint

        payload = handle["trainer"].state_payload()
        manifest = publish_checkpoint(
            run_dir=os.path.dirname(os.path.dirname(path)),
            run_id=f"{handle.get('seed')}-{fraction}",
            update_index=handle["trainer"].counters.optimizer_updates,
            payload=payload,
            config_identity=payload.get("config_identity") or "k8",
            data_identity="k8-bundle",
            code_identity="k8")
        return manifest.checkpoint_id

    def free_generation(self, handle, *, prompt, max_new_tokens):
        from bramastra_lab.research.runtime.inference import generate_free_form

        return generate_free_form(
            handle["model"], handle["config"], prompt,
            max_new_tokens=max_new_tokens).__dict__

    def optimizer_updates(self, handle) -> int:
        try:
            return int(handle["trainer"].counters.optimizer_updates)
        except Exception:
            return 0


class RecordingDoubleOps:
    """Deterministic test double: records calls, performs zero updates."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._handles: dict[int, dict] = {}
        self._next = 0

    def init_model(self, *, seed, profile, device):
        handle_id = self._next
        self._next += 1
        handle = {"double_id": handle_id, "seed": seed, "profile": profile,
                  "device": device, "updates": 0, "attempted": 0,
                  "exposure": 0, "init_state": f"init-{seed}-{profile}"}
        self._handles[handle_id] = handle
        self.calls.append(("init_model", {"seed": seed, "profile": profile,
                                          "device": device}))
        return handle

    def snapshot_state(self, handle):
        self.calls.append(("snapshot_state", {"id": handle["double_id"]}))
        return dict(handle)

    def states_equal(self, left, right) -> bool:
        self.calls.append(("states_equal", {}))
        if isinstance(left, dict) and isinstance(right, dict):
            return left.get("init_state") == right.get("init_state") \
                and left.get("seed") == right.get("seed")
        return left == right

    def training_update(self, handle, *, batch, window, extra=None):
        weights = dict(getattr(window, "weights", {}))
        enabled = sorted(getattr(window, "enabled_terms", frozenset()))
        self.calls.append(("training_update", {"weights": weights,
                                               "enabled": enabled,
                                               "extra_terms": sorted((extra or {})),
                                               "targets": getattr(batch, "target_count", 0)}))
        handle["updates"] += 1
        handle["attempted"] += 1
        exposure = int(getattr(batch, "target_count", 0))
        handle["exposure"] += exposure
        return {"committed": 1, "attempted": 1, "exposure": exposure,
                "optimizer_update": handle["updates"]}

    def save_checkpoint(self, handle, *, path, fraction):
        self.calls.append(("save_checkpoint", {"id": handle["double_id"],
                                               "fraction": fraction,
                                               "path": path}))
        return f"double-ckpt-{handle['double_id']}-{fraction}"

    def free_generation(self, handle, *, prompt, max_new_tokens):
        self.calls.append(("free_generation", {"id": handle["double_id"],
                                               "max_new_tokens": max_new_tokens}))
        return {"answer": "double", "stopped_on_eos": True, "tokens": []}

    def optimizer_updates(self, handle) -> int:
        return 0
