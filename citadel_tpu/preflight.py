"""Single preflight command for the TPU path: `python -m citadel_tpu.preflight`.

Verifies, before any XLA compile or device use: Citadel + Cymek runtime SHAs,
platform, PJRT setting, torch versions, citadel_tpu imports, Cymek file
presence AND real Cymek imports, TPU availability, and XLA API compatibility.
Prints the READY block and exits 0 on READY_FOR_T0=YES, 1 otherwise. A local
CPU box correctly reports READY_FOR_T0=NO with TPU available: FAIL (that is
the expected fail-closed result, not an error in preflight itself).
"""

from __future__ import annotations

import importlib
import os
import sys


CITADEL_MODULES = (
    "citadel_tpu.environment",
    "citadel_tpu.xla_backend",
    "citadel_tpu.runtime_bootstrap",
    "citadel_tpu.one_update",
    "citadel_tpu.calculator_data",
    "citadel_tpu.calculator_train",
    "citadel_tpu.checkpoint",
    "citadel_tpu.throughput",
)

CYMEK_MODULES = (
    "anra_v5.miniature_run",
    "v5_model.config",
    "v5_model.core",
    "v5_contracts.model_spec",
    "v5_objectives.causal_lm",
    "v5_training.optimizer",
    "v5_training.distributed",
)

XLA_API_CHECKS = (
    ("torch_xla.device", "torch_xla", "device"),
    ("torch_xla.runtime.world_size", "torch_xla.runtime", "world_size"),
    ("torch_xla.runtime.device_type", "torch_xla.runtime", "device_type"),
    ("xm.mark_step", "torch_xla.core.xla_model", "mark_step"),
    ("xm.optimizer_step", "torch_xla.core.xla_model", "optimizer_step"),
    ("rendezvous(xr|xm)", None, None),
)


def _try_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def _xla_api_status() -> tuple[str, list[str]]:
    try:
        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
    except Exception as exc:
        return "UNAVAILABLE", [f"torch_xla import failed: {type(exc).__name__}"]
    missing: list[str] = []
    for label, modname, attr in XLA_API_CHECKS:
        try:
            if label.startswith("rendezvous"):
                if not callable(getattr(xr, "rendezvous", None)) and not callable(getattr(xm, "rendezvous", None)):
                    missing.append(label)
                continue
            module = importlib.import_module(modname)
            if not callable(getattr(module, attr, None)):
                missing.append(label)
        except Exception:
            missing.append(label)
    return ("PASS", []) if not missing else ("FAIL", missing)


def _versions() -> dict[str, str]:
    try:
        import torch

        torch_v = getattr(torch, "__version__", "unknown")
    except Exception:
        torch_v = "unavailable"
    try:
        import torch_xla

        xla_v = getattr(torch_xla, "__version__", "unknown")
    except Exception:
        xla_v = "unavailable"
    return {"torch": torch_v, "torch_xla": xla_v}


def main() -> int:
    from citadel_tpu import runtime_bootstrap as rb

    lines: list[str] = []
    ok = True

    csha = rb.citadel_sha()
    lines.append(f"CITADEL_SHA={csha}")
    try:
        rt_root, rt_sha = rb.ensure_cymek_runtime()
        lines.append(f"CYMEK_RUNTIME_SHA={rt_sha}")
        lines.append(f"CYMEK_RUNTIME_PATH={rt_root}")
        runtime_ok = True
    except RuntimeError as exc:
        lines.append(f"CYMEK_RUNTIME=FAIL {exc}")
        rt_sha, runtime_ok = None, False
        ok = False

    from citadel_tpu import environment as env_mod

    lines.append(f"PLATFORM={env_mod._detect_platform()}")
    lines.append(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE', 'unset')}")
    versions = _versions()
    lines.append(f"torch={versions['torch']}")
    lines.append(f"torch_xla={versions['torch_xla']}")

    bad_mods = []
    for name in CITADEL_MODULES:
        passed, detail = _try_import(name)
        if not passed:
            bad_mods.append(f"{name} ({detail})")
    lines.append(f"citadel_tpu imports: {'PASS' if not bad_mods else 'FAIL ' + '; '.join(bad_mods)}")
    if bad_mods:
        ok = False

    if runtime_ok:
        missing_files = [rel for rel, present in rb.verify_files(rt_root) if not present]
        lines.append(f"Cymek runtime files: {'PASS' if not missing_files else 'FAIL ' + ', '.join(missing_files)}")
        if missing_files:
            ok = False
        bad_cymek = []
        for name in CYMEK_MODULES:
            passed, detail = _try_import(name)
            if not passed:
                bad_cymek.append(f"{name} ({detail})")
        lines.append(f"Cymek imports: {'PASS' if not bad_cymek else 'FAIL ' + '; '.join(bad_cymek)}")
        cymek_imports_ok = not bad_cymek
        if not cymek_imports_ok:
            ok = False
    else:
        lines.append("Cymek runtime files: SKIP (no runtime)")
        lines.append("Cymek imports: SKIP (no runtime)")
        cymek_imports_ok = False

    try:
        env = env_mod.probe(require_tpu=False)
        tpu_available = bool(env.get("tpu_present"))
        lines.append(f"TPU available: {'PASS' if tpu_available else 'FAIL'} "
                     f"(hw={env.get('accelerator_detected')}, n={env.get('xla_device_count')})")
    except Exception as exc:
        tpu_available = False
        lines.append(f"TPU available: FAIL ({type(exc).__name__})")
    if not tpu_available:
        ok = False

    api_status, api_missing = _xla_api_status()
    lines.append(f"XLA API compatibility: {api_status}"
                 + (f" (missing: {', '.join(api_missing)})" if api_missing else ""))
    if api_status == "FAIL":
        ok = False

    lines.append(f"READY_FOR_T0: {'YES' if ok else 'NO'}")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
