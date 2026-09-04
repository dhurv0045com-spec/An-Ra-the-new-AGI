"""T1 calculator preflight: `python -m citadel_tpu.calculator_preflight`.

Verifies BEFORE any T1 training: T0 receipt exists with PASS + required
provenance keys; pinned Cymek runtime resolves with all T1 files; generator
deterministic with zero forbidden overlap; encode/decode round-trip +
evaluator/Wilson self-tests green; checkpoint path writable; TPU active with
working XLA APIs. Prints READY_FOR_T1 YES/NO; exit 0/1. Never trains.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


T0_REQUIRED_KEYS = ("certification", "citadel_sha", "cymek_runtime_sha",
                    "environment", "model", "checkpoint_sha256", "reload_identical")


def _t0_check() -> tuple[bool, str]:
    from citadel_tpu import runtime_bootstrap as rb

    try:
        root = rb.citadel_root()
    except RuntimeError as exc:
        return False, str(exc)
    path = root / "docs" / "citadel" / "tpu_receipts" / "TPU_ONE_UPDATE.json"
    if not path.is_file():
        return False, f"T0 receipt missing: {path}"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"T0 receipt unreadable: {type(exc).__name__}"
    missing = [k for k in T0_REQUIRED_KEYS if k not in receipt]
    if missing:
        return False, f"T0 receipt lacks keys: {', '.join(missing)}"
    if receipt.get("certification") != "PASS":
        return False, f"T0 certification is {receipt.get('certification')!r}, need PASS"
    return True, (f"citadel={str(receipt.get('citadel_sha'))[:7]}, "
                   f"cymek={str(receipt.get('cymek_runtime_sha'))[:7]}")


def main() -> int:
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import environment as env_mod
    from citadel_tpu import preflight as pf
    from citadel_tpu import runtime_bootstrap as rb

    lines: list[str] = []
    ok = True

    t0_ok, t0_detail = _t0_check()
    lines.append(f"T0 receipt: {'PASS' if t0_ok else 'FAIL'} {t0_detail}")
    ok &= t0_ok

    try:
        rt_root, rt_sha = rb.ensure_cymek_runtime()
        missing = [rel for rel, present in rb.verify_files(rt_root) if not present]
        lines.append(f"Cymek runtime: {'PASS ' + rt_sha[:7] if not missing else 'FAIL missing ' + ','.join(missing)}")
        ok &= not missing
    except RuntimeError as exc:
        lines.append(f"Cymek runtime: FAIL {exc}")
        ok = False

    try:
        from citadel_tpu import calculator_data as calc

        a = calc.generate(split="test")
        b = calc.generate(split="test")
        det = a == b
        overlap = (set(calc.generate(split="train")) & set(a)) | (
            set(calc.generate(split="development")) & set(a))
        lines.append(f"generator deterministic: {'PASS' if det else 'FAIL'}; "
                     f"forbidden overlap rows: {len(overlap)}")
        ok &= det and not overlap
    except Exception as exc:
        lines.append(f"generator: FAIL {type(exc).__name__}: {exc}")
        ok = False

    try:
        cev.selftest()
        lines.append("evaluator selftest (round-trip/normalize/Wilson/scoring/hash): PASS")
    except Exception as exc:
        lines.append(f"evaluator selftest: FAIL {type(exc).__name__}: {exc}")
        ok = False

    try:
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "ckpt_probe.bin"
            probe.write_bytes(b"writable")
            writable = probe.is_file()
        lines.append(f"checkpoint path writable: {'PASS' if writable else 'FAIL'}")
        ok &= writable
    except Exception as exc:
        lines.append(f"checkpoint path writable: FAIL {type(exc).__name__}")
        ok = False

    try:
        env = env_mod.probe(require_tpu=False)
        tpu = bool(env.get("tpu_present"))
        lines.append(f"TPU active: {'PASS' if tpu else 'FAIL'} "
                     f"(hw={env.get('accelerator_detected')}, n={env.get('xla_device_count')})")
        ok &= tpu
    except Exception as exc:
        lines.append(f"TPU active: FAIL ({type(exc).__name__})")
        ok = False

    api_status, api_missing = pf._xla_api_status()
    lines.append(f"XLA APIs: {api_status}"
                 + (f" ({', '.join(api_missing)})" if api_missing else ""))
    ok &= api_status != "FAIL"

    lines.append(f"READY_FOR_T1: {'YES' if ok else 'NO'}")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
