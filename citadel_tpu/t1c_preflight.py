"""T1C session preflight: `python -m citadel_tpu.t1c_preflight`.

Verifies BEFORE the substantial session: T0 PASS + T1 context receipts present;
pinned runtime resolves with every T1C file; arithmetic generator deterministic;
all materialized DEV/TEST prompts fit the fixed greedy-generation buffer with
full answer headroom; evaluator selftests pass; MID spec validates against Cymek
contracts; disk >= 2 GB free; arm budgets reproduce the preregistered arithmetic;
TPU + XLA APIs are active. Prints READY_FOR_T1C YES/NO; exit 0/1. Never trains.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


NOMINAL_RATE = 8_700.0
SESSION_CEILING_S = 2 * 3_600
DISK_NEED_BYTES = 2_000_000_000


def _t1c_receipts_ok(root: Path) -> tuple[bool, str]:
    need = {"TPU_ONE_UPDATE.json": "PASS", "TPU_CALCULATOR_CHECKPOINT.json": None}
    notes = []
    for name, want in need.items():
        p = root / "docs" / "citadel" / "tpu_receipts" / name
        if not p.is_file():
            return False, f"missing {name}"
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"{name} unreadable: {type(exc).__name__}"
        if want is not None and r.get("certification") != want and r.get("status") != want:
            notes.append(f"{name} not {want}")
    if notes:
        return False, "; ".join(notes)
    return True, "T0 PASS + T1 context present"


def main() -> int:
    from citadel_tpu import arith_data as ad
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import environment as env_mod
    from citadel_tpu import preflight as pf
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1c_run as t1c

    lines: list[str] = []
    ok = True
    try:
        root = rb.citadel_root()
    except RuntimeError as exc:
        print(f"citadel root: FAIL {exc}\nREADY_FOR_T1C: NO")
        return 1

    t_ok, t_detail = _t1c_receipts_ok(root)
    lines.append(f"context receipts: {'PASS' if t_ok else 'FAIL'} {t_detail}")
    ok &= t_ok

    try:
        rt_root, rt_sha = rb.ensure_cymek_runtime()
        missing = [rel for rel, present in rb.verify_files(rt_root) if not present]
        lines.append(f"cymek runtime: {'PASS ' + rt_sha[:7] if not missing else 'FAIL ' + ','.join(missing)}")
        ok &= not missing
    except RuntimeError as exc:
        lines.append(f"cymek runtime: FAIL {exc}")
        ok = False

    try:
        seen_templates: set[str] = set()
        det = True
        for i in (0, 1, 7, 999, 123456):
            a = ad.row_at("train", i)
            b = ad.row_at("train", i)
            det &= a == b
            seen_templates.add(a[1]["template"])
            ad.parse_arith(a[0])
        for tpl_rows in (("dev", 5), ("test_template", 3)):
            t, _ = ad.row_at(*tpl_rows)
            ad.parse_arith(t)
        lines.append(f"generator deterministic+parses: {'PASS' if det else 'FAIL'}; "
                     f"train templates seen: {sorted(seen_templates)}")
        ok &= det
    except Exception as exc:
        lines.append(f"generator: FAIL {type(exc).__name__}: {exc}")
        ok = False

    # Full materialized evaluation-domain geometry gate.  This would have caught
    # the real Colab failure before any arm: a 25-token word prompt fit the L=32
    # training row but needed 8 more positions for greedy answer generation.
    try:
        cev.selftest()
        max_prompt = 0
        max_required = 0
        max_full = 0
        checked = 0
        for split in ("dev", "test_core", "test_template", "test_range", "test_composition"):
            rows = [ad.row_at(split, i)[0] for i in range(ad.SPLITS[split]["n"])]
            cap = cev.validate_generation_capacity(rows)
            max_prompt = max(max_prompt, cap["max_prompt_tokens"])
            max_required = max(max_required, cap["max_required_tokens"])
            max_full = max(max_full, max((len(cev.encode(r)) for r in rows), default=0))
            checked += len(rows)
        min_train_length = min(ln for _, ln in t1c.CALIBRATION_SHAPES)
        train_geom_ok = max_full <= min_train_length
        lines.append(
            f"eval geometry: {'PASS' if train_geom_ok else 'FAIL'} rows={checked} "
            f"max_prompt={max_prompt} required_with_headroom={max_required} "
            f"eval_L={cev.EVAL_LENGTH} max_full_row={max_full} train_Lmin={min_train_length}"
        )
        ok &= train_geom_ok and max_required <= cev.EVAL_LENGTH
    except Exception as exc:
        lines.append(f"eval geometry/evaluator: FAIL {type(exc).__name__}: {exc}")
        ok = False

    try:
        from v5_contracts.model_spec import ModelSpec

        spec = ModelSpec(**t1c.MID_SPEC_KWARGS)
        spec.assert_valid()
        total = spec.parameter_receipt().total
        match = total == t1c.MID_EXPECTED_PARAMS
        lines.append(f"MID spec validates: {'PASS' if match else 'FAIL'} total={total}")
        ok &= match
    except Exception as exc:
        lines.append(f"MID spec: FAIL {type(exc).__name__}: {str(exc)[:160]}")
        ok = False

    try:
        free = shutil.disk_usage(str(root)).free
        lines.append(f"disk free: {free // 1_000_000} MB "
                     f"({'PASS' if free >= DISK_NEED_BYTES else 'FAIL <2GB'})")
        ok &= free >= DISK_NEED_BYTES
    except Exception as exc:
        lines.append(f"disk: FAIL {type(exc).__name__}")
        ok = False

    try:
        assert set(t1c.ARMS) == {"A", "B", "C", "D"}
        totals, per_arm = 0, {}
        for tag, cfg in t1c.ARMS.items():
            for b, ln in t1c.CALIBRATION_SHAPES:
                used = cfg["budget"] // (b * ln) * (b * ln)
                assert 0 <= cfg["budget"] - used < b * ln, (tag, b, ln)
            per_arm[tag] = cfg["budget"]
            totals += cfg["budget"]
        lines.append(f"budgets: PASS {per_arm} total={totals} cap-tokens")
        est = totals / NOMINAL_RATE
        lines.append(f"session estimate @nominal: {est / 60:.0f} min "
                     f"({'PASS' if est < SESSION_CEILING_S else 'FAIL ceiling'})")
        ok &= est < SESSION_CEILING_S
    except Exception as exc:
        lines.append(f"budgets: FAIL {type(exc).__name__}: {exc}")
        ok = False

    try:
        import torch  # noqa: F401
        torch_state = "present"
    except Exception:
        torch_state = "absent (device imports deferred to session)"
    lines.append(f"torch: {torch_state}")

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

    lines.append(f"READY_FOR_T1C: {'YES' if ok else 'NO'}")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
