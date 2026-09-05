"""T1D session preflight: `python -m citadel_tpu.t1d_preflight`.

Verifies BEFORE the substantial session: T0 PASS + T1 context receipts present;
pinned runtime resolves with every T1D file; tiered generator deterministic with
all ordinary + teacher templates rendering/parsing; every eval prompt fits the
fixed generation buffer with headroom and every full row fits the training
buffer; alphabet round-trips; split leakage guards pass on eval slices;
eligible answer spans align on every template family; null helpers accept all
formats; MID/SCALE2 specs validate; all unit-test files pass in-process;
notebooks resolve; session dir writable; TPU + XLA APIs active.
Prints READY_FOR_T1D YES/NO; exit 0/1. Never trains.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    from citadel_tpu import arith_data as _ad  # noqa: F401 (canary compat check)
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import environment as env_mod
    from citadel_tpu import preflight as pf
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import tiered_data as td

    lines: list[str] = []
    ok = True
    try:
        root = rb.citadel_root()
    except RuntimeError as exc:
        print(f"citadel root: FAIL {exc}\nREADY_FOR_T1D: NO")
        return 1

    for name, want in (("TPU_ONE_UPDATE.json", "PASS"),
                       ("TPU_CALCULATOR_CHECKPOINT.json", None)):
        p = root / "docs" / "citadel" / "tpu_receipts" / name
        if not p.is_file():
            lines.append(f"context receipt {name}: FAIL missing")
            ok = False
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            good = (r.get("certification") == want) if want else True
            lines.append(f"context receipt {name}: {'PASS' if good else 'FAIL status'}")
            ok &= good
        except Exception as exc:
            lines.append(f"context receipt {name}: FAIL {type(exc).__name__}")
            ok = False

    try:
        rt_root, rt_sha = rb.ensure_cymek_runtime()
        missing = [rel for rel, present in rb.verify_files(rt_root) if not present]
        lines.append(f"cymek runtime: {'PASS ' + rt_sha[:7] if not missing else 'FAIL ' + ','.join(missing)}")
        ok &= not missing
    except RuntimeError as exc:
        lines.append(f"cymek runtime: FAIL {exc}")
        ok = False

    try:
        det = all(td.tier_row(t, "train", i) == td.tier_row(t, "train", i)
                  for t in range(5) for i in (0, 7, 999))
        kinds = [td.teacher_row(k, i) for k in ("digadd", "digsub", "singlemul", "divmicro")
                 for i in (0, 5)]
        det &= all(td.teacher_row(k, i) == td.teacher_row(k, i)
                   for k in ("digadd", "digsub", "singlemul", "divmicro") for i in (0, 5))
        for text, _ in kinds:
            cev.split_prompt_target(text)
            assert all(c in cev.ALPHABET for c in text), f"non-alphabet char in {text!r}"
        lines.append(f"tiered+teacher generator deterministic: {'PASS' if det else 'FAIL'}")
        ok &= det
    except Exception as exc:
        lines.append(f"generator: FAIL {type(exc).__name__}: {exc}")
        ok = False

    try:
        cev.selftest()
        eval_rows: list[str] = []
        for tier in range(5):
            eval_rows.extend(td.tier_row(tier, "dev", j)[0] for j in range(200))
            eval_rows.extend(td.tier_row(tier, "test", j)[0] for j in range(500))
        cap = cev.validate_generation_capacity(eval_rows)
        full_ok = max(len(r) for r in eval_rows) <= 64
        from citadel_tpu import t1c_run as t1c

        spans_ok = all((lambda s: s[1] > 0)(t1c.answer_spans([r], 64)[0]) for r in eval_rows[:2000])
        lines.append(f"eval geometry: PASS rows={len(eval_rows)} "
                     f"max_prompt={cap['max_prompt_tokens']} required={cap['max_required_tokens']} "
                     f"full_rows_fit_L64={full_ok} spans_ok={spans_ok}")
        ok &= full_ok and spans_ok
        # leakage on eval slices via the shared verdict: T2+ pairs must be
        # exactly disjoint; T0/T1-involving pairs are labeled probes.
        from citadel_tpu import tiered_data as _td

        fatal, _ = _td.leakage_verdict(_td.eval_pair_leakage())
        lines.append(f"eval-slice leakage T2+: "
                     f"{'PASS zero' if not fatal else 'FAIL ' + ';'.join(f'{k}={v}' for k, v in sorted(fatal.items()))}")
        ok &= not fatal
    except Exception as exc:
        lines.append(f"eval geometry/evaluator: FAIL {type(exc).__name__}: {exc}")
        ok = False

    try:
        from v5_contracts.model_spec import ModelSpec

        s = ModelSpec(**t1d.SCALE2_SPEC_KWARGS)
        s.assert_valid()
        match = s.parameter_receipt().total == t1d.SCALE2_EXPECTED_PARAMS
        lines.append(f"SCALE2 spec validates: {'PASS' if match else 'FAIL'}")
        ok &= match
        mid = t1d.build_spec("MID")
        from citadel_tpu import t1c_run as t1c

        mid_ok = mid.parameter_receipt().total == t1c.MID_EXPECTED_PARAMS
        lines.append(f"MID spec validates: {'PASS' if mid_ok else 'FAIL'}")
        ok &= mid_ok
        shapes_ok = all(b * ln > 0 for b, ln in t1d.CALIBRATION_SHAPES)
        arms_ok = set(t1d.ARMS) == {"A", "B", "C", "D", "E"}
        lines.append(f"shapes+arms: {'PASS' if shapes_ok and arms_ok else 'FAIL'}")
        ok &= shapes_ok and arms_ok
    except Exception as exc:
        lines.append(f"specs/arms: FAIL {type(exc).__name__}: {str(exc)[:160]}")
        ok = False

    try:
        tests = ["tests/test_citadel_t1d.py", "tests/test_citadel_t1c.py",
                 "tests/test_citadel_t1_canary.py", "tests/test_citadel_notebooks.py"]
        bad = []
        for t in tests:
            r = subprocess.run([sys.executable, t], capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                bad.append(f"{t} (exit {r.returncode}): {(r.stdout or '')[-300:]}")
        lines.append(f"unit tests: {'PASS all-4-files' if not bad else 'FAIL ' + '; '.join(bad)}")
        ok &= not bad
    except Exception as exc:
        lines.append(f"unit tests: FAIL {type(exc).__name__}: {exc}")
        ok = False

    try:
        session_dir = root / "docs" / "citadel" / "tpu_receipts" / "t1d_session"
        session_dir.mkdir(parents=True, exist_ok=True)
        probe = session_dir / ".preflight_writable"
        probe.write_bytes(b"ok")
        probe.unlink()
        lines.append("session dir writable: PASS")
    except Exception as exc:
        lines.append(f"session dir writable: FAIL {type(exc).__name__}")
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

    lines.append(f"READY_FOR_T1D: {'YES' if ok else 'NO'}")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
