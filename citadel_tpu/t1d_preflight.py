"""T1D session preflight: `python -m citadel_tpu.t1d_preflight`.

Structured API (the one-shot orchestrator consumes it):

    run_preflight() -> {"status": PASS|FAIL, "gates": [...],
                        "blocking_gates": [...], "environment": {...},
                        "citadel_sha": ..., "cymek_sha": ...}

LIVE preflight (runs on the TPU host): only gates that need the LIVE
environment plus a compact set of high-value pure contract checks. The full
deterministic repository suite is DEVELOPMENT certification
(DEVELOPMENT_CERTIFICATION.json) and runs before handoff; the live gate
verifies the runtime SHA matches that certificate — code newer than
certification fails closed. Prints READY_FOR_T1D YES/NO; exit 0/1.
Never trains.
"""

from __future__ import annotations

import json
import sys
import time


def run_preflight() -> dict:
    """Structured live preflight. Every gate is recorded; failures never
    raise past this function — the caller decides (the orchestrator writes
    the failure bundle and stops before training)."""
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import environment as env_mod
    from citadel_tpu import preflight as pf
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import tiered_data as td

    gates: list[dict] = []

    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"name": name, "status": "PASS" if passed else "FAIL",
                      "detail": detail})

    citadel_sha = None
    cymek_sha = None
    try:
        root = rb.citadel_root()
        citadel_sha = rb.citadel_sha()
    except RuntimeError as exc:
        return {"schema": "citadel-t1d-preflight/v1", "status": "FAIL",
                "gates": [{"name": "citadel_root", "status": "FAIL",
                           "detail": str(exc)}],
                "blocking_gates": ["citadel_root"], "environment": {},
                "citadel_sha": None, "cymek_sha": None, "plan_sha": None,
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())}

    # 1. context receipts (T0/T1 lineage)
    try:
        ok, detail = True, []
        for name, want in (("TPU_ONE_UPDATE.json", "PASS"),
                           ("TPU_CALCULATOR_CHECKPOINT.json", None)):
            p = root / "docs" / "citadel" / "tpu_receipts" / name
            if not p.is_file():
                ok = False
                detail.append(f"{name} missing")
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            if want is not None and r.get("certification") != want:
                ok = False
                detail.append(f"{name} status {r.get('certification')!r}")
        gate("context_receipts", ok, "; ".join(detail) or "T0+checkpoint present")
    except Exception as exc:
        gate("context_receipts", False, f"{type(exc).__name__}: {exc}")

    # 2. development certification identity (fail-closed on code drift)
    try:
        cert_path = root / "docs" / "citadel" / "experiments" / "T1D" / \
            "DEVELOPMENT_CERTIFICATION.json"
        if not cert_path.is_file():
            gate("development_certification", False,
                 "DEVELOPMENT_CERTIFICATION.json missing")
        else:
            cert = json.loads(cert_path.read_text(encoding="utf-8"))
            from citadel_tpu.t1d_one_shot import code_sha

            runtime_code = code_sha()
            certified_code = cert.get("code_sha", cert.get("citadel_sha"))
            if certified_code != runtime_code:
                gate("development_certification", False,
                     f"certified {str(certified_code)[:12]} != executable code "
                     f"{runtime_code[:12]}; regenerate certification")
            elif cert.get("status") != "PASS":
                gate("development_certification", False,
                     "certification status is not PASS")
            else:
                gate("development_certification", True,
                     f"sha {str(cert.get('citadel_sha'))[:12]} tests "
                     f"{cert.get('files_passed')}/{cert.get('files_total')}")
    except Exception as exc:
        gate("development_certification", False, f"{type(exc).__name__}: {exc}")

    # 3. pinned Cymek runtime resolves with every T1D file
    try:
        rt_root, rt_sha = rb.ensure_cymek_runtime()
        missing = [rel for rel, present in rb.verify_files(rt_root) if not present]
        gate("cymek_runtime", not missing,
             f"{rt_sha[:12]}" + (f" missing {','.join(missing)}" if missing else ""))
        cymek_sha = rt_sha
    except RuntimeError as exc:
        gate("cymek_runtime", False, str(exc))

    # 4. tiered + teacher + self generators deterministic and alphabet-safe
    try:
        from citadel_tpu import self_knowledge as sk

        det = all(td.tier_row(t, "train", i) == td.tier_row(t, "train", i)
                  for t in range(5) for i in (0, 7, 999))
        kinds = [td.teacher_row(k, i) for k in ("digadd", "digsub", "singlemul",
                                                "divmicro") for i in (0, 5)]
        det &= all(td.teacher_row(k, i) == td.teacher_row(k, i)
                   for k in ("digadd", "digsub", "singlemul", "divmicro")
                   for i in (0, 5))
        self_det = all(sk.self_row(i) == sk.self_row(i)
                       for i in (0, 1, 199, 11_999))
        for text, _ in kinds:
            cev.split_prompt_target(text)
            assert all(c in cev.ALPHABET for c in text), f"non-alphabet: {text!r}"
        probe_rows, _targets, _meta = sk.self_probe_rows()
        train_rows = [sk.self_row(i)[0] for i in range(0, sk.SELF_TRAIN_N, 37)]
        for text in probe_rows + train_rows:
            cev.split_prompt_target(text)
            assert all(c in cev.ALPHABET for c in text), f"non-alphabet: {text!r}"
        gate("generators_deterministic", det and self_det,
             f"tiered+teacher+self ({sk.GENERATOR_VERSION}) deterministic; "
             f"{len(probe_rows)} probe rows parse")
    except Exception as exc:
        gate("generators_deterministic", False, f"{type(exc).__name__}: {exc}")

    # 5. eval geometry + evaluator selftest + leakage
    try:
        cev.selftest()
        eval_rows: list[str] = []
        for tier in range(5):
            eval_rows.extend(td.tier_row(tier, "dev", j)[0] for j in range(200))
            eval_rows.extend(td.tier_row(tier, "test", j)[0] for j in range(500))
        cap = cev.validate_generation_capacity(eval_rows)
        full_ok = max(len(r) for r in eval_rows) <= 64
        from citadel_tpu import t1c_run as t1c

        spans_ok = all((lambda s: s[1] > 0)(t1c.answer_spans([r], 64)[0])
                       for r in eval_rows[:2000])
        fatal, _ = td.leakage_verdict(td.eval_pair_leakage())
        gate("eval_geometry", full_ok and spans_ok and not fatal,
             f"rows={len(eval_rows)} required={cap['max_required_tokens']} "
             f"spans_ok={spans_ok} leakage_t2+={'zero' if not fatal else fatal}")
    except Exception as exc:
        gate("eval_geometry", False, f"{type(exc).__name__}: {exc}")

    # 6. specs validate (MID + SCALE2) + arms/shape sanity (includes F)
    try:
        from v5_contracts.model_spec import ModelSpec

        s = ModelSpec(**t1d.SCALE2_SPEC_KWARGS)
        s.assert_valid()
        match = s.parameter_receipt().total == t1d.SCALE2_EXPECTED_PARAMS
        mid = t1d.build_spec("MID")
        from citadel_tpu import t1c_run as t1c

        mid_ok = mid.parameter_receipt().total == t1c.MID_EXPECTED_PARAMS
        arms_ok = set(t1d.ARMS) == {"A", "B", "C", "D", "E", "F"}
        shapes_ok = all(b * ln > 0 for b, ln in t1d.CALIBRATION_SHAPES)
        gate("specs_arms", match and mid_ok and arms_ok and shapes_ok,
             f"MID={mid_ok} SCALE2={match} arms={sorted(t1d.ARMS)}")
    except Exception as exc:
        gate("specs_arms", False, f"{type(exc).__name__}: {str(exc)[:160]}")

    # 7. the REAL producer->finalizer schema bridge (legacy shape included)
    try:
        defects = t1d.producer_consumer_contract_probe(legacy_untrained_keys=True)
        gate("producer_finalizer_schema", not defects,
             "PASS" if not defects else "; ".join(defects[:3]))
    except Exception as exc:
        gate("producer_finalizer_schema", False, f"{type(exc).__name__}: {exc}")

    # 8. PRE50M wiring: bundle inventory + fail-closed decision schema
    try:
        from citadel_tpu import pre50m as _p50

        required_bundle = {"SESSION_MANIFEST.json", "DATA_MANIFEST.json",
                           "CALIBRATION.json"} | \
            {f"ARM_{t}.json" for t in t1d.ARM_ORDER} | \
            {"LIFT_OFF_CURVES.json", "CROSS_ARM_SUMMARY.json",
             "PRE50M_TARGET.json", "PRE50M_FEASIBILITY.json",
             "PRE50M_THROUGHPUT.json", "PRE50M_CHECKPOINT_SMOKE.json",
             "PRE50M_DATA_INTERFACE.json", "PRE50M_PACKING.json",
             "DIAGNOSTICS.json", "NEXT_50M_DECISION.json"}
        bundle_ok = required_bundle <= set(t1d.BUNDLE_FILES)
        green_smoke = {"status": "PASS", "reload_output_identity": True,
                       "optimizer_resume": {"moments_preserved": True,
                                            "continued_update_ok": True},
                       "grad_norm": {"max": 1.0}, "losses": [9.0, 8.0],
                       "param_mutation": True, "production_transaction": True,
                       "checkpoint_compat": {"compatible": True},
                       "writer_fence_probe": "rejected-as-required",
                       "token_accounting": {"consistent": True}}
        green_data = {"status": "PASS", "capacity_tokens": 4096,
                      "real_tokens": 4000, "loss_bearing_tokens": 900,
                      "padding_tokens": 96, "scheduled_rows": 64}
        probe_decision = _p50.build_decision(
            target={"understood": True, "type": _p50.PRE50M_TARGET["type"],
                    "value_tokens": _p50.PRE50M_TARGET["value_tokens"],
                    "parameter_count": None},
            smoke=green_smoke, feasibility={"verdict": "FIT"},
            data_interface=green_data, packing={"status": "PASS"},
            recommended_batch=256, recommended_sequence_length=64,
            rate_tok_s=8000.0)
        ready_ok = (probe_decision["ready_for_50m_training"] is True
                    and probe_decision["blocking_reasons"] == [])
        smoke_ok = (_p50.SMOKE_SPEC == "SCALE2" and _p50.SMOKE_UPDATES >= 3
                    and _p50.PRE50M_TARGET["value_tokens"] == 50_000_000)
        gate("pre50m_wiring", bundle_ok and ready_ok and smoke_ok,
             f"bundle_ok={bundle_ok} green_probe_ready={ready_ok}")
    except Exception as exc:
        gate("pre50m_wiring", False, f"{type(exc).__name__}: {str(exc)[:160]}")

    # 9. session dir writable
    try:
        session_dir = root / "docs" / "citadel" / "tpu_receipts" / "t1d_session"
        session_dir.mkdir(parents=True, exist_ok=True)
        probe = session_dir / ".preflight_writable"
        probe.write_bytes(b"ok")
        probe.unlink()
        gate("session_dir_writable", True, "ok")
    except Exception as exc:
        gate("session_dir_writable", False, f"{type(exc).__name__}: {exc}")

    # 10. TPU + XLA APIs (LIVE environment gates)
    env: dict = {}
    try:
        env = env_mod.probe(require_tpu=False)
        tpu = bool(env.get("tpu_present"))
        gate("tpu_active", tpu,
             f"hw={env.get('accelerator_detected')} "
             f"n={env.get('xla_device_count')}")
    except Exception as exc:
        gate("tpu_active", False, f"{type(exc).__name__}: {exc}")
    try:
        api_status, api_missing = pf._xla_api_status()
        gate("xla_apis", api_status in ("OK", "PARTIAL"),
             api_status + (f" ({', '.join(api_missing)})" if api_missing else ""))
    except Exception as exc:
        gate("xla_apis", False, f"{type(exc).__name__}: {exc}")

    blocking = [g["name"] for g in gates if g["status"] != "PASS"]
    return {"schema": "citadel-t1d-preflight/v1",
            "status": "PASS" if not blocking else "FAIL",
            "gates": gates, "blocking_gates": blocking,
            "environment": env,
            "citadel_sha": citadel_sha, "cymek_sha": cymek_sha,
            "plan_sha": t1d.plan_identity(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main() -> int:
    pre = run_preflight()
    for g in pre["gates"]:
        print(f"{g['name']}: {g['status']} {g['detail']}")
    if pre["blocking_gates"]:
        print("BLOCKING: " + ", ".join(pre["blocking_gates"]))
    print(f"READY_FOR_T1D: {'YES' if pre['status'] == 'PASS' else 'NO'}")
    return 0 if pre["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "run_preflight"]
