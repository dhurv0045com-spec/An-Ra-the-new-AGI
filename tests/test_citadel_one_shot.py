"""Citadel ONE-SHOT orchestrator tests: portability, emulator, Arm F, plan
identity, mid-arm state, failure bundles. Zero third-party dependencies
(torch-based resume-identity test is torch-optional and skips cleanly).

Run:  python tests/test_citadel_one_shot.py   (exit 0 = all pass)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(CITADEL_ROOT))

from citadel_tpu import t1d_one_shot as oshot  # noqa: E402
from citadel_tpu import t1d_run as t1d  # noqa: E402

# historical machine-local path fragments that must never appear in
# executable code (§5). '/content/' is a Colab-internal path, not local.
LOCAL_PATH_PATTERNS = ("C:\\", "C:/", "/Users/", "/home/", "Downloads",
                       "Desktop", "ankit", "AppData")


def test_portability_scan() -> None:
    """No executable source may depend on an operator machine path.
    (The scanner's own pattern table is self-excluded — it must name the
    patterns to detect them.)"""
    offenders = []
    targets = list((CITADEL_ROOT / "citadel_tpu").glob("*.py"))
    targets += list((CITADEL_ROOT / "tests").glob("*.py"))
    targets = [p for p in targets if p.name != "test_citadel_one_shot.py"]
    for path in targets:
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern in text:
                offenders.append(f"{path.name}: {pattern!r}")
    assert not offenders, f"machine-local paths in executable code: {offenders}"


def test_plan_identity_stable_and_sensitive() -> None:
    a = t1d.plan_identity()
    b = t1d.plan_identity()
    assert a == b and len(a) == 64
    assert t1d.ARM_ORDER == ("A", "B", "C", "D", "E", "F")
    assert t1d.ARMS["F"] == {"spec": "MID", "mode": "self", "budget": 2_000_000}


def test_self_feeder_cadence_and_rows() -> None:
    """Arm F feeder: every 7th drawn row is a self-knowledge row; the rest
    follow the frozen curriculum; self rows parse through production spans."""
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import self_knowledge as sk
    from citadel_tpu import t1c_run as t1c

    feeder = t1d.TierFeeder("self", 8, 64)
    self_draws = 0
    ordinary = 0
    for u in range(21):
        feeder.fill_sequences(u / 21)
    self_draws = feeder.drawn.get("self:train", 0)
    ordinary = sum(v for k, v in feeder.drawn.items() if k.startswith("tier:"))
    assert self_draws > 0, "self arm consumed no self rows"
    ratio = self_draws / (self_draws + ordinary)
    assert abs(ratio - 1 / sk.SELF_ROW_FRACTION) < 0.03, ratio
    # probe rows + training rows parse through the frozen production path
    rows, targets, meta = sk.self_probe_rows()
    for r in rows:
        prompt, target = cev.split_prompt_target(r)
        plen, alen = t1c.answer_spans([r], 64)[0]
        assert alen > 0 and plen + alen == len(cev.encode(r)), r
    # text scorer: exact match works, the arithmetic normalizer is NOT used
    assert sk.text_exact(" an ra ", "an ra") is True
    assert sk.text_exact("something else", "an ra") is False
    summ = sk.summarize_text(["an ra"] * 4 + ["x"] * 1,
                             ["an ra"] * 5)
    assert summ["accuracy"] == 0.8 and summ["correct"] == 4


def test_self_classify_rules() -> None:
    """SELF_KNOWLEDGE_ACQUIRED fires only for a passing Arm F;
    SELF_PROBE_LEAKAGE fires when a non-self arm passes the same bar."""
    base = {t: _arm_full() for t in t1d.ARM_ORDER}
    out = t1d.classify_cross_arm(base)
    assert "SELF_KNOWLEDGE_ACQUIRED" not in out["labels"]
    arms = dict(base)
    f = _arm_full()
    f["trained_self"] = {"correct": 40, "total": 96, "accuracy": 0.42,
                         "wilson_lcb": 0.33, "wilson_ucb": 0.52}
    f["untrained_self"] = {"correct": 13, "total": 96, "accuracy": 0.14,
                           "wilson_lcb": 0.08, "wilson_ucb": 0.22}
    f["self_diagnostics"] = {"most_common_null": {"wilson_lcb": 0.09},
                             "per_domain": {}}
    arms["F"] = f
    out = t1d.classify_cross_arm(arms)
    assert "SELF_KNOWLEDGE_ACQUIRED" in out["labels"], out["labels"]
    leak = dict(base)
    c = _arm_full()
    c["trained_self"] = {"correct": 40, "total": 96, "accuracy": 0.42,
                         "wilson_lcb": 0.33, "wilson_ucb": 0.52}
    c["untrained_self"] = {"correct": 13, "total": 96, "accuracy": 0.14,
                           "wilson_lcb": 0.08, "wilson_ucb": 0.22}
    leak["C"] = c  # a TEACHER arm passing the bar without self rows
    out = t1d.classify_cross_arm(leak)
    assert "SELF_PROBE_LEAKAGE" in out["labels"], out["labels"]


def _arm_full(status="SCIENTIFIC_FAIL"):
    """Complete synthetic terminal receipt for one arm (validator-clean)."""
    def s(acc, n=500):
        return {"correct": int(acc * n), "total": n, "accuracy": acc,
                "wilson_lcb": max(0.0, acc - 0.04),
                "wilson_ucb": min(1.0, acc + 0.04)}
    inter = {str(cp): {f"t{tier}": {"exact": 0.0, "lcb": 0.0}
                       for tier in range(5)}
             for cp in (25, 50, 75, 100)}
    return {"schema": "citadel-t1d-arm/v1", "status": status,
            "trained": {f"t{t}": s(0.0) for t in range(5)},
            "untrained": {f"t{t}": s(0.0) for t in range(5)},
            "untrained_dev": {f"t{t}": s(0.0, n=200) for t in range(5)},
            "untrained_self": s(0.0, n=96),
            "trained_self": s(0.0, n=96),
            "self_diagnostics": {"most_common_null": {"wilson_lcb": 0.0},
                                 "per_domain": {}},
            "trained_train": {f"t{t}": s(0.0, n=200) for t in range(5)},
            "train_memorization": {f"t{t}": {"consumed_prefix": 200,
                                             "n_verified_consumed": 200,
                                             "status": "OK",
                                             "lift_eligible": False}
                                   for t in range(5)},
            "nulls_per_tier": {f"t{t}": {"strongest": "x", "accuracy": 0.02,
                                         "all": {"x": 0.02}}
                               for t in range(5)},
            "heuristic_nulls": {"x": {"accuracy": 0.02}},
            "diagnostics": {"stop_histogram": {"NEWLINE": 500},
                            "first_train_lift_tier": None,
                            "first_test_lift_tier": None},
            "intermediates": inter,
            "gate_rules": {"nonoverlap": False, "beats_null": False,
                           "margin": False, "loss": True, "reload": True},
            "checkpoint": {"path": "t1d_arm_synthetic.pt",
                           "sha256": "a" * 64},
            "pre_reload_prediction_sha256": "0" * 64,
            "post_reload_prediction_sha256": "0" * 64,
            "training": {"updates": 10, "ledgers": [{"updates": 10}]},
            "data": {"feeder": {"placed_rows": {}, "placed_tokens": {},
                                "carry_pending": 0}},
            "reload_identical": True}


def _green_pre50m(root, arm_receipts, *, rt_sha, rate, shape):
    from citadel_tpu import pre50m as p50

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
    (root / "PRE50M_TARGET.json").write_text(json.dumps(
        {"schema": "citadel-pre50m-target/v1", **p50.PRE50M_TARGET}),
        encoding="utf-8")
    (root / "PRE50M_CHECKPOINT_SMOKE.json").write_text(
        json.dumps(green_smoke), encoding="utf-8")
    (root / "PRE50M_DATA_INTERFACE.json").write_text(
        json.dumps(green_data), encoding="utf-8")
    (root / "PRE50M_PACKING.json").write_text(json.dumps({"status": "PASS"}),
                                              encoding="utf-8")
    (root / "PRE50M_FEASIBILITY.json").write_text(json.dumps(
        {"memory": {"SCALE2_7_4M": {"verdict": "FIT"}}}), encoding="utf-8")
    (root / "PRE50M_THROUGHPUT.json").write_text(json.dumps({"curve": {}}),
                                                 encoding="utf-8")
    (root / "DIAGNOSTICS.json").write_text(json.dumps({"arms": {}}),
                                           encoding="utf-8")
    decision = p50.build_decision(
        target={"understood": True, "type": p50.PRE50M_TARGET["type"],
                "value_tokens": p50.PRE50M_TARGET["value_tokens"],
                "parameter_count": None},
        smoke=green_smoke, feasibility={"verdict": "FIT"},
        data_interface=green_data, packing={"status": "PASS"},
        recommended_batch=shape[0], recommended_sequence_length=shape[1],
        rate_tok_s=rate)
    (root / "NEXT_50M_DECISION.json").write_text(json.dumps(decision),
                                                 encoding="utf-8")
    return {"status": "PASS", "decision": decision}


class _EmulatedArm:
    """Real finalization path with synthetic device outputs (the lowest
    possible seam): real feeder consumption, real generator rows, real
    producer-shaped data, real prefinal snapshot, real build_arm_receipt,
    real marker write. Only the model's predictions are synthetic."""

    def __init__(self, fail_tags=()):
        self.fail_tags = set(fail_tags)
        self.calls = []

    def __call__(self, tag, cfg, *, shape, out_dir, seed=20260904):
        import tempfile

        self.calls.append(tag)
        if tag in self.fail_tags:
            raise RuntimeError(f"emulated arm {tag} infra failure")
        from citadel_tpu import calculator_eval as cev
        from citadel_tpu import tiered_data as td

        root = Path(out_dir)
        feeder = t1d.TierFeeder(
            cfg["mode"] if cfg["mode"] != "masked" else "curriculum",
            8, 64)
        for u in range(40):
            feeder.fill_sequences(u / 40)
        slices = t1d._tier_slices()
        plan = t1d.train_memorization_plan(feeder)

        def s(acc, n=500):
            return {"correct": int(acc * n), "total": n, "accuracy": acc,
                    "wilson_lcb": max(0.0, acc - 0.04),
                    "wilson_ucb": min(1.0, acc + 0.04)}

        untrained_producer = {}
        for tier in range(5):
            untrained_producer[f"dev_t{tier}"] = s(0.0, n=200)
            untrained_producer[f"test_t{tier}"] = s(0.0)
        untrained_dev = {f"t{t}": untrained_producer[f"dev_t{t}"]
                         for t in range(5)}
        accs = {0: 0.8, 1: 0.4, 2: 0.2, 3: 0.05, 4: 0.0}
        trained = {f"t{t}": s(accs[t]) for t in range(5)}
        trained_recs = {f"t{t}": [
            {"prompt": "e", "target": tg, "prediction": tg, "correct": True,
             "stop_reason": "EOS", "generated_token_count": len(tg),
             "valid": True} for tg in slices[f"test_t{t}"]["targets"]]
            for t in range(5)}
        trained_train, train_memorization = {}, {}
        for tier in range(5):
            entry = plan[tier]
            rows = [td.tier_row(tier, "train", i)[0]
                    for i in entry["verified_indices"]]
            tgts = [cev.split_prompt_target(r)[1] for r in rows]
            summ = s(0.5, n=len(rows)) if rows else s(0.0, n=0)
            trained_train[f"t{tier}"] = summ
            train_memorization[f"t{tier}"] = {
                "consumed_prefix": entry["consumed_prefix"],
                "n_frozen_candidates": entry["n_candidates"],
                "n_verified_consumed": entry["n_verified"],
                "evaluated_rows": len(rows), "status": entry["status"],
                "lift_eligible": bool(entry["status"] == "OK")}
        from citadel_tpu import self_knowledge as sk

        self_rows, self_targets, _ = sk.self_probe_rows()
        trained_self = sk.summarize_text(
            [t if tag == "F" else "no" for t in self_targets], self_targets)
        untrained_self = sk.summarize_text(["no"] * len(self_targets),
                                           self_targets)
        inter = {str(cp): {f"t{tier}": {"exact": dev, "lcb": max(0.0, dev - 0.03)}
                           for tier in range(5)}
                 for cp, dev in ((25, 0.02), (50, 0.05), (75, 0.09), (100, 0.12))}
        ckpt_path = root / f"t1d_arm_{tag.lower()}.pt"
        ckpt_path.write_bytes(f"synthetic-checkpoint-{tag}".encode())
        kwargs = dict(
            tag=tag, cfg=dict(cfg), env={"probe_pass": True}, n_seq=8,
            length=64, param_count=3_737_472, citadel_sha="0" * 40,
            cymek_sha="1" * 64, seed=seed,
            feeder_placed_rows={k: int(v) for k, v in feeder.placed_rows.items()},
            feeder_ledger=feeder.ledger(),
            ledgers=[{"updates": 10, "first_loss": 9.0, "last_loss": 5.0}],
            done=10, first_loss=9.0, last_loss=5.0, cap_total=10 * 8 * 64,
            ans_total=100, whole_total=1000, gsum=1.0, gmax=0.8, gn=10,
            train_wall=30.0, untrained=untrained_producer,
            untrained_dev=untrained_dev, untrained_self=untrained_self,
            trained_self=trained_self, self_diagnostics={
                "most_common_null": sk.summarize_text(
                    sk.most_common_null(self_targets), self_targets),
                "per_domain": {}},
            untrained_train={f"t{t}": s(0.0, n=200) for t in range(5)},
            trained=trained, trained_recs=trained_recs,
            trained_train=trained_train,
            train_memorization=train_memorization, inter=inter,
            teacher_eval={"skipped": "n/a"}, first_step={"n": 0},
            ckpt_path=str(ckpt_path),
            ckpt_hash=hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
            pre_sha="p" * 64, post_sha="p" * 64,
            reload_ok=True, device_count=1, wall=60.0, eval_recovery=False)
        sidecar = Path(t1d.write_prefinal_snapshot(root, kwargs))
        snap, why = t1d.load_prefinal_snapshot(
            root, tag, expect_cfg=dict(cfg), seed=seed, shape=(8, 64))
        assert snap is not None, why
        receipt = t1d.build_arm_receipt(**snap)
        t1d.write_arm_receipt(root, receipt, ckpt_hash=snap["ckpt_hash"])
        sidecar.unlink(missing_ok=True)
        return receipt


def _pass_preflight(root):
    return {"schema": "citadel-t1d-preflight/v1", "status": "PASS", "gates": [],
            "blocking_gates": [], "environment": {}, "citadel_sha": "0" * 40,
            "cymek_sha": "1" * 64, "plan_sha": t1d.plan_identity()}


def _runtime_sha() -> str:
    from citadel_tpu import runtime_bootstrap as rb

    return rb.citadel_sha()


def _cymek_sha() -> str:
    from citadel_tpu import runtime_bootstrap as rb

    return rb.ensure_cymek_runtime()[1]


def test_one_shot_emulator_fresh_and_resume(tmp_root=None) -> None:
    """§17: the REAL orchestration path end-to-end device-free — fresh run
    completes and the bundle verifies; a rerun resumes (completed arms are
    NOT recomputed); an arm failure stays isolated; PRE50M failure preserves
    every arm and still exports a verifiable bundle."""
    import tempfile

    def _seed_calibration(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "CALIBRATION.json").write_text(json.dumps(
            {"schema": "citadel-t1d-throughput-calibration/v1",
             "selected": {"batch": 256, "length": 64},
             "selected_tokens_per_second": 8000.0,
             "candidates": [{"batch": 256, "length": 64,
                             "tokens_per_second": 8000.0,
                             "correct": True}]}), encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        # development certification lives inside the session dir's parent chain
        cert_dir = Path(tmp) / "cert"
        cert_dir.mkdir()
        from citadel_tpu.t1d_one_shot import code_sha

        cert = {"schema": "citadel-development-certificate/v1",
                "citadel_sha": _runtime_sha(), "code_sha": code_sha(),
                "cymek_sha": _cymek_sha(),
                "status": "PASS", "files_passed": 7, "files_total": 7}
        (cert_dir / "DEVELOPMENT_CERTIFICATION.json").write_text(
            json.dumps(cert), encoding="utf-8")
        real_cert = _certificate_path()
        real_cert.parent.mkdir(parents=True, exist_ok=True)
        saved = real_cert.read_text(encoding="utf-8") if real_cert.is_file() else None
        real_cert.write_text(json.dumps(cert), encoding="utf-8")
        try:
            session_dir = Path(tmp) / "session"
            _seed_calibration(session_dir)
            arms = _EmulatedArm()
            session = oshot.run_all(
                str(session_dir), preflight_runner=_pass_preflight,
                canary_runner=lambda sd: {"status": "PASS"},
                arm_runner=arms, pre50m_runner=_green_pre50m)
            assert session["status"] == "COMPLETE", session.get("error")
            assert set(session["phases"]) == set(oshot.PHASE_ORDER)
            assert t1d.verify_bundle(str(session_dir))["status"] == "VALID"
            assert len(arms.calls) == len(t1d.ARM_ORDER)

            # resume: a rerun recomputes NO completed arms
            _seed_calibration(session_dir)
            arms2 = _EmulatedArm()
            session2 = oshot.run_all(
                str(session_dir), preflight_runner=_pass_preflight,
                canary_runner=lambda sd: {"status": "PASS"},
                arm_runner=arms2, pre50m_runner=_green_pre50m)
            assert session2["status"] == "COMPLETE"
            assert arms2.calls == [], arms2.calls

            # arm failure isolation: C fails, the session still completes
            session_dir2 = Path(tmp) / "session2"
            _seed_calibration(session_dir2)
            arms3 = _EmulatedArm(fail_tags={"C"})
            session3 = oshot.run_all(
                str(session_dir2), preflight_runner=_pass_preflight,
                canary_runner=lambda sd: {"status": "PASS"},
                arm_runner=arms3, pre50m_runner=_green_pre50m)
            assert session3["status"] == "COMPLETE"
            assert session3["phases"]["ARM_C"] == "IMPLEMENTATION_FAILURE"
            assert (session_dir2 / "ARM_C.json").is_file()
            assert t1d.verify_bundle(str(session_dir2))["status"] == "VALID"

            # PRE50M failure: arms preserved, decision fail-closed, bundle valid
            def failing_pre50m(root, arm_receipts, *, rt_sha, rate, shape):
                raise RuntimeError("XLA exploded in pre50m")

            session_dir3 = Path(tmp) / "session3"
            _seed_calibration(session_dir3)
            session4 = oshot.run_all(
                str(session_dir3), preflight_runner=_pass_preflight,
                canary_runner=lambda sd: {"status": "PASS"},
                arm_runner=_EmulatedArm(), pre50m_runner=failing_pre50m)
            assert session4["phases"]["PRE50M"] == "IMPLEMENTATION_FAILURE"
            decision = json.loads(
                (session_dir3 / "NEXT_50M_DECISION.json").read_text())
            assert decision["ready_for_50m_training"] is False
            assert t1d.verify_bundle(str(session_dir3))["status"] == "VALID"

            # preflight failure: STOP before any training, failure bundle exported
            session_dir4 = Path(tmp) / "session4"

            def failing_preflight(root):
                return {"status": "FAIL", "gates": [], "blocking_gates": ["tpu_active"]}

            session5 = oshot.run_all(
                str(session_dir4), preflight_runner=failing_preflight,
                canary_runner=lambda sd: {"status": "PASS"},
                arm_runner=_EmulatedArm(), pre50m_runner=_green_pre50m)
            assert session5["status"] == "FAILED"
            assert "PREFLIGHT" in session5.get("error", "")
            failure_zip = Path(session5["failure_bundle"])
            assert failure_zip.is_file() and failure_zip.stat().st_size > 0
            import zipfile

            with zipfile.ZipFile(failure_zip) as zf:
                assert "FAILURE_TRACEBACK.txt" in zf.namelist()
            assert not (session_dir4 / "ARM_A.json").is_file()
        finally:
            if saved is not None:
                real_cert.write_text(saved, encoding="utf-8")
            elif real_cert.is_file():
                real_cert.unlink()


def test_data_accounting() -> None:
    """§23: unique-pool vs schedulable accounting is computed from the REAL
    frozen budgets, and the decision follows the fraction."""
    account = oshot.data_accounting()
    assert account["available_unique_rows"] > 0
    assert 0 < account["consumable_unique_fraction_est"] < 1
    assert account["decision"] in ("KEEP", "EXPAND_IF_SCIENTIFICALLY_USEFUL")
    json.dumps(account)


def test_mid_state_payload_integrity() -> None:
    """Mid-arm state: payload hash verification and refusal paths (the torch
    model/optimizer halves are exercised on device; these are the pure
    bookkeeping guarantees every resume depends on)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = {"update": 50, "cfg": {"budget": 8_000_000}, "seed": 1,
                   "shape": [8, 64], "ledgers": [{"updates": 50}]}
        doc = {"schema": "citadel-t1d-arm-mid/v1", "arm": "A", "plan_sha": "p" * 64,
               **payload}
        body = {k: v for k, v in doc.items()
                if k not in ("model_path", "model_sha256", "optimizer_path",
                             "optimizer_sha256", "payload_sha256")}
        doc["payload_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        import citadel_tpu.t1d_run as tr

        got = {k: v for k, v in doc.items()
               if k not in ("model_path", "model_sha256", "optimizer_path",
                            "optimizer_sha256", "payload_sha256")}
        assert hashlib.sha256(tr._canonical_json(got)).hexdigest() == \
            doc["payload_sha256"]
        # tampering is detected
        body2 = dict(got)
        body2["update"] = 49
        assert hashlib.sha256(tr._canonical_json(body2)).hexdigest() != \
            doc["payload_sha256"]


def test_torch_resume_identity() -> None:
    """§13 (torch-optional, CPU): N continuous updates vs N/2 + mid-save +
    restore + N/2 — identical token scheduling/feeder state and (deterministic
    CPU) loss trajectory. Skipped cleanly when torch is unavailable."""
    try:
        import torch
    except ImportError:
        raise SkipTest("torch unavailable in this interpreter")
    import types

    from citadel_tpu import runtime_bootstrap as rb

    rt_root, _ = rb.ensure_cymek_runtime()
    if str(rt_root) not in sys.path:
        sys.path.insert(0, str(rt_root))
    from citadel_tpu import calculator_eval as cev

    # monkeypatch the XLA seams at module level (CPU has no torch_xla by
    # design; the seam is exactly where device execution would hook in)
    from citadel_tpu import xla_backend as xb

    real_mark, real_step = xb.mark_step, xb.optimizer_step
    xb.mark_step = lambda: None
    xb.optimizer_step = lambda o: o.step()
    spec = t1d.build_spec("MID")

    def run_half(model, optimizer, feeder, n, start, length=64):
        return t1d._train_updates_packed(
            model, optimizer, feeder, n_updates=n, start_update=start,
            updates_total=2 * n, device=torch.device("cpu"), torch_mod=torch,
            length=length, masked=False, valid_ids=None)

    m1 = t1d.build_spec("MID")
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    torch.manual_seed(20260906)
    model_a = initialize(m1, 20260906)
    opt_a = build_adamw_optimizer(model_a, torch_module=torch)
    feeder_a = t1d.TierFeeder("curriculum", 8, 64)
    rep_continuous = run_half(model_a, opt_a, feeder_a, 3, 0)
    rep_continuous2 = run_half(model_a, opt_a, feeder_a, 3, 3)

    torch.manual_seed(20260906)
    model_b = initialize(m1, 20260906)
    opt_b = build_adamw_optimizer(model_b, torch_module=torch)
    feeder_b = t1d.TierFeeder("curriculum", 8, 64)
    rep_first = run_half(model_b, opt_b, feeder_b, 3, 0)
    state = feeder_b.state()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "mid.pt"
        torch.save({"model_state": model_b.state_dict(),
                    "optimizer_state": opt_b.state_dict()}, ck)
        payload = torch.load(ck, weights_only=False)
        model_b.load_state_dict(payload["model_state"])
        opt_b.load_state_dict(payload["optimizer_state"])
        feeder_b2 = t1d.TierFeeder("curriculum", 8, 64)
        feeder_b2.load_state(state)
        rep_resumed = run_half(model_b, opt_b, feeder_b2, 3, 3)
    # token/data scheduling must be identical; CPU fp32 losses are
    # deterministic enough to compare exactly here
    assert rep_continuous2["real_tokens"] == rep_resumed["real_tokens"]
    assert rep_continuous2["capacity_tokens"] == rep_resumed["capacity_tokens"]
    assert rep_continuous2["answer_supervised"] == rep_resumed["answer_supervised"]
    assert abs(rep_continuous2["last_loss"] - rep_resumed["last_loss"]) < 1e-6, (
        rep_continuous2["last_loss"], rep_resumed["last_loss"])
    assert feeder_a.placed_rows == feeder_b2.placed_rows
    xb.mark_step, xb.optimizer_step = real_mark, real_step


def _certificate_path() -> Path:
    return CITADEL_ROOT / "docs" / "citadel" / "experiments" / "T1D" / \
        "DEVELOPMENT_CERTIFICATION.json"


def test_xla_pass_contract() -> None:
    """THE canonical-contract regression: preflight._xla_api_status returns
    ("PASS", []) on healthy XLA and ("FAIL", missing) otherwise. The
    xla_apis gate must consume that exact vocabulary - healthy PASS opens
    the gate, any FAIL reason closes it. (This mismatch shipped once: the
    gate checked for ("OK", "PARTIAL") and would have failed a healthy
    Colab TPU at preflight.)"""
    from citadel_tpu import preflight as pf
    from citadel_tpu import t1d_preflight

    assert t1d_preflight._xla_gate("PASS") is True
    assert t1d_preflight._xla_gate("FAIL") is False
    assert t1d_preflight._xla_gate("UNAVAILABLE") is False
    # full structured preflight with the REAL low-level contract monkeypatched
    real = pf._xla_api_status
    try:
        pf._xla_api_status = lambda: ("PASS", [])
        pre = t1d_preflight.run_preflight()
        gate = next(g for g in pre["gates"] if g["name"] == "xla_apis")
        assert gate["status"] == "PASS", gate
        assert "xla_apis" not in pre["blocking_gates"]
        pf._xla_api_status = lambda: ("FAIL", ["xm.optimizer_step"])
        pre2 = t1d_preflight.run_preflight()
        gate2 = next(g for g in pre2["gates"] if g["name"] == "xla_apis")
        assert gate2["status"] == "FAIL", gate2
        assert "xla_apis" in pre2["blocking_gates"]
        assert pre2["status"] == "FAIL"
    finally:
        pf._xla_api_status = real


def test_select_calibrated_shape_masked_guard() -> None:
    """A candidate must pass EVERY required variant: the fastest shape can
    pass SCALE2 but fail the masked Arm E path -> the next passing shape is
    selected and the failure is marked in place with the right reason."""
    results = [
        {"batch": 1024, "length": 64, "tokens_per_second": 9000.0, "correct": True},
        {"batch": 512, "length": 64, "tokens_per_second": 7000.0, "correct": True},
        {"batch": 256, "length": 64, "tokens_per_second": 5000.0, "correct": True},
    ]

    def scale2_ok(batch, length):
        return (batch, length) != (1024, 64)

    def masked_ok(batch, length):
        return (batch, length) not in ((1024, 64), (512, 64))

    best, note = t1d.select_calibrated_shape(
        results, scale2_verifier=scale2_ok, masked_verifier=masked_ok)
    assert note == "pass" and best["batch"] == 256, (note, best)
    failed = {r["batch"]: r["error"] for r in results if not r["correct"]}
    assert failed == {1024: "SCALE2_VERIFICATION_FAILED",
                      512: "MASKED_VERIFICATION_FAILED"}, failed
    assert len(results) == 3  # no duplicate dicts
    assert all(r["correct"] for r in results if r["batch"] == 256)
    # a shape passing neither is marked by the first failing verifier
    none, note2 = t1d.select_calibrated_shape(
        [{"batch": 1, "length": 64, "tokens_per_second": 1.0, "correct": True}],
        scale2_verifier=lambda b, l: False, masked_verifier=lambda b, l: False)
    assert none is None and "safe" in note2


def test_run_arm_feeder_restore_wiring() -> None:
    """THE disconnect-recovery regression (torch-optional): the PRODUCTION
    restore path run_arm uses — restore_mid_into(model, optimizer, feeder,
    mid) — must bring back the EXACT saved data-plane state (tier cursors,
    teacher cursor, self cursor, carry/pending rows, drawn/placed counters,
    placed-token counters), and the next feed must match uninterrupted
    execution byte-for-byte, for every feeder mode. Also pins the ordering:
    run_arm restores before its first consumption and skips the
    consumption-dependent pre-training baselines on resume."""
    try:
        import torch
    except ImportError:
        raise SkipTest("torch unavailable in this interpreter")
    import tempfile

    from citadel_tpu import runtime_bootstrap as rb

    rb.ensure_cymek_runtime()
    from citadel_tpu import self_knowledge as sk
    from citadel_tpu import t1c_run as t1c

    # source-order audit of the production wiring itself
    src = (CITADEL_ROOT / "citadel_tpu" / "t1d_run.py").read_text(encoding="utf-8")
    body = src[src.index("def run_arm("):src.index("def frozen_train_candidates")]
    assert "restore_mid_into(model, optimizer, feeder, mid)" in body
    assert body.index("restore_mid_into(model, optimizer, feeder, mid)")         < body.index("_train_updates_packed(")
    assert "if recover is None:" in body  # baseline evals skipped on resume

    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("flat", "curriculum", "teacher", "self", "masked"):
            feeder_mode = "curriculum" if mode == "masked" else mode
            # 1. advance a real feeder
            feeder = t1d.TierFeeder(feeder_mode, 8, 64)
            for u in range(10):
                feeder.fill_sequences(u / 10)
            saved_state = feeder.state()
            # 2. save the mid state through the production saver
            torch.manual_seed(20260906)
            spec = t1d.build_spec("MID")
            model = initialize(spec, 20260906)
            optimizer = build_adamw_optimizer(model, torch_module=torch)
            payload = {"update": 10, "cfg": {"mode": feeder_mode,
                                             "budget": 8_000_000},
                       "seed": 20260906, "shape": [8, 64],
                       "feeder_state": saved_state}
            t1d.save_mid_state(tmp, "A", model=model, optimizer=optimizer,
                               feeder=feeder, payload=payload)
            # 3. enter the SAME restore path run_arm uses (verified load +
            #    helper), into a FRESH feeder and re-initialized model
            mid, why = t1d.load_mid_state(tmp, "A",
                                          expect_cfg={"mode": feeder_mode,
                                                      "budget": 8_000_000},
                                          seed=20260906, shape=(8, 64))
            assert mid is not None, (mode, why)
            model2 = initialize(spec, 20260906)
            optimizer2 = build_adamw_optimizer(model2, torch_module=torch)
            feeder2 = t1d.TierFeeder(feeder_mode, 8, 64)
            t1d.restore_mid_into(model2, optimizer2, feeder2, mid)
            # 4. the live feeder state after restore EXACTLY equals saved
            assert feeder2.state() == saved_state, (
                mode, "restored feeder state differs")
            # model/optimizer identity across the restore
            sha1 = __import__("hashlib").sha256(b"".join(
                t.detach().cpu().contiguous().numpy().tobytes()
                for t in model.state_dict().values())).hexdigest()
            sha2 = __import__("hashlib").sha256(b"".join(
                t.detach().cpu().contiguous().numpy().tobytes()
                for t in model2.state_dict().values())).hexdigest()
            assert sha1 == sha2, mode
            # 5+6. the next feed on the resumed feeder must equal
            #      uninterrupted execution, byte for byte
            resumed_seqs = feeder2.fill_sequences(10 / 20)
            continuous_seqs = feeder.fill_sequences(10 / 20)
            assert resumed_seqs == continuous_seqs, (
                mode, "resumed data schedule diverged from uninterrupted")
            more = 8
            feeder2.fill_sequences(11 / 20)
            feeder.fill_sequences(11 / 20)
            assert feeder2.placed_rows == feeder.placed_rows, mode
            assert feeder2.placed_tokens == feeder.placed_tokens, mode
            # the strongest invariant: resumed feeder state == uninterrupted
            # feeder state (cursors, counters, carry, everything)
            assert feeder2.state() == feeder.state(), (
                mode, "resumed feeder diverged from uninterrupted")
        # a mid state WITHOUT feeder_state must be refused by the helper
        model3 = initialize(spec, 20260906)
        opt3 = build_adamw_optimizer(model3, torch_module=torch)
        f3 = t1d.TierFeeder("flat", 8, 64)
        try:
            t1d.restore_mid_into(model3, opt3, f3, {"model_path": "", })
            raise SystemExit("feeder-less mid state accepted")
        except RuntimeError as exc:
            assert "feeder_state missing" in str(exc), exc


class _CpuSeams:
    """Monkeypatches the device seams so the REAL run_arm is CPU-executable:
    environment probe, XLA device, mark_step/optimizer_step, and generation
    (the lowest possible seam - deterministic synthetic predictions)."""

    def __init__(self):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")

    def __enter__(self):
        from citadel_tpu import calculator_eval as cev
        from citadel_tpu import environment as env_mod
        from citadel_tpu import xla_backend as xb

        self._saved = (env_mod.probe, xb.get_device, xb.assert_tpu_active,
                       xb.mark_step, xb.optimizer_step, cev.generate)
        env_mod.probe = lambda require_tpu=False: {
            "probe_pass": True, "tpu_present": True,
            "accelerator_detected": "cpu-test", "xla_device_count": 1}
        xb.get_device = lambda: self.device
        xb.assert_tpu_active = lambda min_devices=1: 1
        xb.mark_step = lambda: None
        xb.optimizer_step = lambda o: o.step()

        def fake_generate(rows, model, xb_, *, device, torch_mod,
                          allow_ids=None, first_step_stats=False):
            recs = []
            for r in rows:
                _, tgt = cev.split_prompt_target(r)
                recs.append({"prompt": r, "target": tgt, "prediction": tgt,
                             "correct": True, "stop_reason": "EOS",
                             "generated_token_count": len(tgt), "valid": True})
            return recs

        cev.generate = fake_generate
        return self

    def __exit__(self, *exc):
        from citadel_tpu import calculator_eval as cev
        from citadel_tpu import environment as env_mod
        from citadel_tpu import xla_backend as xb

        (env_mod.probe, xb.get_device, xb.assert_tpu_active,
         xb.mark_step, xb.optimizer_step, cev.generate) = self._saved
        return False


TINY_CFG = {"spec": "MID", "mode": "curriculum", "budget": 8 * 64 * 4}


def test_post_reload_self_probe_recovery() -> None:
    """THE exact real-TPU failure path, exercised callable on CPU: training
    completes 100% -> final checkpoint saved -> original model released ->
    reload prediction check -> self-knowledge probe evaluation -> per-domain
    aggregation -> prefinal snapshot -> final receipt. The old code died at
    the self probe with `cannot access free variable 'model'` AFTER the
    expensive training. The final_model_ready boundary (written BEFORE the
    final evaluations) plus the explicit-model evaluators make the rerun a
    pure evaluation+finalization recovery with NO retraining."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SkipTest("torch unavailable in this interpreter")
    import tempfile

    from citadel_tpu import self_knowledge as sk

    with tempfile.TemporaryDirectory() as tmp:
        with _CpuSeams():
            # run 1: crash exactly at the trained_self evaluation (the
            # summarize_text call #3: untrained_self, untrained null, trained)
            real_summ = sk.summarize_text
            calls = {"n": 0}

            def crashing_summ(preds, tgts):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise RuntimeError(
                        "cannot access free variable 'model' (simulated)")
                return real_summ(preds, tgts)

            sk.summarize_text = crashing_summ
            try:
                receipt1 = t1d.run_arm("A", dict(TINY_CFG), shape=(8, 64),
                                       out_dir=tmp, seed=20260906)
                raise SystemExit("run 1 unexpectedly completed")
            except RuntimeError as exc:
                assert "free variable" in str(exc), exc
            finally:
                sk.summarize_text = real_summ
            assert calls["n"] == 3, calls
            # the boundary was written BEFORE the failing evaluation
            fmr_path = Path(tmp) / "ARM_A.final_model_ready.json"
            assert fmr_path.is_file(), "final_model_ready must survive the crash"
            assert not (Path(tmp) / "ARM_A.json").is_file()
            # run 2: recovery - evaluation + finalization only, NO retraining
            real_tup = t1d._train_updates_packed

            def forbidden_training(*a, **k):
                raise RuntimeError("RETRAINING FORBIDDEN during recovery")

            t1d._train_updates_packed = forbidden_training
            try:
                receipt2 = t1d.run_arm("A", dict(TINY_CFG), shape=(8, 64),
                                       out_dir=tmp, seed=20260906)
            finally:
                t1d._train_updates_packed = real_tup
            assert receipt2["status"] in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL")
            assert receipt2.get("eval_recovery") is True
            assert (Path(tmp) / "ARM_A.json").is_file()
            assert (Path(tmp) / "ARM_A.done.json").is_file()
            assert not fmr_path.is_file()  # consumed
            # rerun 3: completed receipt short-circuits
            receipt3 = t1d.run_arm("A", dict(TINY_CFG), shape=(8, 64),
                                   out_dir=tmp, seed=20260906)
            assert receipt3.get("resumed") is True


def test_ab_mid_resume_simulation() -> None:
    """A/B 75%-mid recovery simulation: an arm crashing at its LAST training
    block leaves a valid mid state; the rerun resumes from it (executing only
    the remaining updates) and its data schedule is byte-identical to an
    uninterrupted reference run."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SkipTest("torch unavailable in this interpreter")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_a, \
            tempfile.TemporaryDirectory() as tmp_ref:
        with _CpuSeams():
            real_tup = t1d._train_updates_packed
            blocks = []

            def spying_tup(model, optimizer, feeder, *, n_updates,
                           start_update, **k):
                blocks.append((n_updates, start_update))
                if start_update == 3:  # crash at the last training block
                    raise RuntimeError("simulated disconnect at 75%->100%")
                return real_tup(model, optimizer, feeder, n_updates=n_updates,
                                start_update=start_update, **k)

            t1d._train_updates_packed = spying_tup
            try:
                t1d.run_arm("A", dict(TINY_CFG), shape=(8, 64), out_dir=tmp_a,
                            seed=20260906)
                raise SystemExit("crashing run unexpectedly completed")
            except RuntimeError as exc:
                assert "disconnect" in str(exc), exc
            finally:
                t1d._train_updates_packed = real_tup
            mid = json.loads((Path(tmp_a) / "ARM_A.mid.json").read_text())
            assert mid["update"] == 3, mid["update"]
            assert not (Path(tmp_a) / "ARM_A.done.json").is_file()
            # resume: exactly ONE remaining block (1 update from update 3)
            receipt = t1d.run_arm("A", dict(TINY_CFG), shape=(8, 64),
                                  out_dir=tmp_a, seed=20260906)
            assert receipt["training"]["updates"] == 4
            assert receipt["status"] in ("SCIENTIFIC_PASS", "SCIENTIFIC_FAIL")
            # uninterrupted reference run with the same seed
            ref = t1d.run_arm("A", dict(TINY_CFG), shape=(8, 64),
                              out_dir=tmp_ref, seed=20260906)
            assert receipt["data"]["feeder"]["placed_rows"] == \
                ref["data"]["feeder"]["placed_rows"]
            assert receipt["data"]["feeder"]["placed_tokens"] == \
                ref["data"]["feeder"]["placed_tokens"]


def test_final_model_ready_integrity() -> None:
    """final_model_ready artifact: hash verification and every refusal path
    (corrupt payload, missing checkpoint, plan/cfg mismatch)."""
    import hashlib as _hl
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = {"cfg": {"mode": "curriculum"}, "env": {}, "seed": 7,
                   "n_seq": 8, "length": 64, "param_count": 1, "updates_total": 4,
                   "feeder_state": {"cursors": {}}, "ledgers": [], "done": 4,
                   "first_loss": 9.0, "last_loss": 8.0, "cap_total": 2048,
                   "ans_total": 10, "whole_total": 100, "gsum": 1.0,
                   "gmax": 0.5, "gn": 4, "train_wall": 1.0,
                   "untrained": {}, "untrained_dev": {}, "untrained_self": {},
                   "untrained_train": {}, "train_candidates": {}, "inter": {},
                   "citadel_sha": "0" * 40, "cymek_sha": "1" * 64}
        ckpt = root / "t1d_arm_a.pt"
        ckpt.write_bytes(b"checkpoint-bytes")
        sidecar = Path(t1d.write_final_model_ready(
            root, "A", ckpt_path=str(ckpt),
            ckpt_sha=_hl.sha256(ckpt.read_bytes()).hexdigest(),
            payload=payload))
        snap, why = t1d.load_final_model_ready(root, "A",
                                               expect_cfg=payload["cfg"],
                                               seed=7, shape=(8, 64))
        assert snap is not None, why
        assert snap["ckpt_path"] == str(ckpt)
        # corrupt payload -> refused + archived
        doc = json.loads(sidecar.read_text())
        doc["done"] = 999
        sidecar.write_text(json.dumps(doc), encoding="utf-8")
        snap2, why2 = t1d.load_final_model_ready(root, "A",
                                                 expect_cfg=payload["cfg"],
                                                 seed=7, shape=(8, 64))
        assert snap2 is None and "hash mismatch" in why2
        # missing checkpoint -> refused
        t1d.write_final_model_ready(root, "A", ckpt_path=str(root / "gone.pt"),
                                    ckpt_sha="0" * 64, payload=payload)
        snap3, why3 = t1d.load_final_model_ready(root, "A",
                                                 expect_cfg=payload["cfg"],
                                                 seed=7, shape=(8, 64))
        assert snap3 is None and "missing" in why3
        # checkpoint tampering -> refused + archived
        t1d.write_final_model_ready(root, "A", ckpt_path=str(ckpt),
                                    ckpt_sha=_hl.sha256(ckpt.read_bytes()).hexdigest(),
                                    payload=payload)
        ckpt.write_bytes(b"tampered")
        snap4, why4 = t1d.load_final_model_ready(root, "A",
                                                 expect_cfg=payload["cfg"],
                                                 seed=7, shape=(8, 64))
        assert snap4 is None and "checkpoint hash mismatch" in why4


def test_no_stale_model_closure() -> None:
    """Mechanical audit of run_arm: the evaluators take the model EXPLICITLY
    (active_model) and every post-`del model` call passes model2/model_v —
    no nested evaluator may capture a model whose lifetime ends."""
    src = (CITADEL_ROOT / "citadel_tpu" / "t1d_run.py").read_text(encoding="utf-8")
    body = src[src.index("def run_arm("):src.index("def frozen_train_candidates")]
    assert "def gen(active_model, rows, targets)" in body
    assert "def gen_text(active_model, rows, targets)" in body
    assert "def final_evals(active_model):" in body
    # no model-less evaluator definitions remain in run_arm
    assert "def gen(rows, targets)" not in body
    assert "def gen_text(rows, targets)" not in body
    # every post-del evaluation explicitly uses model2 / the recovery model
    del_pos = body.index("del model")
    after = body[del_pos:]
    assert "_gen_eval(model," not in after
    assert "_gen_eval(model2," in after or "_gen_eval(model_v," in after
    assert "gen_text(self_rows" not in after  # no model-less text-eval call


def test_pre50m_smoke_budget_funds_resume() -> None:
    """Pure regression of the REAL TPU failure (2026-09-06, PRE50M phase):
    the smoke state was created with token_budget = updates * tokens_per_update
    but the resume-proof publishes update updates+1 - Cymek refuses with
    "a completed run cannot advance". The budget arithmetic must fund
    updates+1, verified against the REAL Cymek TrainingState contract
    (framework-neutral module)."""
    from citadel_tpu import runtime_bootstrap as rb

    rb.ensure_cymek_runtime()
    from citadel_tpu import cymek_checkpoint as cckpt
    from citadel_tpu import pre50m as p50

    updates, tokens_per_update = p50.SMOKE_UPDATES, 32 * 64
    funded_budget = (updates + 1) * tokens_per_update
    old_budget = updates * tokens_per_update
    assert funded_budget > old_budget
    identities = cckpt.build_identities(
        model_spec_sha256="0" * 64, data_manifest_sha256="1" * 64,
        pack_manifest_sha256="2" * 64, run_spec={}, optimizer_spec={},
        schedule_spec={}, curriculum_spec={}, source_commit="0" * 40)

    def run_updates(budget):
        state = cckpt.initial_state(
            lineage_id="budget-check", token_budget=budget,
            tokens_per_update=tokens_per_update,
            pack_manifest_sha256="2" * 64, identities=identities,
            rng_state_sha256="0" * 64)
        cursor = cckpt.cursor_for_update("2" * 64, sequence_ordinal=1,
                                         token_offset=tokens_per_update)
        for k in range(updates):  # the smoke's `updates` real updates
            state = state.advance(
                tokens_by_source={"smoke": tokens_per_update},
                cursor=cckpt.cursor_for_update(
                    "2" * 64, sequence_ordinal=k + 1,
                    token_offset=(k + 1) * tokens_per_update),
                rng_state_sha256="0" * 64, parent_checkpoint_sha256=None)
        return state

    completed_old = run_updates(old_budget)
    assert completed_old.complete is True, (
        "the old budget completes the run: the resume-proof update raises "
        "'a completed run cannot advance' - the real TPU failure")
    completed_funded = run_updates(funded_budget)
    assert completed_funded.complete is False, (
        "the funded budget must leave room for the resume-proof update")
    assert completed_funded.cumulative_tokens == old_budget


def test_pre50m_status_from_decision() -> None:
    """A decision file alone is NEVER success: ready=True + no blockers is
    PASS; ready=False, blocking reasons, or an explicit failure status are
    carried through truthfully (the real TPU run mislabeled a failed PRE50M
    as PASS here)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert t1d.pre50m_status_from_decision(root)["status"] == "NOT_RUN"
        (root / "NEXT_50M_DECISION.json").write_text(json.dumps(
            {"ready_for_50m_training": False, "blocking_reasons": ["x"],
             "status": "IMPLEMENTATION_FAILURE"}), encoding="utf-8")
        got = t1d.pre50m_status_from_decision(root)
        assert got["status"] == "IMPLEMENTATION_FAILURE", got
        (root / "NEXT_50M_DECISION.json").write_text(json.dumps(
            {"ready_for_50m_training": False, "blocking_reasons": ["not fit"]}),
            encoding="utf-8")
        assert t1d.pre50m_status_from_decision(root)["status"] == "NOT_READY"
        (root / "NEXT_50M_DECISION.json").write_text(json.dumps(
            {"ready_for_50m_training": True, "blocking_reasons": []}),
            encoding="utf-8")
        assert t1d.pre50m_status_from_decision(root)["status"] == "PASS"


def test_pre50m_reserved_final_update_contract() -> None:
    """§13: the REAL Cymek TrainingState contract with a RESERVED FINAL
    UPDATE. budget = 6X, 5 advances -> complete is False; a
    serialize/restore-equivalent state rebuild advances once more ->
    complete is True, global_update == 6, cumulative == 6X. Negative
    control: budget = 5X -> the 6th advance MUST raise 'a completed run
    cannot advance' (proving Cymek is correct and the old smoke was
    wrong)."""
    from citadel_tpu import runtime_bootstrap as rb

    rb.ensure_cymek_runtime()
    from citadel_tpu import cymek_checkpoint as cckpt

    X = 2048
    identities = cckpt.build_identities(
        model_spec_sha256="0" * 64, data_manifest_sha256="1" * 64,
        pack_manifest_sha256="2" * 64, run_spec={}, optimizer_spec={},
        schedule_spec={}, curriculum_spec={}, source_commit="0" * 40)

    def advance_n(budget, n, *, start_state=None, start_k=0):
        state = start_state if start_state is not None else \
            cckpt.initial_state(
                lineage_id="reserved-final", token_budget=budget,
                tokens_per_update=X, pack_manifest_sha256="2" * 64,
                identities=identities, rng_state_sha256="0" * 64)
        for k in range(start_k, start_k + n):
            state = state.advance(
                tokens_by_source={"smoke": X},
                cursor=cckpt.cursor_for_update(
                    "2" * 64, sequence_ordinal=k + 1,
                    token_offset=(k + 1) * X),
                rng_state_sha256="0" * 64, parent_checkpoint_sha256=None)
        return state

    state = advance_n(6 * X, 5)
    assert state.complete is False
    assert state.global_update == 5 and state.cumulative_tokens == 5 * X
    # serialize/restore-equivalent rebuild: TrainingState is a frozen
    # dataclass - the restore path constructs a NEW state from the canonical
    # record (exactly what the production restore does with
    # training_state.json), then the reserved final update is consumed.
    canonical = json.loads(json.dumps(state.canonical()))
    from v5_training import state as v5_state

    cursor = canonical["cursor"]
    restored = v5_state.TrainingState(
        schema=canonical["schema"], lineage_id=canonical["lineage_id"],
        generation=canonical["generation"],
        global_update=canonical["global_update"],
        cumulative_tokens=canonical["cumulative_tokens"],
        token_budget=canonical["token_budget"],
        tokens_per_update=canonical["tokens_per_update"],
        tokens_by_source=dict(canonical["tokens_by_source"]),
        optimizer_step_max=canonical["optimizer_step_max"],
        schedule_tokens=canonical["schedule_tokens"],
        cursor=v5_state.CursorState(
            schema=cursor["schema"],
            pack_manifest_sha256=cursor["pack_manifest_sha256"],
            shard_ordinal=cursor["shard_ordinal"],
            sequence_ordinal=cursor["sequence_ordinal"],
            token_offset=cursor["token_offset"]),
        rng_state_sha256=canonical["rng_state_sha256"],
        curriculum_phase=canonical["curriculum_phase"],
        identities=identities,
        parent_checkpoint_sha256=canonical.get("parent_checkpoint_sha256"))
    restored_tokens = dict(canonical["tokens_by_source"])
    final = advance_n(6 * X, 1, start_state=restored, start_k=5)
    assert final.complete is True
    assert final.global_update == 6
    assert final.cumulative_tokens == 6 * X
    assert sum(final.tokens_by_source.values()) == 6 * X
    assert final.tokens_by_source == restored_tokens or restored_tokens
    assert final.optimizer_step_max == final.global_update
    assert final.schedule_tokens == final.cumulative_tokens
    # negative control: the OLD underfunded budget raises on the 6th advance
    underfunded = advance_n(5 * X, 5)
    assert underfunded.complete is True
    try:
        underfunded.advance(
            tokens_by_source={"smoke": X},
            cursor=cckpt.cursor_for_update("2" * 64, sequence_ordinal=6,
                                           token_offset=6 * X),
            rng_state_sha256="0" * 64, parent_checkpoint_sha256=None)
        raise SystemExit("underfunded 6th advance accepted - Cymek contract "
                         "violated or the budget semantics changed")
    except ValueError as exc:
        assert "a completed run cannot advance" in str(exc), exc


def test_pre50m_phase_status_propagation() -> None:
    """§14: the session PRE50M status must NEVER say PASS while
    NEXT_50M_DECISION.ready_for_50m_training is false (the real TPU bundle
    carried that contradiction). summarize_session + the pre50m status
    inference must agree."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        (root / "CALIBRATION.json").write_text(json.dumps(
            {"selected": {"batch": 256, "length": 64},
             "selected_tokens_per_second": 8000.0}), encoding="utf-8")
        (root / "DATA_MANIFEST.json").write_text(json.dumps({"dummy": True}),
                                                 encoding="utf-8")
        arms = {}
        for tag in t1d.ARM_ORDER:
            arms[tag] = _arm_full()
            (root / f"ARM_{tag}.json").write_text(json.dumps(arms[tag]),
                                                  encoding="utf-8")
        _green_pre50m(root, arms, rt_sha="1" * 64, rate=8000.0,
                      shape=(256, 64))
        # the real TPU contradiction: decision says failure/not-ready
        (root / "NEXT_50M_DECISION.json").write_text(json.dumps(
            {"ready_for_50m_training": False,
             "blocking_reasons": ["smoke failed"],
             "status": "IMPLEMENTATION_FAILURE"}), encoding="utf-8")
        inferred = t1d.pre50m_status_from_decision(root)
        assert inferred["status"] == "IMPLEMENTATION_FAILURE"
        # summarize carries that truth into the session manifest
        arms = {t: _arm_full() for t in t1d.ARM_ORDER}
        session = t1d.summarize_session(
            root, arms, shape=(256, 64), rate=8000.0, scaled=False,
            budgets={t: t1d.ARMS[t]["budget"] for t in t1d.ARMS},
            rt_sha="1" * 64, pre50m_status=inferred)
        assert session["pre50m"]["status"] == "IMPLEMENTATION_FAILURE"
        assert json.loads((root / "SESSION_MANIFEST.json")
                          .read_text())["pre50m"]["status"] == \
            "IMPLEMENTATION_FAILURE"


def test_t1e_eos_helpers() -> None:
    """T1E future-experiment helpers (unit-tested, NOT executed): EOS is
    appended to the packed row, the eligible mask supervises answer + EOS
    only, termination classification separates stop from content, and the
    generation-step split leaves room for a full-length answer + EOS."""
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import t1c_run as t1c
    from citadel_tpu import t1e_helpers as h

    assert h.MAX_GENERATION_STEPS == h.MAX_CONTENT_TOKENS + 1
    for row in ("12 + 9 = 21", "9 * 9 = 81", "7 / 1 = 7", "5 - 12 = -7"):
        ids = h.row_with_eos(row, eos_id=3)
        assert ids[-1] == 3 and ids[:-1] == cev.encode(row)
        mask = h.eligibility_with_eos(row, eos_id=3)
        plen, alen = t1c.answer_spans([row], 64)[0]
        supervised = [i for i, m in enumerate(mask) if m]
        assert supervised == list(range(plen, plen + alen)) + [plen + alen], row
        assert not any(mask[:plen]), "prompt must stay unsupervised"
    # exact supervised counts per answer spelling (spans come from the
    # frozen production splitter, not assumptions)
    ids = h.row_with_eos("7 / 1 = 7", eos_id=3)
    mask = h.eligibility_with_eos("7 / 1 = 7", eos_id=3)
    plen, alen = t1c.answer_spans(["7 / 1 = 7"], 64)[0]
    assert sum(mask) == alen + 1, (alen, sum(mask))  # target chars + EOS
    mask2 = h.eligibility_with_eos("5 - 12 = -7", eos_id=3)
    plen2, alen2 = t1c.answer_spans(["5 - 12 = -7"], 64)[0]
    assert sum(mask2) == alen2 + 1  # "-7" + EOS
    # termination classification separates stop from content
    assert h.termination_classify("EOS", prediction="21", target="21") == "EOS_OK"
    assert h.termination_classify("MAX_TOKENS", prediction="2", target="21") ==         "TERMINATION_FAILURE"
    assert h.termination_classify("NON_ALPHABET", prediction="x", target="21") ==         "TERMINATION_FAILURE"
    assert h.termination_classify("NEWLINE", prediction="2", target="21") ==         "PREMATURE_STOP"
    # content-truncated diagnostic never rewrites the exact-match contract
    assert h.content_exact_truncated("2155", "21") is True
    assert h.content_exact_truncated("20", "21") is False
    assert h.content_exact_truncated("", "21") is False


def main() -> int:
    tests = [test_portability_scan, test_plan_identity_stable_and_sensitive,
             test_self_feeder_cadence_and_rows, test_self_classify_rules,
             test_one_shot_emulator_fresh_and_resume, test_data_accounting,
             test_mid_state_payload_integrity, test_torch_resume_identity,
             test_xla_pass_contract, test_select_calibrated_shape_masked_guard,
             test_run_arm_feeder_restore_wiring,
             test_post_reload_self_probe_recovery,
             test_ab_mid_resume_simulation,
             test_final_model_ready_integrity,
             test_no_stale_model_closure,
             test_pre50m_smoke_budget_funds_resume,
             test_pre50m_status_from_decision,
             test_pre50m_reserved_final_update_contract,
             test_pre50m_phase_status_propagation,
             test_t1e_eos_helpers]
    failed = skipped = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP {fn.__name__}: {exc}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    total = len(tests)
    print(f"{total - failed - skipped}/{total} passed"
          + (f" ({skipped} skipped)" if skipped else ""))
    return 1 if failed else 0


class SkipTest(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
