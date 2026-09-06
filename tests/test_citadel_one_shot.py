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
            reload_ok=True, device_count=1, wall=60.0)
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


def main() -> int:
    tests = [test_portability_scan, test_plan_identity_stable_and_sensitive,
             test_self_feeder_cadence_and_rows, test_self_classify_rules,
             test_one_shot_emulator_fresh_and_resume, test_data_accounting,
             test_mid_state_payload_integrity, test_torch_resume_identity,
             test_xla_pass_contract, test_select_calibrated_shape_masked_guard]
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
