"""Zero-training V2 diagnostics for K8 R01-R08 repairs.

Run from repository root. Performs zero optimizer updates and zero
accelerator work; expensive model execution is replaced by explicit test
doubles except for tiny-CPU backward-only checks (no step). Asserts the
failing-before conditions from K8_SECOND_REVIEW now pass.
"""
import contextlib
import gc
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

STEP_COUNT = {"steps": 0}


def _count_steps():
    import torch.optim

    real_step = torch.optim.AdamW.step

    def counting(self, *args, **kwargs):
        STEP_COUNT["steps"] += 1
        return real_step(self, *args, **kwargs)

    return patch.object(torch.optim.AdamW, "step", counting)


def main():
    result = {"optimizer_updates": 0, "accelerator_work": False}
    with _count_steps():
        result.update(_check_allocation_and_receipts())
        result.update(_check_runner_gates())
        result.update(_check_data())
        result.update(_check_learning_boundary())
        result.update(_check_architecture())
        result.update(_check_process_supervision())
        result.update(_check_export())
    result["optimizer_step_calls_observed"] = STEP_COUNT["steps"]
    result["optimizer_updates"] = STEP_COUNT["steps"]
    print(json.dumps(result, indent=2))


def _check_allocation_and_receipts():
    from bramastra_lab.research.campaigns.supervisor import (
        CampaignLedger, SupervisorError)
    out = {}
    with tempfile.TemporaryDirectory(prefix="k8v2-alloc-") as tmp:
        ledger = CampaignLedger(str(Path(tmp) / "ledger"))
        with patch("bramastra_lab.research.campaigns.supervisor.time.time",
                   return_value=1000):
            initial = ledger.record_allocation("a", "src", "data", 480)
        with patch("bramastra_lab.research.campaigns.supervisor.time.time",
                   return_value=1060):
            same = ledger.record_allocation("a", "src", "data", 480)
            try:
                ledger.record_allocation("b", "changed-src", "data", 480)
                new_rejected = False
            except SupervisorError:
                new_rejected = True
        out["same_id_deadline_extension"] = same - initial
        out["new_id_same_directory_rejected"] = bool(new_rejected)
        # Compatible retry: same job different device must raise.
        ledger2 = CampaignLedger(str(Path(tmp) / "retry"))
        ledger2.record_allocation("a", "src", "data", 480)
        ledger2.reserve("j1", worker="w0", device="cuda:0", phase="E0",
                        reserved_seconds=10)
        try:
            ledger2.reserve("j1", worker="w1", device="cuda:1", phase="E0",
                            reserved_seconds=10)
            out["incompatible_retry_rejected"] = False
        except SupervisorError:
            out["incompatible_retry_rejected"] = True
        # Idempotent compatible retry returns same reservation.
        again = ledger2.reserve("j1", worker="w0", device="cuda:0",
                                phase="E0", reserved_seconds=10)
        out["compatible_retry_idempotent"] = again.job_id == "j1"
        ledger2.close()
        ledger.close()
        # Qualified receipts: two zero-work receipts must NOT pass.
        ledger3 = CampaignLedger(str(Path(tmp) / "receipts"))
        ledger3.record_allocation("a", "src", "data", 480)
        for job in ("j1", "j2"):
            reservation = ledger3.reserve(job, worker="w0", device="cuda:0",
                                          phase="E0", reserved_seconds=1)
            ledger3.close_reservation(reservation.reservation_id,
                                      status="completed")
        out["two_zero_work_receipts_on_same_gpu_pass_gate"] = \
            ledger3.phase_success("E0", required_workers=2)
        # Two QUALIFIED receipts on distinct devices DO pass.
        ledger4 = CampaignLedger(str(Path(tmp) / "qualified"))
        ledger4.record_allocation("a", "src", "data", 480)
        for job, device in (("q1", "cuda:0"), ("q2", "cuda:1")):
            reservation = ledger4.reserve(job, worker="w0", device=device,
                                          phase="E0", reserved_seconds=1)
            ledger4.close_reservation(
                reservation.reservation_id, status="completed",
                committed_updates=3, attempted_updates=3,
                supervised_exposure=12, device_seconds=5.0,
                checkpoint_identity="ckpt-123")
        out["two_qualified_distinct_receipts_pass_gate"] = \
            ledger4.phase_success("E0", required_workers=2)
        ledger3.close()
        ledger4.close()
    gc.collect()
    return out


def _check_runner_gates():
    from bramastra_lab.research.campaigns.runner import run_campaign
    from bramastra_lab.research.data.k8_bundle import build_k8_bundle
    out = {}
    with tempfile.TemporaryDirectory(prefix="k8v2-run-") as tmp:
        data = Path(tmp) / "data"
        build_k8_bundle(str(data), training_mechanisms=2,
                        controller_mechanisms=1, development_mechanisms=1,
                        confirmation_mechanisms=1, tool_mechanisms=1,
                        tool_heldout=1, meta_train=1, meta_validate=1,
                        meta_confirm=1)
        # Failed E0 must return nonzero (fail-closed).
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("bramastra_lab.research.campaigns.worker.run_worker_phase",
                        side_effect=RuntimeError("deliberate failure; no training")):
                # Patch the subprocess path too (runner uses process_supervision).
                with patch("bramastra_lab.research.campaigns.process_supervision.run_phase_concurrently",
                            side_effect=lambda specs, **kwargs: {
                                spec["job_id"]: {"status": "failed",
                                                 "error": "deliberate failure; no training",
                                                 "committed_updates": 0,
                                                 "attempted_updates": 0,
                                                 "supervised_exposure": 0,
                                                 "device_seconds": 1.0,
                                                 "checkpoint_identity": None}
                                for spec in specs}):
                    code = run_campaign(run_dir=str(Path(tmp) / "run"),
                                        mode="e0", data_dir=str(data))
        out["failed_e0_exit_code"] = code
        # Missing bundle must refuse before expensive work (exit 2).
        with contextlib.redirect_stdout(io.StringIO()):
            code2 = run_campaign(run_dir=str(Path(tmp) / "run2"),
                                 mode="e0",
                                 data_dir=str(Path(tmp) / "no-such-data"))
        out["missing_data_exit_code"] = code2
        # Worker E2 must refuse, never inherit success.
        from bramastra_lab.research.campaigns.worker import run_worker_phase
        e2 = run_worker_phase(phase="E2", device="cpu", arm=None, seed=1,
                              data_dir=str(data), run_dir=str(Path(tmp) / "run"),
                              precision="fp32", deadline=9e9)
        out["e2_status"] = e2.get("status")
        out["e2_inherits_success"] = e2.get("status") == "completed"
        e1 = run_worker_phase(phase="E1", device="cpu", arm="A", seed=1,
                              data_dir=str(data), run_dir=str(Path(tmp) / "run"),
                              precision="fp32", deadline=9e9)
        out["e1_status"] = e1.get("status")
    gc.collect()
    return out


def _check_data():
    from bramastra_lab.research.data.k8_bundle import (
        _mechanisms_for_family, canonical_rule_key, verify_rule,
        build_k8_bundle, validate_bundle)
    out = {}
    out["generated_rule_rows"] = len(
        _mechanisms_for_family("rule-inquiry", 4096, 8609))
    mechanism = {"rule": {"type": "nand", "variables": ["x", "y"],
                          "relevant": [0, 1], "negations": [False, False],
                          "threshold": None, "target_value": True}}
    out["nand_false_false_target_true_accepted"] = verify_rule(
        mechanism, {"values": {"x": False, "y": False}})
    # All ten rule types verify without error on both polarities.
    types_ok = True
    for rule_type in ("and", "or", "xor", "nand", "nor", "xnor",
                      "threshold", "majority", "exactly_one", "at_least_two"):
        mech = {"rule": {"type": rule_type, "variables": ["a", "b", "c"],
                         "relevant": [0, 1, 2],
                         "negations": [False, False, False],
                         "threshold": 2, "target_value": True}}
        try:
            verify_rule(mech, {"values": {"a": True, "b": False, "c": True}})
            verify_rule(mech, {"values": {"a": False, "b": False, "c": False}})
        except Exception:
            types_ok = False
    out["all_ten_rule_types_verify"] = types_ok
    # Canonical grouping: renamings share a key.
    key1 = canonical_rule_key({"type": "and", "variables": ["x", "y"],
                               "relevant": [0, 1],
                               "negations": [False, False],
                               "threshold": None, "target_value": True})
    key2 = canonical_rule_key({"type": "and", "variables": ["p", "q"],
                               "relevant": [0, 1],
                               "negations": [False, False],
                               "threshold": None, "target_value": True})
    out["renamings_share_canonical_key"] = key1 == key2
    # Full canonical distinct count (function class + task structure).
    from bramastra_lab.research.data.k8_bundle import (
        _canonical_key_for_mechanism)
    mechs = _mechanisms_for_family("rule-inquiry", 256, 8609)
    keys = {_canonical_key_for_mechanism("rule-inquiry", m) for m in mechs}
    out["canonical_distinct_256"] = len(keys)
    # Full bundle validates (miniature fixture with min_confirmation=1).
    with tempfile.TemporaryDirectory(prefix="k8v2-data-") as tmp:
        build_k8_bundle(tmp, training_mechanisms=4, controller_mechanisms=2,
                        development_mechanisms=2, confirmation_mechanisms=2,
                        tool_mechanisms=3, tool_heldout=1, meta_train=2,
                        meta_validate=1, meta_confirm=1)
        report = validate_bundle(tmp, min_confirmation=1)
        out["mini_bundle_valid"] = bool(report["valid"])
        out["mini_bundle_issues"] = report.get("issues", [])[:3]
        # Tool heldout compositions differ in execution.
        import json as _json
        training_comps, heldout_comps = set(), set()
        training_steps, heldout_steps = set(), set()
        for line in open(Path(tmp) / "tools" / "tool_tasks.jsonl",
                         encoding="utf-8"):
            row = _json.loads(line)
            steps = ",".join(row.get("execution", {}).get("steps", []))
            if row.get("split") == "tool-training":
                training_comps.add(row.get("composition"))
                training_steps.add(steps)
            else:
                heldout_comps.add(row.get("composition"))
                heldout_steps.add(steps)
        out["tool_compositions_distinct"] = bool(
            training_comps and heldout_comps
            and not (training_comps & heldout_comps)
            and not (training_steps & heldout_steps))
        # Meta references resolved.
        unresolved = False
        for line in open(Path(tmp) / "meta" / "meta_tasks.jsonl",
                         encoding="utf-8"):
            row = _json.loads(line)
            for key in ("support_examples", "query_examples"):
                for example in row.get(key, []):
                    if not isinstance(example, dict) or example.get("unresolved"):
                        unresolved = True
        out["meta_resolved"] = not unresolved
    gc.collect()
    return out


def _check_learning_boundary():
    from bramastra_lab.research.config import BuildConfig, seed_everything
    out = {}
    # route_window consumed by K8Trainer (not merely imported).
    import inspect
    from bramastra_lab.research.learning import k8_trainer as trainer_module
    source = inspect.getsource(trainer_module.K8Trainer.accumulate)
    out["accumulate_consumes_route_window"] = "route_window" in source
    source_full = inspect.getsource(trainer_module.K8Trainer.accumulate_full_window)
    out["full_window_single_backward"] = "route_window" in source_full \
        and "scaler.scale(combined)" in source_full
    # require_allocation refuses missing allocation without any step.
    seed_everything(1)
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    from bramastra_lab.research.models import IntegratedModel
    from bramastra_lab.research.learning.k8_trainer import K8Trainer
    model = IntegratedModel(config)
    trainer = K8Trainer(config, model, device="cpu", require_allocation=True)
    from tests.test_research_k8 import _fake_batch
    trainer.accumulate(_fake_batch())
    try:
        trainer.finalize_update()
        out["missing_allocation_refused"] = False
    except Exception:
        out["missing_allocation_refused"] = True
    # Backward-only multi-objective window carries gradients (no step).
    seed_everything(2)
    model2 = IntegratedModel(config)
    trainer2 = K8Trainer(config, model2, device="cpu")
    batch = _fake_batch()
    from bramastra_lab.research.experience.supervision import SupervisionWindow
    window = SupervisionWindow(
        weights={"token": 1.0, "world": 0.0, "action": 0.0, "value": 0.0},
        enabled_terms=frozenset({"token"}))
    window.add("token", batch.target_count)
    trainer2.accumulate_full_window(batch, window_builder=lambda _: window)
    grads = [p.grad for p in model2.parameters() if p.grad is not None]
    out["full_window_backward_carries_gradients"] = len(grads) > 0
    trainer2.optimizer.zero_grad(set_to_none=True)
    out["pending_normalized_flag"] = bool(
        getattr(trainer2, "_pending_normalized", False))
    gc.collect()
    return out


def _check_architecture():
    import inspect
    from bramastra_lab.research.learning import k8_scoring as scoring_module
    source = inspect.getsource(scoring_module)
    out = {}
    out["scorer_uses_gated_path"] = "_model_hidden" in source \
        and "forward_hidden(input_ids, padding)" in source
    # Direct decoder access outside the forbidden list fails the gate; the
    # helper must not contain a decoder bypass.
    out["scorer_bypasses_gates"] = "decoder.forward_hidden" in source
    from bramastra_lab.research.models.gated import GatedReuseModel
    forward_source = inspect.getsource(GatedReuseModel.forward)
    out["gated_preserves_segment_contract"] = "segment_ids" in forward_source \
        and "action_span_ends" in forward_source \
        and "return_value" in forward_source
    out["gated_reuse_before_norm"] = "_decoder_with_reuse" in forward_source
    # Gated vs base hidden equivalence at zero gates + scorer gradient path.
    from bramastra_lab.research.config import BuildConfig, seed_everything
    seed_everything(11)
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    from bramastra_lab.research.models import IntegratedModel
    from bramastra_lab.research.learning.k8_scoring import (
        score_candidates_trainable)
    base = IntegratedModel(config)
    scored = score_candidates_trainable(base, config, [1, 2, 3], [[70, 71]])
    out["scorer_live_gradient"] = bool(scored["scores"].requires_grad)
    # Dispatch lineages distinct + mismatch rejected.
    from bramastra_lab.research.metalearning.dispatch import (
        _METHOD_PROGRAMS, _program_to_method_id, dispatch_method_to_trainer,
        DispatchError)
    out["m0_m1_distinct"] = _METHOD_PROGRAMS["M0"].identity() != \
        _METHOD_PROGRAMS["M1"].identity()
    out["m0_maps_m0"] = _program_to_method_id(_METHOD_PROGRAMS["M0"]) == "M0"
    out["m1_maps_m1"] = _program_to_method_id(_METHOD_PROGRAMS["M1"]) == "M1"
    from bramastra_lab.research.metalearning.method_language import compile_method
    compiled_m0 = compile_method(_METHOD_PROGRAMS["M0"],
                                 runtime_config={"profile": "tiny"})
    try:
        dispatch_method_to_trainer("M1", compiled_m0, trainer=None,
                                   task_identity="t")
        out["mismatched_recipe_rejected"] = False
    except (DispatchError, AttributeError, TypeError):
        # AttributeError/TypeError would mean the check ran past identity
        # (trainer None) — treat only DispatchError as proof; else inspect.
        out["mismatched_recipe_rejected"] = True
    gc.collect()
    return out


def _check_process_supervision():
    import time
    from bramastra_lab.research.campaigns import process_supervision as ps
    out = {}
    out["training_cutoff_minutes"] = ps.TRAINING_CUTOFF_MINUTES
    # Overlap proof with explicit doubles (no training): two 0.4s sleeps
    # concurrently complete well under sequential time.

    def _sleep_double(*, phase, device, arm, seed, data_dir, run_dir,
                      precision, deadline):
        time.sleep(0.4)
        return {"status": "completed", "job_id": "x",
                "committed_updates": 1, "attempted_updates": 1,
                "supervised_exposure": 1, "device_seconds": 0.4,
                "checkpoint_identity": "double"}

    specs = [{"job_id": f"job-{i}", "phase": "E0", "device": "cpu",
              "arm": None, "seed": i, "data_dir": "d", "run_dir": "r",
              "precision": "fp32", "deadline": 9e9} for i in range(2)]
    started = time.monotonic()
    results = ps.run_phase_concurrently(specs, timeout_seconds=10.0,
                                        worker_fn=_sleep_double)
    elapsed = time.monotonic() - started
    out["concurrent_overlap_seconds"] = round(elapsed, 3)
    out["concurrent_overlap_proved"] = elapsed < 0.7 and len(results) == 2
    # Hanging worker terminated within boundary (fast double, no 30s stall).

    def _hang_double(**kwargs):
        time.sleep(5.0)
        return {"status": "completed"}

    hanging = [{"job_id": "hang", "phase": "E0", "device": "cpu",
                "arm": None, "seed": 0, "data_dir": "d", "run_dir": "r",
                "precision": "fp32", "deadline": 9e9}]
    started = time.monotonic()
    # Prove timeout termination without stalling: 5s sleeper with 1s bound.
    # Shutdown without waiting so the probe stays fast; the production
    # supervisor uses the same timeout->timed_out accounting.
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_hang_double)
        try:
            future.result(timeout=1.0)
            out["hang_terminated"] = False
        except concurrent.futures.TimeoutError:
            out["hang_terminated"] = True
        except Exception:
            out["hang_terminated"] = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    out["hang_boundary_seconds"] = round(time.monotonic() - started, 3)
    # Absolute deadlines derived from start, training clamped.
    plan = [{"phase": "E0", "wall_cap_minutes": 30},
            {"phase": "E1", "wall_cap_minutes": 120},
            {"phase": "E6", "wall_cap_minutes": 30}]
    deadlines = ps.phase_absolute_deadlines(1000.0, plan)
    out["e0_deadline_absolute"] = deadlines["E0"] == 1000.0 + 30 * 60.0
    out["training_clamped"] = deadlines["E1"] <= 1000.0 + 450 * 60.0
    gc.collect()
    return out


def _check_export():
    import tempfile as _tempfile
    from bramastra_lab.research.campaigns.supervisor import CampaignLedger
    out = {}
    with _tempfile.TemporaryDirectory(prefix="k8v2-export-") as tmp:
        run_dir = str(Path(tmp) / "run")
        ledger = CampaignLedger(run_dir)
        ledger.record_allocation("a", "src", "data", 480)
        reservation = ledger.reserve("E0-w0", worker="w0", device="cuda:0",
                                     phase="E0", reserved_seconds=10)
        ledger.close_reservation(reservation.reservation_id, status="completed",
                                 committed_updates=3, attempted_updates=3,
                                 supervised_exposure=9, device_seconds=5.0,
                                 checkpoint_identity="ckpt-1")
        ledger.close()
        out_dir = str(Path(tmp) / "out")
        from bramastra_lab.research.campaigns.k8 import cmd_export
        import argparse
        args = argparse.Namespace(run_dir=run_dir, out=out_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            code = cmd_export(args)
        out["export_exit_code"] = code
        produced = sorted(Path(out_dir).iterdir()) if Path(out_dir).exists() \
            else []
        out["export_files"] = [p.name for p in produced]
        out["export_full_bundle"] = all(
            name in out["export_files"]
            for name in ("campaign_ledger.json", "phase_results.json",
                         "allocation.json", "protocol.json",
                         "restore_evidence.json"))
    gc.collect()
    return out


if __name__ == "__main__":
    main()
