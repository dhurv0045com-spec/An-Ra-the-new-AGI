"""Bounded integrated no-update rehearsal (F22, FINAL-K8 section 23).

One connected engineering rehearsal through the production campaign
control interfaces with explicit NO-STEP optimizer boundaries:

- real campaign ledger and allocation wiring (fake-clock deadline; no
  waiting, no production allocation authority);
- real checkpoint payload publication and parent wiring through the
  production ledger receipts;
- phase executors E1->E5 driven in order on one run directory (fixture
  doubles where local hardware cannot train, explicitly labeled);
- real randomly initialized campaign-geometry model calls (live episode,
  no-step optimizer boundary) with zero optimizer commits;
- statistics consumers fed by real phase outputs;
- the export CLI exercised end-to-end, which must report honestly.

This is an engineering rehearsal, not scientific evidence: every fixture
leg stays labeled, no learned-campaign evidence is produced, and the
result proves connected execution rather than capability.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any


def run_rehearsal(*, repo_root: str | None = None,
                  live_data_dir: str | None = None,
                  timeout_seconds: float = 900.0) -> dict[str, Any]:
    """Execute the bounded rehearsal and return its receipt."""
    started = time.monotonic()
    if repo_root is None:
        repo_root = _repo_root()
    steps: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="bramastra-rehearsal-") as tmp:
        # 1. Real generated bundle with full-threshold confirmation pools.
        from bramastra_lab.research.data.k8_bundle import build_k8_bundle

        data_dir = os.path.join(tmp, "bundle")
        manifest = build_k8_bundle(
            data_dir, training_mechanisms=16, controller_mechanisms=4,
            development_mechanisms=4, confirmation_mechanisms=32,
            tool_mechanisms=8, tool_heldout=4, meta_train=4,
            meta_validate=2, meta_confirm=4)
        steps["bundle"] = {"identity": manifest["identity"],
                           "families": manifest["families"]}

        # 2. Real campaign ledger: one 480-minute fake-clock allocation.
        run_dir = os.path.join(tmp, "run")
        os.makedirs(run_dir, exist_ok=True)
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger

        ledger = CampaignLedger(run_dir)
        ledger.record_allocation(
            "rehearsal-alloc", manifest["identity"],
            manifest["identity"], 600.0)
        ledger.close()
        steps["allocation"] = {"allocation_id": "rehearsal-alloc",
                               "wall_minutes": 600.0,
                               "clock": "fake-deadline-no-waiting"}

        # 3. E1 parent wiring: real tiny payloads through the production
        # checkpoint code and production ledger receipts.
        parent_ids = _wire_e1_parents(run_dir)
        steps["e1_parents"] = parent_ids

        # 4. Phase executors in order on the same run dir (doubles; the
        # executors' own contracts are separately tested).
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.campaigns.phases import e2, e4, e5

        ops = RecordingDoubleOps()
        phase_results: dict[str, Any] = {}

        def _job(phase: str, parent: str | None) -> JobInput:
            return JobInput(phase=phase, slot=0, arm=None, seed=1701,
                            parent=parent, physical_device="cpu",
                            local_device="cpu", data_dir=data_dir,
                            run_dir=run_dir, precision="fp32", deadline=9e9)

        e2_res = e2.execute(
            _job("E2", "E1-B-1701/E1-A-1701"), ops=ops,
            eval_cases=len(e2.EVAL_FAMILIES))
        phase_results["E2"] = {"status": e2_res.status,
                               "error": e2_res.error,
                               "evidence": e2_res.evidence_kind}
        e4_results = {}
        for arm in ("S0", "S1"):
            job = _job("E4", "E1-B-1701")
            job = JobInput(phase="E4", slot=0, arm=arm, seed=1701,
                           parent="E1-B-1701", physical_device="cpu",
                           local_device="cpu", data_dir=data_dir,
                           run_dir=run_dir, precision="fp32", deadline=9e9)
            res = e4.execute(job, ops=ops, update_target=2)
            e4_results[arm] = {"status": res.status, "error": res.error,
                               "evidence": res.evidence_kind}
            if res.status != "completed":
                break
        # Copied per-arm records (never aliased: a cyclic receipt would
        # make the report body unhashable for content_identity).
        phase_results["E4"] = {
            "status": e4_results.get("S1", {}).get("status", "missing"),
            "error": e4_results.get("S1", {}).get("error"),
            "evidence": e4_results.get("S1", {}).get("evidence"),
            "both_arms": {arm: dict(record)
                          for arm, record in e4_results.items()}}
        e5_res = e5.execute(_job("E5", "E1-B-1701"), ops=ops,
                            tasks_per_block=1)
        phase_results["E5"] = {"status": e5_res.status,
                               "error": e5_res.error,
                               "evidence": e5_res.evidence_kind}
        for phase in ("E2", "E4", "E5"):
            if phase_results[phase]["status"] != "completed":
                raise RuntimeError(
                    f"rehearsal phase {phase} failed: "
                    f"{phase_results[phase]['error']}")
        for arm, record in e4_results.items():
            if record["status"] != "completed":
                raise RuntimeError(
                    f"rehearsal E4 arm {arm} failed: {record['error']}")
        steps["phase_executors"] = phase_results

        # 5. Real no-step model legs (zero optimizer commits).
        from bramastra_lab.research.campaigns.verify_build import (
            exercise_live_episode_no_update, exercise_no_update_boundary)

        episode_data_dir = live_data_dir or data_dir
        steps["no_update_boundary"] = exercise_no_update_boundary(
            episode_data_dir)
        steps["live_episode"] = exercise_live_episode_no_update(
            episode_data_dir)

        # 6. Statistics consumers fed by the E2 phase output.
        with open(os.path.join(run_dir, "phase_outputs", "E2", "E2-1701.json"),
                  encoding="utf-8") as handle:
            artifact = json.load(handle)
        deltas = artifact.get("paired_deltas", [])
        if not deltas:
            raise RuntimeError("E2 artifact carries no paired rows for "
                               "the statistics consumer")
        steps["statistics"] = _exercise_statistics(deltas)

        # 7. Export CLI: must run and report honestly (a rehearsal run has
        # fixture receipts, so a COMPLETE success would be a fake).
        export_out = os.path.join(tmp, "export")
        proc = subprocess.run(
            [sys.executable, "-m", "bramastra_lab.research.campaigns.k8",
             "export", "--run-dir", run_dir, "--out", export_out],
            cwd=repo_root, capture_output=True, text=True,
            timeout=timeout_seconds)
        try:
            payload = json.loads(proc.stdout)
        except Exception:
            payload = {"raw": (proc.stdout or "")[-400:]}
        steps["export"] = {"exit_code": proc.returncode,
                           "status": payload.get("status"),
                           "complete": payload.get("complete"),
                           "missing_phases": payload.get("missing_phases"),
                           "honest_incomplete": (
                               payload.get("status") == "EXPORTED_PARTIAL"
                               and payload.get("complete") is False)}
        if not steps["export"]["honest_incomplete"]:
            raise RuntimeError(
                "rehearsal export claimed complete success on a fixture "
                "run; export completeness is not honest")

        # 8. Zero-commit audit across the ledger.
        ledger = CampaignLedger(run_dir)
        try:
            committed = 0
            for phase in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"):
                consumption = ledger.phase_consumption(phase)
                committed += int(consumption.get("committed_updates", 0)
                                 or 0)
        finally:
            ledger.close()
        if committed != 0:
            raise RuntimeError(
                f"rehearsal ledger recorded {committed} committed updates; "
                "the no-update guarantee is violated")
        steps["optimizer_updates_committed"] = committed
    return {"rehearsal": "passed", "duration_seconds": round(
                time.monotonic() - started, 3),
            "steps": steps,
            "note": "engineering rehearsal; not learned-campaign evidence"}


def _wire_e1_parents(run_dir: str) -> dict[str, str]:
    """Publish real E1-A/E1-B tiny payloads + production ledger receipts."""
    import torch

    from bramastra_lab.research.campaigns.supervisor import CampaignLedger
    from bramastra_lab.research.runtime import checkpoint as ckpt

    checkpoints: dict[str, str] = {}
    ledger = CampaignLedger(run_dir)
    try:
        for arm, index in (("A", 1), ("B", 2)):
            manifest = ckpt.publish_checkpoint(
                run_dir=run_dir,
                run_id=f"E1-{arm}-1701", update_index=index,
                payload={"model": {"w": torch.arange(4, dtype=torch.float32)
                                   + (0.0 if arm == "A" else 1.0)},
                         "counters": {"optimizer_updates": index}},
                config_identity="cfg-rehearsal",
                tokenizer_identity="tok-rehearsal",
                data_identity="data-rehearsal",
                parent_checkpoint_id=None, code_identity="code-rehearsal")
            checkpoints[arm] = manifest.checkpoint_id
            reservation = ledger.reserve(
                f"E1-{arm}-1701", worker="rehearsal", device="cpu",
                phase="E1", arm=arm, seed=1701, reserved_seconds=1.0)
            ledger.close_reservation(
                reservation.reservation_id, status="completed",
                committed_updates=0, attempted_updates=0,
                supervised_exposure=0, device_seconds=0.0,
                checkpoint_identity=manifest.checkpoint_id)
    finally:
        ledger.close()
    return {"E1-A": checkpoints["A"], "E1-B": checkpoints["B"],
            "payloads": "real-random-init-zero-commits"}


def _exercise_statistics(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    """Feed real E2 paired rows through the registered statistics consumer.

    Paired cases share the case identity and the ground-truth label; the two
    policies differ in their raw predictions. b_policy_minus_a_fixed encodes
    which policy won the matched episode: candidate (B) wins at +1,
    reference (A) wins at -1, ties are 0.
    """
    from bramastra_lab.research.evaluation.scoring import (
        RawOutcome, clustered_bootstrap_delta)

    def _outcome(outcome_id: str, row: dict[str, Any],
                 *, correct: bool) -> RawOutcome:
        return RawOutcome(
            outcome_id=outcome_id, pool="measurement", split="rehearsal",
            family=str(row["family"]),
            task_semantic_id=str(row["mechanism_id"]),
            prediction="true" if correct else "false",
            stopped_on_eos=False, label="true", cost=1.0,
            case_id=str(row["mechanism_id"]))

    reference = []
    candidate = []
    for row in deltas:
        delta = int(row["b_policy_minus_a_fixed"])
        reference.append(_outcome(
            f"ref-{row['mechanism_id']}", row, correct=(delta == -1)))
        candidate.append(_outcome(
            f"cand-{row['mechanism_id']}", row, correct=(delta == 1)))
    result = clustered_bootstrap_delta(
        reference, candidate, iterations=200, seed=1701)
    return {"consumer": "clustered_bootstrap_delta",
            "rows_consumed": len(deltas),
            "delta_estimate": result.get("delta"),
            "ci": result.get("ci")}


def _repo_root() -> str:
    import bramastra_lab

    package_init = os.path.abspath(bramastra_lab.__file__)
    return os.path.dirname(os.path.dirname(package_init))
