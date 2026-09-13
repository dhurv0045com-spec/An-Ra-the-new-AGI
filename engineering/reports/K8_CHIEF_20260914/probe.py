"""Reproduce K8 control-plane defects without models, learning or CUDA.

Run from the repository root. Worker replacement is explicitly a test double.
Observed failures describe reviewed source 2a1e35a, not experimental results.
"""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from bramastra_lab.research.campaigns.supervisor import CampaignLedger
from bramastra_lab.research.campaigns.runner import run_campaign
from bramastra_lab.research.campaigns.worker import run_worker_phase


def main():
    results = {"source_reviewed": "2a1e35a", "model_instantiations": 0,
               "optimizer_updates": 0, "accelerator_work": False}
    from bramastra_lab.research.data.k8_bundle import _mechanisms_for_family
    mechanisms = _mechanisms_for_family("rule-inquiry", 4096, 8609)
    results["rule_inventory_requested"] = 4096
    results["rule_inventory_returned"] = len(mechanisms)
    from bramastra_lab.research.metalearning.dispatch import MethodArchive
    try:
        MethodArchive(rows=(), cutoff_event_index=0).identity()
    except Exception as exc:
        results["archive_identity_error"] = f"{type(exc).__name__}: {exc}"
    with tempfile.TemporaryDirectory(prefix="k8-chief-") as tmp:
        ledger = CampaignLedger(str(Path(tmp) / "allocation"))
        with patch("bramastra_lab.research.campaigns.supervisor.time.time", return_value=1000):
            first = ledger.record_allocation("same", "source", 480)
        with patch("bramastra_lab.research.campaigns.supervisor.time.time", return_value=1060):
            second = ledger.record_allocation("same", "source", 480)
        results["same_allocation_deadline_extension_seconds"] = second - first
        ledger.close()

        ledger = CampaignLedger(str(Path(tmp) / "jobs"))
        ledger.record_allocation("jobs", "source", 480)
        kwargs = dict(worker="w0", device="cuda:0", phase="E0", arm=None,
                      seed=None, reserved_seconds=1)
        a = ledger.reserve("duplicate-job", **kwargs)
        b = ledger.reserve("duplicate-job", **kwargs)
        results["duplicate_job_admitted_twice"] = a.reservation_id != b.reservation_id
        ledger.close()

        data = Path(tmp) / "data"
        data.mkdir()
        run = str(Path(tmp) / "runner")
        called = []
        def fake_worker(**kwargs):
            called.append(kwargs["phase"])
            if kwargs["phase"] == "E0":
                raise RuntimeError("deliberate E0 failure, no training")
            return {"status": "completed"}
        with patch("bramastra_lab.research.campaigns.worker.run_worker_phase", fake_worker):
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_campaign(run_dir=run, mode="full", data_dir=str(data))
        results["fake_failed_e0_run_exit_code"] = code
        results["phases_called_after_failed_e0"] = sorted(set(called) - {"E0"})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run_campaign(run_dir=run, mode="e0", data_dir=str(data))
        except Exception as exc:
            results["restart_error"] = f"{type(exc).__name__}: {exc}"

        results["real_nonlearning_worker_statuses"] = {}
        for phase in ("E1", "E2", "E3", "E4", "E5", "E6"):
            out = run_worker_phase(phase=phase, device="cpu", arm=None, seed=1701,
                                   data_dir=str(data), run_dir=run, precision="fp32",
                                   deadline=0)
            results["real_nonlearning_worker_statuses"][phase] = out["status"]
        # Reviewed runner does not close its SQLite connection explicitly.
        # Collect unreachable connections before Windows temporary cleanup.
        import gc
        gc.collect()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
