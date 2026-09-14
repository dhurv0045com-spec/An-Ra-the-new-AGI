"""Integration counterexamples for 30d0fcd. No models or optimizer work."""
import json
import os
import sys
import tempfile
import types
import time
from pathlib import Path
from unittest.mock import patch
from bramastra_lab.research.campaigns.runner import _phase_plan
from bramastra_lab.research.campaigns.supervisor import CampaignLedger, SupervisorLease
from bramastra_lab.research.campaigns.process_supervision import _child_execute, _run_one_in_process


def finite_worker(**kwargs):
    time.sleep(0.5)
    Path(kwargs["run_dir"], "worker-survived.txt").write_text("finished", encoding="utf-8")
    return {"status": "completed"}


def main():
    out = {"source_reviewed": "30d0fcd", "model_instantiations": 0,
           "optimizer_updates": 0, "accelerator_work": False}
    with tempfile.TemporaryDirectory(prefix="k8-v2-chief-") as tmp:
        ledger = CampaignLedger(str(Path(tmp) / "slots"))
        deadline = ledger.record_allocation("a", "s", "d", 480)
        e1 = next(p for p in _phase_plan("full", deadline, ["cuda:0", "cuda:1"])
                  if p["phase"] == "E1")
        admitted = []
        for job in e1["workers"]:
            try:
                ledger.reserve(job["job_id"], worker=job["worker"], device=job["device"],
                               phase="E1", arm=job["arm"], seed=job["seed"],
                               reserved_seconds=job["wall_seconds"])
                admitted.append(job["job_id"])
            except Exception as exc:
                out["e1_reservation_error"] = f"{type(exc).__name__}: {exc}"
                break
        out["e1_jobs_reserved_together"] = admitted
        ledger.close()
        started = time.monotonic()
        try:
            _run_one_in_process({"phase": "E0", "device": "cpu", "arm": None,
                                 "seed": 1, "data_dir": tmp, "run_dir": tmp,
                                 "precision": "fp32", "deadline": 9e9},
                                timeout_seconds=0.05,
                                worker_fn_path="engineering.reports.K8_V2_CHIEF_20260914.probe:finite_worker")
        except Exception as exc:
            out["timeout_error"] = f"{type(exc).__name__}: {exc}"
        out["timeout_elapsed_seconds"] = round(time.monotonic() - started, 3)
        # Finite child writes after 0.5 seconds; keep its sandbox alive long
        # enough to distinguish returning early from actually terminating it.
        time.sleep(1.5)
        out["worker_finished_after_timeout"] = Path(tmp, "worker-survived.txt").exists()
        lockdir = Path(tmp) / "lease"
        lockdir.mkdir()
        lease = SupervisorLease(str(lockdir))
        lease.acquire()
        try:
            with patch.object(SupervisorLease, "_pid_alive", return_value=True):
                with patch.object(SupervisorLease, "_lock_probe", return_value=(f"{os.getpid()}:0", 0)):
                    try:
                        SupervisorLease(str(lockdir)).acquire()
                        out["old_live_lease_reclaimed"] = True
                    except Exception:
                        out["old_live_lease_reclaimed"] = False
        finally:
            lease.release()
    fake = types.ModuleType("k8_device_probe")
    fake.run = lambda **kwargs: {"status": "completed", "received_device": kwargs["device"]}
    with patch.dict(sys.modules, {"k8_device_probe": fake}), patch.dict(os.environ):
        result = _child_execute({"phase": "E0", "device": "cuda:1", "arm": None,
                                 "seed": 1, "data_dir": "unused", "run_dir": "unused",
                                 "precision": "fp32", "deadline": 9e9}, "k8_device_probe:run")
        out["worker_device_after_visibility_isolation"] = result["received_device"]
        out["cuda_visible_devices"] = result["supervision"]["visible_devices"]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
