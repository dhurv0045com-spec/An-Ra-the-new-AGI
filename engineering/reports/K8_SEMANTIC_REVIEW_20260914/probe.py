"""Exercise executor claim boundaries at 5c0567f without model training."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from bramastra_lab.research.campaigns.phases.ops import RecordingDoubleOps
from bramastra_lab.research.campaigns.phases.types import JobInput
from bramastra_lab.research.campaigns.phases import e5, e6
from bramastra_lab.research.campaigns.supervisor import CampaignLedger
from bramastra_lab.research.runtime import checkpoint


class NoTrainingOps(RecordingDoubleOps):
    def training_update(self, *args, **kwargs):
        raise AssertionError("No optimizer work authorized in this diagnostic")


def main():
    out = {"source_reviewed": "5c0567f", "model_instantiations": 0,
           "optimizer_updates": 0, "accelerator_work": False}
    with tempfile.TemporaryDirectory(prefix="k8-semantics-") as tmp:
        data = Path(tmp) / "data"
        data.mkdir()
        (data / "manifest.json").write_text("{}", encoding="utf-8")
        run = Path(tmp) / "run"
        run.mkdir()
        ops = NoTrainingOps()
        job = JobInput(phase="E5", slot=0, arm=None, seed=1701, parent=None,
                       physical_device="cpu", local_device="cpu", data_dir=str(data),
                       run_dir=str(run), precision="fp32", deadline=9e9)
        result = e5.execute(job, ops=ops, tasks_per_block=1)
        out["e5_status_with_training_forbidden"] = result.status
        out["e5_reported_committed_updates"] = result.committed_updates
        out["e5_training_calls"] = sum(name == "training_update" for name, _ in ops.calls)
        ledger = CampaignLedger(str(run))
        ledger.record_allocation("probe", "source", "data", 480)
        for seed in (1701, 1702):
            r = ledger.reserve(f"E1-B-{seed}", worker=str(seed), device=str(seed),
                               phase="E1", arm="B", seed=seed, reserved_seconds=1)
            ledger.close_reservation(r.reservation_id, status="completed",
                                     committed_updates=1, attempted_updates=1,
                                     supervised_exposure=1, device_seconds=1,
                                     checkpoint_identity=f"nonexistent-{seed}")
        ledger.close()
        export_job = JobInput(phase="E6", slot=0, arm=None, seed=None, parent=None,
                              physical_device="cpu", local_device="cpu", data_dir=str(data),
                              run_dir=str(run), precision="fp32", deadline=9e9)
        with contextlib.redirect_stdout(io.StringIO()):
            result = e6.execute(export_job)
        out["e6_status_without_payloads"] = result.status
        out["payload_files_present"] = len(list(run.rglob("*.pt")))
    out["publish_checkpoint_api_exists"] = hasattr(checkpoint, "publish_checkpoint")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
