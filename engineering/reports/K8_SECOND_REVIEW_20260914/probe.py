"""Zero-learning diagnostics for 3914dc4; run from repository root."""
import contextlib
import gc
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from bramastra_lab.research.campaigns.supervisor import CampaignLedger
from bramastra_lab.research.campaigns.runner import run_campaign
from bramastra_lab.research.data.k8_bundle import verify_rule, _mechanisms_for_family


def main():
    result = {"source_reviewed": "3914dc4", "optimizer_updates": 0,
              "model_instantiations": 0, "accelerator_work": False}
    with tempfile.TemporaryDirectory(prefix="k8-second-") as tmp:
        ledger = CampaignLedger(str(Path(tmp) / "ledger"))
        with patch("bramastra_lab.research.campaigns.supervisor.time.time", return_value=1000):
            initial = ledger.record_allocation("a", "src", "data", 480)
        with patch("bramastra_lab.research.campaigns.supervisor.time.time", return_value=1060):
            same = ledger.record_allocation("a", "src", "data", 480)
            changed = ledger.record_allocation("b", "changed-src", "data", 480)
        result["same_id_deadline_extension"] = same - initial
        result["new_id_same_directory_deadline_extension"] = changed - initial
        ledger.close()
        ledger = CampaignLedger(str(Path(tmp) / "receipts"))
        ledger.record_allocation("a", "src", "data", 480)
        for job in ("j1", "j2"):
            reservation = ledger.reserve(job, worker="w0", device="cuda:0", phase="E0",
                                         reserved_seconds=1)
            ledger.close_reservation(reservation.reservation_id, status="completed")
        result["two_zero_work_receipts_on_same_gpu_pass_gate"] = ledger.phase_success("E0", required_workers=2)
        ledger.close()
        data = Path(tmp) / "data"
        data.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("bramastra_lab.research.campaigns.worker.run_worker_phase",
                       side_effect=RuntimeError("deliberate failure; no training")):
                result["failed_e0_exit_code"] = run_campaign(
                    run_dir=str(Path(tmp) / "run"), mode="e0", data_dir=str(data))
        gc.collect()
    result["generated_rule_rows"] = len(_mechanisms_for_family("rule-inquiry", 4096, 8609))
    mechanism = {"rule": {"type": "nand", "variables": ["x", "y"], "target_value": True}}
    result["nand_false_false_target_true_accepted"] = verify_rule(
        mechanism, {"values": {"x": False, "y": False}})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
