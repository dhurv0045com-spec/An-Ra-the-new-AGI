"""The launch gate must refuse without verified build evidence (F21).

The evidence-bound gate admits only when a valid build report matches the
current source closure. These tests drive the refusal and the admission
paths explicitly instead of depending on repo-level report state.
"""
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

from bramastra_lab.research.campaigns.runner import run_campaign


def _refusal_readiness(**_kwargs):
    return {"ready": False, "blocked_phases": ["E1", "E2", "E3", "E4",
                                               "E5", "E6"],
            "reason": "build evidence unavailable: (test refusal)"}


@pytest.mark.parametrize("mode", ["e0", "full"])
def test_unverified_build_refuses_before_allocation(capsys, mode):
    """Without verified build evidence the gate refuses before any
    allocation authority exists (no ledger, no lease)."""
    with TemporaryDirectory(prefix="k8-launch-gate-") as directory, \
         patch("bramastra_lab.research.campaigns.readiness."
               "implementation_readiness", _refusal_readiness), \
         patch("bramastra_lab.research.campaigns.runner.CampaignLedger",
               side_effect=AssertionError("must not create allocation")), \
         patch("bramastra_lab.research.campaigns.runner.SupervisorLease",
               side_effect=AssertionError("must not acquire lease")):
        tmp_path = Path(directory)
        code = run_campaign(run_dir=str(tmp_path / "run"), mode=mode,
                            data_dir=str(tmp_path / "data"))
        assert code == 2
        assert "IMPLEMENTATION_NOT_READY" in capsys.readouterr().out
        assert not list(tmp_path.rglob("campaign_ledger.sqlite"))


def test_verified_build_admission_proceeds_to_bundle_preflight(capsys):
    """A ready gate no longer blocks: preflight continues to the bundle
    check and refuses on the missing manifest (a DIFFERENT refusal — the
    IMPLEMENTATION_NOT_READY gate itself must not fire)."""
    with TemporaryDirectory(prefix="k8-launch-gate-") as directory, \
         patch("bramastra_lab.research.campaigns.readiness."
               "implementation_readiness",
               lambda **_k: {"ready": True, "blocked_phases": [],
                             "runtime_checks_pending": [{"id": "G01"}]}), \
         patch("bramastra_lab.research.campaigns.runner.CampaignLedger",
               side_effect=AssertionError("must not create allocation")), \
         patch("bramastra_lab.research.campaigns.runner.SupervisorLease",
               side_effect=AssertionError("must not acquire lease")):
        tmp_path = Path(directory)
        code = run_campaign(run_dir=str(tmp_path / "run"), mode="e0",
                            data_dir=str(tmp_path / "data"))
        assert code == 2
        out = capsys.readouterr().out
        assert "IMPLEMENTATION_NOT_READY" not in out
        assert "prepared bundle manifest missing" in out
