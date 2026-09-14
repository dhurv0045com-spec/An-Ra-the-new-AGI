"""Known incomplete executors must not consume the owner's E0 allowance."""
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

from bramastra_lab.research.campaigns.runner import run_campaign


@pytest.mark.parametrize("mode", ["e0", "full"])
def test_unqualified_implementation_refuses_before_allocation(capsys, mode):
    with TemporaryDirectory(prefix="k8-launch-gate-") as directory, \
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
