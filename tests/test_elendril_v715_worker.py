"""V7.15 frozen registration and safety contracts; no GPU training."""
from anra_v5.elendril.v715_protocol import PROTOCOL_SHA, registered_jobs, protocol
from anra_v5.elendril.v71_protocol import jobs as old_jobs
from tools.elendril_v715_worker import REPO, run


def test_registration_reuses_nine_real_v71_checkpoints():
    names = [x["job_id"] for x in registered_jobs()]
    assert names == [x["job_id"] for x in old_jobs("core")]
    assert len(names) == 9 and len(set(names)) == 9
    assert len(PROTOCOL_SHA) == 64
    assert protocol()["limits"]["new_optimizer_updates"] == 0


def test_invalid_request_rejected_before_cuda_or_checkpoint():
    try:
        run({"protocol_sha": "0"*64, "job": {}})
    except ValueError as e:
        assert "protocol mismatch" in str(e)
    else:
        raise AssertionError("unregistered request accepted")


def test_checkout_identity_exists():
    assert (REPO / "anra_v5" / "elendril" / "v715_diagnostics.py").is_file()
