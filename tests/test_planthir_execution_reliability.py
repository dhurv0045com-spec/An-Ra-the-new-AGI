import json
import pytest

from anra_v5.planthir.execution import append_update_log, read_update_log, run_session


def test_session_failure_resume_and_identity_binding(tmp_path):
    work = tmp_path / "run"
    with pytest.raises(RuntimeError, match="stopped"):
        with run_session(work, {"seed": 4}) as state:
            state["phase"] = "training"
            latest = work / "arm" / "LATEST"
            latest.parent.mkdir()
            latest.write_text("abc")
            raise RuntimeError("stopped")
    status = json.loads((work / "RUN_STATUS.json").read_text())
    assert status["status"] == "FAILED"
    assert status["phase"] == "training"
    assert status["checkpoint_latest_paths"] == ["arm/LATEST"]
    prior = (work / "RUN_STATUS.json").read_bytes()
    with pytest.raises(ValueError, match="identity mismatch"):
        with run_session(work, {"seed": 5}, resume=True):
            pass
    assert (work / "RUN_STATUS.json").read_bytes() == prior
    with run_session(work, {"seed": 4}, resume=True):
        pass
    assert json.loads((work / "RUN_STATUS.json").read_text())["status"] == "COMPLETE"


def test_reject_unknown_nonempty_directory_and_interrupt(tmp_path):
    (tmp_path / "unrelated").write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        with run_session(tmp_path, {}):
            pass
    work = tmp_path / "other"
    with pytest.raises(KeyboardInterrupt):
        with run_session(work, {}):
            raise KeyboardInterrupt()
    assert json.loads((work / "RUN_STATUS.json").read_text())["status"] == "INTERRUPTED"


def test_journal_quarantine_and_missing_updates(tmp_path):
    path = tmp_path / "updates.jsonl"
    for i in range(1, 4):
        append_update_log(path, {"update": i, "loss": 1 / i})
    assert [r["update"] for r in read_update_log(path, 2)] == [1, 2]
    orphan, = tmp_path.glob("updates.jsonl.orphan-*")
    assert json.loads(orphan.read_text())["update"] == 3
    assert [r["update"] for r in read_update_log(path, 2)] == [1, 2]
    with pytest.raises(ValueError, match="missing"):
        read_update_log(path, 3)
    path.write_text('{"update":1}\n{"update":3}\n')
    with pytest.raises(ValueError, match="sequence"):
        read_update_log(path, 1)


def test_torn_tail_is_quarantined_after_checkpoint(tmp_path):
    import os
    path = tmp_path / "updates.jsonl"
    append_update_log(path, {"update": 1})
    append_update_log(path, {"update": 2})
    with path.open("ab") as handle:
        handle.write(b'{"update":3,"loss":')
        handle.flush()
        os.fsync(handle.fileno())
    assert read_update_log(path, 1) == [{"update": 1}]
    orphan, = tmp_path.glob("updates.jsonl.orphan-*")
    assert orphan.read_bytes() == b'{"update":2}\n{"update":3,"loss":'
    assert path.read_bytes() == b'{"update":1}\n'
    path.write_bytes(b'{"update":1}\n{"update":')
    with pytest.raises(ValueError, match="sequence"):
        read_update_log(path, 2)
    path.write_bytes(b'{"update":1}\n{bad}\n')
    with pytest.raises(ValueError, match="sequence"):
        read_update_log(path, 1)


def test_timeout_is_interrupted_and_resume_preserves_attempt(tmp_path):
    with pytest.raises(TimeoutError):
        with run_session(tmp_path, {"seed": 1}):
            raise TimeoutError("deadline")
    old = json.loads((tmp_path / "RUN_STATUS.json").read_text())
    assert old["status"] == "INTERRUPTED"
    with run_session(tmp_path, {"seed": 1}, resume=True):
        pass
    assert json.loads((tmp_path / "ATTEMPTS.jsonl").read_text()) == old
    assert json.loads((tmp_path / "RUN_STATUS.json").read_text())["status"] == "COMPLETE"
