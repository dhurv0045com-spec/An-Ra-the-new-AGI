import json
from dataclasses import fields

from anra_v5.planthir import __main__ as cli
from anra_v5.planthir.cartridge import ExperimentCartridge
from anra_v5.planthir.execution import run_session


def call(args, capsys):
    code = cli.main(args)
    return code, json.loads(capsys.readouterr().out)


def test_agent_info_is_derived_and_read_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_fault_injection_suite", lambda: (_ for _ in ()).throw(AssertionError("training called")))
    code, result = call(["agent-info"], capsys)
    assert code == 0 and result["schema"] == "anra-planthir-agent-info-v1"
    assert [x["name"] for x in result["inputs"]["ExperimentCartridge"]] == [f.name for f in fields(ExperimentCartridge)]
    assert "NOT_QUALIFIED" in result["legacy_commands"]["run-gpu-validation"]
    assert not list(tmp_path.iterdir())


def test_inspect_is_read_only_and_reports_unverified(tmp_path, capsys):
    work = tmp_path / "work"
    with run_session(work, {"test": 1}):
        pass
    before = {p.name: p.read_bytes() for p in work.iterdir() if p.is_file()}
    code, result = call(["inspect-run", "--work-root", str(work)], capsys)
    assert code == 0 and result["status"]["status"] == "COMPLETE"
    assert result["verification"] == "NOT_CHECKED" and not result["final_receipt_present"]
    assert before == {p.name: p.read_bytes() for p in work.iterdir() if p.is_file()}


def test_inspect_missing_and_corrupt_metadata_are_localized(tmp_path, capsys):
    work = tmp_path / "missing"
    code, result = call(["inspect-run", "--work-root", str(work)], capsys)
    assert code == 1 and result["error"]["file"] == "RUN_IDENTITY.json" and not work.exists()
    with run_session(work, {"test": 1}):
        pass
    (work / "RUN_STATUS.json").write_text("{")
    code, result = call(["inspect-run", "--work-root", str(work)], capsys)
    assert code == 1 and result["error"]["file"] == "RUN_STATUS.json"


def test_legacy_gpu_command_stops_before_execution(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_exp_004_live_gpu_cartridge_validation", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("launch called")))
    code, result = call(["run-gpu-validation"], capsys)
    assert code == 1 and result["error"] == "legacy_launcher_not_qualified"


def test_inspect_nonfinite_receipt_is_localized(tmp_path, capsys):
    with run_session(tmp_path / 'work', {'test': 1}):
        pass
    (tmp_path / 'work' / 'FINAL_RECEIPT.json').write_text('{"cost": NaN}')
    code, result = call(['inspect-run', '--work-root', str(tmp_path / 'work')], capsys)
    assert code == 1 and result['error']['file'] == 'FINAL_RECEIPT.json'
