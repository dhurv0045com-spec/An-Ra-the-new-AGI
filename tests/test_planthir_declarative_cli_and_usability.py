"""Stage 6 verification: declarative data-only spec CLI, fresh-agent E2E workflow, and productivity study analyzer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from anra_v5.planthir import __main__ as cli
from anra_v5.planthir.productivity import (
    SIX_BENCHMARK_TASKS,
    PairedTaskObservation,
    analyze_productivity_study,
    default_protocol_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cli(args: list[str], capsys) -> tuple[int, dict]:
    code = cli.main(args)
    return code, json.loads(capsys.readouterr().out)


def _subproc_cli(args: list[str], *, extra_env: dict[str, str] | None = None) -> tuple[int, dict | None, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-m", "anra_v5.planthir", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = None
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except Exception:
            parsed = None
    return proc.returncode, parsed, proc.stderr


def test_schema_init_and_data_only_spec_security_and_confounds(tmp_path: Path, capsys) -> None:
    code, schema = _cli(["schema"], capsys)
    assert code == 0 and schema["schema"] == "anra-planthir-spec-v1"
    assert set(schema["backends"]) >= {"v8", "x1_rd2"}

    x1_spec_path = tmp_path / "x1_spec.json"
    code, out_init = _cli(["init", "--backend", "x1_rd2", "--out", str(x1_spec_path)], capsys)
    assert code == 0 and x1_spec_path.is_file()
    code, val_ok = _cli(["validate", "--spec", str(x1_spec_path)], capsys)
    assert code == 0 and val_ok["valid"] is True

    v8_spec_path = tmp_path / "v8_spec.json"
    code, _ = _cli(["init", "--backend", "v8", "--out", str(v8_spec_path)], capsys)
    assert code == 0
    code, val_v8 = _cli(["validate", "--spec", str(v8_spec_path)], capsys)
    assert code == 0 and val_v8["valid"] is True

    base_doc = json.loads(v8_spec_path.read_text(encoding="utf-8"))

    # 1. Unknown field rejected
    bad_unknown = dict(base_doc, illegal_hook="bad")
    bad_path = tmp_path / "bad_unknown.json"
    bad_path.write_text(json.dumps(bad_unknown), encoding="utf-8")
    code, res = _cli(["validate", "--spec", str(bad_path)], capsys)
    assert code == 1 and any(e["code"] == "UNKNOWN_FIELD" for e in res["errors"])

    # 2. Eval / shell fragment rejected
    bad_eval = json.loads(json.dumps(base_doc))
    bad_eval["program"] = "eval('__import__(\"os\")')"
    bad_eval_path = tmp_path / "bad_eval.json"
    bad_eval_path.write_text(json.dumps(bad_eval), encoding="utf-8")
    code, res = _cli(["validate", "--spec", str(bad_eval_path)], capsys)
    assert code == 1 and any(e["code"] == "FORBIDDEN_CODE_OR_SHELL_FRAGMENT" for e in res["errors"])

    # 3. Dotted Python import reference rejected
    bad_import = json.loads(json.dumps(base_doc))
    bad_import["program"] = "anra_v5.planthir.backend:V8PlanthirBackend"
    bad_import_path = tmp_path / "bad_import.json"
    bad_import_path.write_text(json.dumps(bad_import), encoding="utf-8")
    code, res = _cli(["validate", "--spec", str(bad_import_path)], capsys)
    assert code == 1 and any(e["code"] == "FORBIDDEN_DOTTED_IMPORT" for e in res["errors"])

    # 4. Confounded precision rejected & explained cleanly via explain-error
    bad_prec = json.loads(json.dumps(base_doc))
    bad_prec["treatment_arms"][0]["recipe"]["precision"] = "fp16"
    bad_prec_path = tmp_path / "bad_prec.json"
    bad_prec_path.write_text(json.dumps(bad_prec), encoding="utf-8")
    code, res = _cli(["validate", "--spec", str(bad_prec_path)], capsys)
    assert code == 1 and any("PRECISION_MISMATCH" in str(e) for e in res["errors"])

    code, exp = _cli(["explain-error", "--code", "PRECISION_MISMATCH"], capsys)
    assert code == 0 and exp["known"] is True and len(exp["remediation"]) >= 1


def test_fresh_agent_declarative_cli_workflow_e2e(tmp_path: Path) -> None:
    spec_path = tmp_path / "workflow_spec.json"
    work_dir = tmp_path / "work"
    once_file = tmp_path / "journal.once"
    replay_out = tmp_path / "CLI_REPLAY_RECEIPT.json"

    # 1. init -> validate
    rc, out_init, err = _subproc_cli(["init", "--backend", "x1_rd2", "--out", str(spec_path)])
    assert rc == 0, err
    assert out_init["status"] == "TEMPLATE_WRITTEN"

    rc, out_val, err = _subproc_cli(["validate", "--spec", str(spec_path), "--canary"])
    assert rc == 0 and out_val["valid"] is True, err

    # 2. run interrupted via real os._exit fault injection -> inspect-run -> resume
    fault_env = {
        "PLANTHIR_FAULT_INJECTION_POINT": "after_journal_append",
        "PLANTHIR_FAULT_ONCE_FILE": str(once_file),
        "PLANTHIR_FAULT_EXIT_CODE": "99",
    }
    rc, _, _ = _subproc_cli(["run", "--spec", str(spec_path), "--work-root", str(work_dir), "--canary"], extra_env=fault_env)
    assert rc == 99

    rc, out_insp, err = _subproc_cli(["inspect-run", "--work-root", str(work_dir)])
    assert rc == 0, err
    assert out_insp["provenance_identity_present"] is True

    # Running without --resume must fail closed with RESUME_REQUIRED explanation
    rc, out_no_res, _ = _subproc_cli(["run", "--spec", str(spec_path), "--work-root", str(work_dir), "--canary"])
    assert rc == 1 and out_no_res["error"] == "RESUME_REQUIRED"

    # 3. resume -> verify-bundle -> replay-bundle
    rc, out_res, err = _subproc_cli(["resume", "--spec", str(spec_path), "--work-root", str(work_dir), "--canary"])
    assert rc == 0 and out_res["status"] == "COMPLETE" and out_res["gates_passed"] is True, err

    bdir = work_dir / "result_bundle"
    rc, out_ver, err = _subproc_cli(["verify-bundle", "--bundle-dir", str(bdir)])
    assert rc == 0 and out_ver["verified"] is True and out_ver["custody"]["verified"] is True, err

    rc, out_rep, err = _subproc_cli(
        ["replay-bundle", "--bundle-dir", str(bdir), "--output", str(replay_out)]
    )
    assert rc == 0 and out_rep["verified"] is True and out_rep["independent_environment"] is True, err
    assert replay_out.is_file()


def test_legacy_cli_gates_and_productivity_study_honesty(capsys) -> None:
    code, out_val = _cli(["run-validation-study"], capsys)
    assert code == 1 and out_val["error"] == "legacy_launcher_deprecated"
    assert len(out_val["migrate_to"]) >= 3

    code, out_gpu = _cli(["run-gpu-validation"], capsys)
    assert code == 1 and out_gpu["error"] == "legacy_launcher_not_qualified"

    # Pending protocol never fabricates efficiency %
    manifest = default_protocol_manifest()
    assert manifest["status"] == "STUDY_OBSERVATIONS_PENDING"
    assert manifest["claimed_efficiency_percent"] is None
    assert len(manifest["tasks"]) == 6
    empty_res = analyze_productivity_study([])
    assert empty_res["status"] == "STUDY_OBSERVATIONS_PENDING"
    assert empty_res["claimed_efficiency_percent"] is None

    # Counterbalanced complete 6-task analysis succeeds accurately
    obs = [
        PairedTaskObservation(
            task_id=task["task_id"],
            participant_id=f"P{idx % 2}",
            order="baseline_first" if idx % 2 == 0 else "planthir_first",
            baseline_tokens=10_000,
            planthir_tokens=3_000,
            baseline_wall_seconds=120.0,
            planthir_wall_seconds=30.0,
            baseline_quality_score=0.90,
            planthir_quality_score=0.95,
            planthir_passed=True,
        )
        for idx, task in enumerate(SIX_BENCHMARK_TASKS)
    ]
    analyzed = analyze_productivity_study(obs)
    assert analyzed["passed"] is True and analyzed["status"] == "STUDY_ANALYZED"
    assert analyzed["mean_token_reduction_fraction"] == 0.7
    assert analyzed["mean_wall_time_reduction_fraction"] == 0.75
    assert analyzed["mean_quality_delta"] > 0.0


def test_cli_docs_align_with_live_declarative_schema() -> None:
    from anra_v5.planthir.agent_tools import planthir_spec_schema

    agent_doc = (REPO_ROOT / "agent.md").read_text(encoding="utf-8")
    schema_cmds = set(planthir_spec_schema()["cli_subcommands"])
    for required_cmd in (
        "schema",
        "init",
        "validate",
        "run",
        "resume",
        "inspect-run",
        "verify-bundle",
        "replay-bundle",
        "explain-error",
    ):
        assert required_cmd in schema_cmds
        assert required_cmd in agent_doc
    assert "verify-bundle --dir" not in agent_doc

