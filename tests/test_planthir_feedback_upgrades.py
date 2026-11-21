"""Tests for upstream Palantiri upgrades motivated by Avendi Y3 & Elendril X3 feedback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch
import torch.nn as nn

from anra_v5.planthir import (
    CampaignOrchestrator,
    LogitsPolicy,
    SessionJob,
    StaticPromptKernel,
    UniversalAction,
    batched_rollout,
    build_pinned_kaggle_notebook,
    safe_extract_planthir_archive,
    win_long_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_win_long_path_and_safe_extract_over_265_chars(tmp_path: Path) -> None:
    src_zip = tmp_path / "nested.zip"
    payload = json.dumps({"ok": True, "len": 280})
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr("jobs/j0/JOB_RECEIPT.json", payload)

    deep_dest = tmp_path / ("a" * 60) / ("b" * 60) / ("c" * 60) / ("d" * 60) / "out"
    assert len(str(deep_dest)) > 265
    lp = win_long_path(deep_dest)
    assert lp == win_long_path(lp)
    if os.name == "nt":
        assert str(lp).startswith("\\\\?\\")

    extracted = safe_extract_planthir_archive(src_zip, deep_dest)
    assert extracted == ["jobs/j0/JOB_RECEIPT.json"]
    receipt_file = win_long_path(deep_dest / "jobs" / "j0" / "JOB_RECEIPT.json")
    assert receipt_file.is_file()
    assert json.loads(receipt_file.read_text(encoding="utf-8"))["len"] == 280


def test_cli_backend_module_flag_without_env_var(tmp_path: Path) -> None:
    plug_dir = tmp_path / "cli_plug_pkg"
    plug_dir.mkdir(parents=True, exist_ok=True)
    (plug_dir / "my_cli_plug.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import torch, torch.nn as nn, torch.nn.functional as F",
                "from anra_v5.planthir import ExperimentModule, GenericDatasetBundle",
                "",
                "class TinyPlugNet(nn.Module):",
                "    def __init__(self, width: int = 8, seed: int = 7) -> None:",
                "        super().__init__()",
                "        torch.manual_seed(seed)",
                "        self.fc1 = nn.Linear(4, width)",
                "        self.fc2 = nn.Linear(width, 2)",
                "    def forward(self, x: torch.Tensor):",
                "        h = torch.tanh(self.fc1(x))",
                "        return self.fc2(h), h",
                "",
                "def _ds(cfg, cpu_canary):",
                "    def _mk(pfx, n):",
                "        return [{'world_id': f'{pfx}_{i}', 'sig': f'{pfx}_{i}', 'hops': 1, 'feat': [1.0 if i % 2 == 0 else -1.0, 0.3, -0.2, 0.1], 'label': i % 2, 'hidden': {'expected': 'A' if i % 2 == 0 else 'B'}} for i in range(n)]",
                "    tr, ev, mh, pr = _mk('tr', 8), _mk('ev', 4), _mk('mh', 2), _mk('pr', 4)",
                "    return {'train_data': GenericDatasetBundle(dataset_name='cli_p', items=tuple(tr), data_identity='cli-p-v1', manifest={'n': len(tr)}), 'train_worlds': tr, 'eval_worlds': ev, 'multihop_worlds': mh, 'probe_train_worlds': pr}",
                "",
                "def _loss(m, chunk, recipe):",
                "    x = torch.tensor([c['feat'] for c in chunk], dtype=torch.float32)",
                "    y = torch.tensor([c['label'] for c in chunk], dtype=torch.int64)",
                "    return F.cross_entropy(m(x)[0], y)",
                "",
                "module = ExperimentModule(",
                "    backend_id='cli_plug_exp', architectures=('base', 'trt'), objectives=('ce',),",
                "    model_factory=lambda a, c, s: TinyPlugNet(int(c.get('width', 8)), s),",
                "    dataset_factory=_ds, loss_fn=_loss, action_labels=('A', 'B'),",
                "    default_model_config={'width': 8}, treatment_model_config={'width': 8}, extra_model_keys=('width',),",
                ").register()",
                "",
                "def register_planthir_backends():",
                "    return module",
            ]
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.pop("PLANTHIR_BACKEND_MODULES", None)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{plug_dir}"

    def _run(args: list[str]) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "anra_v5.palantiri", *args],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"args={args}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        return json.loads(proc.stdout)

    spec_p = tmp_path / "spec.json"
    _run(["--backend-module", "my_cli_plug", "init", "--backend", "cli_plug_exp", "--out", str(spec_p)])
    val = _run(["validate", "--backend-module", "my_cli_plug", "--spec", str(spec_p), "--canary"])
    assert val["valid"] is True
    work_dir = tmp_path / "run_cli"
    run_res = _run(
        ["--backend-module", "my_cli_plug", "run", "--spec", str(spec_p), "--work-root", str(work_dir), "--canary", "--device", "cpu"]
    )
    assert run_res["gates_passed"] is True
    rep = _run(
        ["replay-bundle", "--backend-module", "my_cli_plug", "--bundle-dir", str(work_dir / "result_bundle")]
    )
    assert rep["verified"] is True and rep["prediction_agreement_status"] == "MATCHED"


def test_run_registered_plan_gates_blocked_parents_preemption_multigroup_and_u0000(tmp_path: Path) -> None:
    commit = "e" * 40
    # 1. stage_gates LOCKED + BLOCKED_MISSING_PARENT isolation + intra-session depends_on ordering
    jobs_a = [
        SessionJob("j_t0", "ARCH_A", 7, 7, 10, session="S0", extra={"stage": "T0"}),
        SessionJob("j_t1", "ARCH_A", 19, 19, 10, session="S0", extra={"stage": "T1", "depends_on": ["j_t0"]}),
        SessionJob("j_bad_parent", "ARCH_A", 41, 41, 10, session="S0", extra={"stage": "T1", "depends_on": ["missing_p"]}),
        SessionJob("j_s50", "ARCH_B", 7, 7, 10, session="S1", extra={"stage": "S50"}),
    ]
    exec_order: list[str] = []

    def _runner_a(job: SessionJob, wroot: Path, dev: str) -> dict:
        exec_order.append(job.job_id)
        return {"exit_code": 0, "completed_updates": job.updates, "eval_rows": 16, "metric_score": 0.25}

    orch_a = CampaignOrchestrator(campaign_id="X3-GATE-LOCK", source_commit=commit, work_dir=tmp_path / "gate_lock", min_disk_bytes=1024)
    res_lock = orch_a.run_registered_plan(
        jobs_a,
        job_runner=_runner_a,
        devices=("cuda:0", "cuda:1"),
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        stage_gates={
            "S50": lambda ctx: (
                ctx["receipts_by_id"].get("j_t0", {}).get("metric_score", 0.0) >= 0.6,
                f"t0_score={ctx['receipts_by_id'].get('j_t0', {}).get('metric_score', 0.0)}",
            )
        },
        u0000_smoke_fn=lambda arch, rep, sdir: {"passed": True, "arch": arch},
    )
    assert exec_order == ["j_t0", "j_t1"]
    assert res_lock["status"] == "PARTIAL"
    assert res_lock["blocked_job_ids"] == ["j_bad_parent"]
    assert res_lock["blocked_jobs"][0]["status"] == "BLOCKED_MISSING_PARENT"
    assert res_lock["stage_gate_results"]["S50"]["status"] == "GATED_LOCKED"
    assert res_lock["gated_locked_job_ids"] == ["j_s50"]
    assert (tmp_path / "gate_lock" / "BLOCKED_JOBS.json").is_file()
    assert (tmp_path / "gate_lock" / "STAGE_GATES.json").is_file()
    assert res_lock["u0000_smoke_receipt"]["passed"] is True

    # 2. stage_gates UNLOCKED when threshold is met
    orch_b = CampaignOrchestrator(campaign_id="X3-GATE-UNLOCK", source_commit=commit, work_dir=tmp_path / "gate_unlock", min_disk_bytes=1024)
    res_unlock = orch_b.run_registered_plan(
        [jobs_a[0], jobs_a[1], jobs_a[3]],
        job_runner=lambda j, w, d: {"exit_code": 0, "completed_updates": j.updates, "eval_rows": 16, "metric_score": 0.85},
        devices=("cuda:0", "cuda:1"),
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        stage_gates={"S50": lambda ctx: (ctx["receipts_by_id"]["j_t0"]["metric_score"] >= 0.6, "t0>=0.6")},
    )
    assert res_unlock["status"] == "COMPLETE"
    assert res_unlock["stage_gate_results"]["S50"]["status"] == "UNLOCKED"
    assert set(res_unlock["completed_jobs"]) == {"j_t0", "j_t1", "j_s50"}

    # 3. Live inter-wave wall-clock preemption (clock_fn jumps into reserve) + Session 2 resume
    pre_jobs = [SessionJob(f"p{i}", "ARCH_A", 7 + i, 7 + i, 25, session="S0") for i in range(4)]
    now_sec = [0.0]

    def _wave_clock_runner(job: SessionJob, wroot: Path, dev: str) -> dict:
        if job.job_id == "p0":
            now_sec[0] = 260.0 * 60.0  # Elapsed becomes 260m; remaining = 300 - 260 = 40m <= reserve(45m)
        return {"exit_code": 0, "completed_updates": job.updates, "eval_rows": 16}

    res_pre = CampaignOrchestrator(
        campaign_id="X3-PREEMPT",
        source_commit=commit,
        work_dir=tmp_path / "pre_s1",
        max_minutes=300.0,
        reserve_minutes=45.0,
        min_disk_bytes=1024,
    ).run_registered_plan(
        pre_jobs,
        job_runner=_wave_clock_runner,
        devices=("cuda:0", "cuda:1"),
        measured_seconds_per_update=1.0,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        clock_fn=lambda: now_sec[0],
        export_prefix="PRE1",
    )
    assert res_pre["status"] == "PARTIAL"
    assert res_pre["wall_clock_preempted"] is True
    assert "WALL_CLOCK_RESERVE_REACHED" in str(res_pre["preemption_reason"])
    assert res_pre["dispatched_job_ids"] == ["p0", "p1"]
    assert res_pre["deferred_pending_job_ids"] == ["p2", "p3"]

    res_pre_s2 = CampaignOrchestrator(
        campaign_id="X3-PREEMPT",
        source_commit=commit,
        work_dir=tmp_path / "pre_s2",
        max_minutes=300.0,
        reserve_minutes=45.0,
        min_disk_bytes=1024,
    ).run_registered_plan(
        pre_jobs,
        job_runner=lambda j, w, d: {"exit_code": 0, "completed_updates": j.updates, "eval_rows": 16},
        devices=("cuda:0", "cuda:1"),
        resume_input=res_pre["export_receipt"]["archives"]["full"],
        measured_seconds_per_update=1.0,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        clock_fn=lambda: 0.0,
    )
    assert res_pre_s2["status"] == "COMPLETE"
    assert res_pre_s2["dispatched_job_ids"] == ["p2", "p3"]
    assert set(res_pre_s2["completed_jobs"]) == {"p0", "p1", "p2", "p3"}

    # 4. Multi-group profiling + per_job_seconds_per_update + u0000 failure blocking Wave 0
    prof_calls: list[str] = []

    def _group_profile(rep: SessionJob, prof_dir: Path) -> float:
        grp = str(rep.extra["profile_group"])
        prof_calls.append(grp)
        return 0.5 if grp == "S20" else 2.5

    grp_jobs = [
        SessionJob("g_s20_a", "S20_ARCH", 7, 7, 10, extra={"profile_group": "S20"}),
        SessionJob("g_s20_b", "S20_ARCH", 19, 19, 10, extra={"profile_group": "S20"}),
        SessionJob("g_s50_a", "S50_ARCH", 7, 7, 10, extra={"profile_group": "S50"}),
    ]
    res_grp = CampaignOrchestrator(
        campaign_id="X3-GROUPS", source_commit=commit, work_dir=tmp_path / "groups", min_disk_bytes=1024
    ).run_registered_plan(
        grp_jobs,
        profile_fn=_group_profile,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        inventory_only=True,
    )
    assert prof_calls == ["S20", "S50"]
    assert res_grp["admission_plan"]["per_job_seconds_per_update"] == {
        "g_s20_a": 0.5,
        "g_s20_b": 0.5,
        "g_s50_a": 2.5,
    }

    worker_calls = 0

    def _counting_runner(job: SessionJob, wroot: Path, dev: str) -> int:
        nonlocal worker_calls
        worker_calls += 1
        return 0

    with pytest.raises(RuntimeError, match="U0000_SMOKE_FAILED:S50_ARCH"):
        CampaignOrchestrator(
            campaign_id="X3-U0FAIL", source_commit=commit, work_dir=tmp_path / "u0fail", min_disk_bytes=1024
        ).run_registered_plan(
            grp_jobs,
            job_runner=_counting_runner,
            measured_seconds_per_update=0.5,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
            u0000_smoke_fn=lambda arch, rep, sdir: {"passed": arch != "S50_ARCH", "err": "bad_s50"},
        )
    assert worker_calls == 0
    assert (tmp_path / "u0fail" / "U0000_SMOKE_RECEIPT.json").is_file()


def test_batched_rollout_serial_parity_and_notebook_preflight_commands() -> None:
    torch.manual_seed(7)
    mod = nn.Linear(4, 2)
    pol = LogitsPolicy(mod, action_labels=("A", "B"))
    kern = StaticPromptKernel(action_labels=("A", "B"))
    worlds = [
        {"world_id": f"w{i}", "sig": f"s{i}", "feat": [float(i % 2) * 2 - 1.0, 0.2, -0.1, 0.4], "expected": "A" if i % 2 else "B"}
        for i in range(6)
    ]

    batched_res = batched_rollout(pol, kern, worlds, batch_size=2, verify_serial_parity_n=4, parity_atol=1e-4)
    assert len(batched_res) == 6

    class _DivergentPolicy(LogitsPolicy):
        def propose_batch(self, obs_batch):
            acts = super().propose_batch(obs_batch)
            # Simulate a multi-item padding/masking bug that only manifests for len(obs_batch) > 1.
            if len(obs_batch) > 1:
                return [
                    UniversalAction(a.operation, a.argument, float(a.log_probability) - 0.5, a.action_distribution, a.mode, a.raw_operation, a.raw_argument)
                    for a in acts
                ]
            return acts

    bad_pol = _DivergentPolicy(mod, action_labels=("A", "B"))
    with pytest.raises(ValueError, match="BATCHED_SERIAL_PARITY_MISMATCH"):
        batched_rollout(bad_pol, kern, worlds, batch_size=2, verify_serial_parity_n=2, parity_atol=1e-4)

    nb = build_pinned_kaggle_notebook(
        source_commit="f" * 40,
        title="X3 Preflight Test",
        launch_command="python -m experiments.x3.runner --mode FULL",
        preflight_commands=["python -m experiments.x3.smoke --u0000"],
        require_dual_gpu=True,
    )
    gate_src = "".join(nb["cells"][2]["source"])
    assert "PREFLIGHT_COMMANDS" in gate_src
    assert "'--u0000'" in gate_src
