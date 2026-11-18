"""Tests for Planthir v2 audit upgrades, ExperimentModule plugins, and Hitchhiker fixes."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.planthir import (
    BytePromptTrainer,
    CampaignOrchestrator,
    CampaignStageSpec,
    ExperimentModule,
    ExperimentRegistration,
    GenericDatasetBundle,
    GenericTrainRecipe,
    IndependentWorkerScheduler,
    LegacyByteObservablePolicy,
    LogitsPolicy,
    PlanthirRegistry,
    SessionJob,
    StaticPromptKernel,
    UniversalAction,
    UniversalObservation,
    WorkerJobSpec,
    analyze_paired_seed_runs,
    analyze_productivity_study,
    audit_pair_margin_gradient_sign_reversal,
    audit_sampled_strata,
    batched_rollout,
    bootstrap_paired_families,
    build_cartridge_from_spec,
    build_deterministic_retraining_recipe,
    build_pinned_kaggle_notebook,
    build_verification_matrix,
    classify_parameter_group,
    compute_ce_and_pair_margin_terms,
    export_session_archives,
    frame_byte_prompt_sample,
    get_backend,
    independent_replay_bundle,
    init_spec_template,
    load_legacy_byte_checkpoint,
    locate_checkpoint_store_root,
    planthir_doctor,
    prepare_x1_rd2_training_data,
    render_campaign_results_markdown,
    run_fault_injection_suite,
    run_scientific_shortcut_calibration,
    safe_extract_planthir_archive,
    score_crossover_panel,
    validate_spec_document,
    verify_crossover_fixture,
)
from anra_v5.planthir.backend import _BACKENDS
from anra_v5.planthir.cartridge import CartridgeRunner
from anra_v5.planthir.productivity import PairedTaskObservation, SIX_BENCHMARK_TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]


class _ToyDepthModel(nn.Module):
    def __init__(self, width: int, toy_head_width: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(4, width)
        self.proj.weight.data.normal_(0.0, 0.2, generator=g)
        self.head = nn.Linear(width, 2)
        self.toy_head = nn.Linear(width, toy_head_width)
        self.toy_out = nn.Linear(toy_head_width, 2)
        self.trained_compute_steps = 2
        self._ablate = False
        self._depth = 2

    @contextmanager
    def ablate_mechanism(self):
        self._ablate = True
        try:
            yield "toy_extra_head"
        finally:
            self._ablate = False

    @contextmanager
    def compute_depth_context(self, steps: int):
        old = self._depth
        self._depth = int(steps)
        try:
            yield
        finally:
            self._depth = old

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.proj(x))
        for _ in range(max(0, self._depth)):
            h = h + 0.05 * torch.tanh(h)
        logits = self.head(h)
        if not self._ablate:
            logits = logits + 0.25 * self.toy_out(torch.tanh(self.toy_head(h)))
        return logits, h


class _ToyKernel:
    kernel_id = "toy-kernel-v1"

    def structural_signature(self, world: dict) -> str:
        return f"sig:{world['sig']}"

    def inject_distractor(self, world: dict) -> dict:
        w = dict(world)
        w["world_id"] = f"{world['world_id']}::dist"
        w["feat"] = list(world["feat"])
        return w

    def permute_world(self, world: dict, shift: int = 1) -> dict:
        w = dict(world)
        w["world_id"] = f"{world['world_id']}::perm{shift}"
        return w

    def reverse_world(self, world: dict) -> dict:
        w = dict(world)
        w["world_id"] = f"{world['world_id']}::rev"
        return w

    def initial_state(self, world: dict) -> dict:
        return {"world": world, "step": 0, "terminal": False}

    def observe(self, state: dict) -> UniversalObservation:
        w = state["world"]
        return UniversalObservation(
            obs_id=str(w["world_id"]),
            rendered_text=f"TOY sig={w['sig']} feat={w['feat']}",
            available_tools={"query": False, "memory": False},
            consumed_tools=frozenset(),
            decision_index=0,
            budget_remaining=1,
            metadata={"world_id": str(w["world_id"])},
        )

    def ideal_action(self, state: dict) -> tuple[str, str | None]:
        return ("ANSWER", str(state["world"]["target"]))

    def step(self, state: dict, operation: str, argument: str | None) -> dict:
        return {**state, "terminal": True}

    def verify(self, world: dict, actions) -> dict:
        op, arg = actions[0] if actions else ("INVALID", None)
        return {"correct": bool(op == "ANSWER" and arg == str(world["target"])), "terminal_status": "answered"}

    def decompose_subgoals(self, world: dict):
        return ()


class _ToyPolicy:
    policy_id = "toy-policy-v1"

    def __init__(self, model: _ToyDepthModel, model_spec, *, model_identity: str, mode: str = "raw", copy_enabled_override=None) -> None:
        self.model = model
        self.model_spec = model_spec
        self.model_identity = model_identity
        self.mode = mode
        self.batched_calls = 0

    def _feat_from_obs(self, obs: UniversalObservation) -> torch.Tensor:
        raw = obs.rendered_text.split("feat=", 1)[1]
        vals = json.loads(raw)
        return torch.tensor(vals, dtype=torch.float32)

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return self.propose_batch([obs])[0]

    def propose_batch(self, obs_batch) -> list[UniversalAction]:
        self.batched_calls += 1
        x = torch.stack([self._feat_from_obs(o) for o in obs_batch], dim=0)
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = F.softmax(logits, dim=-1)
        out: list[UniversalAction] = []
        for row in probs:
            pred = "A" if float(row[0]) >= float(row[1]) else "B"
            lp = float(torch.log(row[0 if pred == "A" else 1].clamp_min(1e-8)).item())
            out.append(
                UniversalAction(
                    operation="ANSWER",
                    argument=pred,
                    log_probability=lp,
                    action_distribution={"ANSWER:A": float(row[0]), "ANSWER:B": float(row[1])},
                    mode=self.mode,
                    raw_operation="ANSWER",
                    raw_argument=pred,
                )
            )
        return out

    def extract_representations(self, obs_batch) -> dict[str, torch.Tensor]:
        x = torch.stack([self._feat_from_obs(o) for o in obs_batch], dim=0)
        with torch.no_grad():
            _, h = self.model(x)
        return {"decision_hidden": h}


def _make_toy_plugin(backend_id: str = "toy_v2") -> ExperimentModule:
    def _model_factory(arch: str, mcfg: dict, seed: int) -> nn.Module:
        return _ToyDepthModel(
            width=int(mcfg.get("width", 8)),
            toy_head_width=int(mcfg.get("toy_head_width", 4)),
            seed=seed,
        )

    def _ds_factory(dcfg: dict, cpu_canary: bool) -> dict:
        n = int(dcfg.get("num_items", 8))
        items = [
            {"world_id": f"tr_{i}", "sig": f"tr_{i}", "feat": [1.0 if i % 2 == 0 else -1.0, 0.5, -0.2, 0.1], "label": i % 2, "target": "A" if i % 2 == 0 else "B"}
            for i in range(n)
        ]
        ev = [
            {"world_id": f"ev_{i}", "sig": f"ev_{i}", "feat": [1.0 if i % 2 == 0 else -1.0, 0.5, -0.2, 0.1], "label": i % 2, "target": "A" if i % 2 == 0 else "B"}
            for i in range(6)
        ]
        mh = [
            {"world_id": f"mh_{i}", "sig": f"mh_{i}", "feat": [1.0 if i % 2 == 0 else -1.0, 0.5, -0.2, 0.1], "label": i % 2, "target": "A" if i % 2 == 0 else "B"}
            for i in range(4)
        ]
        pr = [
            {"world_id": f"pr_{i}", "sig": f"pr_{i}", "feat": [1.0 if i % 2 == 0 else -1.0, 0.5, -0.2, 0.1], "label": i % 2, "target": "A" if i % 2 == 0 else "B"}
            for i in range(6)
        ]
        return {
            "train_data": GenericDatasetBundle(dataset_name="toy", items=tuple(items), data_identity="toy-v1", manifest={"n": n}),
            "train_worlds": items,
            "eval_worlds": ev,
            "multihop_worlds": mh,
            "probe_train_worlds": pr,
        }

    def _loss_fn(model: _ToyDepthModel, chunk, recipe) -> torch.Tensor:
        x = torch.tensor([it["feat"] for it in chunk], dtype=torch.float32)
        y = torch.tensor([it["label"] for it in chunk], dtype=torch.int64)
        logits, _ = model(x)
        return F.cross_entropy(logits, y)

    return ExperimentModule(
        backend_id=backend_id,
        architectures=("toy_base", "toy_deep"),
        objectives=("ce",),
        model_factory=_model_factory,
        dataset_factory=_ds_factory,
        loss_fn=_loss_fn,
        kernel_factory=_ToyKernel,
        policy_factory=_ToyPolicy,
        default_dataset_config={"dataset_name": "toy", "num_items": 8},
        default_model_config={"width": 8, "toy_head_width": 4},
        treatment_architecture="toy_deep",
        treatment_model_config={"width": 8, "toy_head_width": 6},
        extra_model_keys=("width", "toy_head_width"),
        extra_dataset_keys=("dataset_name", "num_items"),
        extra_arch_head_keys=("toy_head_width",),
    )


def test_experiment_module_e2e_concise_spec_and_confounder_rejection(tmp_path: Path) -> None:
    plugin = _make_toy_plugin("toy_v2").register()
    try:
        tpl = init_spec_template(backend_id="toy_v2", canary=True)
        val_ok = validate_spec_document(tpl, canary_override=True)
        assert val_ok["valid"] is True, val_ok

        # Omitting contrasts cleanly works for 1 varying factor; adding 2nd hidden confound fails Gate 2!
        bad_doc = json.loads(json.dumps(tpl))
        bad_doc["treatment_arms"][0]["recipe"]["lr"] = 0.005
        val_bad = validate_spec_document(bad_doc, canary_override=True)
        assert val_bad["valid"] is False
        assert any("LR_MISMATCH" in str(e) for e in val_bad["errors"])

        # Custom backend runs full 5-Gate CartridgeRunner, Gate 4 hooks, and bundle replay without core edits
        cart, run_kwargs = build_cartridge_from_spec(tpl, canary_override=True)
        work = tmp_path / "toy_run"
        res = CartridgeRunner().run(cart, work_root=work, **run_kwargs)
        assert res["gates"]["all_5_gates_passed"] is True
        assert res["bundle_receipt"]["status"] == "COMPLETE"
        trt_diag = res["arms"]["TRT"]["causal_diagnostics"]
        assert trt_diag["surgical_ablation"]["ablation_supported"] is True
        assert trt_diag["compute_depth_scaling"]["supports_variable_depth"] is True

        rep = independent_replay_bundle(work / "result_bundle", worlds=cart.eval_worlds)
        assert rep["verified"] is True and rep["restore_verified"] is True

        # Batched rollout exercises propose_batch
        pol = plugin.create_policy(_ToyDepthModel(8, 4, 1), get_backend("toy_v2").resolve_model_spec(cart.control_arm), model_identity="m1")
        br = batched_rollout(pol, _ToyKernel(), cart.eval_worlds, batch_size=3)
        assert len(br) == len(cart.eval_worlds) and pol.batched_calls == 2

        # Checkpoint milestone deduplication
        trainer = plugin.create_trainer(
            cart.control_arm,
            cart.train_data,
            source_commit="a" * 40,
            root=tmp_path / "ck",
            lineage="toy-s7",
            device="cpu",
            cpu_canary=True,
            resume=False,
        )
        m_dir = trainer.save_inference_milestone_weights(
            tmp_path / "milestones", update=0, shared_backbone_dir=tmp_path / "shared", backbone_prefix="proj."
        )
        assert (m_dir / "head_weights.bin").is_file()
        assert any((tmp_path / "shared").glob("init_backbone_*.bin"))
    finally:
        _BACKENDS.pop("toy_v2", None)


def test_env_backend_module_loading_and_doctor(tmp_path: Path) -> None:
    mod_file = tmp_path / "dyn_plugin.py"
    mod_file.write_text(
        "\n".join(
            [
                "from tests.test_planthir_v2_audit_upgrades import _make_toy_plugin",
                "def register_planthir_backends():",
                "    _make_toy_plugin('dyn_toy').register()",
            ]
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{tmp_path}"
    env["PLANTHIR_BACKEND_MODULES"] = "dyn_plugin"
    proc = subprocess.run(
        [sys.executable, "-m", "anra_v5.planthir", "init", "--backend", "dyn_toy"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["backend_id"] == "dyn_toy"
    assert planthir_doctor()["green_signal"] is True


def test_scheduler_cuda_ordinal_remap_and_orchestrator_archives(tmp_path: Path) -> None:
    observed: list[tuple[str, str | None, list[str]]] = []

    def _fast_runner(job: WorkerJobSpec, env, cmd):
        observed.append((job.worker_id, env.get("CUDA_VISIBLE_DEVICES"), list(cmd)))
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    jobs = [
        SessionJob("job_a", "R", 7, 7, 10, device="cuda:0"),
        SessionJob("job_b", "R", 19, 19, 10, device="cuda:1"),
    ]
    orch = CampaignOrchestrator(
        campaign_id="TEST-X1",
        source_commit="a" * 40,
        work_dir=tmp_path / "camp",
        max_minutes=330,
        reserve_minutes=45,
        min_disk_bytes=1024,
        in_process_runner=_fast_runner,
    )
    orch.prepare(jobs)
    adm = orch.admit(jobs, est_seconds_per_update=0.1, per_job_disk_bytes=1024, available_disk_bytes=10**9)
    assert adm["admitted"] is True
    with pytest.raises(RuntimeError):
        orch.admit(jobs, est_seconds_per_update=0.1, per_job_disk_bytes=1024, available_disk_bytes=512)

    wave_res = orch.run_waves(
        jobs,
        command_builder=lambda j, wr: ("python", "-m", "worker", "--work-root", str(wr), "--device", j.device),
        devices=("cuda:0", "cuda:1"),
    )
    assert wave_res["status"] == "COMPLETE" and len(observed) == 2
    by_id = {wid: (vis, cmd) for wid, vis, cmd in observed}
    assert by_id["job_b"][0] == "1"
    idx = by_id["job_b"][1].index("--device")
    assert by_id["job_b"][1][idx + 1] == "cuda:0"

    # Resume skips already completed jobs
    observed.clear()
    resumed = orch.run_waves(
        jobs,
        command_builder=lambda j, wr: ("python", "-m", "worker", "--device", j.device),
        resume=True,
    )
    assert len(observed) == 0 and resumed["completed_count"] == 2

    # Archive export + traversal/duplicate rejection
    src_f = tmp_path / "metrics.json"
    src_f.write_text('{"ok": true}\n', encoding="utf-8")
    rec = export_session_archives(tmp_path, tmp_path / "out", prefix="S0", analysis_files={"METRICS.json": src_f})
    assert rec["verified"] is True
    extracted = safe_extract_planthir_archive(tmp_path / "out" / "S0_ANALYSIS.zip", tmp_path / "ext")
    assert extracted == ["METRICS.json"]

    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "boom")
    with pytest.raises(ValueError, match="unsafe archive path"):
        safe_extract_planthir_archive(bad_zip, tmp_path / "ext_bad")

    nb = build_pinned_kaggle_notebook(source_commit="a" * 40, title="Test NB", launch_command="python -m tool")
    assert "OWNER_AUTHORIZED = False" in "".join(nb["cells"][1]["source"])


def test_hitchhiker_audit_fixes_registry_replay_stats_strata_and_docs(tmp_path: Path) -> None:
    # 1. Registry sha invariance across status transitions + tamper detection
    reg = PlanthirRegistry()
    sha1 = reg.register(
        ExperimentRegistration(
            experiment_id="EXP-1",
            program="X1",
            architecture_family="R",
            parameter_count=100,
            dataset_identity="ds1",
            source_commit="a" * 40,
            hypotheses=(),
            contrasts=(),
        )
    )
    reg.update_status("EXP-1", "COMPLETE", result_ref="ckpt123")
    reg_path = reg.save(tmp_path / "REGISTRY.json")
    loaded = PlanthirRegistry.load(reg_path)
    assert loaded.get("EXP-1")["registration_sha256"] == sha1
    assert loaded.get("EXP-1")["status"] == "COMPLETE"

    doc = json.loads(reg_path.read_text(encoding="utf-8"))
    doc["entries"]["EXP-1"]["registration"]["program"] = "TAMPERED"
    tampered = tmp_path / "BAD_REG.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        PlanthirRegistry.load(tampered)

    doc_ev = json.loads(reg_path.read_text(encoding="utf-8"))
    doc_ev["entries"]["EXP-1"]["status_events"][0]["status"] = "FORGED"
    tampered_ev = tmp_path / "BAD_EV.json"
    tampered_ev.write_text(json.dumps(doc_ev), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch|status_events_sha256 mismatch"):
        PlanthirRegistry.load(tampered_ev)

    # 2. Paired seed degenerate variance flag, sign-flip exact probability, ties, and per-seed reports
    stats_rep = analyze_paired_seed_runs(
        [
            {"contrast_id": "C1", "pair_id": "p1", "init_seed": 7, "order_seed": 7, "control_value": 0.5, "treatment_value": 0.75, "family_se": 0.04},
            {"contrast_id": "C1", "pair_id": "p2", "init_seed": 19, "order_seed": 19, "control_value": 0.5, "treatment_value": 0.75, "family_se": 0.06},
            {"contrast_id": "C1", "pair_id": "p3", "init_seed": 41, "order_seed": 41, "control_value": 0.5, "treatment_value": 0.75, "family_se": 0.05},
            {"contrast_id": "C_TIES", "pair_id": "t1", "init_seed": 7, "order_seed": 7, "control_value": 0.5, "treatment_value": 0.5},
            {"contrast_id": "C_TIES", "pair_id": "t2", "init_seed": 19, "order_seed": 19, "control_value": 0.5, "treatment_value": 0.75},
            {"contrast_id": "C_TIES", "pair_id": "t3", "init_seed": 41, "order_seed": 41, "control_value": 0.5, "treatment_value": 0.25},
        ]
    )
    c1 = stats_rep["contrasts"]["C1"]
    assert c1["variance_status"] == "DEGENERATE_ZERO_VARIANCE"
    assert c1["t_stat"] is None
    assert c1["p_raw"] == pytest.approx(0.25)
    assert c1["sign_flip_p_value"] == pytest.approx(0.25)
    assert c1["significant_holm"] is False
    assert c1["independent_unit"] == "seed_pair_cluster"
    assert len(c1["per_seed_effects"]) == 3
    assert c1["family_uncertainty_conditional_on_seeds"]["mean_family_se"] == pytest.approx(0.05)
    cties = stats_rep["contrasts"]["C_TIES"]
    assert cties["tie_count"] == 1 and cties["positive_count"] == 1 and cties["negative_count"] == 1
    assert cties["significant_holm"] is False

    # 3. X1 RD2 stratified group selection across 4, 8, 32 + strata audit
    for n_grp in (4, 8, 32):
        ds, _ = prepare_x1_rd2_training_data(num_groups=n_grp)
        counts = ds.manifest["sampled_strata"]["canonical_operation_counts"]
        assert set(counts) >= {"ANSWER", "QUERY", "RETRIEVE", "ABSTAIN"}
        assert ds.manifest["sampled_strata"]["interactive_claim_eligible"] is True

    strata_bad = audit_sampled_strata(
        [{"gold_op": 0, "family": "X01"}, {"gold_op": 0, "family": "X02"}],
        required_operations=("ANSWER", "QUERY"),
    )
    assert strata_bad["passed"] is False and strata_bad["missing_required_operations"] == ["QUERY"]
    strata_ok = audit_sampled_strata(
        [{"gold_op": 0, "family": "X01"}, {"gold_op": 1, "family": "X03"}, {"gold_op": 3, "family": "X04"}],
        required_operations=("ANSWER", "QUERY", "ABSTAIN"),
    )
    assert strata_ok["passed"] is True and strata_ok["interactive_claim_eligible"] is True

    # 4. Custom param map in observatory
    assert classify_parameter_group("my_weird_tensor", custom_map={"my_weird": "custom_mech"}) == "custom_mech"

    # 5. Replay bundle case-by-case prediction agreement (exact match, swapped equal-accuracy rejection, missing case)
    plugin = _make_toy_plugin("toy_rep").register()
    try:
        tpl = init_spec_template(backend_id="toy_rep", canary=True)
        cart, run_kwargs = build_cartridge_from_spec(tpl, canary_override=True)
        work = tmp_path / "rep_work"
        CartridgeRunner().run(cart, work_root=work, **run_kwargs)
        lin = f"{cart.campaign_id.lower()}-{cart.control_arm.arm_id}-s{cart.control_arm.order_seed}"
        bdir = work / "result_bundle"
        probe_rec = independent_replay_bundle(bdir, worlds=cart.eval_worlds)
        assert probe_rec["prediction_agreement_status"] == "PROBE_ONLY_NO_REFERENCE"
        lineages = {row["lineage_id"]: row for row in probe_rec["restored_lineages"]}
        assert lin in lineages
        lin_ckpt = lineages[lin]["checkpoint_sha256"]

        from hashlib import sha256 as _sha256
        from anra_v5.planthir import CheckpointCatalog

        cat = CheckpointCatalog.discover_checkpoints([work / "checkpoints"])
        kernel = _ToyKernel()
        ref_by_lin: dict[str, list[dict[str, Any]]] = {}
        for row in probe_rec["restored_lineages"]:
            r_lin = row["lineage_id"]
            target_rec = max([r for r in cat.records if r.lineage_id == r_lin], key=lambda r: r.global_update)
            loaded_ref = CheckpointCatalog.load_checkpoint_for_causal_probing(target_rec, mode="raw", device="cpu")
            ref_pol = loaded_ref["policy"]
            lin_rows = []
            for w in cart.eval_worlds:
                init_obs = kernel.observe(kernel.initial_state(w))
                act = ref_pol.propose_with_telemetry(init_obs)
                lin_rows.append(
                    {
                        "world_id": w["world_id"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "decoder": "raw",
                        "rendered_input_sha256": _sha256(init_obs.rendered_text.encode("utf-8")).hexdigest(),
                        "operation": act.operation,
                        "argument": act.argument,
                    }
                )
            ref_by_lin[r_lin] = lin_rows
        ref_exact = ref_by_lin[lin]
        exact_rec = independent_replay_bundle(
            bdir,
            worlds=cart.eval_worlds,
            reference_predictions=ref_by_lin,
        )
        assert exact_rec["prediction_agreement_status"] == "MATCHED"
        assert exact_rec["original_prediction_agreement"]["matched"] is True

        # Exchange two case predictions (preserving aggregate accuracy) -> must fail!
        exchanged = json.loads(json.dumps(ref_exact))
        diff_pair = next(((i, j) for i in range(len(exchanged)) for j in range(i + 1, len(exchanged)) if exchanged[i]["argument"] != exchanged[j]["argument"]), None)
        if diff_pair is not None:
            i, j = diff_pair
            exchanged[i]["argument"], exchanged[j]["argument"] = exchanged[j]["argument"], exchanged[i]["argument"]
        else:
            exchanged[0]["argument"] = "B" if exchanged[0]["argument"] == "A" else "A"
        with pytest.raises(ValueError, match="ORIGINAL_PREDICTION_MISMATCH"):
            independent_replay_bundle(
                bdir,
                worlds=cart.eval_worlds,
                reference_predictions={**ref_by_lin, lin: exchanged},
            )
        missing_case = ref_exact + [{**ref_exact[0], "world_id": "nonexistent_world"}]
        with pytest.raises(ValueError, match="ORIGINAL_PREDICTION_MISMATCH"):
            independent_replay_bundle(
                bdir,
                worlds=cart.eval_worlds,
                reference_predictions={**ref_by_lin, lin: missing_case},
            )
    finally:
        _BACKENDS.pop("toy_rep", None)

    # 6. Scientific shortcut calibration & planted-error timing in productivity study
    sci_cal = run_scientific_shortcut_calibration(families=16)
    assert sci_cal["passed"] is True
    assert sci_cal["policies"]["timestamp_rank_shortcut"]["shortcut_detected"] is True
    assert sci_cal["policies"]["terminal_value_copy_shortcut"]["shortcut_detected"] is True
    assert sci_cal["policies"]["traversal_oracle"]["qualifies_for_claims"] == ["general_traversal", "interactive_tool_decision"]
    full_cal = run_fault_injection_suite()
    assert full_cal["total_faults"] == 6 and full_cal["detected_faults"] == 6
    assert full_cal["scientific_shortcuts_rejected"] is True and full_cal["all_detected"] is True

    obs_err = [
        PairedTaskObservation(
            task_id=t["task_id"],
            participant_id=f"P{i % 2}",
            order="baseline_first" if i % 2 == 0 else "planthir_first",
            baseline_tokens=10_000,
            planthir_tokens=2_000,
            baseline_wall_seconds=100.0,
            planthir_wall_seconds=20.0,
            baseline_quality_score=0.9,
            planthir_quality_score=0.95,
            planthir_passed=True,
            baseline_error_detection_seconds=600.0,
            planthir_error_detection_seconds=60.0,
        )
        for i, t in enumerate(SIX_BENCHMARK_TASKS)
    ]
    study_res = analyze_productivity_study(obs_err)
    assert study_res["passed"] is True
    assert study_res["mean_error_detection_time_reduction_fraction"] == pytest.approx(0.9)

    # 7. agent.md sections present
    doc_text = (REPO_ROOT / "agent.md").read_text(encoding="utf-8")
    for needle in (
        "Path 1 — Varying Hyperparams / Existing Arms",
        "Path 2 — New Architecture / New Environment / New Loss",
        "Fast Test Budget & Anti-Slowdown Rules",
        "Gotchas & Exact Core Contracts",
        "optimizer_spec_sha256",
        "CUDA_VISIBLE_DEVICES",
        "vocabulary",
    ):
        assert needle in doc_text


def test_avendi_postmortem_upgrades_static_kernel_byte_loader_crossover_and_v6_archives(tmp_path: Path) -> None:
    from anra_v5.elendril.representation import UTF8
    from tools.elendril_v6_export import verify_zip as v6_verify_zip
    from v5_contracts.model_spec import ModelSpec

    # 1. ExperimentModule with omitted kernel_factory, policy_factory, and probe_train_worlds/multihop_worlds
    def _static_model(arch: str, mcfg: dict, seed: int) -> nn.Module:
        return _ToyDepthModel(width=int(mcfg.get("width", 8)), toy_head_width=int(mcfg.get("toy_head_width", 4)), seed=seed)

    def _static_ds(dcfg: dict, cpu_canary: bool) -> dict:
        tr = [
            {
                "world_id": f"tr_{i}",
                "sig": f"s_{i}",
                "feat": [1.0 if i % 2 == 0 else -1.0, 0.5, -0.2, 0.1],
                "label": i % 2,
                "hidden": {"expected": "A" if i % 2 == 0 else "B", "expected_operation": "ANSWER"},
            }
            for i in range(int(dcfg.get("num_items", 8)))
        ]
        ev = [
            {
                "world_id": f"ev_{i}",
                "sig": f"ev_{i}",
                "feat": [1.0 if i % 2 == 0 else -1.0, 0.5, -0.2, 0.1],
                "label": i % 2,
                "hidden": {"expected": "A" if i % 2 == 0 else "B", "expected_operation": "ANSWER"},
            }
            for i in range(6)
        ]
        # Intentionally omit probe_train_worlds and multihop_worlds to prove no self-inflicted leakage!
        return {
            "train_data": GenericDatasetBundle(dataset_name="static_toy", items=tuple(tr), data_identity="st-1", manifest={"n": len(tr)}),
            "train_worlds": tr,
            "eval_worlds": ev,
        }

    def _static_loss(model: _ToyDepthModel, chunk, recipe) -> torch.Tensor:
        logits, _ = model(torch.tensor([it["feat"] for it in chunk], dtype=torch.float32))
        return F.cross_entropy(logits, torch.tensor([it["label"] for it in chunk], dtype=torch.int64))

    static_mod = ExperimentModule(
        backend_id="toy_static_default",
        architectures=("toy_base", "toy_deep"),
        objectives=("ce",),
        model_factory=_static_model,
        dataset_factory=_static_ds,
        loss_fn=_static_loss,
        action_labels=("A", "B"),
        default_dataset_config={"dataset_name": "static_toy", "num_items": 8},
        default_model_config={"width": 8, "toy_head_width": 4},
        treatment_architecture="toy_deep",
        treatment_model_config={"width": 8, "toy_head_width": 6},
        extra_arch_head_keys=("toy_head_width",),
    ).register()
    try:
        tpl = init_spec_template(backend_id="toy_static_default", canary=True)
        val_rep = validate_spec_document(tpl, canary_override=True)
        assert val_rep["valid"] is True, val_rep
        assert val_rep["cohort_transitivity"]["passed"] is True
        cart, rkw = build_cartridge_from_spec(tpl, canary_override=True)
        assert isinstance(cart.kernel, StaticPromptKernel)
        res = CartridgeRunner().run(cart, work_root=tmp_path / "static_run", **rkw)
        assert res["gates"]["all_5_gates_passed"] is True
        pol = static_mod.create_policy(
            _ToyDepthModel(8, 4, 1),
            get_backend("toy_static_default").resolve_model_spec(cart.control_arm),
            model_identity="m_static",
        )
        assert isinstance(pol, LogitsPolicy)
        roll = batched_rollout(pol, cart.kernel, cart.eval_worlds, batch_size=3)
        assert len(roll) == len(cart.eval_worlds)
    finally:
        _BACKENDS.pop("toy_static_default", None)

    # 2. BytePromptSample + gradient sign-reversal negative control + BytePromptTrainer + LegacyByteObservablePolicy restore
    grad_ctrl = audit_pair_margin_gradient_sign_reversal()
    assert grad_ctrl["passed"] is True and grad_ctrl["gradient_cosine"] <= -0.999

    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_threads = torch.get_num_threads()
    byte_root = tmp_path / "byte_ckpts"
    try:
        byte_spec = ModelSpec(
            schema="anra-v5-model-spec/v1",
            family="dense-decoder-transformer",
            vocabulary_size=UTF8.vocabulary_size,
            width=16,
            layers=1,
            query_heads=2,
            kv_heads=1,
            head_dimension=8,
            ffn_width=32,
            context_length=64,
            rope_base=10_000.0,
            norm_epsilon=1e-5,
            tied_embeddings=True,
            qk_norm=True,
            qk_norm_affine=True,
            linear_bias=False,
            dropout=0.0,
        )
        byte_samples = [
            frame_byte_prompt_sample(
                f"Q=A;A={'E' if i % 2 == 0 else 'F'}",
                "E" if i % 2 == 0 else "F",
                rival_answer="F" if i % 2 == 0 else "E",
                task_id=f"bs_{i}",
                world_id=f"w_{i}",
                family_id=f"fam_{i // 2}",
                cell_key="q0_b0" if i % 2 == 0 else "q0_b1",
            )
            for i in range(8)
        ]
        b_trainer = BytePromptTrainer(
            byte_samples,
            spec=byte_spec,
            recipe=GenericTrainRecipe(seed=7, total_updates=2, batch_size=4, micro_batch_size=2, objective="ce_pair_margin", cf_weight=0.5, cf_margin=1.0),
            source_commit="a" * 40,
            root=byte_root,
            lineage="y1-s3-s7",
            data_provenance={"split": "train", "license": "MIT"},
            device="cpu",
            cpu_canary=True,
        )
        b_trainer.step()
        b_trainer.save()
        b_trainer.step()
        ckpt_sha = b_trainer.save()
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)

    assert locate_checkpoint_store_root(byte_root, expected_lineages=("y1-s3-s7",)) == byte_root
    restored = load_legacy_byte_checkpoint(byte_root, lineage_id="y1-s3-s7", checkpoint_sha256=ckpt_sha, device="cpu")
    assert isinstance(restored["policy"], LegacyByteObservablePolicy)
    obs_probe = UniversalObservation(
        obs_id="probe0",
        rendered_text="Q=A;A=E",
        available_tools={"query": False, "memory": False},
        consumed_tools=frozenset(),
        decision_index=0,
        budget_remaining=1,
        metadata={"world_id": "probe0"},
    )
    act_probe = restored["policy"].propose_with_telemetry(obs_probe)
    assert act_probe.operation in ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")
    assert len(act_probe.metadata["first_logits"]) == UTF8.vocabulary_size
    reps = restored["policy"].extract_representations([obs_probe])
    assert reps["decision_hidden"].shape == (1, UTF8.vocabulary_size)

    # 3. CampaignOrchestrator stage DAG, missing & present frozen checkpoint evaluation, and results markdown
    orch = CampaignOrchestrator(campaign_id="AVENDI-TEST", source_commit="a" * 40, work_dir=tmp_path / "orch")
    miss_rec = orch.evaluate_frozen_checkpoints(tmp_path / "absent_dir", expected_lineages=("y1-s3-s7",))
    assert miss_rec["status"] == "MISSING_CHECKPOINT_ARTIFACT"
    assert miss_rec["missing_lineages"] == ["y1-s3-s7"]
    assert miss_rec["retraining_recipe"]["status"] == "REQUIRES_FRESH_IDENTITY_AND_OWNER_COMPUTE_APPROVAL"

    ok_rec = orch.evaluate_frozen_checkpoints(byte_root, expected_lineages=("y1-s3-s7",))
    assert ok_rec["status"] == "COMPLETE" and ok_rec["evaluated_lineages"]["y1-s3-s7"]["policy_loaded"] is True

    dag_rec = orch.run_stage_dag(
        [
            CampaignStageSpec("p0", "fixture", lambda ctx: {"status": "COMPLETE", "one_hop_acc": 0.5}),
            CampaignStageSpec(
                "p3",
                "acquisition",
                lambda ctx: {"status": "COMPLETE", "ran_p3": True},
                predicate_fn=lambda ctx: (ctx.get("one_hop_acc", 1.0) < 0.8, "one_hop_acc < 0.8"),
            ),
            CampaignStageSpec(
                "p4",
                "transition",
                lambda ctx: {"status": "COMPLETE", "ran_p4": True},
                predicate_fn=lambda ctx: (ctx.get("one_hop_acc", 0.0) >= 0.8, "requires one_hop_acc >= 0.8"),
            ),
        ],
        selected_phase="all",
    )
    assert dag_rec["status"] == "COMPLETE"
    by_stage = {s["stage_id"]: s for s in dag_rec["stages"]}
    assert by_stage["p3"]["status"] == "COMPLETE" and by_stage["p4"]["status"] == "NOT_APPLICABLE"

    vmat = build_verification_matrix(checkpoint_restore="PASS", regenerated_per_case_agreement="PASS")
    assert len(vmat) == 6
    md = render_campaign_results_markdown(
        title="Test Report",
        campaign_id="AVENDI-TEST",
        source_commit="a" * 40,
        status="COMPLETE",
        verification_matrix=vmat,
        stage_summaries={"p0": {"one_hop_acc": 0.5}},
    )
    assert "| `checkpoint_restore` | `PASS` |" in md

    # 4. verify_crossover_fixture, score_crossover_panel, and bootstrap_paired_families
    fams = [
        {
            "family_id": "f0",
            "cells": [
                {"world_id": "c00", "cell_key": "q0_b0", "public_prompt": "goal=G1 G1->X X->A", "hidden": {"expected": "A"}},
                {"world_id": "c01", "cell_key": "q0_b1", "public_prompt": "goal=G1 G1->Y Y->B", "hidden": {"expected": "B"}},
                {"world_id": "c10", "cell_key": "q1_b0", "public_prompt": "goal=G2 G2->Z Z->C", "hidden": {"expected": "C"}},
                {"world_id": "c11", "cell_key": "q1_b1", "public_prompt": "goal=G2 G2->W W->D", "hidden": {"expected": "D"}},
            ],
        },
        {
            "family_id": "f1",
            "cells": [
                {"world_id": "d00", "cell_key": "q0_b0", "public_prompt": "goal=G3 G3->X X->E", "hidden": {"expected": "E"}},
                {"world_id": "d01", "cell_key": "q0_b1", "public_prompt": "goal=G3 G3->Y Y->F", "hidden": {"expected": "F"}},
                {"world_id": "d10", "cell_key": "q1_b0", "public_prompt": "goal=G4 G4->Z Z->G", "hidden": {"expected": "G"}},
                {"world_id": "d11", "cell_key": "q1_b1", "public_prompt": "goal=G4 G4->W W->H", "hidden": {"expected": "H"}},
            ],
        },
    ]
    fix_ok = verify_crossover_fixture(
        fams,
        oracle_fn=lambda c: c["hidden"]["expected"] if "goal=" in c["public_prompt"] else "!",
        blind_policies={"const_A": lambda _c: "A"},
        erase_goal_fn=lambda c: {**c, "public_prompt": c["public_prompt"].replace("goal=", "x=")},
        erase_bridge_fn=lambda c: {**c, "public_prompt": "erased"},
        splits_by_name={"train": fams[:1], "eval": fams[1:]},
    )
    assert fix_ok["passed"] is True
    assert fix_ok["oracle"]["complete_family_accuracy"] == 1.0
    assert fix_ok["blind_policies"]["const_A"]["shortcut_rejected"] is True

    fix_leak = verify_crossover_fixture(fams, splits_by_name={"train": fams[:1], "eval": fams[:1]})
    assert fix_leak["passed"] is False
    assert any(d["code"] == "INSTANTIATED_STATE_OVERLAP" for d in fix_leak["defects"])

    preds = {
        "c00": {"raw_exact_correct": True, "signed_margin": 1.0, "endpoint_margin": 2.0},
        "c01": {"raw_exact_correct": True, "signed_margin": 1.0, "endpoint_margin": -2.0},
        "c10": {"raw_exact_correct": True, "signed_margin": 1.0, "endpoint_margin": -2.0},
        "c11": {"raw_exact_correct": True, "signed_margin": 1.0, "endpoint_margin": 2.0},
        "d00": {"raw_exact_correct": False, "ap_constrained_correct": True, "signed_margin": 0.5, "endpoint_margin": 1.0},
        "d01": {"raw_exact_correct": True, "signed_margin": 0.5, "endpoint_margin": -1.0},
        "d10": {"raw_exact_correct": True, "signed_margin": 0.5, "endpoint_margin": -1.0},
        "d11": {"raw_exact_correct": True, "signed_margin": 0.5, "endpoint_margin": 1.0},
    }
    panel = score_crossover_panel(fams, preds)
    assert panel["all4_raw_accuracy"] == pytest.approx(0.5)
    assert panel["ap_constrained_all4_accuracy"] == pytest.approx(1.0)
    assert panel["decoding_gap"] == pytest.approx(0.5)
    assert panel["latent_flip_rate"] == pytest.approx(1.0)
    assert panel["mean_signed_interaction"] == pytest.approx(1.5)

    boot_res = bootstrap_paired_families({"f0": 0.0, "f1": 0.5, "f2": 0.25}, {"f0": 1.0, "f1": 1.0, "f2": 0.75}, resamples=500, seed=7)
    assert boot_res["mean_delta"] == pytest.approx(2.0 / 3.0)
    assert boot_res["positive_count"] == 3 and boot_res["ci_low"] > 0.0

    # 5. export_session_archives unified v6-verified archives + pinned Kaggle helper + agent.md mirror parity
    f_res = tmp_path / "RESULTS.md"
    f_res.write_text(md, encoding="utf-8")
    f_ckpt = tmp_path / "weights.bin"
    f_ckpt.write_bytes(b"weights")
    exp_out = tmp_path / "v6_bundles"
    exp_rec = export_session_archives(
        tmp_path,
        exp_out,
        prefix="AVENDI",
        summary_files={"RESULTS.md": f_res},
        evidence_files={"RESULTS.md": f_res},
        analysis_files={"RESULTS.md": f_res},
        full_files={"RESULTS.md": f_res, "weights.bin": f_ckpt},
        source_commit="a" * 40,
    )
    assert exp_rec["verified"] is True and exp_rec["v6_manifest_embedded"] is True
    for zname in ("AVENDI_SUMMARY.zip", "AVENDI_EVIDENCE.zip", "AVENDI_ANALYSIS.zip", "AVENDI_FULL.zip"):
        zver = v6_verify_zip(exp_out / zname)
        assert zver["status"] == "EXPORT_VERIFIED"

    nb_cfg = build_pinned_kaggle_notebook(
        source_commit="a" * 40,
        title="NB",
        launch_command="python -m tools.avendi_campaign",
        discover_kaggle_inputs=("ELENDRIL_V8_Y1_FULL.zip",),
        mirror_outputs_to_working=True,
    )
    nb_src = "".join(nb_cfg["cells"][1]["source"])
    assert "ELENDRIL_V8_Y1_FULL.zip" in nb_src and "/kaggle/working" in nb_src

    root_doc = (REPO_ROOT / "agent.md").read_text(encoding="utf-8")
    mirror_doc = (REPO_ROOT / "research" / "planthir" / "agent.md").read_text(encoding="utf-8")
    assert root_doc == mirror_doc
    for contract_needle in (
        "UTF8.special(SpecialToken.ANSWER)",
        "TokenSample",
        "DEGENERATE_ZERO_VARIANCE",
        "StaticPromptKernel",
    ):
        assert contract_needle in root_doc
