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
    PairedSeedObservation,
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
    build_analyst_first_page_report,
    build_cartridge_from_spec,
    build_deterministic_retraining_recipe,
    build_paired_seed_observations,
    build_pinned_kaggle_notebook,
    build_verification_matrix,
    classify_parameter_group,
    compare_paired_episode_records,
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
    rescore_compact_analysis_archive,
    run_fault_injection_suite,
    run_paired_comparison_preflight,
    run_scientific_shortcut_calibration,
    run_surgical_mechanism_ablation,
    safe_extract_planthir_archive,
    score_crossover_panel,
    stratify_intervention_worlds,
    summarize_diagnostic_coverage,
    synthesize_scientific_interpretation,
    validate_spec_document,
    verify_crossover_fixture,
    verify_full_mode_dispatch,
    verify_session_archive,
    verify_shared_backbone_and_head_overhead,
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
    assert "OWNER_AUTHORIZED = True" in "".join(nb["cells"][1]["source"])
    nb_dry = build_pinned_kaggle_notebook(
        source_commit="a" * 40, title="Test NB", launch_command="python -m tool", owner_authorized=False
    )
    assert "OWNER_AUTHORIZED = False" in "".join(nb_dry["cells"][1]["source"])


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


def test_codex_x1_a0_review_feedback_coverage_pairing_preflight_first_page_and_rescore(tmp_path: Path) -> None:
    # 1. Paired episode disagreement comparison (treatment-only vs control-only, by family/view, deterministic exemplars)
    ctrl_eps = [
        {"world_id": "w00", "base_world_id": "b0", "family": "X01", "form": "canonical", "correct": True, "steps": [{"executed_op": "ANSWER", "executed_arg": "A"}]},
        {"world_id": "w01", "base_world_id": "b0", "family": "X01", "form": "alt_renderer", "correct": False, "steps": [{"executed_op": "ABSTAIN", "executed_arg": None}]},
        {"world_id": "w02", "base_world_id": "b1", "family": "X02", "form": "canonical", "correct": True, "steps": [{"executed_op": "QUERY", "executed_arg": None}]},
        {"world_id": "w03", "base_world_id": "b1", "family": "X02", "form": "alt_renderer", "correct": False, "steps": [{"executed_op": "ANSWER", "executed_arg": "B"}]},
    ]
    trt_eps = [
        {"world_id": "w00", "base_world_id": "b0", "family": "X01", "form": "canonical", "correct": True, "steps": [{"executed_op": "ANSWER", "executed_arg": "A"}]},
        {"world_id": "w01", "base_world_id": "b0", "family": "X01", "form": "alt_renderer", "correct": True, "steps": [{"executed_op": "ANSWER", "executed_arg": "A"}]},
        {"world_id": "w02", "base_world_id": "b1", "family": "X02", "form": "canonical", "correct": False, "steps": [{"executed_op": "ABSTAIN", "executed_arg": None}]},
        {"world_id": "w03", "base_world_id": "b1", "family": "X02", "form": "alt_renderer", "correct": False, "steps": [{"executed_op": "ANSWER", "executed_arg": "C"}]},
    ]
    ep_cmp = compare_paired_episode_records(ctrl_eps, trt_eps)
    assert ep_cmp["both_solved"] == 1
    assert ep_cmp["treatment_only_solved"] == 1
    assert ep_cmp["control_only_solved"] == 1
    assert ep_cmp["neither_solved"] == 1
    assert ep_cmp["first_operation_disagreement_count"] == 2
    assert ep_cmp["contrasting_trace_ids"]["treatment_only_example_id"] == "w01"
    assert ep_cmp["contrasting_trace_ids"]["control_only_example_id"] == "w02"
    assert ep_cmp["by_view"]["alt_renderer"]["treatment_only_solved"] == 1
    assert ep_cmp["by_family"]["X02"]["control_only_solved"] == 1

    # 2. Diagnostic coverage + stratified world selection + unresolved-coverage firewall in interpretation
    x01_only = [
        {"world_id": f"x01_b{b}_v{v}", "base_world_id": f"x01_b{b}", "family": "X01", "form": f"v{v}", "sig": f"s{b}", "feat": [1.0, 0.5, -0.2, 0.1], "hidden": {"expected": "A", "expected_operation": "ANSWER"}}
        for b in range(8)
        for v in range(6)
    ]
    r_mech = ("X02", "X03", "X05", "X06", "X07", "X08", "X09")
    cov_x01 = summarize_diagnostic_coverage(x01_only, target_families=r_mech, seed_ids=(7, 19, 41))
    assert cov_x01["distinct_base_world_count"] == 8
    assert cov_x01["distinct_families"] == ["X01"]
    assert cov_x01["distinct_view_count"] == 6
    assert cov_x01["distinct_seed_count"] == 3
    assert cov_x01["target_family_coverage_ratio"] == "0/7"
    assert cov_x01["coverage_complete"] is False
    assert "Mechanism unresolved." in cov_x01["coverage_summary"]

    class _ShamBypassModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))
            self._ablate = False

        @contextmanager
        def ablate_mechanism(self):
            self._ablate = True
            try:
                yield "x1_relational_slot_readout"
            finally:
                self._ablate = False

        def forward(self, x):
            logits = torch.tensor([[5.0, 0.0]] * x.shape[0])
            return logits, x

    ab_pol = LogitsPolicy(_ShamBypassModel(), action_labels=("A", "B"))
    ab_res = run_surgical_mechanism_ablation(
        ab_pol,
        StaticPromptKernel(action_labels=("A", "B")),
        x01_only[:12],
        target_families=r_mech,
        seed_ids=(7,),
    )
    assert ab_res["ornamental_mechanism_bypass"] is False
    assert ab_res["mechanism_status"] == "MECHANISM_UNRESOLVED_MISSING_TARGET_FAMILIES"
    assert "Mechanism unresolved." in ab_res["headline"]
    assert ab_res["paired_changes"]["gained_correct_count"] == 0

    interp = synthesize_scientific_interpretation(
        experiment_id="X1-A0-TEST",
        instrument_validation={"passed": True, "status": "PASSED"},
        surgical_ablation=ab_res,
        slice_comparisons={"canonical_vs_alt": {"slice_name": "canonical_vs_alt", "baseline_accuracy": 0.744, "slice_accuracy": 0.161}},
    )
    assert any(c.claim_id == "U-INCOMPLETE-DIAGNOSTIC-COVERAGE" for c in interp.unresolved_questions)
    assert any(c.claim_id == "S-RENDERER-OR-SLICE-TRANSFER-GAP" for c in interp.suspected_mechanisms)
    assert interp.recommended_next_experiments[0].action_id == "EVALUATE_FROZEN_LESIONS_ON_TARGET_FAMILIES"

    all_12_fams = [
        {"world_id": f"{fam}_b{b}_v{v}", "base_world_id": f"{fam}_b{b}", "family": fam, "form": f"v{v}"}
        for fam in ("X01", "X02", "X03", "X04", "X05", "X06", "X07", "X08", "X09", "X10", "X11", "X12")
        for b in range(4)
        for v in range(6)
    ]
    strat = stratify_intervention_worlds(all_12_fams, max_worlds=48)
    cov_strat = summarize_diagnostic_coverage(strat, target_families=r_mech)
    assert len(strat) == 48 and cov_strat["coverage_complete"] is True and cov_strat["target_family_coverage_ratio"] == "7/7"

    # 3. Typed pair constructor + preflight comparison fixture + doctor integration
    pf = run_paired_comparison_preflight()
    assert pf["passed"] is True
    assert pf["complete_fixture"]["complete_pair_count"] == 3
    assert pf["dropped_arm_fixture"]["status"] == "INCOMPLETE_PAIRS_REPORTED"
    assert len(pf["dropped_arm_fixture"]["preserved_arm_records"]) == 5
    assert "41" in pf["dropped_arm_fixture"]["missing_pairs"][0]["explanation"]
    doc_rep = planthir_doctor()
    assert doc_rep["green_signal"] is True and doc_rep["checks"]["paired_comparison_preflight"]["passed"] is True

    # 4. Analyst's first page + compact analysis archive rebuild via rescore-archive
    arm_rows = [
        {"architecture": "P", "init_seed": 7, "order_seed": 7, "value": 0.5951},
        {"architecture": "R", "init_seed": 7, "order_seed": 7, "value": 0.6363},
        {"architecture": "P", "init_seed": 19, "order_seed": 19, "value": 0.6589},
        {"architecture": "R", "init_seed": 19, "order_seed": 19, "value": 0.5998},
        {"architecture": "P", "init_seed": 41, "order_seed": 41, "value": 0.6068},
        {"architecture": "R", "init_seed": 41, "order_seed": 41, "value": 0.6133},
    ]
    bundle = build_paired_seed_observations(arm_rows, contrast_id="R_VS_P", control_arm="P", treatment_arm="R")
    metrics_payload = {
        "title": "X1 A0 Test Report",
        "campaign_id": "X1-A0-TEST",
        "source_commit": "a" * 40,
        "decision_summary": "Retain P as comparator; R missed the registered +5.0 pt development threshold (-0.38 pt).",
        "control_arm": "P",
        "treatment_arm": "R",
        "arm_records": arm_rows,
        "episodes_by_arm_seed": {"P": {"7": ctrl_eps}, "R": {"7": trt_eps}},
        "predeclared_slices": [
            {
                "slice_name": "canonical_vs_alt_renderer",
                "baseline_accuracy": 0.744,
                "slice_accuracy": 0.161,
                "control_first_op_disagree_rate": 0.142,
                "treatment_first_op_disagree_rate": 0.455,
            }
        ],
        "interventions": {"zero_r_on_x01": ab_res},
        "retrospective_findings": ["R gained +4.12 pts on seed 7 but dropped -5.90 pts on seed 19."],
        "next_experiment": {
            "title": "Cross frozen surgical lesions with canonical vs narrative views across all 7 mechanistic families",
            "requires_artifact": "X1_A0_FULL.zip",
            "discriminates_between": ("renderer_fragility", "relational_routing_failure"),
            "protocol": "Evaluate frozen checkpoints without retraining.",
        },
        "full_archive": {"name": "X1_A0_FULL.zip", "sha256": "b" * 64, "verified": True},
    }
    m_src = tmp_path / "src" / "METRICS.json"
    m_src.parent.mkdir(parents=True, exist_ok=True)
    m_src.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    arch_dir = tmp_path / "archives"
    exp = export_session_archives(
        tmp_path / "src",
        arch_dir,
        prefix="X1A0",
        analysis_files={"METRICS.json": m_src},
        include_v6_manifest=True,
        source_commit="a" * 40,
    )
    zip_path = Path(exp["archives"]["analysis"])
    fresh_dir = tmp_path / "fresh_rebuild"
    rescore_rec = rescore_compact_analysis_archive(zip_path, fresh_dir)
    assert rescore_rec["verified"] is True and rescore_rec["v6_zip_verified"] is True
    assert rescore_rec["complete_pair_count"] == 3 and rescore_rec["rescored_seed_count"] == 1
    rebuilt_md = (fresh_dir / "RESULTS.md").read_text(encoding="utf-8")
    for expected_piece in (
        "X1A0_ANALYSIS.zip",
        "X1_A0_FULL.zip",
        "rescore-archive",
        "Retain P as comparator",
        "`[PREDECLARED]`",
        "`[RETROSPECTIVE]`",
        "w01",
        "w02",
        "0/7",
        "Mechanism unresolved.",
    ):
        assert expected_piece in rebuilt_md

    # Verify CLI `python -m anra_v5.planthir rescore-archive`
    cli_out = tmp_path / "cli_rebuild"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "anra_v5.planthir", "rescore-archive", "--archive", str(zip_path), "--out", str(cli_out)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["verified"] is True
    assert (cli_out / "RESULTS.md").is_file() and (cli_out / "METRICS.json").is_file()


def test_replay_release_blocker_x1_s20_and_stateful_plugin_cli_and_tamper(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    def _run_cli(args: list[str], *, extra_env: dict[str, str] | None = None, expect_ok: bool = True) -> dict:
        run_env = {**env, **(extra_env or {})}
        proc = subprocess.run(
            [sys.executable, "-m", "anra_v5.palantiri", *args],
            cwd=str(REPO_ROOT),
            env=run_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if expect_ok:
            assert proc.returncode == 0, f"cmd={args}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        else:
            assert proc.returncode != 0, f"expected failure for {args}, got exit 0: {proc.stdout}"
        return json.loads(proc.stdout)

    # A. Stock x1_s20 CLI chain: init -> validate -> run -> verify-bundle -> replay-bundle
    x1_dir = tmp_path / "x1_s20_cli"
    x1_spec = x1_dir / "spec.json"
    _run_cli(["init", "--backend", "x1_s20", "--out", str(x1_spec)])
    val = _run_cli(["validate", "--spec", str(x1_spec), "--canary"])
    assert val["valid"] is True
    run_res = _run_cli(["run", "--spec", str(x1_spec), "--work-root", str(x1_dir / "cpu"), "--canary", "--device", "cpu"])
    assert run_res["gates_passed"] is True
    bdir_x1 = x1_dir / "cpu" / "result_bundle"
    ver_x1 = _run_cli(["verify-bundle", "--bundle-dir", str(bdir_x1)])
    assert ver_x1["verified"] is True
    rep_x1 = _run_cli(["replay-bundle", "--bundle-dir", str(bdir_x1), "--output", str(x1_dir / "REPLAY_X1.json")])
    assert rep_x1["verified"] is True and rep_x1["prediction_agreement_status"] == "MATCHED"
    assert all(r["backend_id"] == "x1_s20" for r in rep_x1["restored_lineages"])

    # B. Stateful 2-step plugin (step-2 observation depends on step-1 QUERY action + reset_episode + hidden["expected_answer"])
    plug_dir = tmp_path / "plug_pkg"
    plug_dir.mkdir(parents=True, exist_ok=True)
    (plug_dir / "stateful_p.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import torch, torch.nn as nn, torch.nn.functional as F",
                "from anra_v5.planthir import ExperimentModule, GenericDatasetBundle, UniversalAction, UniversalObservation",
                "",
                "class StatefulModel(nn.Module):",
                "    def __init__(self, width: int = 8, seed: int = 7) -> None:",
                "        super().__init__()",
                "        torch.manual_seed(seed)",
                "        self.net = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, 3))",
                "    def forward(self, x: torch.Tensor):",
                "        logits = self.net(x)",
                "        return logits, x",
                "",
                "class StatefulKernel:",
                "    kernel_id = 'stateful-t14-test-v1'",
                "    max_actions = 3",
                "    def structural_signature(self, w): return f\"st:{w['sig']}\"",
                "    def inject_distractor(self, w): return {**w, 'world_id': w['world_id'] + '::dist', 'sig': w['sig'] + '::dist'}",
                "    def permute_world(self, w, shift=1): return {**w, 'world_id': w['world_id'] + f'::p{shift}', 'sig': w['sig'] + f'::p{shift}'}",
                "    def reverse_world(self, w): return {**w, 'world_id': w['world_id'] + '::rev', 'sig': w['sig'] + '::rev'}",
                "    def initial_state(self, w):",
                "        return {'w': w, 'queried': False, 'step': 0, 'terminal': False}",
                "    def observe(self, st):",
                "        w = st['w']",
                "        qa = not st['queried']",
                "        txt = f\"WORLD {w['sig']} queried={st['queried']} hint={w['hidden']['cue'] if st['queried'] else 'NONE'}\"",
                "        return UniversalObservation(",
                "            obs_id=f\"{w['world_id']}:s{st['step']}:q{int(qa)}\",",
                "            rendered_text=txt,",
                "            available_tools={'query': qa, 'memory': False},",
                "            consumed_tools=frozenset({'query'}) if st['queried'] else frozenset(),",
                "            decision_index=st['step'],",
                "            budget_remaining=max(1, 3 - st['step']),",
                "            metadata={'queried': st['queried'], 'feat': w['feat']},",
                "        )",
                "    def ideal_action(self, st):",
                "        return ('QUERY', None) if not st['queried'] else ('ANSWER', st['w']['hidden']['expected_answer'])",
                "    def step(self, st, op, arg):",
                "        if op == 'QUERY' and not st['queried']:",
                "            return {'w': st['w'], 'queried': True, 'step': st['step'] + 1, 'terminal': False}",
                "        return {'w': st['w'], 'queried': st['queried'], 'step': st['step'] + 1, 'terminal': True}",
                "    def verify(self, w, actions):",
                "        op, arg = actions[-1] if actions else ('INVALID', None)",
                "        return {'correct': op == 'ANSWER' and arg == w['hidden']['expected_answer'], 'terminal_status': 'answered'}",
                "    def decompose_subgoals(self, w): return ()",
                "",
                "class StatefulPolicy:",
                "    def __init__(self, model, model_spec, *, model_identity='sp', mode='raw', copy_enabled_override=None):",
                "        self.model = model",
                "        self.model_spec = model_spec",
                "        self.model_identity = model_identity",
                "        self.mode = mode",
                "        self.step_count = 0",
                "    def reset_episode(self): self.step_count = 0",
                "    def propose_with_telemetry(self, obs):",
                "        x = torch.tensor([obs.metadata['feat']], dtype=torch.float32)",
                "        with torch.no_grad(): logits, _ = self.model(x)",
                "        probs = F.softmax(logits[0], dim=-1)",
                "        if obs.available_tools.get('query') and self.step_count == 0:",
                "            self.step_count += 1",
                "            return UniversalAction('QUERY', None, float(torch.log(probs[2].clamp_min(1e-6))), {'QUERY:QUERY': float(probs[2]), 'ANSWER:A': float(probs[0]), 'ANSWER:B': float(probs[1])}, self.mode, 'QUERY', None)",
                "        idx = int(torch.argmax(probs[:2]).item())",
                "        lbl = 'A' if idx == 0 else 'B'",
                "        self.step_count += 1",
                "        return UniversalAction('ANSWER', lbl, float(torch.log(probs[idx].clamp_min(1e-6))), {'ANSWER:A': float(probs[0]), 'ANSWER:B': float(probs[1]), 'QUERY:QUERY': float(probs[2])}, self.mode, 'ANSWER', lbl)",
                "    def extract_representations(self, obs_batch):",
                "        x = torch.tensor([o.metadata['feat'] for o in obs_batch], dtype=torch.float32)",
                "        with torch.no_grad(): _, h = self.model(x)",
                "        return {'decision_hidden': h}",
                "",
                "def _ds(cfg, cpu_canary):",
                "    def _mk(pfx, n):",
                "        return [{'world_id': f'{pfx}_{i}', 'sig': f'{pfx}_{i}', 'hops': 2, 'feat': [1.0 if i % 2 == 0 else -1.0, 0.2, 0.1, -0.3], 'label': i % 2, 'hidden': {'expected_answer': 'A' if i % 2 == 0 else 'B', 'cue': f'C{i}'}} for i in range(n)]",
                "    tr, ev, mh, pr = _mk('tr', 8), _mk('ev', 4), _mk('mh', 3), _mk('pr', 4)",
                "    return {'train_data': GenericDatasetBundle(dataset_name='sp', items=tuple(tr), data_identity='sp-v1', manifest={'n': len(tr)}), 'train_worlds': tr, 'eval_worlds': ev, 'multihop_worlds': mh, 'probe_train_worlds': pr}",
                "",
                "def _loss(m, chunk, recipe):",
                "    x = torch.tensor([c['feat'] for c in chunk], dtype=torch.float32)",
                "    y = torch.tensor([c['label'] for c in chunk], dtype=torch.int64)",
                "    return F.cross_entropy(m(x)[0], y)",
                "",
                "def register_planthir_backends():",
                "    ExperimentModule(",
                "        backend_id='stateful_p', architectures=('base', 'trt'), objectives=('ce',),",
                "        model_factory=lambda a, c, s: StatefulModel(int(c.get('width', 8)), s),",
                "        dataset_factory=_ds, loss_fn=_loss, kernel_factory=StatefulKernel, policy_factory=StatefulPolicy,",
                "        default_model_config={'width': 8}, treatment_model_config={'width': 8}, extra_model_keys=('width',),",
                "    ).register()",
            ]
        ),
        encoding="utf-8",
    )
    plug_env = {"PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{plug_dir}", "PLANTHIR_BACKEND_MODULES": "stateful_p"}
    sp_dir = tmp_path / "sp_cli"
    sp_spec = sp_dir / "spec.json"
    _run_cli(["init", "--backend", "stateful_p", "--out", str(sp_spec)], extra_env=plug_env)
    _run_cli(["run", "--spec", str(sp_spec), "--work-root", str(sp_dir / "cpu"), "--canary", "--device", "cpu"], extra_env=plug_env)
    bdir_sp = sp_dir / "cpu" / "result_bundle"
    rep_sp = _run_cli(["replay-bundle", "--bundle-dir", str(bdir_sp)], extra_env=plug_env)
    assert rep_sp["verified"] is True and rep_sp["prediction_agreement_status"] == "MATCHED"

    # Missing plugin fails loudly with MISSING_BACKEND_PLUGIN (no Windows PermissionError masking)
    miss_rep = _run_cli(["replay-bundle", "--bundle-dir", str(bdir_sp)], expect_ok=False)
    assert "MISSING_BACKEND_PLUGIN" in miss_rep["detail"]

    # Wrong expected backend rejected
    bad_be = _run_cli(["replay-bundle", "--bundle-dir", str(bdir_sp), "--expected-backend", "wrong_id"], extra_env=plug_env, expect_ok=False)
    assert "BACKEND_MISMATCH" in bad_be["detail"]

    # Deliberate input change fails with offending world identity
    import importlib
    sys.path.insert(0, str(plug_dir))
    try:
        sp_mod = importlib.import_module("stateful_p")
        sp_mod.register_planthir_backends()
        ev_worlds = json.loads((bdir_sp / "REPLAY_COHORT.json").read_text(encoding="utf-8"))["worlds"]
        bad_worlds = json.loads(json.dumps(ev_worlds))
        bad_worlds[0]["sig"] = "TAMPERED_INPUT"
        bad_worlds_path = sp_dir / "bad_worlds.json"
        bad_worlds_path.write_text(json.dumps(bad_worlds), encoding="utf-8")
        bad_inp = _run_cli(["replay-bundle", "--bundle-dir", str(bdir_sp), "--worlds-json", str(bad_worlds_path)], extra_env=plug_env, expect_ok=False)
        assert "ORIGINAL_PREDICTION_MISMATCH" in bad_inp["detail"]

        # Action / observation history mismatch check
        arch_refs = json.loads((bdir_sp / "REPLAY_COHORT.json").read_text(encoding="utf-8"))["reference_predictions"]
        lin_0 = sorted(arch_refs)[0]
        tampered_hist = json.loads(json.dumps(arch_refs))
        tampered_hist[lin_0][0]["observation_sha256_history"][-1] = "f" * 64
        with pytest.raises(ValueError, match="ORIGINAL_PREDICTION_MISMATCH"):
            independent_replay_bundle(bdir_sp, worlds=ev_worlds, reference_predictions=tampered_hist)

        # Require reference on unreferenced call returns INSUFFICIENT_REPLAY_EVIDENCE
        with pytest.raises(ValueError, match="INSUFFICIENT_REPLAY_EVIDENCE"):
            independent_replay_bundle(bdir_sp, worlds=ev_worlds, require_reference=True)
    finally:
        sys.path.remove(str(plug_dir))
        _BACKENDS.pop("stateful_p", None)


def test_campaign_dual_clock_verified_resume_backbone_notebook_and_agent_docs(tmp_path: Path) -> None:
    from hashlib import sha256 as _sha

    # 1. Dual-clock admit_slice + PENDING_QUEUE.json preservation
    orch = CampaignOrchestrator(
        campaign_id="X2-DUAL-CLOCK",
        source_commit="a" * 40,
        work_dir=tmp_path / "camp_dual",
        max_minutes=330.0,
        reserve_minutes=45.0,
        min_disk_bytes=1024,
    )
    jobs = [SessionJob(f"j{i}", "DELTA", 7 + i, 7 + i, 100, device="cuda:0") for i in range(6)]
    plan = orch.admit_slice(
        jobs,
        measured_seconds_per_update=24.0,
        per_job_disk_bytes=1024,
        parallel_workers=2,
        available_disk_bytes=10**8,
        outer_remaining_minutes=130.0,
    )
    assert plan["admitted_job_ids"] == ["j0", "j1", "j2", "j3"]
    assert plan["deferred_pending_job_ids"] == ["j4", "j5"]
    assert plan["next_resumable_job_id"] == "j4"
    assert (tmp_path / "camp_dual" / "PENDING_QUEUE.json").is_file()

    # 2. Verified resume reconciliation (swapped spec & missing archive member rejected)
    good_zip = tmp_path / "j0.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("FINAL_STATUS.json", '{"status": "COMPLETE"}\n')
    good_sha = _sha(good_zip.read_bytes()).hexdigest()
    reg_rows = [
        {"job_id": "j0", "source_commit": "a" * 40, "spec_sha256": "s0", "updates": 100},
        {"job_id": "j1", "source_commit": "a" * 40, "spec_sha256": "s1", "updates": 100},
        {"job_id": "j2", "source_commit": "a" * 40, "spec_sha256": "s2", "updates": 100},
    ]
    rec_res = orch.reconcile_verified_resume(
        [
            {"job_id": "j0", "source_commit": "a" * 40, "spec_sha256": "s0", "completed_updates": 100, "eval_rows": 64, "archive_path": str(good_zip), "archive_sha256": good_sha},
            {"job_id": "j1", "source_commit": "a" * 40, "spec_sha256": "SWAPPED", "completed_updates": 100, "eval_rows": 64, "archive_path": str(good_zip), "archive_sha256": good_sha},
            {"job_id": "j2", "source_commit": "a" * 40, "spec_sha256": "s2", "completed_updates": 100, "eval_rows": 64, "archive_path": str(good_zip), "archive_sha256": good_sha, "required_archive_members": ["MISSING.json"]},
        ],
        reg_rows,
        required_archive_members=("FINAL_STATUS.json",),
    )
    assert rec_res["verified_completed_jobs"] == ["j0"]
    assert rec_res["next_resumable_job_id"] == "j1"
    rej_by_id = {r["job_id"]: r["reasons"] for r in rec_res["rejected_jobs"]}
    assert "SWAPPED_SPEC_SHA256" in rej_by_id["j1"]
    assert "MISSING_ARCHIVE_MEMBER:MISSING.json" in rej_by_id["j2"]

    # 3. verify_shared_backbone_and_head_overhead
    class _Wrap(nn.Module):
        def __init__(self, seed: int, head_out: int) -> None:
            super().__init__()
            torch.manual_seed(seed)
            self.core = nn.Linear(10, 10)
            self.head = nn.Linear(10, head_out)
    ok_bb = verify_shared_backbone_and_head_overhead(_Wrap(7, 1), _Wrap(7, 1), backbone_prefix="core.", max_head_overhead_ratio=0.15, max_control_parity_ratio=0.02)
    assert ok_bb["passed"] is True and ok_bb["backbone_digest_match"] is True
    bad_bb = verify_shared_backbone_and_head_overhead(_Wrap(7, 1), _Wrap(8, 1), backbone_prefix="core.")
    assert bad_bb["passed"] is False and "BACKBONE_DIGEST_MISMATCH" in bad_bb["defects"]

    # 4. Upgraded build_pinned_kaggle_notebook
    nb = build_pinned_kaggle_notebook(
        source_commit="a" * 40,
        title="V8-X2 Kaggle",
        launch_command="python -m experiments.elendril_x2.module",
        extra_pinned_repos={"x1_worker": {"repo_url": "https://example.invalid/repo.git", "commit": "b" * 40}},
        preflight_canary_backend="x1_s20",
        require_dual_gpu=True,
        emit_file_links=True,
    )
    assert len(nb["cells"]) == 5
    nb_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for tok in ("OWNER_AUTHORIZED = True", "device_count() >= 2", "replay-bundle", "FileLink"):
        assert tok in nb_src

    # 5. Root agent.md == anra_v5/planthir/agent.md with Two-Agent Framework Protocol
    root_doc = (REPO_ROOT / "agent.md").read_text(encoding="utf-8")
    pkg_doc = (REPO_ROOT / "anra_v5" / "planthir" / "agent.md").read_text(encoding="utf-8")
    assert root_doc == pkg_doc
    for needle in (
        "Two-Agent Framework Protocol",
        "6-Rule Zero-Degradation Specification",
        "MANDATORY PALANTIRI BUILDER CONTRACT",
        "policy.reset_episode()",
        "in_process_runner",
        "admit_slice",
        "reconcile_verified_resume",
        "verify_shared_backbone_and_head_overhead",
    ):
        assert needle in root_doc


def test_avendi_y3_postmortem_run_registered_plan_full_dispatch_session_verify_notebook_and_docs(tmp_path: Path) -> None:
    import anra_v5.palantiri as pal
    import anra_v5.planter as plt
    import anra_v5.planthir as pla

    commit_sha = "c" * 40
    ds_counts = {"train_families": 1024, "val_families": 256, "test_families": 256}
    jobs = [
        SessionJob(
            job_id=f"j_{arch}_s{seed}",
            architecture=arch,
            init_seed=seed,
            order_seed=seed,
            updates=100,
            session="S1",
            device="cuda:0",
            extra={"eval_rows": 256, "dataset_counts": ds_counts},
        )
        for seed in (7, 19, 41)
        for arch in ("CTRL", "TRT")
    ]

    def _runner(job: SessionJob, wroot: Path, device: str) -> dict:
        ckpt = wroot / "ckpt.json"
        ckpt.write_text(json.dumps({"job_id": job.job_id, "device": device}), encoding="utf-8")
        return {
            "exit_code": 0,
            "completed_updates": job.updates,
            "eval_rows": 256,
            "dataset_counts": ds_counts,
        }

    # 1. CampaignOrchestrator.run_registered_plan: inventory_only + disposable profile + full run + no-full export
    prof_seen_dirs: list[Path] = []

    def _profile(sample_job: SessionJob, prof_dir: Path) -> float:
        prof_seen_dirs.append(prof_dir)
        (prof_dir / "prof_marker.txt").write_text(sample_job.job_id, encoding="utf-8")
        return 24.0

    inv_dir = tmp_path / "inv_run"
    orch_inv = CampaignOrchestrator(
        campaign_id="Y3-INV",
        source_commit=commit_sha,
        work_dir=inv_dir,
        max_minutes=330.0,
        reserve_minutes=45.0,
        min_disk_bytes=1024,
    )
    inv_res = orch_inv.run_registered_plan(
        jobs,
        profile_fn=_profile,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        inventory_only=True,
    )
    assert inv_res["status"] == "INVENTORY_ONLY"
    assert inv_res["dispatched_job_ids"] == []
    assert prof_seen_dirs and not (inv_dir in prof_seen_dirs[0].parents or prof_seen_dirs[0] == inv_dir)
    assert (inv_dir / "PREPARED_MANIFEST.json").is_file()
    assert (inv_dir / "PENDING_QUEUE.json").is_file()
    assert (inv_dir / "INVENTORY_PLAN.json").is_file()

    # Full run with write_full_zip=False AND then default full run with all 4 archives
    no_full_dir = tmp_path / "no_full_run"
    res_no_full = CampaignOrchestrator(
        campaign_id="Y3-NOFULL",
        source_commit=commit_sha,
        work_dir=no_full_dir,
        max_minutes=330.0,
        reserve_minutes=45.0,
        min_disk_bytes=1024,
    ).run_registered_plan(
        jobs,
        job_runner=_runner,
        measured_seconds_per_update=24.0,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        export_prefix="NOFULL",
        write_full_zip=False,
    )
    assert res_no_full["status"] == "COMPLETE"
    assert "full" not in res_no_full["export_receipt"]["archives"]
    assert set(res_no_full["archive_verifications"]) == {"summary", "evidence", "analysis"}

    # 2. verify_full_mode_dispatch passing on healthy multi-session dual-GPU run_registered_plan wrapper
    def _healthy_dispatch(work_dir: Path, *, session_minutes: float, resume_input=None) -> dict:
        orch = CampaignOrchestrator(
            campaign_id="Y3-DISPATCH",
            source_commit=commit_sha,
            work_dir=work_dir,
            max_minutes=330.0,
            reserve_minutes=45.0,
            min_disk_bytes=1024,
        )
        return orch.run_registered_plan(
            jobs,
            job_runner=_runner,
            devices=("cuda:0", "cuda:1"),
            resume_input=resume_input,
            profile_fn=_profile,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
            outer_remaining_minutes=float(session_minutes),
            export_prefix="Y3",
        )

    ok_disp = verify_full_mode_dispatch(
        _healthy_dispatch,
        expected_job_ids=[j.job_id for j in jobs],
        expected_seeds=(7, 19, 41),
        required_devices=("cuda:0", "cuda:1"),
        min_dataset_counts=ds_counts,
        tight_session_minutes=130.0,
        full_session_minutes=300.0,
        work_root=tmp_path / "verify_ok",
    )
    assert ok_disp["passed"] is True and ok_disp["defects"] == []
    assert ok_disp["full_pass"]["dispatched_count"] == 6
    assert set(ok_disp["full_pass"]["dispatched_devices"]) == {"cuda:0", "cuda:1"}
    assert len(ok_disp["multi_session_resume"]["session_1_deferred"]) == 2
    assert set(ok_disp["multi_session_resume"]["session_2_dispatched"]) == set(
        ok_disp["multi_session_resume"]["session_1_deferred"]
    )
    assert json.loads((tmp_path / "verify_ok" / "session_2" / "PENDING_QUEUE.json").read_text(encoding="utf-8"))[
        "pending_job_ids"
    ] == []

    # 3. Tampered spec_sha256 inside resume archive fails closed in run_registered_plan
    s1_zip = Path(tmp_path / "verify_ok" / "session_1" / "exports" / "Y3_ANALYSIS.zip")
    tamper_zip = tmp_path / "tampered_resume.zip"
    ext_tamp = tmp_path / "ext_tamp"
    safe_extract_planthir_archive(s1_zip, ext_tamp)
    jr_doc = json.loads((ext_tamp / "JOB_RECEIPTS.json").read_text(encoding="utf-8"))
    jr_doc["receipts"][0]["spec_sha256"] = "0" * 64
    (ext_tamp / "JOB_RECEIPTS.json").write_text(json.dumps(jr_doc), encoding="utf-8")
    with zipfile.ZipFile(tamper_zip, "w") as zf:
        for f in sorted(ext_tamp.rglob("*")):
            if f.is_file() and f.name not in ("BUNDLE_MANIFEST.json", "README_RESULTS.md", "SHA256SUMS"):
                zf.write(f, arcname=f.relative_to(ext_tamp).as_posix())
    with pytest.raises(ValueError, match="verified resume rejected jobs"):
        CampaignOrchestrator(
            campaign_id="Y3-TAMPER",
            source_commit=commit_sha,
            work_dir=tmp_path / "tamper_work",
            min_disk_bytes=1024,
        ).run_registered_plan(
            jobs,
            job_runner=_runner,
            resume_input=tamper_zip,
            measured_seconds_per_update=24.0,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
        )

    # 4. verify_full_mode_dispatch fails closed on every FULL-mode compression/dispatch defect
    def _bad_missing_and_serial(work_dir: Path, *, session_minutes: float, resume_input=None) -> dict:
        return CampaignOrchestrator(
            campaign_id="Y3-BAD1",
            source_commit=commit_sha,
            work_dir=work_dir,
            max_minutes=330.0,
            reserve_minutes=45.0,
            min_disk_bytes=1024,
        ).run_registered_plan(
            [j for j in jobs if j.init_seed == 7],
            job_runner=lambda job, wroot, dev: {
                "exit_code": 0,
                "completed_updates": job.updates,
                "eval_rows": 32,
                "dataset_counts": {"train_families": 32, "val_families": 8, "test_families": 8},
            },
            devices=("cuda:0",),
            measured_seconds_per_update=1.0,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
            outer_remaining_minutes=float(session_minutes),
        )

    with pytest.raises(ValueError, match="FULL-mode dispatch verification failed"):
        verify_full_mode_dispatch(
            _bad_missing_and_serial,
            expected_job_ids=[j.job_id for j in jobs],
            expected_seeds=(7, 19, 41),
            required_devices=("cuda:0", "cuda:1"),
            min_dataset_counts=ds_counts,
            tight_session_minutes=60.0,
        )
    bad1_res = verify_full_mode_dispatch(
        _bad_missing_and_serial,
        expected_job_ids=[j.job_id for j in jobs],
        expected_seeds=(7, 19, 41),
        required_devices=("cuda:0", "cuda:1"),
        min_dataset_counts=ds_counts,
        tight_session_minutes=60.0,
        raise_on_failure=False,
    )
    defect_text_1 = "\n".join(bad1_res["defects"])
    for code in (
        "MISSING_REGISTERED_JOBS",
        "MISSING_GPU_LANES",
        "SERIAL_DISPATCH_DETECTED",
        "MISSING_REQUIRED_SEEDS",
        "COMPRESSED_DATASET_DENOMINATOR:train_families:32<1024",
        "TIGHT_SESSION_DID_NOT_DEFER_OVERFLOW",
    ):
        assert code in defect_text_1

    def _bad_reexec_resume(work_dir: Path, *, session_minutes: float, resume_input=None) -> dict:
        return CampaignOrchestrator(
            campaign_id="Y3-BAD2",
            source_commit=commit_sha,
            work_dir=work_dir,
            max_minutes=330.0,
            reserve_minutes=45.0,
            min_disk_bytes=1024,
        ).run_registered_plan(
            jobs,
            job_runner=_runner,
            devices=("cuda:0", "cuda:1"),
            resume_input=None,  # Bug: ignores resume_input and re-runs all Session-1 jobs
            measured_seconds_per_update=24.0,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
            outer_remaining_minutes=float(session_minutes),
        )

    bad2_res = verify_full_mode_dispatch(
        _bad_reexec_resume,
        expected_job_ids=[j.job_id for j in jobs],
        expected_seeds=(7, 19, 41),
        required_devices=("cuda:0", "cuda:1"),
        tight_session_minutes=130.0,
        raise_on_failure=False,
    )
    assert any(d.startswith("RESUME_REEXECUTED_COMPLETED_JOBS") for d in bad2_res["defects"])

    # 5. verify_session_archive checks v6 manifest, .sha256 sidecar, max_bytes, source commit & rejects plain ZIPs
    ok_sum = Path(tmp_path / "verify_ok" / "full_pass" / "exports" / "Y3_SUMMARY.zip")
    ver_ok = verify_session_archive(ok_sum, max_bytes=5 * 1024 * 1024, expected_source_commit=commit_sha)
    assert ver_ok["verified"] is True and ver_ok["v6_manifest_verified"] is True and ver_ok["sidecar_verified"] is True

    with pytest.raises(ValueError, match="exceeds cap"):
        verify_session_archive(ok_sum, max_bytes=10)
    with pytest.raises(ValueError, match="SOURCE_COMMIT_MISMATCH"):
        verify_session_archive(ok_sum, expected_source_commit="d" * 40)
    with pytest.raises(ValueError, match="MISSING_ARCHIVE_MEMBER"):
        verify_session_archive(ok_sum, required_members=("DOES_NOT_EXIST.json",))

    (ok_sum.parent / f"{ok_sum.name}.sha256").write_text("0" * 64 + f"  {ok_sum.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SIDECAR_SHA256_MISMATCH"):
        verify_session_archive(ok_sum)

    plain_zip = tmp_path / "plain_cartridge_style.zip"
    with zipfile.ZipFile(plain_zip, "w") as zf:
        zf.writestr("ARCHIVE_MANIFEST.json", "{}")
    with pytest.raises(ValueError, match="MISSING_SIDECAR_SHA256"):
        verify_session_archive(plain_zip, require_sidecar_sha256=True)
    with pytest.raises(ValueError, match="MISSING_V6_BUNDLE_MANIFEST"):
        verify_session_archive(plain_zip, require_sidecar_sha256=False, require_v6_manifest=True)

    # 6. Upgraded build_pinned_kaggle_notebook handles repo_url, FULL controls, and OWNER_AUTHORIZED=False dry-run
    nb = build_pinned_kaggle_notebook(
        source_commit=commit_sha,
        title="Avendi V8-Y3 Kaggle",
        launch_command="python -m experiments.avendi_y3.module --mode FULL",
        owner_authorized=False,
        repo_url="https://example.invalid/anra.git",
        repo_dir="anra_repo",
        campaign_mode="FULL",
        session_minutes=300,
        resume_input="/kaggle/input/s1/Y3_FULL.zip",
        inventory_command="python -m experiments.avendi_y3.module --inventory-only",
        preflight_canary_backend="x1_s20",
        require_dual_gpu=True,
        emit_file_links=True,
    )
    assert len(nb["cells"]) == 4
    c1_src = "".join(nb["cells"][1]["source"])
    gate_src = "".join(nb["cells"][2]["source"])
    for needle in (
        "'clone', '--no-checkout'",
        "CAMPAIGN_MODE = 'FULL'",
        "SESSION_MINUTES = 300",
        "RESUME_INPUT = '/kaggle/input/s1/Y3_FULL.zip'",
        "--inventory-only",
    ):
        assert needle in c1_src
    assert "if REQUIRE_DUAL_GPU and globals().get('OWNER_AUTHORIZED', False):" in gate_src

    # Executing the gate cell's dual-GPU guard under OWNER_AUTHORIZED=False with 0 visible GPUs does not raise
    class _FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 0

    class _FakeTorch:
        cuda = _FakeCuda()

    guard_only_src = gate_src.split("subprocess.check_call", 1)[0]
    guard_only_src = "\n".join(ln for ln in guard_only_src.splitlines() if "import os," not in ln)
    exec_ns = {"OWNER_AUTHORIZED": False, "torch": _FakeTorch()}
    exec(guard_only_src, exec_ns)

    # 7. Three-way agent.md parity + new Avendi Y3 post-mortem contract needles + alias parity
    root_doc = (REPO_ROOT / "agent.md").read_text(encoding="utf-8")
    pkg_doc = (REPO_ROOT / "anra_v5" / "planthir" / "agent.md").read_text(encoding="utf-8")
    res_doc = (REPO_ROOT / "research" / "planthir" / "agent.md").read_text(encoding="utf-8")
    assert root_doc == pkg_doc == res_doc
    for needle in (
        "run_registered_plan",
        "verify_full_mode_dispatch",
        "verify_session_archive",
        "SubgoalProbeSpec",
        "forbidden_metadata_key",
        "archive_sha256",
        "job.worker_id",
        "v5_model.core.initialize",
        "UTF8.decode",
        "pytest -n auto",
        "2,500–3,200",
    ):
        assert needle in root_doc

    assert pal.verify_full_mode_dispatch is pla.verify_full_mode_dispatch
    assert plt.verify_full_mode_dispatch is pla.verify_full_mode_dispatch
    assert pal.verify_session_archive is pla.verify_session_archive
    assert plt.verify_session_archive is pla.verify_session_archive
