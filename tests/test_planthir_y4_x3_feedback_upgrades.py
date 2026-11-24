"""Verification tests for the 2026-11-24 Avendi Y4 + Elendril X3 PALANTIRI_FEEDBACK Level-6 upgrades."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.v8.contracts import SYMBOLS
from anra_v5.planthir import (
    CheckpointRecord,
    ExperimentModule,
    GenericExperimentTrainer,
    GenericModelSpec,
    GenericTrainRecipe,
    LogitsPolicy,
    Palantiri,
    SubgoalProbeSpec,
    UniversalAction,
    UniversalObservation,
    analyze_checkpoint_trajectory,
    build_pinned_kaggle_notebook,
    build_x1_spec,
    compute_effective_rank,
    compute_representation_isotropy,
    InternalTelemetryCollector,
    run_representation_linear_probe,
    scaffold_experiment_module,
)
from anra_v5.planthir.cartridge import _collect_capsule_arrays, _probe_initial_actions_and_hidden
from anra_v5.planthir.causal import _eval_ridge_classifier, _fit_and_eval_mlp_probe, _fit_ridge_classifier
from anra_v5.planthir.validation_study import _collect_capsule_arrays as _val_collect_capsule_arrays

REPO_ROOT = Path(__file__).resolve().parents[1]


_FORBIDDEN_CUDA_FUNCS = {
    torch.cat,
    torch.stack,
    torch.linalg.solve,
    torch.linalg.svdvals,
    torch.matmul,
    torch.Tensor.__matmul__,
    F.normalize,
    torch.sub,
    torch.Tensor.sub,
    torch.Tensor.__sub__,
    torch.Tensor.mean,
    torch.Tensor.scatter_,
}


class _SimulatedCudaTensor(torch.Tensor):
    """CPU-runnable Tensor subclass simulating `cuda:0` placement for device-handoff tests."""

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        def _has_sim(obj: Any) -> bool:
            if isinstance(obj, _SimulatedCudaTensor):
                return True
            if isinstance(obj, (list, tuple)):
                return any(_has_sim(item) for item in obj)
            if isinstance(obj, dict):
                return any(_has_sim(val) for val in obj.values())
            return False

        kw = kwargs or {}
        if func is torch.Tensor.cpu:
            base = args[0].as_subclass(torch.Tensor).detach().clone()
            return base
        if func in _FORBIDDEN_CUDA_FUNCS and (_has_sim(args) or _has_sim(kw)):
            raise RuntimeError(f"wrapper_CUDA_cat simulated device mismatch in {func}")
        out = super().__torch_function__(func, types, args, kw)
        if func in (torch.Tensor.detach, torch.Tensor.float, torch.Tensor.long) and isinstance(out, torch.Tensor):
            return _wrap_sim_cuda(out.as_subclass(torch.Tensor), require_grad=False)
        return out


def _wrap_sim_cuda(tensor: torch.Tensor, *, require_grad: bool = False) -> _SimulatedCudaTensor:
    return torch.Tensor._make_subclass(_SimulatedCudaTensor, tensor.detach().clone(), require_grad=require_grad)  # type: ignore[arg-type]


class _SimKernel:
    kernel_id = "sim_cuda_kernel"
    max_actions = 2

    def initial_state(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return {"world": dict(world), "terminal": False}

    def observe(self, state: Mapping[str, Any]) -> UniversalObservation:
        w = state["world"]
        return UniversalObservation(
            obs_id=str(w["world_id"]),
            rendered_text=str(w["prompt"]),
            available_tools={"query": False, "retrieve": False},
            consumed_tools=frozenset(),
            decision_index=0,
            budget_remaining=2,
            metadata={"world_tag": str(w["world_id"]), "feat": list(w["feat"])},
        )

    def ideal_action(self, state: Mapping[str, Any]) -> tuple[str, str | None]:
        return ("ANSWER", str(state["world"]["ans"]))

    def step(self, state: Mapping[str, Any], op: str, arg: str | None) -> dict[str, Any]:
        return {"world": state["world"], "terminal": True}

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        expected = ("ANSWER", str(world["ans"]))
        return {"correct": bool(actions and tuple(actions[-1]) == expected)}

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        return (
            SubgoalProbeSpec(
                subgoal_id=f"{world['world_id']}_h1",
                stage_name="hop_1",
                hop_index=1,
                modified_world=dict(world),
                expected_operation="ANSWER",
                expected_argument=str(world["ans"]),
            ),
        )

    def structural_signature(self, world: Mapping[str, Any]) -> str:
        return str(world["struct_sig"])

    def inject_oracle_information(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return dict(world)

    def target_symbol(self, world: Mapping[str, Any]) -> str:
        return str(world["ans"])


class _SimCudaPolicy:
    policy_id = "sim_cuda_policy"

    def __init__(self) -> None:
        self.resets = 0

    def reset_episode(self) -> None:
        self.resets += 1

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return UniversalAction(
            operation="ANSWER",
            argument=SYMBOLS[0],
            log_probability=-0.1,
            action_distribution={"ANSWER": 1.0},
            mode="raw",
            raw_operation="ANSWER",
            raw_argument=SYMBOLS[0],
            telemetry={"pointer_gate": 0.5},
        )

    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        n = len(obs_batch)
        feats = torch.tensor(
            [
                obs.metadata["feat"] if "feat" in obs.metadata else [float(i + 1), -0.5, 0.25, 0.1]
                for i, obs in enumerate(obs_batch)
            ],
            dtype=torch.float32,
        )
        gates = torch.linspace(0.1, 0.9, max(1, n), dtype=torch.float32)
        ents = torch.linspace(0.2, 0.8, max(1, n), dtype=torch.float32)
        return {
            "decision_hidden": _wrap_sim_cuda(feats, require_grad=True),
            "pointer_gate": _wrap_sim_cuda(gates, require_grad=True),
            "pointer_entropy": _wrap_sim_cuda(ents, require_grad=True),
        }


def test_simulated_cuda_representation_consumers_enforce_detach_cpu() -> None:
    torch.manual_seed(11)
    x_tr = _wrap_sim_cuda(torch.randn((8, 4)), require_grad=True)
    y_tr = _wrap_sim_cuda(torch.tensor([i % 4 for i in range(8)], dtype=torch.int64))
    x_ev = _wrap_sim_cuda(torch.randn((4, 4)), require_grad=True)
    y_ev = _wrap_sim_cuda(torch.tensor([i % 4 for i in range(4)], dtype=torch.int64))

    w_ridge = _fit_ridge_classifier(x_tr, y_tr, 4)
    assert type(w_ridge) is torch.Tensor and not w_ridge.requires_grad
    w_cuda = _wrap_sim_cuda(w_ridge, require_grad=True)
    acc = _eval_ridge_classifier(w_cuda, x_ev, y_ev)
    assert 0.0 <= acc <= 1.0

    mlp_acc = _fit_and_eval_mlp_probe(x_tr, y_tr, x_ev, y_ev, 4, hidden_dim=8, steps=4, seed=7)
    assert 0.0 <= mlp_acc <= 1.0

    # Confirm the simulated guard actually fires on un-CPU'd tensors
    with pytest.raises(RuntimeError, match="wrapper_CUDA_cat"):
        torch.cat([x_tr, torch.ones((8, 1))], dim=1)

    kernel = _SimKernel()
    policy = _SimCudaPolicy()
    symbols = SYMBOLS[:4]
    tr_worlds = [
        {
            "world_id": f"tr_{i}",
            "struct_sig": f"sig_tr_{i}",
            "prompt": f"p tr {i}",
            "ans": symbols[i % 4],
            "hops": 1,
            "feat": [1.0 if i % 2 == 0 else -1.0, float((i % 4) - 1.5), 0.3, -0.2],
            "hidden": {"expected": symbols[i % 4]},
        }
        for i in range(6)
    ]
    ev_worlds = [
        {
            "world_id": f"ev_{i}",
            "struct_sig": f"sig_ev_{i}",
            "prompt": f"p ev {i}",
            "ans": symbols[i % 4],
            "hops": 2,
            "feat": [1.0 if i % 2 == 0 else -1.0, float((i % 4) - 1.5), 0.25, -0.15],
            "hidden": {"expected": symbols[i % 4]},
        }
        for i in range(4)
    ]

    probe_out = run_representation_linear_probe(
        policy, kernel, tr_worlds, ev_worlds, include_nonlinear_probe=True, nonlinear_hidden_dim=8, nonlinear_steps=5
    )
    assert probe_out["probe_operation_accuracy"] == pytest.approx(1.0)
    assert probe_out["probe_final_symbol_accuracy"] is not None
    assert probe_out["nonlinear_probe_final_symbol_accuracy"] is not None

    cap_cart = _collect_capsule_arrays(policy, kernel, ev_worlds)
    cap_val = _val_collect_capsule_arrays(policy, kernel, ev_worlds)
    for cap in (cap_cart, cap_val):
        assert type(cap["decision_hidden"]) is torch.Tensor and not cap["decision_hidden"].requires_grad
        assert type(cap["pointer_gate"]) is torch.Tensor and not cap["pointer_gate"].requires_grad

    obs_list = [kernel.observe(kernel.initial_state(w)) for w in ev_worlds]
    actions, probed_hidden = _probe_initial_actions_and_hidden(policy, obs_list)
    assert len(actions) == len(ev_worlds)
    assert type(probed_hidden) is torch.Tensor and not probed_hidden.requires_grad

    rank = compute_effective_rank(x_tr)
    iso = compute_representation_isotropy(x_tr)
    assert rank >= 1.0 and iso["isotropy"] is not None

    tiny_mod = nn.Linear(4, 2)
    snap = InternalTelemetryCollector().measure_snapshot(
        tiny_mod, step=1, lr=1e-3, policy=policy, probe_observations=obs_list
    )
    assert snap["representation"]["hidden_dim"] == 4
    assert "pointer_gate_mean" in snap["mechanism_telemetry"]

    rec = CheckpointRecord(
        checkpoint_sha256="a" * 64,
        lineage_id="main",
        checkpoint_root="ckpt",
        generation=1,
        global_update=1,
        cumulative_tokens=8,
        curriculum_phase="p1",
        source_commit="b" * 40,
        model_spec_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
        program_family="sim",
        architecture="F",
        parameter_count=10,
        total_byte_size=128,
        is_latest=True,
        is_milestone=True,
    )
    traj = analyze_checkpoint_trajectory(
        [rec],
        kernel=kernel,
        eval_worlds=ev_worlds,
        probe_train_worlds=tr_worlds,
        load_fn=lambda _rec, mode="raw": {"policy": policy},
        probe_fn=run_representation_linear_probe,
    )
    assert traj["checkpoints_evaluated"] == 1

    class _SimForwardNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = self.lin(x)
            return logits, _wrap_sim_cuda(x, require_grad=True)

    lp = LogitsPolicy(_SimForwardNet(), action_labels=("A", "B"))
    lp_reps = lp.extract_representations(obs_list)
    assert type(lp_reps["decision_hidden"]) is torch.Tensor and not lp_reps["decision_hidden"].requires_grad


def test_trainer_post_step_and_train_loop_save_final_and_post_train_hook(tmp_path: Path) -> None:
    items = [
        {"x": [1.0, 0.5, -0.5, 0.25], "y": 0},
        {"x": [-1.0, -0.5, 0.5, -0.25], "y": 1},
        {"x": [0.8, 0.4, -0.4, 0.2], "y": 0},
        {"x": [-0.8, -0.4, 0.4, -0.2], "y": 1},
    ]

    def _loss(m: nn.Module, chunk: Sequence[Mapping[str, Any]], _rcp: Any) -> torch.Tensor:
        x = torch.tensor([row["x"] for row in chunk], dtype=torch.float32)
        y = torch.tensor([row["y"] for row in chunk], dtype=torch.int64)
        return F.cross_entropy(m(x), y)

    def _radial_cap(trainer: GenericExperimentTrainer, max_norm: float = 0.5) -> None:
        with torch.no_grad():
            for p in trainer.model.parameters():
                nrm = float(p.norm().item())
                if nrm > max_norm:
                    p.mul_(max_norm / nrm)

    recipe = GenericTrainRecipe(seed=7, total_updates=4, batch_size=2, micro_batch_size=2, lr=0.05)

    # A. post_step_hook runs after every optimizer step
    hook_updates: list[int] = []

    def _per_step(tr: GenericExperimentTrainer, meta: dict[str, Any]) -> None:
        hook_updates.append(int(meta["update"]))
        _radial_cap(tr, max_norm=0.5)

    m_a = nn.Linear(4, 2)
    spec_a = GenericModelSpec("linear", {"w": 4}, sum(p.numel() for p in m_a.parameters()), "test_hooks")
    tr_a = GenericExperimentTrainer(
        backend_id="test_hooks",
        model=m_a,
        model_spec=spec_a,
        recipe=recipe,
        loss_fn=_loss,
        items=items,
        data_provenance={"id": "tiny"},
        order_seed=7,
        source_commit="a" * 40,
        root=tmp_path / "run_a",
        lineage="main",
        post_step_hook=_per_step,
    )
    res_a = tr_a.train_loop(save_final=True)
    assert hook_updates == [1, 2, 3, 4]
    assert res_a["updates_completed"] == 4 and res_a["saved_final"] is True
    assert all(float(p.norm().item()) <= 0.5 + 1e-6 for p in tr_a.model.parameters())

    # B. save_final=False avoids double-save failures before manual post-train intervention + single save()
    m_b = nn.Linear(4, 2)
    spec_b = GenericModelSpec("linear", {"w": 4}, sum(p.numel() for p in m_b.parameters()), "test_hooks")
    tr_b = GenericExperimentTrainer(
        backend_id="test_hooks",
        model=m_b,
        model_spec=spec_b,
        recipe=recipe,
        loss_fn=_loss,
        items=items,
        data_provenance={"id": "tiny"},
        order_seed=7,
        source_commit="a" * 40,
        root=tmp_path / "run_b",
        lineage="main",
    )
    res_b = tr_b.train_loop(checkpoint_every=2, save_final=False)
    assert res_b["saved_final"] is False and tr_b.published_update == 2
    _radial_cap(tr_b, max_norm=0.25)
    sha_b = tr_b.save()
    assert sha_b == tr_b.parent and tr_b.published_update == 4
    with pytest.raises(ValueError, match="cannot publish faulted or repeated update"):
        tr_b.save()

    restored_b = GenericExperimentTrainer(
        backend_id="test_hooks",
        model=nn.Linear(4, 2),
        model_spec=spec_b,
        recipe=recipe,
        loss_fn=_loss,
        items=items,
        data_provenance={"id": "tiny"},
        order_seed=7,
        source_commit="a" * 40,
        root=tmp_path / "run_b",
        lineage="main",
        resume=True,
    )
    for k, v in tr_b.model.state_dict().items():
        assert torch.allclose(restored_b.model.state_dict()[k], v)

    # C. checkpoint_every dividing total_updates + post_train_hook + save_final=True publishes capped weights cleanly
    post_calls = 0

    def _post_cap(tr: GenericExperimentTrainer) -> None:
        nonlocal post_calls
        post_calls += 1
        _radial_cap(tr, max_norm=0.15)

    m_c = nn.Linear(4, 2)
    spec_c = GenericModelSpec("linear", {"w": 4}, sum(p.numel() for p in m_c.parameters()), "test_hooks")
    tr_c = GenericExperimentTrainer(
        backend_id="test_hooks",
        model=m_c,
        model_spec=spec_c,
        recipe=recipe,
        loss_fn=_loss,
        items=items,
        data_provenance={"id": "tiny"},
        order_seed=7,
        source_commit="a" * 40,
        root=tmp_path / "run_c",
        lineage="main",
    )
    res_c = tr_c.train_loop(checkpoint_every=2, post_train_hook=_post_cap, save_final=True)
    assert post_calls == 1 and res_c["saved_final"] is True and tr_c.published_update == 4
    assert all(float(p.norm().item()) <= 0.15 + 1e-6 for p in tr_c.model.parameters())


def test_build_x1_spec_canary_and_full_aliases() -> None:
    all_archs = ("B0", "J", "F", "F-cap", "P", "P-capR", "R", "R-noedge", "U-1", "U-2", "U-4", "U2-untied")
    for arch in all_archs:
        spec = build_x1_spec(arch, cpu_canary=True)
        assert spec.architecture == arch
        assert spec.total_parameters() <= 100_000

    for alias, expected in (
        ("X1-P", "P"),
        ("X1-R", "R"),
        ("X1-R-noedge", "R-noedge"),
        ("X1-P-capR", "P-capR"),
    ):
        spec = build_x1_spec(alias, cpu_canary=True)
        assert spec.architecture == expected
        assert spec.total_parameters() <= 100_000

    r_canary = build_x1_spec("X1-R", cpu_canary=True)
    for ctrl_arch in ("R-noedge", "P-capR"):
        ctrl_spec = build_x1_spec(ctrl_arch, cpu_canary=True)
        rel_diff = abs(ctrl_spec.added_parameters() - r_canary.added_parameters()) / r_canary.added_parameters()
        assert rel_diff <= 0.01

    full_r = Palantiri.build_x1_spec("X1-R", cpu_canary=False)
    assert full_r.architecture == "R" and full_r.base_spec.parameter_receipt().total == 19_871_616

    with pytest.raises(ValueError, match="cleanly divisible"):
        build_x1_spec("P", width=33, query_heads=2)


def test_notebook_default_owner_authorized_true_and_explicit_false_and_verify_scaffold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nb_default = build_pinned_kaggle_notebook(
        source_commit="a" * 40,
        title="Default Auto NB",
        launch_command="python -m tool",
    )
    assert "OWNER_AUTHORIZED = True" in "".join(nb_default["cells"][1]["source"])

    nb_dry = build_pinned_kaggle_notebook(
        source_commit="a" * 40,
        title="Explicit Dry Run NB",
        launch_command="python -m tool",
        owner_authorized=False,
    )
    assert "OWNER_AUTHORIZED = False" in "".join(nb_dry["cells"][1]["source"])

    scaf_root = tmp_path / "scaf"
    res = scaffold_experiment_module("audit_pkg", scaf_root, include_verify_module=True)
    verify_path = scaf_root / "audit_pkg" / "verify.py"
    assert verify_path.is_file()
    assert str(verify_path) in res["created_files"]

    monkeypatch.syspath_prepend(str(scaf_root))
    sys.modules.pop("audit_pkg", None)
    sys.modules.pop("audit_pkg.verify", None)
    import importlib

    vmod = importlib.import_module("audit_pkg.verify")
    audit_res = vmod.verify_campaign_claims({"j1": {"status": "COMPLETE"}}, expected_job_ids=["j1"])
    assert audit_res["passed"] is True and audit_res["records_audited"] == 1

    # Confirm 3-way manual mirror parity and new feedback documentation
    for name in ("orchestrator_agent.md", "execution_agent.md", "agent.md"):
        root_b = (REPO_ROOT / name).read_bytes()
        assert root_b == (REPO_ROOT / "anra_v5" / "planthir" / name).read_bytes()
        assert root_b == (REPO_ROOT / "research" / "planthir" / name).read_bytes()
        text = root_b.decode("utf-8")
        for needle in ("verify.py", "build_x1_spec", "train_loop", "post_step_hook", "owner_authorized=True", ".detach().cpu()"):
            assert needle in text, f"missing {needle!r} in {name}"
        if name != "orchestrator_agent.md":
            assert "cannot publish faulted or repeated update" in text
            assert "zipfile" in text
        if name == "orchestrator_agent.md":
            assert "--format=zip" in text and "zipfile" in text
