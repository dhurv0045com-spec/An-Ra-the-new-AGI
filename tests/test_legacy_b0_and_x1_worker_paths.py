"""Failing-first regression tests for legacy X1 B0 V8BytePolicy adapter & all distinct worker paths."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch

from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import POLICY_MODES, PublicInput, V8ModelSpec
from anra_v5.elendril.v8.policies import V8BytePolicy
from anra_v5.elendril.v8x.contracts import build_x1_model_spec
from anra_v5.elendril.v8x.data import build_x1_d1h_training_dataset, build_x1_d2_paired_dataset, verify_x1_episode
from anra_v5.elendril.v8x.evaluation import (
    X1S20EnvironmentKernel,
    X1S20ObservablePolicy,
    evaluate_x1_worlds_batched,
    run_x1_lesion_and_intervention_battery,
)
from anra_v5.elendril.v8x.models import build_x1_s20_model
from anra_v5.elendril.v8x.training import create_x1_s20_trainer
from tools.elendril_v8x_worker import _load_training_corpus, _save_and_eval_milestone, find_x1_job, prepare_splits

REPO_ROOT = Path(__file__).resolve().parents[1]


def _micro_b0_specs():
    tiny_base = replace(
        MODEL_PROFILES["S20"],
        family="v5-s20-cpu-fixture",
        layers=1,
        width=16,
        query_heads=2,
        kv_heads=1,
        head_dimension=8,
        ffn_width=32,
        context_length=640,
    )
    x1_spec = build_x1_model_spec("B0", base_spec=tiny_base)
    v8_spec = V8ModelSpec(
        architecture="B0",
        base_spec=tiny_base,
        vocabulary_size=tiny_base.vocabulary_size,
        max_sequence_length=x1_spec.max_sequence_length,
    )
    return x1_spec, v8_spec


def test_b0_policy_parity_with_reference_v8_byte_policy() -> None:
    """Compare X1S20ObservablePolicy(B0) against V8BytePolicy on identical weights & public prompts."""
    x1_spec, v8_spec = _micro_b0_specs()
    model = build_x1_s20_model(x1_spec, init_seed=7)
    kernel = X1S20EnvironmentKernel()
    ds = build_x1_d1h_training_dataset(quota_per_family=1)
    worlds = ds.worlds[:4]

    for mode in POLICY_MODES:
        x1_pol = X1S20ObservablePolicy(model, x1_spec, model_identity="b0-parity", mode=mode)
        ref_pol = V8BytePolicy(model, v8_spec, model_identity="b0-parity", mode=mode)

        assert x1_pol.v8_spec is not None
        assert x1_pol.v8_spec.base_spec.sha256() == v8_spec.base_spec.sha256()
        assert x1_pol.v8_spec.vocabulary_size == v8_spec.vocabulary_size
        assert x1_pol.v8_spec.max_sequence_length == v8_spec.max_sequence_length

        obs_list = [kernel.observe(kernel.initial_state(w)) for w in worlds]
        pubs = [
            PublicInput(
                o.rendered_text,
                bool(o.available_tools.get("query", False)),
                bool(o.available_tools.get("retrieve", False)),
                max(1, o.budget_remaining),
            )
            for o in obs_list
        ]

        batched = x1_pol.propose_batch(obs_list)
        scalar = [x1_pol.propose_with_telemetry(o) for o in obs_list]
        ref_batched = ref_pol.propose_batch(pubs)

        for b_act, s_act, (ref_ex, ref_raw), obs in zip(batched, scalar, ref_batched, obs_list):
            assert (b_act.operation, b_act.argument) == (s_act.operation, s_act.argument)
            assert (b_act.operation, b_act.argument) == (ref_ex.operation, ref_ex.argument)
            assert (b_act.raw_operation, b_act.raw_argument) == (ref_raw.operation, ref_raw.argument)
            if not obs.available_tools.get("query", False):
                assert b_act.operation != "QUERY" or mode in ("raw", "grammar")

        # Compare full episode terminal outcomes
        ep_recs = evaluate_x1_worlds_batched(x1_pol, kernel, worlds, batch_size=2)
        assert len(ep_recs) == len(worlds)
        for w, rec in zip(worlds, ep_recs):
            st = kernel.initial_state(w)
            for _ in range(kernel.max_actions):
                if st.terminal:
                    break
                o = kernel.observe(st)
                p = PublicInput(
                    o.rendered_text,
                    bool(o.available_tools.get("query", False)),
                    bool(o.available_tools.get("retrieve", False)),
                    max(1, o.budget_remaining),
                )
                ex, _ = ref_pol.propose(p)
                st = kernel.step(st, ex.operation, ex.argument)
            ref_out = verify_x1_episode(w, st.executed_actions, action_budget=kernel.max_actions)
            assert rec["terminal_status"] == ref_out.terminal_status
            assert rec["correct"] == ref_out.correct
            assert rec["epistemic_supported"] == ref_out.epistemic_supported


def test_real_x1_worker_b0_j_paired_and_u_paths() -> None:
    """Exercise REAL tools/elendril_v8x_worker.py on B0 (w/ u0000+restore+eval), J, D2-R-PAIR-EQ-CF, and U-2."""
    with tempfile.TemporaryDirectory(prefix="x1leg_") as td:
        wdir = Path(td)
        subprocess.run(
            [sys.executable, "-m", "tools.elendril_v8x_worker", "prepare-splits", "--work-dir", str(wdir), "--cpu-fixtures"],
            cwd=str(REPO_ROOT),
            check=True,
        )

        # 1. Fresh full 0 -> u0000 -> u0001 -> u0002 -> final eval via REAL CLI on B0
        b0_fresh_dir = wdir / "fresh_b0"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.elendril_v8x_worker",
                "run-job",
                "--job-id",
                "X1-A3-S20-B0-standard-s7",
                "--work-dir",
                str(b0_fresh_dir),
                "--source",
                "a" * 40,
                "--device",
                "cpu",
                "--checkpoint-interval",
                "1",
                "--cpu-fixtures",
                "--updates-override",
                "2",
            ],
            cwd=str(REPO_ROOT),
            check=True,
        )
        b0_job = json.loads(
            (b0_fresh_dir / "jobs" / "X1-A3-S20-B0-standard-s7" / "job.json").read_text(encoding="utf-8")
        )
        assert b0_job["status"] == "COMPLETE"
        assert b0_job["completed_updates"] == 2
        assert b0_job["milestones_recorded"] == ["u0000", "u0001", "u0002"]

        # 2. Interrupt/restore lifecycle on B0: u0000 -> step(1) -> save() -> REAL CLI --resume -> update 2 -> eval
        b0_res_dir = wdir / "res_b0"
        splits = prepare_splits(work_dir=b0_res_dir, cpu_fixtures=True)
        from tools.elendril_v8x_worker import _load_or_build_splits, _resolve_x1_spec

        loaded_splits = _load_or_build_splits(b0_res_dir, cpu_fixtures=True)
        job_b0 = find_x1_job("X1-A3-S20-B0-standard-s7")
        spec_b0 = _resolve_x1_spec("B0", cpu_fixtures=True)
        ds_b0 = _load_training_corpus(str(job_b0["dataset"]), objective="standard", cpu_fixtures=True)
        tr = create_x1_s20_trainer(
            ds_b0,
            job_b0,
            source_commit="b" * 40,
            root=b0_res_dir / "checkpoints",
            device="cpu",
            cpu_canary=True,
            x1_spec_override=spec_b0,
            updates_override=2,
            batch_size_override=4,
        )
        jdir = b0_res_dir / "jobs" / "X1-A3-S20-B0-standard-s7"
        jdir.mkdir(parents=True, exist_ok=True)
        _save_and_eval_milestone(
            trainer=tr,
            spec=spec_b0,
            job=job_b0,
            update=0,
            job_dir=jdir,
            shared_bb_dir=b0_res_dir / "shared_backbones",
            splits=loaded_splits,
            is_final=False,
        )
        st1 = tr.step()
        ckpt1_sha = tr.save(prune_old=True)
        (jdir / "updates.jsonl").write_text(json.dumps(st1, sort_keys=True) + "\n", encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.elendril_v8x_worker",
                "run-job",
                "--job-id",
                "X1-A3-S20-B0-standard-s7",
                "--work-dir",
                str(b0_res_dir),
                "--source",
                "b" * 40,
                "--device",
                "cpu",
                "--checkpoint-interval",
                "1",
                "--resume",
                "--cpu-fixtures",
                "--updates-override",
                "2",
            ],
            cwd=str(REPO_ROOT),
            check=True,
        )
        res_doc = json.loads((jdir / "job.json").read_text(encoding="utf-8"))
        assert res_doc["status"] == "COMPLETE"
        assert res_doc["completed_updates"] == 2
        assert res_doc["latest_checkpoint_sha256"] != ckpt1_sha
        assert (jdir / "milestones" / "u0002" / "summary.json").is_file()
        assert (jdir / "milestones" / "u0002" / "episodes.jsonl").is_file()

        # 3. Remaining distinct X1 worker paths: J and U-2 via REAL run-job CLI
        for jid in (
            "X1-A3-S20-J-standard-s7",
            "X1-U0-S20-U-2-standard-s7",
        ):
            sub_dir = wdir / f"run_{jid}"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.elendril_v8x_worker",
                    "run-job",
                    "--job-id",
                    jid,
                    "--work-dir",
                    str(sub_dir),
                    "--source",
                    "c" * 40,
                    "--device",
                    "cpu",
                    "--checkpoint-interval",
                    "1",
                    "--cpu-fixtures",
                    "--updates-override",
                    "2",
                ],
                cwd=str(REPO_ROOT),
                check=True,
            )
            jrec = json.loads((sub_dir / "jobs" / jid / "job.json").read_text(encoding="utf-8"))
            assert jrec["status"] == "COMPLETE"
            assert jrec["completed_updates"] == 2
            diag = json.loads(
                (sub_dir / "jobs" / jid / "milestones" / "u0002" / "diagnostics.json").read_text(encoding="utf-8")
            )
            if jid.endswith("-U-2-standard-s7"):
                assert set(diag["lesions"]["u_depth_sweep"]) == {"1", "2", "4", "8"}

        # 4. Paired D2 worker path: REAL canary CLI on R PAIR-EQ-CF & P PAIR-CE + R lesions
        for arm_name, obj_name in (("R", "PAIR-EQ-CF"), ("P", "PAIR-CE")):
            can_json = wdir / f"canary_{arm_name}_{obj_name}.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.elendril_v8x_worker",
                    "canary",
                    "--arm",
                    arm_name,
                    "--objective",
                    obj_name,
                    "--work-dir",
                    str(wdir / f"can_{arm_name}"),
                    "--source",
                    "d" * 40,
                    "--device",
                    "cpu",
                    "--cpu-fixtures",
                    "--json-out",
                    str(can_json),
                ],
                cwd=str(REPO_ROOT),
                check=True,
            )
            crec = json.loads(can_json.read_text(encoding="utf-8"))
            assert crec["passed"] is True
            assert crec["resume_weights_match"] is True
            assert crec["resume_loss_diff"] <= 1e-6

        d1h = build_x1_d1h_training_dataset(quota_per_family=1)
        d2 = build_x1_d2_paired_dataset(base_worlds=d1h.worlds[:4])
        eligible_eq = sum(1 for g in d2.groups for p in g.eq_pairs if p.eligible)
        eligible_cf = sum(1 for g in d2.groups for p in g.cf_pairs if p.eligible)
        assert len(d2.groups) > 0 and eligible_eq > 0 and eligible_cf > 0

        from tools.elendril_v8x_worker import _cpu_micro_base

        r_spec = build_x1_model_spec(
            "R",
            base_spec=_cpu_micro_base(),
            pointer_dim=8,
            slot_dim=8,
            num_slots=4,
            slot_edge_hidden=8,
        )
        r_job = find_x1_job("X1-D2-S20-R-PAIR-EQ-CF-s7")
        from dataclasses import replace as dc_replace

        d2_sub = dc_replace(d2, groups=d2.groups[:4])
        r_tr = create_x1_s20_trainer(
            d2_sub,
            r_job,
            source_commit="d" * 40,
            root=wdir / "r_paired_ckpts",
            device="cpu",
            cpu_canary=True,
            x1_spec_override=r_spec,
            updates_override=2,
            batch_size_override=4,
        )
        r_tr.step()
        r_tr.save()
        r_tr.step()
        r_tr.save()
        lesions = run_x1_lesion_and_intervention_battery(
            r_tr.model,
            r_spec,
            loaded_splits["dev_views"],
            model_identity="X1-D2-S20-R-PAIR-EQ-CF-s7:test",
        )
        assert {"zero_r", "permute_edges", "sham", "disable_copy", "permute_val_bins"} <= set(lesions["interventions"])
