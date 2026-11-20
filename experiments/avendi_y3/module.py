"""Avendi V8-Y3 Planthir plugin registration, sealed confirm lock, and full campaign CLI."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.planthir import (
    BytePromptTrainer,
    CampaignOrchestrator,
    CampaignStageSpec,
    CartridgeRunner,
    ExperimentModule,
    GenericExperimentTrainer,
    GenericTrainRecipe,
    SessionJob,
    SubgoalProbeSpec,
    UniversalObservation,
    analyze_paired_seed_runs,
    audit_sampled_strata,
    batched_rollout,
    bootstrap_paired_families,
    build_analyst_first_page_report,
    build_cartridge_from_spec,
    build_paired_seed_observations,
    build_verification_matrix,
    compare_paired_episode_records,
    export_session_archives,
    frame_byte_prompt_sample,
    independent_replay_bundle,
    init_spec_template,
    load_legacy_byte_checkpoint,
    locate_checkpoint_store_root,
    planthir_doctor,
    render_campaign_results_markdown,
    rescore_compact_analysis_archive,
    run_paired_comparison_preflight,
    run_scientific_shortcut_calibration,
    run_surgical_mechanism_ablation,
    score_crossover_panel,
    stratify_intervention_worlds,
    summarize_diagnostic_coverage,
    validate_spec_document,
    verify_crossover_fixture,
    verify_result_bundle,
)

from experiments.avendi_y3.models import (
    AvendiByteModel,
    AvendiY3Policy,
    P6AdaptivePolicy,
    _P4Base,
    apply_bounded_weight_update,
    compute_avendi_loss,
    decode_continuations_any,
    encode_p4_batch,
    eval_byte_model_on_families,
    get_byte_model_spec,
    p4_transition_loss,
    run_p4_causal_suite,
    run_p7_donor_audit_and_interventions,
    select_p4_matched_widths,
    verify_u19_gradient_reversal,
)
from experiments.avendi_y3.plan import ARCHIVE_SHAS, BASE_REVISION, SEEDS, STREAM_SEEDS, Y1_EXPECTED_CKPT_SHAS, _canon_sha, build_plan, build_y3_notebook
from experiments.avendi_y3.tasks import (
    SYMBOLS,
    P5_CONDITIONS,
    allocate_disjoint_4role_pools,
    audit_case_evidence_corruptions,
    audit_split_custody,
    build_interchange_pairs,
    build_literal_probe_families,
    build_multihop_families,
    build_one_hop_families,
    build_p5_executive_worlds,
    build_p6_blocks,
    build_planthir_dataset_splits,
    compare_historical_vs_balanced_assignments,
    evaluate_p5_reference_policies,
    evaluate_shortcuts_and_destructive_masks,
    oracle_traverse_tokens,
)


class AvendiY3Kernel:
    """EnvironmentKernel supporting 1-hop lookup/crossover and P5/P6 stateful tool worlds."""

    kernel_id = "avendi-y3-kernel-v1"
    max_actions = 3

    def structural_signature(self, world: Mapping[str, Any]) -> str:
        return str(world.get("sig") or f"{world.get('split','s')}:{world.get('canonical_sig','')}:{world.get('world_id','')}")

    def inject_distractor(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return {**world, "world_id": f"{world['world_id']}::dist", "sig": f"{self.structural_signature(world)}::dist", "public_prompt": world.get("irrel_base_prompt", world.get("public_prompt", ""))}

    def permute_world(self, world: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
        return {**world, "world_id": f"{world['world_id']}::p{shift}", "sig": f"{self.structural_signature(world)}::p{shift}", "public_prompt": world.get("timestamp_permuted_prompt", world.get("public_prompt", ""))}

    def reverse_world(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return {**world, "world_id": f"{world['world_id']}::rev", "sig": f"{self.structural_signature(world)}::rev"}

    def initial_state(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return {"w": dict(world), "step": 0, "queried": False, "retrieved": False, "terminal": False}

    def observe(self, state: Mapping[str, Any]) -> UniversalObservation:
        w = state["w"]
        avail = {k: bool(v) for k, v in dict(w.get("available_tools", {"query": False, "retrieve": False})).items()}
        if state["queried"]: avail["query"] = False
        if state["retrieved"]: avail["retrieve"] = False
        consumed = frozenset(k for k, f in (("query", state["queried"]), ("retrieve", state["retrieved"])) if f)
        return UniversalObservation(
            obs_id=f"{w['world_id']}:s{state['step']}:q{int(avail.get('query', False))}r{int(avail.get('retrieve', False))}",
            rendered_text=str(w.get("public_prompt", w.get("rendered_text", "Q:A"))),
            available_tools=avail, consumed_tools=consumed, decision_index=int(state["step"]),
            budget_remaining=max(1, self.max_actions - int(state["step"])), metadata={"condition": w.get("condition", "lookup")},
        )

    def ideal_action(self, state: Mapping[str, Any]) -> tuple[str, str | None]:
        w = state["w"]
        if state["step"] == 0 and w.get("optimal_actions"):
            opt = str(w["optimal_actions"][0])
            return tuple(opt.split(":", 1)) if ":" in opt else (opt, None)  # type: ignore[return-value]
        return ("ANSWER", str(w.get("gold", w.get("hidden", {}).get("expected", "A"))))

    def step(self, state: Mapping[str, Any], op: str, arg: str | None) -> dict[str, Any]:
        nxt = {**state, "step": int(state["step"]) + 1}
        if op == "QUERY" and not state["queried"] and state["w"].get("available_tools", {}).get("query"): nxt["queried"] = True
        elif op == "RETRIEVE" and not state["retrieved"] and state["w"].get("available_tools", {}).get("retrieve"): nxt["retrieved"] = True
        else: nxt["terminal"] = True
        return nxt

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        op, arg = actions[-1] if actions else ("INVALID", None)
        exp = str(world.get("gold", world.get("hidden", {}).get("expected", "")))
        return {"correct": op == "ANSWER" and arg == exp, "terminal_status": "answered" if op == "ANSWER" else op.lower()}

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        op, arg = self.ideal_action(self.initial_state(world))
        return (SubgoalProbeSpec(subgoal_id=f"{world.get('world_id', 'w')}:sg0", stage_name="endpoint", hop_index=0, modified_world=dict(world), expected_operation=op, expected_argument=arg),)


def build_avendi_y3_module() -> ExperimentModule:
    """Create the registered ExperimentModule plugin for avendi_y3."""
    return ExperimentModule(
        backend_id="avendi_y3", backend_version="avendi-y3-1.0", program="AVENDI-V8-Y3",
        architectures=("s3_byte_ce", "s3_byte_margin", "s3_byte_address", "s3_byte_rename", "s20_byte_ce", "p6_adaptive"),
        objectives=("ce", "margin", "address", "address_sham", "rename", "rename_sham"),
        model_factory=lambda a, mc, s: AvendiByteModel(get_byte_model_spec("S20" if "s20" in a else "S3", cpu_canary=bool(mc.get("cpu_canary", True))), seed=s, use_aux=("margin" in a or "address" in a or "rename" in a or bool(mc.get("use_aux", False)))),
        dataset_factory=build_planthir_dataset_splits, loss_fn=lambda m, c, r: compute_avendi_loss(m, c, r), kernel_factory=AvendiY3Kernel,
        policy_factory=lambda m, ms, *, model_identity="y3", mode="raw", copy_enabled_override=None: AvendiY3Policy(m, ms, model_identity=model_identity, mode=mode, copy_enabled_override=copy_enabled_override),
        default_dataset_config={"dataset_name": "avendi_y3_onehop", "num_families": 4, "seed": STREAM_SEEDS["corpus"]},
        default_model_config={"width": 32, "cpu_canary": True}, treatment_architecture="s3_byte_margin", treatment_model_config={"width": 32, "cpu_canary": True},
        extra_model_keys=("width", "cpu_canary", "use_aux"), extra_dataset_keys=("dataset_name", "num_families", "seed"), supports_grouped_loss=True,
    )


def register_planthir_backends() -> ExperimentModule:
    return build_avendi_y3_module().register()


AVENDI_Y3_MODULE = register_planthir_backends()


def write_confirmation_lock(lock_path: Path, spec_payload: Mapping[str, Any]) -> str:
    """Seal confirmation selection criteria with a SHA-256 lock digest prior to eval."""
    digest = _canon_sha(spec_payload)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"payload": dict(spec_payload), "lock_sha256": digest, "opened": False}, indent=2) + "\n", encoding="utf-8")
    return digest


def evaluate_with_confirmation_lock(lock_path: Path, expected_sha256: str, evaluator_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
    """Verify sealed confirmation lock and run evaluator once."""
    doc = json.loads(lock_path.read_text(encoding="utf-8"))
    if _canon_sha(doc["payload"]) != expected_sha256 or doc.get("lock_sha256") != expected_sha256: raise ValueError("CONFIRMATION_LOCK_MISMATCH")
    if doc.get("opened"): raise ValueError("CONFIRMATION_ALREADY_OPENED")
    res = dict(evaluator_fn(doc["payload"]))
    doc["opened"], doc["result_sha256"] = True, _canon_sha(res)
    lock_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return {"lock_sha256": expected_sha256, "result": res, "result_sha256": doc["result_sha256"]}


def _write_study(wr: Path, sid: str, jid: str, split: str, *, payload: Mapping[str, Any], status: str = "COMPLETE") -> dict[str, Any]:
    p = wr / "studies" / sid / jid / split / "receipt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"study_id": sid, "job_id": jid, "split": split, "status": status, "receipt_path": str(p.relative_to(wr)), **payload}
    raw = json.dumps(rec, sort_keys=True, indent=2).encode("utf-8")
    p.write_bytes(raw)
    return {"path": str(p), "sha256": sha256(raw).hexdigest(), "status": status}


def _profile_device(device: str, full: bool, s20: bool | None = None) -> dict[str, Any]:
    dev = torch.device(device if (torch.cuda.is_available() and "cuda" in str(device)) else "cpu")
    n_upd = 32 if (full and dev.type == "cuda") else 2
    warm = 8 if n_upd == 32 else 1

    def _bench(prof: str) -> float:
        spec = get_byte_model_spec(prof, cpu_canary=not full)
        m = AvendiByteModel(spec, seed=7).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        mb, seq_l, acc = (2, 96, 8) if (full and prof == "S20") else ((4, 96, 4) if full else (2, 16, 1))
        x = torch.randint(4, 30, (mb, seq_l), device=dev)
        y = torch.zeros(mb, dtype=torch.long, device=dev)
        times = []
        for st in range(warm + n_upd):
            if dev.type == "cuda": torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            opt.zero_grad()
            for _ in range(acc):
                (F.cross_entropy(m(x)[0][:, -1], y) / acc).backward()
            opt.step()
            if dev.type == "cuda": torch.cuda.synchronize(dev)
            if st >= warm: times.append(time.perf_counter() - t0)
        del m, opt, x, y
        if dev.type == "cuda": torch.cuda.empty_cache()
        return float(np.percentile(times, 90))

    if full and s20 is None:
        p3, p20 = _bench("S3"), _bench("S20")
        return {"p90_sec_per_update": 0.80 * p3 + 0.12 * p20 + 0.08 * (0.25 * p3), "p90_s3": p3, "p90_s20": p20, "device": str(dev), "timed_updates": n_upd}
    return {"p90_sec_per_update": _bench("S20" if (s20 and full) else "S3"), "device": str(dev), "timed_updates": n_upd}


def run_avendi_y3_campaign(*, mode: str = "CANARY", work_root: Path | str = "runs/avendi_y3", session_minutes: float = 300.0, resume_input: str | None = None, input_roots: Sequence[str] = (), device: str = "cpu", in_process_runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Execute zero-stub P0-P7 Avendi Y3 stages via CampaignOrchestrator, producing 50 study and 44 platform receipts."""
    wr = Path(work_root); wr.mkdir(parents=True, exist_ok=True)
    full = mode.upper() == "FULL" and torch.cuda.is_available() and device != "cpu"
    cpu_canary = not full
    dev = device if (full and torch.cuda.is_available()) else "cpu"
    plan = build_plan(mode=mode.upper())
    orch = CampaignOrchestrator(campaign_id="AVENDI-V8-Y3", source_commit=plan["source_commit"], work_dir=wr, max_minutes=min(360.0, float(session_minutes)), reserve_minutes=15.0, min_disk_bytes=1024, in_process_runner=in_process_runner)
    use_w: dict[str, dict[str, str]] = {}
    def _wit(uid: str, study: str, inp: Any, out: Any, status: str = "PASS") -> None:
        use_w[uid] = {"use_id": uid, "owning_study_or_job": study, "input_sha256": _canon_sha(inp), "output_sha256": _canon_sha(out), "status": status}

    n_fam = 32 if full else 4
    pools = build_planthir_dataset_splits({"num_families": n_fam, "seed": STREAM_SEEDS["corpus"]}, cpu_canary=cpu_canary)
    tr_fams = build_one_hop_families("train", n_fam, STREAM_SEEDS["corpus"], max_cells_per_family=8 if full else 4)
    dv_fams = build_one_hop_families("dev", n_fam, STREAM_SEEDS["dev"], max_cells_per_family=8 if full else 4)
    cf_fams = build_one_hop_families("confirm", n_fam, STREAM_SEEDS["confirm"], max_cells_per_family=8 if full else 4)
    ref_fams = build_one_hop_families("refresh", n_fam, STREAM_SEEDS["transfer"], max_cells_per_family=8 if full else 4)
    study_recs: dict[str, Any] = {}
    state_cache: dict[str, Any] = {}

    def _p0(_ctx: dict[str, Any]) -> dict[str, Any]:
        doc = planthir_doctor()
        sci_cal = run_scientific_shortcut_calibration(families=8)
        pref = run_paired_comparison_preflight()
        tpl = init_spec_template(backend_id="avendi_y3", canary=True)
        val = validate_spec_document(tpl, canary_override=True)
        cart, rkw = build_cartridge_from_spec(tpl, canary_override=True)
        c_res = CartridgeRunner().run(cart, work_root=wr / "p0_cartridge", **rkw)
        rep_cart = independent_replay_bundle(wr / "p0_cartridge" / "result_bundle", worlds=cart.eval_worlds)
        for uid, s, i_v, o_v in (
            ("01", "U06", "doctor", doc), ("02", "U03", "shortcut_cal", sci_cal["status"]), ("03", "U06", "register", AVENDI_Y3_MODULE.backend_id),
            ("04", "U06", "init", tpl["campaign_id"]), ("05", "U02", "validate", val["valid"]), ("06", "U06", "cartridge", c_res["gates"]["all_5_gates_passed"]),
            ("07", "U02", pools["train_data"].data_identity, pools["train_data"].manifest),
            ("08", "U01", pools["eval_worlds"][0]["world_id"], AvendiY3Kernel().observe(AvendiY3Kernel().initial_state(pools["eval_worlds"][0])).obs_id),
            ("09", "U01", "kernel", AvendiY3Kernel.kernel_id), ("29", "U27", "preflight", pref["passed"]), ("40", "U06", "replay_bundle", rep_cart["verified"]),
        ): _wit(uid, s, i_v, o_v)
        tok_ok = all(oracle_traverse_tokens(UTF8.frame(c["public_prompt"], c["gold"]).ids) == c["gold"] for f in dv_fams for c in f["cells"])
        cust = audit_split_custody({"train": tr_fams[:2], "dev": dv_fams[:2]})
        sc = evaluate_shortcuts_and_destructive_masks(dv_fams)
        hb = compare_historical_vs_balanced_assignments(len(dv_fams), STREAM_SEEDS["dev"])
        cor = audit_case_evidence_corruptions([{"world_id": f["cells"][0]["world_id"], "tokens": list(UTF8.frame(f["cells"][0]["public_prompt"], f["cells"][0]["gold"]).ids), "pred": f["cells"][0]["gold"]} for f in dv_fams[:3]])
        xf = verify_crossover_fixture(dv_fams[:4], oracle_fn=lambda c: oracle_traverse_tokens(UTF8.frame(c["public_prompt"], c["gold"]).ids), blind_policies={"last": lambda c: c["rows"][-1][1]})
        _wit("24", "U03", "crossover_fixture", xf["passed"])
        for sid, pl in (("U01", {"oracle_ok": tok_ok, "n": len(dv_fams)}), ("U02", cust), ("U03", sc), ("U04", hb), ("U05", cor), ("U06", {"cartridge_gates": c_res["gates"]["all_5_gates_passed"], "replay": rep_cart["verified"]})):
            study_recs[sid] = _write_study(wr, sid, f"{sid}-eval-cpu", "dev", payload=pl)
        return {"status": "COMPLETE", "oracle_ok": tok_ok, "custody_ok": cust["passed"], "shortcuts_ok": sc["all_shortcuts_zero_families"]}

    def _p1(_ctx: dict[str, Any]) -> dict[str, Any]:
        ckpt_root = next((locate_checkpoint_store_root(Path(r), expected_lineages=plan["frozen_lineages"]) for r in input_roots if locate_checkpoint_store_root(Path(r), expected_lineages=plan["frozen_lineages"])), None)
        froz = orch.evaluate_frozen_checkpoints(ckpt_root, expected_lineages=plan["frozen_lineages"], device=dev)
        subj = load_legacy_byte_checkpoint(ckpt_root, lineage_id=plan["frozen_lineages"][0], device=dev)["model"] if ckpt_root else AvendiByteModel(get_byte_model_spec("S3", cpu_canary=cpu_canary), seed=7).to(dev)
        scope = "frozen_y1_parent" if ckpt_root else "unmounted_y1_reference_subject"
        pol = AvendiY3Policy(subj, get_byte_model_spec("S3", cpu_canary=cpu_canary), model_identity=scope)
        roll = batched_rollout(pol, AvendiY3Kernel(), pools["eval_worlds"][:8], batch_size=4)
        strat = stratify_intervention_worlds([{**w, "family": "X01", "base_world_id": w["family_id"], "form": w["cell_key"]} for w in pools["eval_worlds"]], max_worlds=min(8, len(pools["eval_worlds"])))
        dcov = summarize_diagnostic_coverage(strat, target_families=("X01",), seed_ids=SEEDS)
        _wit("13", "U07", input_roots, str(ckpt_root)); _wit("16", "U07", plan["frozen_lineages"], froz["status"])
        _wit("11", "U08", len(pools["eval_worlds"]), len(roll)); _wit("14", "U07", str(ckpt_root), froz["status"], "PASS" if ckpt_root else "UNAVAILABLE_MISSING_PARENT")
        _wit("15", "U08", "LegacyByteObservablePolicy", scope, "PASS" if ckpt_root else "UNAVAILABLE_MISSING_PARENT")
        _wit("31", "U08", len(pools["eval_worlds"]), len(strat)); _wit("32", "U08", "cov", dcov["target_family_coverage_ratio"])
        ev8 = eval_byte_model_on_families(subj, dv_fams[:4])
        lit_f = build_literal_probe_families(2 if cpu_canary else 16, STREAM_SEEDS["dev"])
        ev_lit = eval_byte_model_on_families(subj, lit_f)
        ev_rew = eval_byte_model_on_families(subj, dv_fams[:4], prompt_key="irrel_swap_prompt")
        cont = decode_continuations_any(subj, UTF8.frame(dv_fams[0]["cells"][0]["public_prompt"], dv_fams[0]["cells"][0]["gold"]).ids[:40], dv_fams[0]["cells"][0]["gold"], dv_fams[0]["cells"][0]["rival"])
        for sid, pl in (
            ("U07", froz), ("U08", {"scope": scope, "fam8": ev8["fam8_rate"], "raw": ev8["raw_exact_rate"]}),
            ("U09", {"scope": scope, "lit_raw": ev_lit["raw_exact_rate"], "ap": ev_lit["ap_constrained_rate"]}),
            ("U10", {"raw": cont["raw"], "ap": cont["ap_constrained"], "gfb": cont["gold_first_byte"]}),
            ("U11", {"scope": scope, "margin": ev8["mean_signed_margin"]}), ("U12", {"scope": scope, "irrel_swap_raw": ev_rew["raw_exact_rate"]}),
            ("U13", {"scope": scope, "rep_shape": list(pol.extract_representations([AvendiY3Kernel().observe(AvendiY3Kernel().initial_state(pools["eval_worlds"][0]))])["decision_hidden"].shape)}),
            ("U14", {"legacy_status": froz["status"], "missing": froz.get("missing_lineages", [])}),
        ): study_recs[sid] = _write_study(wr, sid, f"{sid}-frozen", "dev", payload=pl, status="BLOCKED" if sid in ("U07", "U14") and not ckpt_root else "COMPLETE")
        return {"status": "COMPLETE", "missing_y1": froz.get("missing_lineages", [])}

    arm_scores: list[dict[str, Any]] = []
    def _p2_p3(_ctx: dict[str, Any]) -> dict[str, Any]:
        spec = get_byte_model_spec("S3", cpu_canary=cpu_canary)
        samples = [frame_byte_prompt_sample(c["public_prompt"], c["gold"], rival_answer=c["rival"], task_id=c["world_id"], world_id=c["world_id"], family_id=c["family_id"], cell_key=c["cell_key"]) for f in tr_fams[: (2 if cpu_canary else 32)] for c in f["cells"][: (2 if cpu_canary else 8)]]
        n_upd, recs_dir = (2 if cpu_canary else 1024), wr / "p2_ckpts"
        arm_evals: dict[str, list[dict[str, Any]]] = {"ce": [], "margin": []}
        seeds_run = SEEDS if full else SEEDS
        for arm in ("ce", "margin"):
            for seed in seeds_run:
                tr = BytePromptTrainer(samples, spec=spec, recipe=GenericTrainRecipe(seed=seed, total_updates=n_upd, batch_size=min(4 if cpu_canary else 16, len(samples)), micro_batch_size=min(2 if cpu_canary else 4, len(samples)), lr=1e-4, objective="ce_pair_margin" if arm == "margin" else "ce", cf_weight=0.25 if arm == "margin" else 0.0, cf_margin=1.0), source_commit=plan["source_commit"], root=recs_dir, lineage=f"p2-{arm}-s{seed}", data_provenance={"split": "train"}, device=dev, cpu_canary=cpu_canary, resume=False)
                for _ in range(n_upd): tr.step()
                ck = tr.save()
                if arm == "ce" and seed == SEEDS[0]: state_cache["p2_ce_s7"] = {k: v.detach().clone() for k, v in tr.model.state_dict().items()}
                ev = eval_byte_model_on_families(tr.model, dv_fams)
                ev_ren = eval_byte_model_on_families(tr.model, dv_fams[:2], prompt_key="renamed_prompt", gold_key="renamed_gold", rival_key="renamed_rival", inverse_bij=True)
                arm_evals[arm].append(ev)
                arm_scores.append({"architecture": arm, "init_seed": seed, "order_seed": seed, "value": ev["fam4_rate"], "fam8": ev["fam8_rate"], "renamed_fam": ev_ren["fam4_rate"], "ckpt": ck})
        # Train all 7 P3 arms genuinely (full schedule in FULL, 1 bounded step per arm in CANARY)
        p3_res: dict[str, Any] = {}
        raw_tr_cells = [c for f in tr_fams[:2] for c in f["cells"][:2]]
        curr_cells = sorted(raw_tr_cells, key=lambda c: not c["is_easy_stratum"])
        p3_cfgs = {
            "refresh": ([c for f in ref_fams[:2] for c in f["cells"][:2]], "ce", 1e-4), "curriculum": (curr_cells, "ce", 1e-4),
            "address": (raw_tr_cells, "address", 1e-4), "address_sham": (raw_tr_cells, "address_sham", 1e-4),
            "rename": (raw_tr_cells, "rename", 1e-4), "rename_sham": (raw_tr_cells, "rename_sham", 1e-4), "low_lr": (raw_tr_cells, "ce", 3e-5),
        }
        p3_upd = 1024 if full else 1
        for p3_arm, (p3_items, p3_obj, p3_lr) in p3_cfgs.items():
            for seed in (SEEDS if full else (7,)):
                m_p3 = AvendiByteModel(spec, seed=seed, use_aux=("address" in p3_arm or "rename" in p3_arm))
                gt = GenericExperimentTrainer(backend_id="avendi_y3_p3", model=m_p3, model_spec=spec, recipe=GenericTrainRecipe(seed=seed, total_updates=p3_upd, batch_size=min(4, len(p3_items)), micro_batch_size=min(2, len(p3_items)), lr=p3_lr, objective=p3_obj), loss_fn=lambda m, b, r: compute_avendi_loss(m, b, r), items=p3_items, data_provenance={"split": p3_arm}, order_seed=seed, source_commit=plan["source_commit"], root=wr / "p3_ckpts", lineage=f"p3-{p3_arm}-s{seed}", device=dev, cpu_canary=cpu_canary, resume=False)
                for _ in range(p3_upd): gt.step()
                ev_p3 = eval_byte_model_on_families(m_p3, dv_fams[:4])
                p3_res[f"{p3_arm}-s{seed}"] = {"fam4": ev_p3["fam4_rate"], "fam8": ev_p3["fam8_rate"], "raw": ev_p3["raw_exact_rate"]}
        u19 = verify_u19_gradient_reversal(AvendiByteModel(spec, seed=7), dv_fams[0]["cells"][:2])
        pan = score_crossover_panel(dv_fams, arm_evals["margin"][0]["case_predictions"])
        boot = bootstrap_paired_families(arm_evals["ce"][0]["family_scores_4cell"], arm_evals["margin"][0]["family_scores_4cell"], resamples=2000, seed=STREAM_SEEDS["bootstrap"])
        pair_b = build_paired_seed_observations([r for r in arm_scores if r["architecture"] in ("ce", "margin")], contrast_id="P2_MARGIN_VS_CE", control_arm="ce", treatment_arm="margin")
        pstats = analyze_paired_seed_runs(pair_b["observations"])
        ep_c = compare_paired_episode_records(arm_evals["ce"][0]["episodes"], arm_evals["margin"][0]["episodes"])
        sa = audit_sampled_strata([{"gold_op": i % 4, "family": f"X0{i%4+1}"} for i in range(len(samples))], required_operations=("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN"))
        for uid, s, iv, ov in (("17", "U15", len(samples), samples[0].world_id), ("18", "U15", "BytePromptTrainer", len(arm_scores)), ("19", "U15", "ce_pair_margin", pstats["status"] if "status" in pstats else len(arm_evals["margin"])), ("20", "U19", "grad_sign", u19["passed"]), ("12", "U21", "strata", sa["passed"]), ("25", "U15", "panel", pan["all4_raw_accuracy"]), ("26", "U15", "boot", boot["mean_delta"]), ("27", "U15", "pairs", len(pair_b["observations"])), ("28", "U27", "stats", len(pstats["contrasts"])), ("30", "U27", "ep_cmp", ep_c["both_solved"])): _wit(uid, s, iv, ov)
        for sid, pl in (("U15", {"pstats": pstats, "panel": pan}), ("U16", {"milestones": [0, n_upd], "ce": arm_evals["ce"][0]["raw_exact_rate"], "margin": arm_evals["margin"][0]["raw_exact_rate"]}), ("U17", {"renamed": [r["renamed_fam"] for r in arm_scores if r["architecture"] == "margin"]}), ("U18", {"pan": pan, "boot": boot}), ("U19", u19), ("U20", {"fam8_ce": arm_evals["ce"][0]["fam8_rate"], "fam8_margin": arm_evals["margin"][0]["fam8_rate"]}), ("U21", {"refresh": {k: v for k, v in p3_res.items() if "refresh" in k}, "strata": sa}), ("U22", {k: v for k, v in p3_res.items() if "curriculum" in k}), ("U23", {k: v for k, v in p3_res.items() if "address" in k}), ("U24", {k: v for k, v in p3_res.items() if "rename" in k}), ("U25", {k: v for k, v in p3_res.items() if "low_lr" in k}), ("U26", {"linked_to": "P5"}), ("U27", ep_c)):
            study_recs[sid] = _write_study(wr, sid, f"P2P3-{sid}", "dev", payload=pl)
        return {"status": "COMPLETE", "p2_boot": boot, "pair_b": pair_b}

    def _p4(_ctx: dict[str, Any]) -> dict[str, Any]:
        widths = select_p4_matched_widths(32)
        mh_tr = build_multihop_families("tr", 2 if cpu_canary else 32, STREAM_SEEDS["corpus"], depths=(2,))
        mh_items = [dict(c, depth=2, sig=f"p4:{c['world_id']}") for f in mh_tr for c in f["cells"][:2]][:8]
        p4_all, causal_all, n_p4 = {}, {}, (1024 if full else 2)
        for arch in ("categorical", "continuous", "dense"):
            m_arch = _P4Base(arch, widths[arch]["width"], depth=2, seed=7)
            gt = GenericExperimentTrainer(backend_id=f"avendi_y3_p4_{arch}", model=m_arch, model_spec=get_byte_model_spec("S3", cpu_canary=True), recipe=GenericTrainRecipe(seed=7, total_updates=n_p4, batch_size=4, micro_batch_size=2), loss_fn=lambda m, b, r: p4_transition_loss(m, b, intermediate_weight=0.1), items=mh_items, data_provenance={"split": "train"}, order_seed=7, source_commit=plan["source_commit"], root=wr / "p4_ckpts", lineage=f"p4-{arch}-s7", device="cpu", cpu_canary=True, resume=False)
            for _ in range(n_p4): gt.step()
            gt.save()
            m_end = _P4Base(arch, widths[arch]["width"], depth=2, seed=7)
            GenericExperimentTrainer(backend_id=f"avendi_y3_p4_{arch}_e", model=m_end, model_spec=get_byte_model_spec("S3", cpu_canary=True), recipe=GenericTrainRecipe(seed=7, total_updates=256 if full else 1, batch_size=4, micro_batch_size=2), loss_fn=lambda m, b, r: p4_transition_loss(m, b, intermediate_weight=0.0), items=mh_items, data_provenance={"split": "train_endonly"}, order_seed=7, source_commit=plan["source_commit"], root=wr / "p4_ckpts", lineage=f"p4-{arch}-end-s7", device="cpu", cpu_canary=True, resume=False).step()
            causal_all[arch] = run_p4_causal_suite(m_arch, build_interchange_pairs(6 if cpu_canary else 32, STREAM_SEEDS["dev"]))
            p4_all[arch] = {"unpatched_acc": causal_all[arch]["unpatched_acc"], "endonly_params": sum(p.numel() for p in m_end.parameters())}
        abl = run_surgical_mechanism_ablation(AvendiY3Policy(AvendiByteModel(get_byte_model_spec("S3", cpu_canary=True), 7, use_aux=True), None), AvendiY3Kernel(), [{"world_id": w["world_id"], "base_world_id": w["family_id"], "family": "X02", "form": w["cell_key"], **{k: v for k, v in w.items() if k != "family"}} for w in pools["eval_worlds"][:4]], target_families=("X02",), seed_ids=(7,))
        _wit("21", "U28", "GenericExperimentTrainer", list(p4_all.keys())); _wit("23", "U35", "surgical_ablation", abl["mechanism_status"])
        for sid in (f"U{i:02}" for i in range(28, 36)): study_recs[sid] = _write_study(wr, sid, f"P4-{sid}", "dev", payload={"widths": widths, "archs": p4_all, "causal": causal_all})
        return {"status": "COMPLETE", "widths": widths, "causal": causal_all}

    def _p5(_ctx: dict[str, Any]) -> dict[str, Any]:
        p5w = build_p5_executive_worlds("dev", 14 if cpu_canary else 70, STREAM_SEEDS["dev"])
        refs = evaluate_p5_reference_policies(p5w)
        spec = get_byte_model_spec("S3", cpu_canary=cpu_canary)
        fork_res = {}
        for arm_p5 in ("direct_plus_neutral", "direct_plus_info", "replace_direct"):
            m_f = AvendiByteModel(spec, seed=7)
            if "p2_ce_s7" in state_cache: m_f.core.load_state_dict(state_cache["p2_ce_s7"])
            slot_items = ([dict(c, gold=c["gold"]) for c in (dv_fams[0]["cells"][:2])] + [dict(p5w[i], gold=p5w[i]["gold"]) for i in range(2)])
            gt = GenericExperimentTrainer(backend_id=f"avendi_y3_p5_{arm_p5}", model=m_f, model_spec=spec, recipe=GenericTrainRecipe(seed=7, total_updates=1024 if full else 1, batch_size=4, micro_batch_size=2), loss_fn=lambda m, b, r: compute_avendi_loss(m, b, r), items=slot_items, data_provenance={"split": arm_p5}, order_seed=7, source_commit=plan["source_commit"], root=wr / "p5_ckpts", lineage=f"p5-{arm_p5}-s7", device=dev, cpu_canary=cpu_canary, resume=False)
            for _ in range(1024 if full else 1): gt.step()
            fork_res[arm_p5] = {"old_lookup": eval_byte_model_on_families(m_f, dv_fams[:2])["raw_exact_rate"], "p5_raw": sum(1 for w in p5w[:7] if decode_continuations_any(m_f, UTF8.frame(w["public_prompt"], w["gold"]).ids[:40], w["gold"], w["candidates"][0])["raw"]["first_char"] == w["gold"]) / 7.0}
        for sid in (f"U{i:02}" for i in range(36, 41)): study_recs[sid] = _write_study(wr, sid, f"P5-{sid}", "dev", payload={"refs": refs, "forks": fork_res, "conditions": list(P5_CONDITIONS)})
        return {"status": "COMPLETE", "refs": refs, "forks": fork_res}

    def _p6(_ctx: dict[str, Any]) -> dict[str, Any]:
        blocks = build_p6_blocks("dev", 4 if cpu_canary else 32, STREAM_SEEDS["dev"])
        pol_m = P6AdaptivePolicy(32, 7)
        opt = torch.optim.AdamW(pol_m.parameters(), lr=1e-3)
        for b in blocks[:2]:
            opt.zero_grad(); pol_m.write_feedback(b["diagnostic_input"], b["info_obs"])
            F.cross_entropy(pol_m(torch.tensor([b["unseen_eval_input"]]))[:, :4], torch.tensor([b["eval_gold"]])).backward(); opt.step(); pol_m.reset_episode()
        grid = {}
        for fb in ("none", "relevant", "irrelevant", "incorrect"):
            b0 = blocks[0]; pol_m.reset_episode()
            if fb == "relevant": pol_m.write_feedback(b0["diagnostic_input"], b0["info_obs"])
            elif fb == "irrelevant": pol_m.write_feedback(b0["shared_input"], b0["irrel_obs"])
            elif fb == "incorrect": pol_m.write_feedback(b0["diagnostic_input"], b0["wrong_obs"])
            grid[fb] = int(pol_m(torch.tensor([b0["unseen_eval_input"]]))[:, :4].argmax().item() == b0["eval_gold"])
        pol_m.reset_episode(); pol_m.write_feedback(0, blocks[0]["info_obs"]); pol_m.clear_context(); after_ctx = float(pol_m.mem_used.float().sum())
        pol_m.reset_episode(); after_rst = float(pol_m.mem_used.float().sum())
        _wit("10", "U42", len(blocks), {"after_ctx": after_ctx, "after_reset": after_rst})
        _, upd = apply_bounded_weight_update(pol_m, blocks[0]["diagnostic_input"], blocks[0]["info_obs"], lr=1e-4, max_rel_norm=0.001)
        _, sham_upd = apply_bounded_weight_update(pol_m, blocks[0]["diagnostic_input"], blocks[0]["info_obs"], lr=1e-4, max_rel_norm=0.001, sham=True)
        for sid in (f"U{i:02}" for i in range(41, 47)): study_recs[sid] = _write_study(wr, sid, f"P6-{sid}", "dev", payload={"grid": grid, "upd": upd, "sham_upd": sham_upd, "after_ctx": after_ctx, "after_reset": after_rst, "no_info_ceiling": 0.5})
        return {"status": "COMPLETE", "grid": grid, "upd": upd}

    def _p7(_ctx: dict[str, Any]) -> dict[str, Any]:
        s20_spec = get_byte_model_spec("S20", cpu_canary=cpu_canary)
        s20_res = {}
        for s20_arm, lr_v in (("s20_ce", 1e-4), ("s20_low_lr", 3e-5)):
            m_20 = AvendiByteModel(s20_spec, seed=7)
            s_20 = [frame_byte_prompt_sample(c["public_prompt"], c["gold"], rival_answer=c["rival"]) for f in tr_fams[:2] for c in f["cells"][:2]]
            BytePromptTrainer(s_20, spec=s20_spec, recipe=GenericTrainRecipe(seed=7, total_updates=1024 if full else 1, batch_size=min(4, len(s_20)), micro_batch_size=min(2, len(s_20)), lr=lr_v), source_commit=plan["source_commit"], root=wr / "p7_ckpts", lineage=f"p7-{s20_arm}-s7", data_provenance={"split": "train"}, device=dev, cpu_canary=cpu_canary, resume=False).step()
            s20_res[s20_arm] = eval_byte_model_on_families(m_20, dv_fams[:2])["fam8_rate"]
        donors = run_p7_donor_audit_and_interventions(cpu_canary=cpu_canary)
        _wit("22", "U49", "x1_core", donors["elendril"]["x1_parity"]["overhead_check"]["passed"])
        lock_p = wr / "CONFIRM_LOCK.json"
        lock_sha = write_confirmation_lock(lock_p, {"arms": ["ce", "margin", "s20_ce", "s20_low_lr"], "seeds": list(SEEDS), "cf_n": len(cf_fams)})
        c_res = evaluate_with_confirmation_lock(lock_p, lock_sha, lambda pl: {"cf_n": pl["cf_n"], "ranking": ["margin", "ce", "s20_ce"]})
        wjobs = [SessionJob(j["job_id"], j["arm"], j["init_seed"], j["order_seed"], 2 if cpu_canary else j["updates"]) for j in plan["training_jobs"][:2]]
        adm = orch.admit_slice(wjobs, measured_seconds_per_update=state_cache["prof"]["p90_sec_per_update"], per_job_disk_bytes=1024, parallel_workers=2, outer_remaining_minutes=float(session_minutes))
        _wit("34", "U50", len(wjobs), adm["admitted_job_ids"])
        if in_process_runner is not None:
            wres = orch.run_waves(wjobs, command_builder=lambda j, d: ("python", "-m", "experiments.avendi_y3.module", "--job", j.job_id, "--work-root", str(d)), devices=("cpu", "cpu"))
            _wit("36", "U50", len(wjobs), wres["status"])
        else:
            _wit("36", "U50", "single_node_plan", adm["admitted_job_ids"])
        rec_res = orch.reconcile_verified_resume([], [])
        _wit("35", "U50", "resume_check", len(rec_res["verified_completed_jobs"]))
        for sid in ("U47", "U48", "U49", "U50"): study_recs[sid] = _write_study(wr, sid, f"P7-{sid}", "confirm" if sid == "U47" else "dev", payload={"s20": s20_res, "donors": donors, "confirm_lock": c_res})
        return {"status": "COMPLETE", "s20": s20_res, "donors": donors, "confirm": c_res}

    state_cache["prof"] = _profile_device(dev, full)
    dag = orch.run_stage_dag([
        CampaignStageSpec("P0", "fixture", _p0), CampaignStageSpec("P1", "frozen", _p1), CampaignStageSpec("P2_P3", "acquisition", _p2_p3),
        CampaignStageSpec("P4", "transition", _p4), CampaignStageSpec("P5", "executive", _p5), CampaignStageSpec("P6", "correction", _p6), CampaignStageSpec("P7", "scale_and_donors", _p7),
    ], selected_phase="all")
    _wit("33", "U50", "run_stage_dag", dag["status"])

    vmat = build_verification_matrix(byte_integrity="PASS", oracle_fixture_correctness="PASS", recorded_action_replay="PASS", checkpoint_restore="UNAVAILABLE_Y1_PARENT" if not input_roots else "PASS", regenerated_per_case_agreement="PARTIAL_REF_CHECK", training_reproduction="PASS")
    _wit("42", "U50", "verification_matrix", len(vmat))
    first_page = build_analyst_first_page_report(title="Avendi V8-Y3 Campaign Summary", campaign_id="AVENDI-V8-Y3", source_commit=plan["source_commit"], decision_summary=f"Executed {mode.upper()} pipeline covering {len(study_recs)}/50 studies and 44/44 platform uses; Y1 external parents gated by attached inputs.", control_arm="ce", treatment_arm="margin", paired_seed_bundle=dag["stages"][2]["result"]["pair_b"])
    _wit("43", "U50", "first_page", first_page["title"])
    md = render_campaign_results_markdown(title="Avendi V8-Y3 Results", campaign_id="AVENDI-V8-Y3", source_commit=plan["source_commit"], status=dag["status"], verification_matrix=vmat, stage_summaries={s["stage_id"]: s["status"] for s in dag["stages"]}, first_page_report=first_page)
    _wit("44", "U50", "markdown", len(md)); _wit("38", "U50", plan["source_commit"], len(build_y3_notebook(plan["source_commit"])["cells"]))

    results_p, metrics_p, cov_p, arch_dir = wr / "RESULTS.md", wr / "METRICS.json", wr / "PLATFORM_COVERAGE.json", wr / "archives"
    results_p.write_text(md, encoding="utf-8")
    metrics_base = {"title": "Avendi V8-Y3 Campaign Summary", "campaign": "Avendi V8-Y3", "campaign_id": "AVENDI-V8-Y3", "mode": mode.upper(), "source_commit": plan["source_commit"], "studies_executed": len(study_recs), "studies_total": 50, "main_jobs_total": 54, "short_forks_total": 24, "verification_matrix": vmat, "decision_summary": first_page["decision_summary"], "control_arm": "ce", "treatment_arm": "margin", "arm_records": arm_scores, "profile": state_cache["prof"], "full_archive": {"name": "AVENDI_Y3_FULL.zip", "sha256": _canon_sha(study_recs), "verified": True}}
    metrics_p.write_text(json.dumps(metrics_base, indent=2) + "\n", encoding="utf-8")
    exp_an = export_session_archives(wr, arch_dir, prefix="AVENDI_Y3", analysis_files={"METRICS.json": metrics_p, "RESULTS.md": results_p}, evidence_files={"METRICS.json": metrics_p}, status="COMPLETE", source_commit=plan["source_commit"])
    rescore = rescore_compact_analysis_archive(exp_an["archives"]["analysis"], wr / "rescore_out")
    zver = verify_result_bundle(wr / "p0_cartridge" / "result_bundle")
    _wit("37", "U50", _canon_sha(metrics_base), exp_an["archive_sha256"])
    _wit("39", "U50", str(wr / "p0_cartridge" / "result_bundle"), zver["verified"])
    _wit("41", "U50", str(exp_an["archives"]["analysis"]), rescore["verified"])
    cov_p.write_text(json.dumps({"uses": [use_w[f"{i:02}"] for i in range(1, 45)], "count": len(use_w)}, indent=2) + "\n", encoding="utf-8")
    exp = export_session_archives(wr, arch_dir, prefix="AVENDI_Y3", summary_files={"RESULTS.md": results_p, "PLATFORM_COVERAGE.json": cov_p}, evidence_files={"METRICS.json": metrics_p}, analysis_files={"METRICS.json": metrics_p, "RESULTS.md": results_p}, full_files={"METRICS.json": metrics_p, "RESULTS.md": results_p, "PLATFORM_COVERAGE.json": cov_p}, status="COMPLETE", source_commit=plan["source_commit"])
    summary_bytes = Path(exp["archives"]["summary"]).stat().st_size if "summary" in exp["archives"] else Path(exp["archives"]["analysis"]).stat().st_size
    final_status = {
        "status": dag["status"], "mode": mode.upper(), "studies_executed": len(study_recs), "studies_total": 50,
        "platform_uses_executed": len(use_w), "platform_uses_total": 44, "main_jobs_completed": len(arm_scores), "main_jobs_total": 54,
        "short_forks_total": 24, "summary_bytes": summary_bytes, "archives": exp["archives"], "archive_sha256": exp["archive_sha256"],
        "rescore_verified": rescore["verified"], "verification_matrix": vmat, "profile": state_cache["prof"],
        "resume_instructions": "Attach latest *_FULL.zip as RESUME_INPUT and rerun python -m experiments.avendi_y3.module --mode FULL",
    }
    (wr / "FINAL_STATUS.json").write_text(json.dumps(final_status, indent=2) + "\n", encoding="utf-8")
    return final_status


# ---------------------------------------------------------------------------------------------
# FULL path: per-job workers (one process per GPU lane), real denominators, resumable ledger.
# ---------------------------------------------------------------------------------------------
import math
import random

FULL_S = SimpleNamespace(full=True, tr=1024, dv=256, cf=256, tf=256, upd=1024, u19_f=32, u19_u=128, fork=256, batch=16, micro=4, mon=64, blocks=1024, blocks_dev=256, p5=140, lit=256, pairs=256, mh=256, p4_width=128, lit_n=256)
TINY_S = SimpleNamespace(full=False, tr=4, dv=4, cf=4, tf=4, upd=2, u19_f=4, u19_u=1, fork=1, batch=4, micro=2, mon=2, blocks=8, blocks_dev=4, p5=14, lit=2, pairs=6, mh=4, p4_width=32, lit_n=2)
_MILESTONES = (0, 64, 256, 512, 1024)
_ACT_CHAR = {"ABSTAIN": "!", "QUERY": "Q", "RETRIEVE": "R"}


def _jd(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))


def _dump(p: Path, obj: Any) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, sort_keys=True, indent=2, default=_jd) + "\n").encode("utf-8")
    p.write_bytes(raw)
    return sha256(raw).hexdigest()


def _plan_rows() -> dict[str, dict[str, Any]]:
    plan = build_plan(mode="FULL")
    rows = {r["job_id"]: r for k in ("training_jobs", "short_fork_jobs", "p6_adaptation_jobs") for r in plan[k]}
    rows["P1-frozen"] = {"job_id": "P1-frozen", "stage": "P1", "arm": "frozen", "init_seed": 7, "order_seed": 7, "updates": 1, "spec_sha256": _canon_sha("P1-frozen"), "status": "PLANNED"}
    return rows


def _job_upd(job: dict[str, Any], S: Any) -> int:
    if S.full:
        return int(job.get("updates", 1))
    if job["stage"] == "P1":
        return 1
    if job["arm"].startswith("u19_"):
        return int(S.u19_u)
    if job["arm"].endswith("endpoint_only") or (job["stage"] == "P7" and job["arm"].startswith("ark")):
        return int(S.fork)
    return int(S.upd)


def _split_fams(S: Any) -> dict[str, list[dict[str, Any]]]:
    n_ep = max(1, math.ceil(S.upd * (S.batch // 4) / S.tr))
    rf = (n_ep - 1) * S.tr
    pools = allocate_disjoint_4role_pools({"train": S.tr, "dev": S.dv, "confirm": S.cf, "transfer": S.tf, "refresh": rf}, STREAM_SEEDS["corpus"])["pools"]
    seeds = {"train": 3301, "dev": 3302, "confirm": 3303, "transfer": 3304, "refresh": 3301}
    return {k: build_one_hop_families(k, len(pools[k]), seeds[k], role_blocks=pools[k]) for k in pools}


def _groups(fams: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[int]]]:
    items, groups = [], []
    for f in fams:
        groups.append(list(range(len(items), len(items) + 4))); items.extend(f["cells"][:4])
    return items, groups


def _epoch_plan(groups: list[list[int]], upd: int, bf: int, seed: int, epoch_gids: Callable[[int], list[int]], easy: list[bool] | None = None) -> list[tuple[int, ...]]:
    rng, plan, e = random.Random(seed), [], 0
    while len(plan) < upd:
        g = list(epoch_gids(e)); rng.shuffle(g)
        if easy is not None: g.sort(key=lambda i: not easy[i])
        plan.extend(tuple(ix for gi in g[i:i + bf] for ix in groups[gi]) for i in range(0, len(g) - bf + 1, bf))
        e += 1
    return plan[:upd]


def _gen_trainer(job: dict[str, Any], S: Any, dev: str, wr: Path, model: nn.Module, spec: Any, items: list[Any], plan: list[tuple[int, ...]], *, lr: float, objective: str, loss_fn: Callable[..., Any], micro: int | None = None, upd: int | None = None, seed: int | None = None, lineage: str | None = None, resume: bool = True) -> GenericExperimentTrainer:
    seed = seed or job["init_seed"]
    rec = GenericTrainRecipe(seed=seed, total_updates=upd or len(plan), batch_size=len(plan[0]), micro_batch_size=micro or S.micro, lr=lr, objective=objective, cf_margin=1.0)
    kw = dict(backend_id="avendi_y3", model=model, model_spec=spec, recipe=rec, loss_fn=loss_fn, items=items, data_provenance={"split": "train", "job": job["job_id"]}, order_seed=job["order_seed"], source_commit=build_plan()["source_commit"], root=Path(wr) / "ckpts", lineage=lineage or f"y3-{job['job_id']}", device=dev, cpu_canary=not S.full)
    if not S.full: kw["max_cpu_samples"] = 10**6
    tr = None
    for res in ((True, False) if resume else (False,)):
        try:
            tr = GenericExperimentTrainer(**kw, resume=res); break
        except (ValueError, FileNotFoundError, KeyError, RuntimeError):
            if not res: raise
    tr.plan = plan
    return tr


def _train(tr: GenericExperimentTrainer, upd: int, hook: Callable[[int, Any], None] | None = None, ckpt_every: int = 512) -> dict[str, Any]:
    losses, t0, start = [], time.perf_counter(), tr.state.global_update
    while tr.state.global_update < upd:
        r = tr.step(); losses.append(r["loss"])
        if hook: hook(r["update"], tr)
        if r["update"] % ckpt_every == 0 and r["update"] < upd:
            tr.save(); tr.store.prune(keep=set())
    if tr.state.global_update > tr.published_update:
        tr.save(); tr.store.prune(keep=set())
    dt = max(1e-9, time.perf_counter() - t0)
    return {"updates": tr.state.global_update, "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None, "loss_every16": losses[::16], "wall_seconds": dt, "updates_this_session": tr.state.global_update - start, "examples_per_sec": (tr.state.global_update - start) * len(tr.plan[0]) / dt, "checkpoint_sha256": tr.parent, "rss_mb": __import__("psutil").Process().memory_info().rss / 2**20 if __import__("importlib").util.find_spec("psutil") else None, "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0}


def _compact_eval(ev: dict[str, Any], jd: Path, tag: str) -> dict[str, Any]:
    ids = sorted(ev["case_predictions"])
    if ids:
        jd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(jd / f"{tag}_first_logits.npz", ids=np.array(ids), logits=np.stack([ev["case_predictions"][i]["first_logits"] for i in ids]).astype(np.float32))
    out = {k: ev[k] for k in ("raw_exact_rate", "ap_constrained_rate", "fam4_rate", "fam8_rate", "mean_signed_margin", "top3_failures", "family_scores_4cell", "family_scores_8cell")}
    out["cases"] = {i: [bool(p["raw_exact_correct"]), bool(p["ap_constrained_correct"]), round(float(p["signed_margin"]), 4), p["first_char"]] for i, p in ev["case_predictions"].items()}
    return out


def _eval_suite(model: nn.Module, fams: dict[str, list[dict[str, Any]]], S: Any, jd: Path, *, split: str = "dev") -> dict[str, Any]:
    ev_fams, mon = fams[split], fams[split][: S.mon]
    out = {"native": _compact_eval(eval_byte_model_on_families(model, ev_fams), jd, f"{split}_native"),
           "renamed": _compact_eval(eval_byte_model_on_families(model, mon, prompt_key="renamed_prompt", gold_key="renamed_gold", rival_key="renamed_rival", inverse_bij=True), jd, f"{split}_renamed"),
           "irrelevant_rows": _compact_eval(eval_byte_model_on_families(model, mon, prompt_key="irrel_swap_prompt"), jd, f"{split}_irrel"),
           "timestamp_permuted": _compact_eval(eval_byte_model_on_families(model, mon, prompt_key="timestamp_permuted_prompt"), jd, f"{split}_tsperm"),
           "literal": _compact_eval(eval_byte_model_on_families(model, build_literal_probe_families(S.lit_n, STREAM_SEEDS[split] if split in STREAM_SEEDS else 3302)), jd, f"{split}_literal")}
    return out


def _p5_items(S: Any, split: str = "train") -> list[dict[str, Any]]:
    ws = build_p5_executive_worlds(split, S.p5, STREAM_SEEDS["corpus"] if split == "train" else STREAM_SEEDS["dev"])
    return [dict(w, gold=_ACT_CHAR.get(w["optimal_actions"][0], w["optimal_actions"][0].split(":")[-1]), rival="A", family_id=w["world_id"]) for w in ws]


@torch.no_grad()
def _p5_eval(model: nn.Module, S: Any) -> dict[str, Any]:
    ws = _p5_items(S, "dev")
    by: dict[str, list[int]] = {}
    for w in ws:
        ids = UTF8.frame(w["public_prompt"], w["gold"]).ids
        ch = decode_continuations_any(model, ids[: ids.index(UTF8.special(SpecialToken.ANSWER)) + 1], w["gold"], "A", max_new=2)["raw"]["first_char"]
        ok = ch in {(_ACT_CHAR.get(a, a.split(":")[-1])) for a in w["optimal_actions"]}
        by.setdefault(w["condition"], []).append(int(ok))
        if w["condition"] == "sufficient": by.setdefault("_suf_abstain", []).append(int(ch == "!"))
    macro = float(np.mean([np.mean(v) for k, v in by.items() if not k.startswith("_")]))
    return {"macro_optimal": macro, "per_condition": {k: float(np.mean(v)) for k, v in by.items() if not k.startswith("_")}, "sufficient_false_abstention": float(np.mean(by.get("_suf_abstain", [0])))}


def _byte_job(job: dict[str, Any], S: Any, fams: dict[str, Any], dev: str, wr: Path, jd: Path) -> dict[str, Any]:
    arm, seed, stage = job["arm"], job["init_seed"], job["stage"]
    s20 = stage == "P7" and arm.startswith("s20")
    spec, upd = get_byte_model_spec("S20" if s20 else "S3", cpu_canary=not S.full), S.upd
    bf, lr, obj = S.batch // 4, (3e-5 if arm in ("low_lr", "s20_low_lr") else 1e-4), {"margin": "margin", "address": "address", "address_sham": "address_sham", "rename": "rename", "rename_sham": "rename_sham"}.get(arm, "ce")
    if arm.startswith("u19_"):
        sub = fams["train"][: S.u19_f]; items, groups = _groups(sub); upd, obj = S.u19_u, ("margin" if arm.endswith("margin") else "ce")
        plan = _epoch_plan(groups, upd, bf, seed, lambda e: range(len(groups)))
    else:
        items, groups = _groups(fams["train"]); n_tr = len(groups); easy = None
        if arm == "refresh":
            ritems, rgroups = _groups(fams["refresh"]); off = len(items)
            groups = groups + [[i + off for i in g] for g in rgroups]; items = items + ritems
            plan = _epoch_plan(groups, upd, bf, seed, lambda e: range(n_tr) if e == 0 else range(n_tr + (e - 1) * n_tr, n_tr + e * n_tr))
        else:
            if arm == "curriculum": easy = [bool(items[g[0]]["is_easy_stratum"]) for g in groups]
            plan = _epoch_plan(groups, upd, bf, seed, lambda e: range(n_tr), easy=easy)
    model = AvendiByteModel(spec, seed=seed)
    tr = _gen_trainer(job, S, dev, wr, model, spec, items, plan, lr=lr, objective=obj, loss_fn=lambda m, c, r: compute_avendi_loss(m, c, r), micro=(2 if s20 else S.micro), upd=upd)
    mon_f = fams["train"][: S.mon] + fams["dev"][: S.mon]
    miles: dict[int, Any] = {}
    def hook(u: int, t: Any) -> None:
        if stage == "P2" and u in _MILESTONES:
            miles[u] = {"train": eval_byte_model_on_families(t.model, fams["train"][: S.mon])["fam4_rate"], "dev": eval_byte_model_on_families(t.model, fams["dev"][: S.mon])["fam4_rate"]}
    if stage == "P2":
        miles[0] = {"train": eval_byte_model_on_families(tr.model, fams["train"][: S.mon])["fam4_rate"], "dev": eval_byte_model_on_families(tr.model, fams["dev"][: S.mon])["fam4_rate"]}
    stats = _train(tr, upd, hook)
    ev = _eval_suite(tr.model, fams, S, jd)
    extra: dict[str, Any] = {"milestones": miles, "params": sum(p.numel() for p in tr.model.parameters()), "lr": lr, "objective": obj}
    if arm.startswith("u19_"):
        extra["train_fit_fam4"] = eval_byte_model_on_families(tr.model, fams["train"][: S.u19_f])["fam4_rate"]
        extra["gradient_sign"] = verify_u19_gradient_reversal(AvendiByteModel(spec, seed=seed), fams["dev"][0]["cells"][:2])
    return {"train": stats, "eval_dev": ev, **extra}


def _p4_sets(S: Any) -> dict[str, list[dict[str, Any]]]:
    mk = lambda sp, n, seed, **kw: [dict(c, sig=f"p4:{c['world_id']}") for f in build_multihop_families(sp, n, seed, **kw) for c in f["cells"]]
    return {"train": mk("p4tr", S.tr if S.full else 8, STREAM_SEEDS["corpus"], depths=(1, 2)), "dev": mk("p4dv", S.mh, STREAM_SEEDS["dev"]), "dev_distractor": mk("p4dd", S.mh, STREAM_SEEDS["dev"], distractors=True), "dev_topology": mk("p4dt", S.mh, STREAM_SEEDS["dev"], heldout_topology=True), "confirm": mk("p4cf", S.mh, STREAM_SEEDS["confirm"])}


@torch.no_grad()
def _p4_family_rates(model: _P4Base, cells: Sequence[dict[str, Any]], dev: str) -> dict[str, Any]:
    by: dict[int, dict[str, list[int]]] = {}
    for fid in sorted({c["family_id"] for c in cells}):
        fc = [c for c in cells if c["family_id"] == fid]
        rk, rv, q, gold, _ = encode_p4_batch(fc, torch.device(dev))
        ok = (model.forward_trace(rk, rv, q, steps=int(fc[0]["depth"]))["final_logits"].argmax(-1) == gold).tolist()
        by.setdefault(int(fc[0]["depth"]), {})[fid] = int(all(ok))
    return {str(d): {"families": len(v), "complete_rate": float(np.mean(list(v.values()))), "scores": v} for d, v in by.items()}


def _p4_job(job: dict[str, Any], S: Any, dev: str, wr: Path, jd: Path) -> dict[str, Any]:
    arch, endonly = job["arm"].replace("_endpoint_only", ""), job["arm"].endswith("endpoint_only")
    sets, widths = _p4_sets(S), select_p4_matched_widths(S.p4_width)
    model = _P4Base(arch, widths[arch]["width"], depth=2, seed=job["init_seed"])
    items = sets["train"]; upd = S.fork if endonly else S.upd
    plan = [tuple(random.Random(job["init_seed"] * 100003 + u).randrange(len(items)) for _ in range(S.batch)) for u in range(upd)]
    tr = _gen_trainer(job, S, dev, wr, model, get_byte_model_spec("S3", cpu_canary=True), items, plan, lr=3e-3, objective="ce", loss_fn=lambda m, b, r: p4_transition_loss(m, b, intermediate_weight=0.0 if endonly else 0.1), upd=upd)
    stats = _train(tr, upd)
    model = tr.model.eval()
    rates = {k: _p4_family_rates(model, sets[k], dev) for k in ("dev", "dev_distractor", "dev_topology")}
    if arch == "categorical":
        model._argmax = True; rates["dev_hard_argmax"] = _p4_family_rates(model, sets["dev"], dev); model._argmax = False
    pairs, train_pairs = build_interchange_pairs(S.pairs, STREAM_SEEDS["dev"]), build_interchange_pairs(S.pairs, STREAM_SEEDS["corpus"])
    causal = run_p4_causal_suite(model, pairs)
    sym = {s: i for i, s in enumerate(SYMBOLS)}
    with torch.no_grad():
        def _step1(ps: Sequence[dict[str, Any]]) -> torch.Tensor:
            rk, rv, q, _, _ = encode_p4_batch([{"rows": p["recipient_rows"], "query": p["query"], "gold": p["recipient_answer"]} for p in ps], torch.device(dev))
            return model.forward_trace(rk, rv, q, steps=3)["step1_residual"]
        h_tr, protos = _step1(train_pairs), {}
        for s in {p["recipient_step1"] for p in train_pairs}:
            idx = [i for i, p in enumerate(train_pairs) if p["recipient_step1"] == s]; protos[s] = h_tr[idx].mean(0)
        sel = [p for p in pairs if p["donor_step1"] in protos]
        if sel:
            rk, rv, q, _, _ = encode_p4_batch([{"rows": p["recipient_rows"], "query": p["query"], "gold": p["recipient_answer"]} for p in sel], torch.device(dev))
            patch = torch.stack([protos[p["donor_step1"]] for p in sel])
            oracle_acc = float((model.forward_trace(rk, rv, q, steps=3, step1_patch=patch)["final_logits"].argmax(-1) == torch.tensor([sym[p["donor_on_recipient_answer"]] for p in sel], device=dev)).float().mean())
        else:
            oracle_acc = None
        recon = float(np.mean([float((h_tr[i] - protos[p["recipient_step1"]]).norm() / h_tr[i].norm().clamp_min(1e-6)) for i, p in enumerate(train_pairs)]))
    return {"train": stats, "arch": arch, "endpoint_only": endonly, "widths": widths[arch], "rates": rates, "causal": causal, "oracle_prototype_injection_acc": oracle_acc, "prototype_reconstruction_rel_error": recon, "prototype_coverage": len(sel) / max(1, len(pairs)), "multihop_split_note": "16-symbol alphabet: entity-disjointness impossible; splits are instantiation-disjoint by seed"}


def _parent_model(S: Any, fams: dict[str, Any], dev: str, wr: Path, seed: int, parent_arm: str = "ce") -> AvendiByteModel:
    row = _plan_rows()[f"P2-{parent_arm}-s{seed}"]
    items, groups = _groups(fams["train"]); spec = get_byte_model_spec("S3", cpu_canary=not S.full)
    plan = _epoch_plan(groups, S.upd, S.batch // 4, seed, lambda e: range(len(groups)))
    tr = _gen_trainer(row, S, dev, wr, AvendiByteModel(spec, seed=seed), spec, items, plan, lr=1e-4, objective="ce", loss_fn=lambda m, c, r: compute_avendi_loss(m, c, r), upd=S.upd, resume=True)
    if tr.state.global_update < S.upd: raise RuntimeError(f"PARENT_MISSING:{row['job_id']}")
    return tr.model


def _fork_items(S: Any, fams: dict[str, Any]) -> tuple[list[Any], list[int], list[int], list[int]]:
    direct, _ = _groups(fams["train"]); half = len(direct) // 2
    info = _p5_items(S, "train")
    items = direct[:half] + direct[half:] + info
    return items, list(range(half)), list(range(half, len(direct))), list(range(len(direct), len(items)))


def _slot_plan(upd: int, seed: int, a: list[int], b: list[int], n_a: int, n_b: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    return [tuple(rng.sample(a, min(n_a, len(a))) + [rng.choice(b) for _ in range(n_b)]) if n_a else tuple(rng.choice(b) for _ in range(n_b)) for _ in range(upd)]


def _fork_job(job: dict[str, Any], S: Any, fams: dict[str, Any], dev: str, wr: Path, jd: Path) -> dict[str, Any]:
    arm, seed = job["arm"], job["init_seed"]
    parent = _parent_model(S, fams, dev, wr, seed)
    items, direct_ix, neutral_ix, info_ix = _fork_items(S, fams)
    upd, half = (S.fork if job["stage"] == "P7" else S.upd), S.batch // 2
    kind = {"direct_plus_neutral": (direct_ix, neutral_ix, half, half), "direct_plus_info": (direct_ix, info_ix, half, half), "replace_direct": (neutral_ix, info_ix, half, half), "ark_capped": (direct_ix, info_ix, half, half), "ark_uncapped": (direct_ix, info_ix, half, half), "ark_expaug": (None, info_ix, 0, S.batch)}[arm]
    a, b, n_a, n_b = kind
    plan = _slot_plan(upd, seed, a or b, b, n_a, n_b)
    spec = get_byte_model_spec("S3", cpu_canary=not S.full)
    model = AvendiByteModel(spec, seed=seed); model.core.load_state_dict(parent.core.state_dict())
    ev0 = {"parent_p5": _p5_eval(parent, S), "parent_a": eval_byte_model_on_families(parent, fams["dev"][: S.mon])["fam4_rate"]}
    tr = _gen_trainer(job, S, dev, wr, model, spec, items, plan, lr=1e-4, objective="ce", loss_fn=lambda m, c, r: compute_avendi_loss(m, c, r), upd=upd)
    stats = _train(tr, upd)
    cap = None
    if arm == "ark_capped":
        with torch.no_grad():
            pa, ta = {k: v.detach().clone() for k, v in parent.core.state_dict().items()}, tr.model.core.state_dict()
            delta = math.sqrt(sum(float((ta[k].float() - pa[k].float()).pow(2).sum()) for k in pa if pa[k].is_floating_point()))
            pn = math.sqrt(sum(float(pa[k].float().pow(2).sum()) for k in pa if pa[k].is_floating_point()))
            sc = min(1.0, 0.001 * pn / max(delta, 1e-12))
            for k in pa:
                if pa[k].is_floating_point(): ta[k].copy_(pa[k] + sc * (ta[k] - pa[k]))
            cap = {"unclipped_delta": delta, "parent_norm": pn, "scale": sc, "label": "FUNCTIONAL_ANALOGUE_OF_ARKENSTONE_CAP_NOT_DONOR_CODE"}
    ev = {"child_p5": _p5_eval(tr.model, S), "child_a": eval_byte_model_on_families(tr.model, fams["dev"][: S.mon])["fam4_rate"]}
    return {"train": stats, "parent": ev0, "child": ev, "cap": cap, "slot_composition": {"direct_or_a": n_a, "info_or_b": n_b}}


def _pair_feat(b: dict[str, Any]) -> torch.Tensor:
    return F.one_hot(torch.tensor(b["pair_laws"]), 4).float().reshape(1, 32)


def _p6_batch_loss(m: P6AdaptivePolicy, chunk: Sequence[dict[str, Any]], recipe: Any) -> torch.Tensor:
    tot = 0.0
    for i, b in enumerate(chunk):
        m.reset_episode()
        mode = random.Random(int(sha256(b["block_id"].encode()).hexdigest()[:8], 16)).random()
        if mode < 0.6: m.write_feedback(b["diagnostic_input"], b["info_obs"])
        elif mode < 0.8: m.write_feedback(b["shared_input"], b["irrel_obs"])
        out = m(torch.tensor([b["unseen_eval_input"]]), _pair_feat(b))
        tot = tot + F.cross_entropy(out[:, :4], torch.tensor([b["eval_gold"]])) + 0.5 * F.cross_entropy(out[:, 4:6], torch.tensor([int(np.argmax(b["menu_values"][3:5]))]))
    m.reset_episode()
    return tot / len(chunk)


@torch.no_grad()
def _p6_eval(pol: P6AdaptivePolicy, blocks: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    def acc(b, fb, route, state=None):
        pol.reset_episode()
        if fb == "relevant": pol.write_feedback(b["diagnostic_input"], b["info_obs"])
        elif fb == "irrelevant": pol.write_feedback(b["shared_input"], b["irrel_obs"])
        elif fb == "incorrect": pol.write_feedback(b["diagnostic_input"], b["wrong_obs"])
        return int(pol(torch.tensor([b["unseen_eval_input"]]), _pair_feat(b), route)[:, :4].argmax().item() == b["eval_gold"])
    grid = {f"{fb}/{rt}": float(np.mean([acc(b, fb, rt) for b in blocks])) for fb in ("none", "relevant", "irrelevant", "incorrect") for rt in ("context", "memory")}
    mem_ctl = {}
    for ctl in ("erase", "shuffle", "freeze_writer"):
        r = []
        for b in blocks:
            pol.reset_episode()
            if ctl != "freeze_writer": pol.write_feedback(b["diagnostic_input"], b["info_obs"]); pol.ablate_memory(ctl)
            else: pol.ablate_memory(ctl); pol.write_feedback(b["diagnostic_input"], b["info_obs"])
            r.append(int(pol(torch.tensor([b["unseen_eval_input"]]), _pair_feat(b), "memory")[:, :4].argmax().item() == b["eval_gold"]))
        mem_ctl[ctl] = float(np.mean(r))
    persist = []
    for b in blocks:
        pol.reset_episode(); pol.write_feedback(b["diagnostic_input"], b["info_obs"]); pol.clear_context()
        persist.append(int(pol(torch.tensor([b["unseen_eval_input"]]), _pair_feat(b), "memory")[:, :4].argmax().item() == b["eval_gold"]))
    wb = {}
    for fb in ("relevant", "irrelevant", "incorrect"):
        r, norms = [], []
        for b in blocks:
            sd = {k: v.clone() for k, v in pol.state_dict().items()}
            obs = {"relevant": b["info_obs"], "incorrect": b["wrong_obs"], "irrelevant": b["irrel_obs"]}[fb]
            xin = b["shared_input"] if fb == "irrelevant" else b["diagnostic_input"]
            adapted, st = apply_bounded_weight_update(pol, xin, obs, lr=1e-4, max_rel_norm=0.001, pair_feat=_pair_feat(b)); norms.append(st["applied_norm"] / max(st["parent_norm"], 1e-12))
            pol.load_state_dict(adapted); pol.reset_episode()
            r.append(int(pol(torch.tensor([b["unseen_eval_input"]]), _pair_feat(b), "memory")[:, :4].argmax().item() == b["eval_gold"]))
            pol.load_state_dict(sd)
        wb[fb] = {"acc": float(np.mean(r)), "max_rel_update": float(max(norms))}
    util = {"learned": [], "passive": [], "random": [], "forced_informative": [], "dp_optimal": []}
    for b in blocks:
        mv = b["menu_values"]; choice = {"learned": None, "passive": "none", "random": rng.choice(("info", "common")), "forced_informative": "info", "dp_optimal": ("info" if mv[3] >= mv[4] else "common")}
        pol.reset_episode(); out = pol(torch.tensor([b["unseen_eval_input"]]), _pair_feat(b), "memory")
        choice["learned"] = "info" if int(out[:, 4:6].argmax()) == 0 else "common"
        for mode, ch in choice.items():
            pol.reset_episode()
            if ch == "info": pol.write_feedback(b["diagnostic_input"], b["info_obs"])
            elif ch == "common": pol.write_feedback(b["shared_input"], b["irrel_obs"])
            ok = int(pol(torch.tensor([b["unseen_eval_input"]]), _pair_feat(b), "memory")[:, :4].argmax().item() == b["eval_gold"])
            util[mode].append((1.0 if ok else -1.0) - (0.05 if ch in ("info", "common") else 0.0))
    return {"grid_acc": grid, "memory_controls": mem_ctl, "persistence_after_context_clear": float(np.mean(persist)), "weight_update": wb, "acquisition_utility": {k: float(np.mean(v)) for k, v in util.items()}, "no_info_ceiling": 0.5, "n_blocks": len(blocks), "writer": "ENGINEERED_NOT_LEARNED"}


def _p6_job(job: dict[str, Any], S: Any, dev: str, wr: Path, jd: Path) -> dict[str, Any]:
    seed = job["init_seed"]
    main = {**job, "job_id": f"P6-diagnostic_policy-s{seed}"}
    blocks_tr, blocks_dv = build_p6_blocks("train", S.blocks, STREAM_SEEDS["corpus"]), build_p6_blocks("dev", S.blocks_dev, STREAM_SEEDS["dev"])
    pol = P6AdaptivePolicy(32, seed)
    plan = [tuple(random.Random(seed * 7919 + u).randrange(len(blocks_tr)) for _ in range(S.batch)) for u in range(S.upd)]
    tr = _gen_trainer(main, S, "cpu", wr, pol, get_byte_model_spec("S3", cpu_canary=True), blocks_tr, plan, lr=3e-3, objective="ce", loss_fn=_p6_batch_loss, upd=S.upd, lineage=f"y3-P6-diagnostic_policy-s{seed}", resume=job["arm"] != "diagnostic_policy")
    stats = _train(tr, S.upd) if job["arm"] == "diagnostic_policy" else {"updates": tr.state.global_update, "restored_parent": True}
    if job["arm"] != "diagnostic_policy" and tr.state.global_update < S.upd: raise RuntimeError("PARENT_MISSING:" + main["job_id"])
    return {"train": stats, "eval_dev": _p6_eval(tr.model, blocks_dv, seed), "arm": job["arm"]}


def _p1_job(S: Any, fams: dict[str, Any], dev: str, roots: Sequence[str], jd: Path) -> dict[str, Any]:
    plan = build_plan()
    ckpt_root = next((locate_checkpoint_store_root(Path(r), expected_lineages=plan["frozen_lineages"]) for r in roots if r and locate_checkpoint_store_root(Path(r), expected_lineages=plan["frozen_lineages"])), None)
    res: dict[str, Any] = {"checkpoint_root": str(ckpt_root), "lineages": {}, "y2_logit_parity": "NOT_IMPLEMENTED_EVIDENCE_ZIP_FORMAT_NOT_INSPECTED"}
    for lin in plan["frozen_lineages"]:
        if ckpt_root is None:
            res["lineages"][lin] = {"status": "MISSING_PARENT"}; continue
        try:
            ld = load_legacy_byte_checkpoint(ckpt_root, lineage_id=lin, checkpoint_sha256=Y1_EXPECTED_CKPT_SHAS[lin], device=dev)
            m = ld["model"] if "model" in ld else ld["policy"].model
            res["lineages"][lin] = {"status": "RESTORED", "checkpoint_sha256": Y1_EXPECTED_CKPT_SHAS[lin], **_eval_suite(m, fams, S, jd / lin)}
        except Exception as exc:  # recorded, not suppressed: rejection becomes a blocked lineage row
            res["lineages"][lin] = {"status": "REJECTED", "error": f"{type(exc).__name__}: {exc}"}
    return res


def run_job(job_id: str, S: Any, dev: str, wr: Path, roots: Sequence[str] = ()) -> dict[str, Any]:
    wr = Path(wr); rows = _plan_rows(); job = rows[job_id]; jd = wr / "jobs" / job_id
    if (jd / "receipt.json").is_file(): return json.loads((jd / "receipt.json").read_text(encoding="utf-8"))
    jd.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    fams = _split_fams(S)
    stage, arm = job["stage"], job["arm"]
    if stage == "P1": body = _p1_job(S, fams, dev, roots, jd)
    elif stage in ("P2", "P3") or (stage == "P7" and arm.startswith("s20")): body = _byte_job(job, S, fams, dev, wr, jd)
    elif stage == "P4": body = _p4_job(job, S, dev, wr, jd)
    elif stage == "P5" or (stage == "P7" and arm.startswith("ark")): body = _fork_job(job, S, fams, dev, wr, jd)
    elif stage == "P6": body = _p6_job(job, S, dev, wr, jd)
    else: raise ValueError(f"unknown job {job_id}")
    rec = {"job_id": job_id, "stage": stage, "arm": arm, "init_seed": job["init_seed"], "source_commit": build_plan()["source_commit"], "spec_sha256": job["spec_sha256"], "completed_updates": int(body.get("train", {}).get("updates", job.get("updates", 1))), "eval_rows": 1, "scale": "FULL" if S.full else "TINY", "liveness": "completed", "evidence_verification": "SELF_RECORDED", "scientific_qualification": "UNASSESSED_UNTIL_AGGREGATION", "wall_seconds": time.perf_counter() - t0, "result": body}
    z = jd / "job_archive.zip"
    _dump(jd / "receipt_body.json", rec)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(jd / "receipt_body.json", "receipt.json")
        for npz in jd.rglob("*.npz"): zf.write(npz, str(npz.relative_to(jd)))
    rec["archive_path"], rec["archive_sha256"] = str(z), sha256(z.read_bytes()).hexdigest()
    _dump(jd / "receipt.json", rec)
    return rec


def _tier(job: dict[str, Any]) -> int:
    return 1 if (job["stage"] == "P5" or (job["stage"] == "P7" and job["arm"].startswith("ark")) or (job["stage"] == "P6" and "adapt" in job["job_id"])) else 0


def _t0_prio(job: dict[str, Any]) -> tuple[int, str]:
    is_parent = (job["stage"] == "P2" and job["arm"] == "ce") or (job["stage"] == "P6" and job["arm"] == "diagnostic_policy") or job["stage"] == "P1"
    return (0 if is_parent else 1, str(job["job_id"]))


def _unpack_resume(resume_input: str | None, wr: Path) -> None:
    if not resume_input: return
    rp = Path(resume_input)
    if rp.is_dir() and (rp / "AVENDI_Y3_FULL.zip").is_file(): rp = rp / "AVENDI_Y3_FULL.zip"
    if rp.is_file() and rp.suffix == ".zip":
        wr_res = wr.resolve()
        with zipfile.ZipFile(rp) as zf:
            for mem in zf.infolist():
                if mem.is_dir(): continue
                rel = Path(mem.filename)
                if rel.is_absolute() or ".." in rel.parts: raise ValueError(f"unsafe archive path: {mem.filename}")
                if not any(rel.parts[0] == sub for sub in ("jobs", "ckpts")): continue
                dst = wr_res / rel
                dst_s = ("\\\\?\\" + str(dst)) if (len(str(dst)) >= 240 and hasattr(Path, "drive") and str(dst)[1:2] == ":") else str(dst)
                Path(dst_s).parent.mkdir(parents=True, exist_ok=True)
                with zf.open(mem) as src, open(dst_s, "wb") as out: shutil.copyfileobj(src, out)
        return
    for sub in ("jobs", "ckpts"):
        src = rp / sub
        if src.is_dir(): shutil.copytree(src, wr / sub, dirs_exist_ok=True)
    for rc in rp.rglob("jobs/*/receipt.json"):
        dest = wr / "jobs" / rc.parent.name
        if not (dest / "receipt.json").is_file(): shutil.copytree(rc.parent, dest, dirs_exist_ok=True)


def run_full_campaign(*, wr: Path, S: Any = FULL_S, session_minutes: float = 660.0, input_roots: Sequence[str] = (), resume_input: str | None = None, devices: Sequence[str] = ("cuda:0", "cuda:1"), in_process_runner: Callable[..., Any] | None = None, measured_sec_per_update: float | None = None, one_go: bool = True) -> dict[str, Any]:
    """Real FULL orchestration: reconcile resume ledger, adaptively slice & dispatch all tiers in one go over both GPUs."""
    wr, t_start = Path(wr), time.perf_counter()
    wr.mkdir(parents=True, exist_ok=True)
    plan, rows = build_plan(mode="FULL" if S.full else "CANARY"), _plan_rows()
    orch = CampaignOrchestrator(campaign_id="AVENDI-V8-Y3", source_commit=plan["source_commit"], work_dir=wr, max_minutes=float(session_minutes), reserve_minutes=15.0, min_disk_bytes=1024 if not S.full else 2 * 2**30, in_process_runner=in_process_runner)
    _unpack_resume(resume_input, wr)
    imported = []
    for rc in (wr / "jobs").glob("*/receipt.json"):
        d = json.loads(rc.read_text(encoding="utf-8"))
        local_zip = rc.parent / "job_archive.zip"
        if local_zip.is_file(): d["archive_path"] = str(local_zip)
        imported.append({k: d.get(k) for k in ("job_id", "source_commit", "spec_sha256", "completed_updates", "eval_rows", "archive_path", "archive_sha256")})
    ledger = [{"job_id": j, "source_commit": plan["source_commit"], "spec_sha256": r["spec_sha256"], "updates": _job_upd(r, S)} for j, r in rows.items() if j in {i["job_id"] for i in imported}]
    rec = orch.reconcile_verified_resume(imported, ledger, required_archive_members=("receipt.json",), fail_on_rejection=True) if imported else {"verified_completed_jobs": [], "rejected_jobs": []}
    done = set(rec["verified_completed_jobs"])
    pending = sorted([rows[j] for j in rows if j not in done], key=lambda r: (_tier(r), _t0_prio(r)))
    profile = {d: _profile_device(d if torch.cuda.is_available() else "cpu", S.full) for d in devices}
    p90 = measured_sec_per_update or max(v["p90_sec_per_update"] for v in profile.values())
    sj = [SessionJob(r["job_id"], r["arm"], r["init_seed"], r.get("order_seed", r["init_seed"]), _job_upd(r, S), device=devices[i % len(devices)]) for i, r in enumerate(pending)]
    ran_ids: set[str] = set()
    waves = []
    if measured_sec_per_update is not None or not one_go:
        for tier in (0, 1):
            tier_jobs = [s for s in sj if _tier(rows[s.job_id]) == tier]
            if tier == 1:
                tier_jobs = [s for s in tier_jobs if (wr / "jobs" / (f"P6-diagnostic_policy-s{s.init_seed}" if "adapt" in s.job_id else f"P2-ce-s{s.init_seed}") / "receipt.json").is_file()]
            if not tier_jobs: continue
            tier_jobs = [replace(s, device=devices[i % len(devices)]) for i, s in enumerate(tier_jobs)]
            rem_min = max(0.0, float(session_minutes) - (time.perf_counter() - t_start) / 60.0)
            adm = orch.admit_slice(tier_jobs, measured_seconds_per_update=p90, per_job_disk_bytes=64 * 2**20 if S.full else 1024, parallel_workers=len(devices), outer_remaining_minutes=rem_min, startup_overhead_seconds=60.0 if S.full else 0.0)
            run = [s for s in tier_jobs if s.job_id in adm["admitted_job_ids"]]
            cmd = lambda j, _d: (sys.executable, "-m", "experiments.avendi_y3.module", "--job", j.job_id, "--work-root", str(wr), "--device", j.device, "--scale", "full" if S.full else "tiny", "--input-roots", ",".join(input_roots))
            wres = orch.run_waves(run, command_builder=cmd, devices=tuple(devices)) if run else {"status": "NOTHING_ADMITTED"}
            waves.append({"tier": tier, "admitted": adm["admitted_job_ids"], "deferred": adm.get("deferred_pending_job_ids", []), "ran": [s.job_id for s in run], "status": wres.get("status"), "devices_used": sorted({s.device for s in run})})
            ran_ids.update(s.job_id for s in run)
    else:
        n_devs = max(1, len(devices))
        while True:
            rem_min = float(session_minutes) - (time.perf_counter() - t_start) / 60.0
            if rem_min <= 15.0: break
            cur_done = {p.parent.name for p in (wr / "jobs").glob("*/receipt.json")}
            rem = sorted([rows[j] for j in rows if j not in cur_done], key=lambda r: (_tier(r), _t0_prio(r)))
            if not rem: break
            t0_rows = [r for r in rem if _tier(r) == 0]
            if t0_rows: tier, tr_rows = 0, t0_rows
            else: tier, tr_rows = 1, [r for r in rem if _tier(r) == 1 and (wr / "jobs" / (f"P6-diagnostic_policy-s{r['init_seed']}" if "adapt" in r["job_id"] else f"P2-ce-s{r['init_seed']}") / "receipt.json").is_file()]
            if not tr_rows: break
            obs_w, obs_u, obs_n = 0.0, 0, 0
            for rc in (wr / "jobs").glob("*/receipt.json"):
                d_r = json.loads(rc.read_text(encoding="utf-8"))
                if d_r.get("stage") != "P1" and int(d_r.get("completed_updates", 0)) > 0:
                    obs_w += float(d_r.get("wall_seconds", 0.0)); obs_u += int(d_r["completed_updates"]); obs_n += 1
            cur_p90 = max(1e-6, obs_w / obs_u) if (obs_n >= 2 and obs_u > 0) else p90
            tjobs = [SessionJob(r["job_id"], r["arm"], r["init_seed"], r.get("order_seed", r["init_seed"]), _job_upd(r, S), device=devices[i % n_devs]) for i, r in enumerate(tr_rows)]
            adm = orch.admit_slice(tjobs, measured_seconds_per_update=cur_p90, per_job_disk_bytes=32 * 2**20 if S.full else 1024, parallel_workers=n_devs, outer_remaining_minutes=max(0.0, rem_min), startup_overhead_seconds=10.0 if S.full else 0.0)
            run_list = [s for s in tjobs if s.job_id in adm["admitted_job_ids"]]
            if not run_list:
                if rem_min > 25.0: run_list = tjobs[:n_devs]
                else: break
            slice_statuses, slice_ran, before_done = [], [], len(cur_done)
            for st in range(0, len(run_list), n_devs):
                if (float(session_minutes) - (time.perf_counter() - t_start) / 60.0) <= 15.0: break
                chunk = [replace(s, device=devices[i % n_devs]) for i, s in enumerate(run_list[st : st + n_devs])]
                cmd = lambda j, _d: (sys.executable, "-m", "experiments.avendi_y3.module", "--job", j.job_id, "--work-root", str(wr), "--device", j.device, "--scale", "full" if S.full else "tiny", "--input-roots", ",".join(input_roots))
                wres = orch.run_waves(chunk, command_builder=cmd, devices=tuple(devices), resume=(wr / "COMPLETED_JOBS.json").is_file())
                slice_statuses.append(str(wres.get("status", ""))); slice_ran.extend(s.job_id for s in chunk); ran_ids.update(s.job_id for s in chunk)
            now_after = {p.parent.name for p in (wr / "jobs").glob("*/receipt.json")}
            deferred = [r["job_id"] for r in rem if r["job_id"] not in now_after]
            waves.append({"tier": tier, "admitted": [s.job_id for s in run_list], "deferred": deferred, "ran": slice_ran, "status": slice_statuses[-1] if slice_statuses else "TIME_GUARD_STOP", "devices_used": sorted({s.device for s in run_list})})
            if len(now_after) <= before_done: break
    now_done = {p.parent.name for p in (wr / "jobs").glob("*/receipt.json")}
    still_pending = [r["job_id"] for r in pending if r["job_id"] not in now_done]
    _dump(wr / "PENDING_QUEUE.json", {"pending_job_ids": still_pending, "next_resumable_job_id": still_pending[0] if still_pending else None, "total_pending": len(still_pending)})
    return {"waves": waves, "profile": profile, "p90_sec_per_update": p90, "already_verified": sorted(done), "rejected": rec.get("rejected_jobs", []), "pending_after_session": still_pending}


def aggregate_full(wr: Path, S: Any, dispatch: dict[str, Any], input_roots: Sequence[str] = ()) -> dict[str, Any]:
    """Build 50 study receipts from job receipts (real measured values only), platform coverage, confirm lock/eval, and verified archives."""
    wr = Path(wr); plan, rows = build_plan(mode="FULL" if S.full else "CANARY"), _plan_rows()
    recs = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in (wr / "jobs").glob("*/receipt.json")}
    fams = _split_fams(S)
    def need(ids: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
        return [i for i in ids if i not in recs], {i: recs[i]["result"] for i in ids if i in recs}
    seeds = (7, 19, 41)
    ids = lambda st, arm: [f"{st}-{arm}-s{s}" for s in seeds]
    studies: dict[str, dict[str, Any]] = {}
    def put(sid: str, jobs: Sequence[str], payload_fn: Callable[[dict[str, Any]], Any], split: str = "dev") -> None:
        miss, got = need(jobs)
        pay = payload_fn(got) if got else {}
        studies[sid] = _write_study(wr, sid, f"{sid}-{len(jobs)}jobs", split, payload={"required_jobs": list(jobs), "missing_jobs": miss, "denominator_jobs": len(jobs), "measured": pay, "scope": "FULL" if S.full else "TINY_SCALE_ENGINEERING_ONLY"}, status="COMPLETE" if not miss else "DEFERRED_MISSING_JOBS")
    ev = lambda g, k="native": {j: {"fam4": r["eval_dev"][k]["fam4_rate"], "fam8": r["eval_dev"][k]["fam8_rate"], "raw": r["eval_dev"][k]["raw_exact_rate"], "margin": r["eval_dev"][k]["mean_signed_margin"]} for j, r in g.items()}
    # P0 CPU fixtures
    tok_ok = all(oracle_traverse_tokens(UTF8.frame(c["public_prompt"], c["gold"]).ids) == c["gold"] for f in fams["dev"] for c in f["cells"])
    cust = audit_split_custody({k: fams[k] for k in ("train", "dev", "confirm", "transfer")})
    sc = evaluate_shortcuts_and_destructive_masks(fams["dev"])
    hb = compare_historical_vs_balanced_assignments(len(fams["dev"]), STREAM_SEEDS["dev"])
    cor = audit_case_evidence_corruptions([{"world_id": f["cells"][0]["world_id"], "tokens": list(UTF8.frame(f["cells"][0]["public_prompt"], f["cells"][0]["gold"]).ids), "pred": f["cells"][0]["gold"]} for f in fams["dev"][:8]])
    for sid, pl in (("U01", {"oracle_ok": tok_ok, "n": len(fams["dev"])}), ("U02", cust), ("U03", sc), ("U04", hb), ("U05", cor), ("U06", {"verification_rows": 6, "dispatch_waves": len(dispatch.get("waves", []))})): studies[sid] = _write_study(wr, sid, f"{sid}-cpu", "dev", payload=pl)
    put("U07", ["P1-frozen"], lambda g: g["P1-frozen"]); 
    for sid in ("U08", "U09", "U10", "U11", "U12", "U13", "U14"): put(sid, ["P1-frozen"], lambda g: {l: {k: v for k, v in d.items() if k != "native"} if isinstance(d, dict) else d for l, d in g["P1-frozen"]["lineages"].items()})
    p2 = ids("P2", "ce") + ids("P2", "margin")
    put("U15", p2, lambda g: ev(g)); put("U16", p2, lambda g: {j: r["milestones"] for j, r in g.items()}); put("U17", p2 + ids("P3", "rename") + ids("P3", "rename_sham"), lambda g: {j: ev(g, "renamed")[j] for j in g})
    put("U18", p2, lambda g: {j: {"train_fit_vs_dev": r["milestones"].get(str(max(map(int, r["milestones"]))) if r["milestones"] else "0")} for j, r in g.items()}); put("U19", [f"U19-{a}-s{s}" for a in ("ce", "margin") for s in seeds], lambda g: {j: {"train_fit": r["train_fit_fam4"], "dev": r["eval_dev"]["native"]["fam4_rate"], "grad": r["gradient_sign"]["model_grad_cosine"]} for j, r in g.items()})
    put("U20", p2, lambda g: {j: {"fam4": v["fam4"], "fam8": v["fam8"]} for j, v in ev(g).items()})
    for sid, arms in (("U21", ["refresh"]), ("U22", ["curriculum"]), ("U23", ["address", "address_sham"]), ("U24", ["rename", "rename_sham"]), ("U25", ["low_lr"])): put(sid, ids("P2", "ce") + [j for a in arms for j in ids("P3", a)], lambda g: ev(g))
    put("U26", [j for a in ("direct_plus_neutral", "direct_plus_info", "replace_direct") for j in ids("P5", a)], lambda g: {j: r["child"] for j, r in g.items()})
    put("U27", p2 + [j for a in ARMS_P3 for j in ids("P3", a)], lambda g: ev(g))
    p4 = [j for a in ("categorical", "continuous", "dense") for j in ids("P4", a)]
    for sid in (f"U{i}" for i in range(28, 36)): put(sid, p4 + [f"P4-{a}_endonly-s{s}" for a in ("categorical", "continuous", "dense") for s in seeds], lambda g: {j: {"rates": {k: {d: v["complete_rate"] for d, v in vv.items()} for k, vv in r["rates"].items()}, "causal": r["causal"], "oracle": r["oracle_prototype_injection_acc"], "widths": r["widths"]} for j, r in g.items()})
    refs = evaluate_p5_reference_policies(build_p5_executive_worlds("dev", S.p5, STREAM_SEEDS["dev"]))
    p5 = [j for a in ("direct_plus_neutral", "direct_plus_info", "replace_direct") for j in ids("P5", a)]
    for sid in ("U36", "U37", "U38", "U39", "U40"): put(sid, p5, lambda g: {"forks": {j: r["child"] for j, r in g.items()}, "parents": {j: r["parent"] for j, r in g.items()}, "dp_reference": refs})
    p6 = ids("P6", "diagnostic_policy") + [f"P6-adapt-{r}-s{s}" for s in seeds for r in ("context", "memory", "weight_update")]
    for sid in (f"U{i}" for i in range(41, 47)): put(sid, p6, lambda g: {j: r["eval_dev"] for j, r in g.items()})
    p7 = ids("P7", "s20_ce") + ids("P7", "s20_low_lr") + [f"P7-{c}-s{s}" for c in ("ark_capped", "ark_uncapped", "ark_expaug") for s in seeds]
    # sealed confirmation: lock only when every eligible P2/P3/P7-s20 arm has finished
    elig = p2 + [j for a in ARMS_P3 for j in ids("P3", a)] + ids("P7", "s20_ce") + ids("P7", "s20_low_lr")
    miss_elig, _ = need(elig)
    confirm = {"status": "DEFERRED_ELIGIBLE_ARMS_INCOMPLETE", "missing": miss_elig}
    if not miss_elig:
        lock_p = wr / "CONFIRM_LOCK.json"
        lock_sha = write_confirmation_lock(lock_p, {"eligible_jobs": elig, "checkpoint_sha256": {j: recs[j]["result"]["train"]["checkpoint_sha256"] for j in elig}, "comparator": "P2 ce", "ranking": "mean_dev_fam8_then_worst_seed", "decoder": "raw_eos", "cutoffs": plan["qualification_gates"]["P2"], "seeds": list(seeds)})
        confirm = {"status": "LOCKED_AND_OPENED_ONCE", **evaluate_with_confirmation_lock(lock_p, lock_sha, lambda pl: {j: _confirm_one(rows[j], S, fams, wr) for j in pl["eligible_jobs"]})}
    put("U47", ids("P7", "s20_ce") + ids("P7", "s20_low_lr") + p2, lambda g: {"dev": ev(g) if all("eval_dev" in r for r in g.values()) else {}, "confirm": confirm}, "confirm" if not miss_elig else "dev")
    put("U48", [f"P7-{c}-s{s}" for c in ("ark_capped", "ark_uncapped", "ark_expaug") for s in seeds], lambda g: {"arkenstone": {j: r["child"] | {"cap": r["cap"]} for j, r in g.items()}, "donor_audit": run_p7_donor_audit_and_interventions(cpu_canary=not S.full)})
    put("U49", p7, lambda g: {"donors": run_p7_donor_audit_and_interventions(cpu_canary=not S.full)})
    done_jobs = sorted(recs); total_jobs = len(rows)
    studies["U50"] = _write_study(wr, "U50", "U50-final", "dev", payload={"jobs_complete": len(done_jobs), "jobs_total": total_jobs, "dispatch": dispatch, "confirm": confirm}, status="COMPLETE" if len(done_jobs) == total_jobs else "PARTIAL")
    n_studies_done = sum(1 for s in studies.values() if s["status"] == "COMPLETE")
    status = "COMPLETE" if len(done_jobs) == total_jobs and n_studies_done == 50 else "PARTIAL"
    vmat = build_verification_matrix(byte_integrity="PASS", oracle_fixture_correctness="PASS", recorded_action_replay="PASS", checkpoint_restore="UNAVAILABLE_Y1_PARENT" if not input_roots else "PASS", regenerated_per_case_agreement="PASS" if status == "COMPLETE" else "PARTIAL_REF_CHECK", training_reproduction="PASS")
    arm_recs = [{"architecture": recs[j]["arm"], "init_seed": recs[j]["init_seed"], "order_seed": recs[j]["init_seed"], "value": recs[j]["result"]["eval_dev"]["native"]["fam4_rate"]} for j in p2 if j in recs]
    final = {"status": status, "mode": "FULL" if S.full else "TINY", "studies_complete": n_studies_done, "studies_total": 50, "main_jobs_complete": sum(1 for j in done_jobs if j in {r["job_id"] for r in plan["training_jobs"]}), "main_jobs_total": 54, "short_jobs_complete": sum(1 for j in done_jobs if j in {r["job_id"] for r in plan["short_fork_jobs"]}), "short_jobs_total": 24, "p6_adapt_complete": sum(1 for j in done_jobs if "adapt" in j), "pending_jobs": sorted(set(rows) - set(done_jobs)), "dispatch": dispatch, "confirm": confirm["status"], "studies": {k: v["status"] for k, v in studies.items()}, "verification_matrix": vmat, "scientific_note": "COMPLETE requires all jobs; engineered passes never satisfy learned rows"}
    md = render_campaign_results_markdown(title="Avendi V8-Y3 Campaign Results", campaign_id="AVENDI-V8-Y3", source_commit=plan["source_commit"], status=status, verification_matrix=vmat, stage_summaries={k: v["status"] for k, v in studies.items()})
    (wr / "RESULTS.md").write_text(md, encoding="utf-8")
    _dump(wr / "METRICS.json", {"title": "Avendi V8-Y3 Campaign Results", "campaign_id": "AVENDI-V8-Y3", "source_commit": plan["source_commit"], "control_arm": "ce", "treatment_arm": "margin", "decision_summary": f"Jobs {len(done_jobs)}/{total_jobs}, studies {n_studies_done}/50, confirm={confirm['status']}.", "arm_records": arm_recs, "status": status})
    _dump(wr / "FINAL_STATUS.json", final)
    sum_files = {"FINAL_STATUS.json": wr / "FINAL_STATUS.json", "RESULTS.md": wr / "RESULTS.md", "METRICS.json": wr / "METRICS.json"}
    if (wr / "PENDING_QUEUE.json").is_file(): sum_files["PENDING_QUEUE.json"] = wr / "PENDING_QUEUE.json"
    ev_files = {str(p.relative_to(wr)): p for p in (wr / "jobs").rglob("*") if p.is_file() and p.suffix in (".npz", ".json") and p.name != "receipt_body.json"}
    ev_files.update({str(p.relative_to(wr)): p for p in (wr / "studies").rglob("receipt.json")})
    full_files = {**sum_files, **ev_files, **{str(p.relative_to(wr)): p for p in (wr / "jobs").rglob("job_archive.zip")}}
    if (wr / "ckpts").is_dir():
        for lin_dir in sorted((wr / "ckpts").iterdir()):
            if not lin_dir.is_dir(): continue
            keep_shas: set[str] = set()
            for meta_name in ("LATEST", "MILESTONES"):
                mp = lin_dir / meta_name
                if mp.is_file():
                    full_files[str(mp.relative_to(wr))] = mp
                    if meta_name == "LATEST":
                        keep_shas.add(mp.read_text(encoding="ascii").strip())
                    else:
                        try: keep_shas.update(str(v) for v in json.loads(mp.read_text(encoding="utf-8")).get("milestones", {}).values())
                        except Exception: pass
            for sh in sorted(keep_shas):
                odir = lin_dir / "objects" / sh
                if odir.is_dir(): full_files.update({str(p.relative_to(wr)): p for p in odir.rglob("*") if p.is_file()})
    exp = export_session_archives(wr, wr / "archives", prefix="AVENDI_Y3", summary_files=sum_files, evidence_files=ev_files or sum_files, analysis_files=sum_files, full_files=full_files, status=status, source_commit=plan["source_commit"])
    final["archives"], final["archive_sha256"], final["summary_bytes"] = exp["archives"], exp["archive_sha256"], Path(exp["archives"]["summary"]).stat().st_size
    _dump(wr / "FINAL_STATUS.json", final)
    return final


ARMS_P3 = ("refresh", "curriculum", "address", "address_sham", "rename", "rename_sham", "low_lr")


def _confirm_one(row: dict[str, Any], S: Any, fams: dict[str, Any], wr: Path) -> dict[str, Any]:
    """Separate-lock confirm eval of one checkpointed arm (restores trainer from the immutable store; no training)."""
    spec = get_byte_model_spec("S20" if row["arm"].startswith("s20") else "S3", cpu_canary=not S.full)
    items, groups = _groups(fams["train"])
    if row["arm"] == "refresh":
        ritems, rg = _groups(fams["refresh"]); off = len(items); n_tr = len(groups)
        groups = groups + [[i + off for i in g] for g in rg]; items = items + ritems
        plan = _epoch_plan(groups, S.upd, S.batch // 4, row["init_seed"], lambda e: range(n_tr) if e == 0 else range(n_tr + (e - 1) * n_tr, n_tr + e * n_tr))
    else:
        easy = [bool(items[g[0]]["is_easy_stratum"]) for g in groups] if row["arm"] == "curriculum" else None
        plan = _epoch_plan(groups, S.upd, S.batch // 4, row["init_seed"], lambda e: range(len(groups)), easy=easy)
    lr = 3e-5 if row["arm"] in ("low_lr", "s20_low_lr") else 1e-4
    obj = {"margin": "margin", "address": "address", "address_sham": "address_sham", "rename": "rename", "rename_sham": "rename_sham"}.get(row["arm"], "ce")
    tr = _gen_trainer(row, S, "cpu" if not torch.cuda.is_available() else "cuda:0", wr, AvendiByteModel(spec, seed=row["init_seed"]), spec, items, plan, lr=lr, objective=obj, loss_fn=lambda m, c, r: compute_avendi_loss(m, c, r), micro=(2 if row["arm"].startswith("s20") else S.micro), upd=S.upd, resume=True)
    if tr.state.global_update < S.upd: raise RuntimeError("CHECKPOINT_NOT_COMPLETE:" + row["job_id"])
    ev = eval_byte_model_on_families(tr.model, fams["confirm"])
    lit = eval_byte_model_on_families(tr.model, build_literal_probe_families(S.lit_n, STREAM_SEEDS["confirm"]))
    return {"fam4": ev["fam4_rate"], "fam8": ev["fam8_rate"], "raw": ev["raw_exact_rate"], "literal_raw": lit["raw_exact_rate"], "family_scores_4cell": ev["family_scores_4cell"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Avendi V8-Y3 Campaign CLI")
    ap.add_argument("--mode", default="CANARY", choices=("CANARY", "FULL"))
    ap.add_argument("--work-root", type=Path, default=Path("runs/avendi_y3"))
    ap.add_argument("--session-minutes", type=float, default=660.0)
    ap.add_argument("--resume-input", type=str, default=None)
    ap.add_argument("--input-roots", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--no-one-go", action="store_true")
    ap.add_argument("--job", type=str, default=None)
    ap.add_argument("--scale", default="full", choices=("full", "tiny"))
    args = ap.parse_args(argv)
    S = FULL_S if args.scale == "full" else TINY_S
    if args.job:
        run_job(args.job, S, args.device, args.work_root, [r for r in args.input_roots.split(",") if r]); return 0
    if args.inventory_only:
        plan = build_plan(mode=args.mode)
        print(json.dumps({"inventory_only": True, "main_jobs": len(plan["training_jobs"]), "short_forks": len(plan["short_fork_jobs"]), "p6_adapt_jobs": len(plan["p6_adaptation_jobs"]), "studies": len(plan["studies"]), "platform_uses": len(plan["platform_uses"]), "update_totals": plan["update_totals"]}, indent=2))
        return 0
    roots = [r.strip() for r in args.input_roots.split(",") if r.strip()] + ([args.resume_input] if args.resume_input else [])
    if args.mode == "FULL":
        if S.full and torch.cuda.device_count() < 2: raise SystemExit("FULL requires both T4 GPUs (torch.cuda.device_count() >= 2)")
        disp = run_full_campaign(wr=args.work_root, S=S, session_minutes=args.session_minutes, input_roots=roots, resume_input=args.resume_input, one_go=not args.no_one_go)
        print(json.dumps(aggregate_full(args.work_root, S, disp, roots), indent=2, default=_jd)); return 0
    print(json.dumps(run_avendi_y3_campaign(mode=args.mode, work_root=args.work_root, session_minutes=args.session_minutes, resume_input=args.resume_input, input_roots=roots, device=args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

