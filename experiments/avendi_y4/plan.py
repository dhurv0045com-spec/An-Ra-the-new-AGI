"""Programmatic 126-job Avendi V8-Y4 registry, DAG validation, and source-pinned Kaggle notebook."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SEEDS_5 = (7, 19, 41, 73, 101)
SEEDS_3 = (7, 19, 41)
TECHNOLOGY_BASELINE = "ba1c0e978f890021a10e2365b61d5403bc486ce5"
Y4_BASE_PIN = "6b2cf407789ef1fcb694df0bc0558f5d85ee65c5"
REPO_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
STREAM_SEEDS = {"corpus": 4401, "dev": 4402, "confirm": 4403, "transfer": 4404, "cpu": 4400, "bootstrap": 4417}

Y1_EXPECTED_CKPT_SHAS = {
    "Y1-t1_factorial-S3-F00-s7": "57ab8b7292a84689ee32d797257eb4a2c9c8cd3c95421df2ca254801c6cd1706",
    "Y1-t1_factorial-S3-F00-s19": "b900fa477499e5a7eca2d5ac4ed717e1f1ff34981a3ae2a85e9aae1a8c3da475",
    "Y1-t1_factorial-S3-F00-s41": "4e8891f6f3d8ef1975d949bc7c96393edda9c4dc304b12cf5a5368e5fdffdf95",
    "Y1-t1_factorial-S3-F11-s7": "65c8506d7359d91c76fd5ac3628f0226c68f76a2edf760f1e62d309e9fa0313f",
    "Y1-t1_factorial-S3-F11-s19": "96fa4f25b62601b8533d602ad5e1ba6587760bc3cb8bb758a0230a2a7bf686ef",
    "Y1-t1_factorial-S3-F11-s41": "f2ce4dfc965559ee3b2e189a80ce3e8e84786c4e23d1de3a78249b544961babf",
}
ARCHIVE_SHAS = {
    "ELENDRIL_V8_Y1_EVIDENCE.zip": "4f76ac0d3a1fc67fac2705325c0909f791325696f9a1162c9e86c783c717173f",
    "AVENDI_V8_Y2_EVIDENCE.zip": "d30607012647c51f6e71c87ea2f21132dccb7e7dcfd738cb99d462aed9033775",
    "AVENDI_Y3_ANALYSIS.zip": "fad1f35312d591cd9381af6fc0a7269b88a909441df8634fda80fc116b153d1f",
    "AVENDI_Y3_EVIDENCE.zip": "a518838ffaa2b937d2639e405e055a087a18bb437ad62560e173da5dd5a3b49a",
}
DONOR_PINS = {
    "cymek": "28bf57a0d299a2c13a99fe0046616c00a1b8530c",
    "arkenstone": "4ae9e3b6dc806d6ada7b7b2f05d32271d072d543",
    "elendril": "cb9a91598c1d831fecb8b7d483f4ceeba1d063a8",
}
STUDIES = {
    "P0": range(1, 7), "H": range(7, 15), "A": range(15, 21), "B": range(21, 28),
    "C": range(28, 36), "D": range(36, 41), "E": range(41, 47), "G": range(47, 51),
}
STUDY_STAGES = {f"U{n:02}": st for st, nums in STUDIES.items() for n in nums}

Y1_DISCOVERY_LINKS = {
    "01": ("U12",), "02": ("U37",), "03": ("U04",), "04": ("U02",), "05": ("U03",),
    "06": ("U11",), "07": ("U09", "U31"), "08": ("U10",), "09": ("A", "C"), "10": ("U07", "U08"),
    "11": ("U26",), "12": ("U39",), "13": ("U21",), "14": ("U14", "U49"), "15": ("U47",),
    "16": ("U16",), "17": ("U18", "U19"), "18": ("U27", "U50"), "19": ("U50",), "20": ("U06", "U07"),
    "21": ("U05", "U06"), "22": ("U02", "U05", "U06"), "23": ("U40",), "24": ("U08", "U14", "U25", "U37"),
    "25": tuple(f"U{i:02}" for i in range(41, 47)),
}

PLATFORM_USES = (
    ("01", "doctor", "U01"), ("02", "calibrate", "U01"), ("03", "plugin_registration", "U06"), ("04", "init", "U06"),
    ("05", "validate", "U06"), ("06", "cartridge_construction", "U06"), ("07", "GenericDatasetBundle", "U02"), ("08", "UniversalObservation", "U02"),
    ("09", "environment_kernel", "U03"), ("10", "stateful_policy_reset", "U42"), ("11", "batched_rollout", "U07"), ("12", "sampled_strata_audit", "U21"),
    ("13", "locate_checkpoint_root", "U07"), ("14", "load_legacy_checkpoint", "U07"), ("15", "LegacyByteObservablePolicy", "U07"), ("16", "evaluate_frozen_checkpoints", "U07"),
    ("17", "byte_framing", "U15"), ("18", "byte_trainer", "U15"), ("19", "ce_and_pair_margin", "U15"), ("20", "gradient_sign_control", "U19"),
    ("21", "GenericExperimentTrainer", "U28"), ("22", "backbone_overhead_verification", "U48"), ("23", "surgical_ablation", "U35"), ("24", "crossover_fixture_verification", "U01"),
    ("25", "crossover_scoring", "U15"), ("26", "family_bootstrap", "U15"), ("27", "seed_pair_builder", "U27"), ("28", "paired_seed_analysis", "U27"),
    ("29", "paired_comparison_preflight", "U06"), ("30", "paired_episode_comparison", "U27"), ("31", "stratify_intervention_worlds", "U12"), ("32", "diagnostic_coverage", "U08"),
    ("33", "stage_dag", "U50"), ("34", "dual_clock_admission", "U50"), ("35", "verified_resume_reconciliation", "U06"), ("36", "independent_worker_waves", "U50"),
    ("37", "session_archive_export", "U50"), ("38", "notebook_generator", "U50"), ("39", "verify_bundle", "U06"), ("40", "replay_bundle", "U05"),
    ("41", "compact_archive_rescoring", "U50"), ("42", "verification_matrix", "U50"), ("43", "analyst_first_page", "U50"), ("44", "markdown_rendering", "U50"),
)

DEFECT_REGISTER = (
    {"status": "Confirmed instrument defect", "y3_observation": "Deterministic walker solved 64,64,0,0 of 64 families (d1 coincide, d3 missing reverse edge, d4 conflicting key).", "y4_disposition": "Require distinct counterfactual endpoints and independent public-byte oracle on all cells; preserve Y3 counterexamples as witnesses."},
    {"status": "Confirmed shortcut", "y3_observation": "Y3 P5 condition prefix recovered optimal action class on 140/140 dev worlds.", "y4_disposition": "Strip condition names, counterbalance flags, match same-flag opposite-action twins, execute tools to termination."},
    {"status": "Confirmed implementation defects", "y3_observation": "Per-slot RNG re-seeding selected 1 item 16x in P4/P6; wrong-donor used rotated target; P6 inherited 1024 counter.", "y4_disposition": "Single persistent stream RNG per job, solve actual inserted donor on current recipient, separate step/eval counters."},
    {"status": "Confirmed operational defects", "y3_observation": "6 Y1 parents hit EROFS on read-only input; case parity unavailable.", "y4_disposition": "Stage actual objects into writable scratch, verify manifest SHA, compare per-case logits (atol=1e-4) and EOS bytes."},
    {"status": "Previously repaired in code", "y3_observation": "Worker/coordinator runtime_identity and silent fresh-init fallback repaired at 60e8cb27.", "y4_disposition": "Repeat subprocess-to-coordinator positive and corrupted-LATEST negative tests on Y4."},
    {"status": "Suspected / test before interpretation", "y3_observation": "Padding clamp in row attention; FP32 grad cosine < -1; first-char P5 scoring.", "y4_disposition": "Pad/perm invariance checks, unpadded serial parity, FP64 reference cosine, full-response/EOS scoring."},
    {"status": "Scientific unknowns", "y3_observation": "P2 margin +5.73pp dev / +3.52pp confirm missed gates; P4 causal promising; S20/cap unsolved acquisition+retention.", "y4_disposition": "Separate byte grounding, transition learning, held-out composition, relevance, memory correction, and norm-cap controls."},
)

QUALIFICATION_GATES = {
    "C_primary": {"iit_gain_min": 0.10, "positive_seeds_min": 4, "seeds_total": 5},
    "C_absolute": {"d1_min": 0.90, "d2_min": 0.90, "d3_min": 0.80, "d4_min": 0.80, "topology_min": 0.75, "donor_min": 0.85, "rewire_min": 0.85, "identity_min": 0.99, "wrong_intended_max": 0.20, "seeds_min": 4},
    "A_byte": {"fam4_min": 0.80, "fam8_min": 0.80, "literal_min": 0.95, "irrel_fam8_min": 0.75, "pointer_gain_min": 0.10, "max_literal_drop": 0.05, "seeds_min": 4},
    "D_exec": {"macro_min": 0.80, "worst_cond_min": 0.70, "max_false_abstain": 0.05, "post_tool_min": 0.80, "utility_gain_min": 0.10, "info_vs_neutral_min": 0.10, "max_retention_drop": 0.05},
    "E_memory": {"post_clear_min": 0.75, "gain_vs_controls_min": 0.15, "erase_drop_min": 0.10, "max_retention_drop": 0.05, "utility_gain_min": 0.05, "weight_rel_norm_cap": 0.001},
    "F_G_scale_donor": {"s20_fam8_gain_min": 0.05, "donor_gain_min": 0.10, "max_literal_drop": 0.05, "max_retention_drop": 0.05},
}


def _canon_sha(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _job_row(
    job_id: str,
    stage: str,
    arm: str,
    seed: int,
    updates: int,
    *,
    backbone: str,
    parent_job_id: str | None = None,
    profile_group: str = "S3",
    is_historical: bool = False,
    bridge_parents: tuple[str, ...] = (),
) -> dict[str, Any]:
    depends = ([parent_job_id] if parent_job_id else []) + list(bridge_parents)
    return {
        "job_id": job_id,
        "stage": stage,
        "arm": arm,
        "init_seed": int(seed),
        "order_seed": int(seed),
        "updates": int(updates),
        "backbone": backbone,
        "model_identity": f"y4-{job_id}",
        "data_identity": f"y4-corpus-{STREAM_SEEDS['corpus']}",
        "intervention": f"{stage}:{arm}",
        "parent_job_id": parent_job_id,
        "depends_on": depends,
        "profile_group": profile_group,
        "is_historical": bool(is_historical),
        "resource_profile": "gpu_t4_s20" if profile_group == "S20" else ("historical_restore" if is_historical else "gpu_t4"),
        "spec_sha256": _canon_sha({"job_id": job_id, "stage": stage, "arm": arm, "seed": seed, "updates": updates, "parent": parent_job_id}),
        "status": "PLANNED",
        "receipt": f"jobs/{job_id}/receipt.json",
    }


def build_plan(source_commit: str = Y4_BASE_PIN, *, mode: str = "FULL") -> dict[str, Any]:
    """Expand the 126-job Avendi V8-Y4 ledger (120 trained + 6 historical), 50 studies, 25 discoveries, 44 platform uses."""
    jobs: list[dict[str, Any]] = []
    for arm in ("F00", "F11"):
        for s in SEEDS_3:
            jobs.append(_job_row(f"H-{arm}-s{s}", "H", arm, s, 2368, backbone="Y1-S3", profile_group="H", is_historical=True))
    for arm in ("ce", "margin", "pointer", "pointer_sham"):
        for s in SEEDS_5:
            jobs.append(_job_row(f"A-{arm}-s{s}", "A", arm, s, 1024, backbone="S3", profile_group="S3"))
    for arm in ("refresh", "curriculum", "address", "address_sham", "rename", "rename_sham", "low_lr"):
        for s in SEEDS_3:
            jobs.append(_job_row(f"B-{arm}-s{s}", "B", arm, s, 1024, backbone="S3", profile_group="S3"))
    for arm in ("ce", "margin"):
        for s in SEEDS_3:
            jobs.append(_job_row(f"M-{arm}-s{s}", "M", arm, s, 128, backbone="S3", profile_group="S3"))
    for arch in ("categorical", "continuous", "dense"):
        for sup in ("endpoint_only", "true_step"):
            for s in SEEDS_5:
                jobs.append(_job_row(f"C-{arch}_{sup}-s{s}", "C", f"{arch}_{sup}", s, 1024, backbone=f"C-{arch}", profile_group="C"))
    for sup in ("true_step_iit", "true_step_sham_iit"):
        for s in SEEDS_5:
            jobs.append(_job_row(f"C-categorical_{sup}-s{s}", "C", f"categorical_{sup}", s, 1024, backbone="C-categorical", profile_group="C"))
    for arm in ("a_to_a", "direct_plus_neutral", "direct_plus_info", "replace_direct"):
        for s in SEEDS_3:
            jobs.append(_job_row(f"D-{arm}-s{s}", "D", arm, s, 1024, backbone="S3", parent_job_id=f"A-ce-s{s}", profile_group="S3"))
    for arm in ("learned_writer", "sham_writer"):
        for s in SEEDS_3:
            jobs.append(_job_row(f"E-{arm}-s{s}", "E", arm, s, 1024, backbone="S3-GRU", profile_group="E"))
    for arm in ("s20_ce", "s20_low_lr"):
        for s in SEEDS_3:
            jobs.append(_job_row(f"F-{arm}-s{s}", "F", arm, s, 1024, backbone="S20", profile_group="S20"))
    for arm in ("uncapped", "true_cap", "sham_cap"):
        for s in SEEDS_3:
            jobs.append(_job_row(f"G-{arm}-s{s}", "G", arm, s, 256, backbone="S3", parent_job_id=f"A-ce-s{s}", profile_group="G"))

    studies = [{"study_id": f"U{n:02}", "home_stage": STUDY_STAGES[f"U{n:02}"], "status": "PLANNED", "receipt_pattern": f"studies/U{n:02}/<job>/<split>/receipt.json"} for n in range(1, 51)]
    trained_jobs = [j for j in jobs if not j["is_historical"]]
    hist_jobs = [j for j in jobs if j["is_historical"]]
    confirm_eligible = [
        j["job_id"] for j in trained_jobs if j["stage"] in ("A", "B", "C", "D", "E", "F", "G")
    ]
    return {
        "campaign": "Avendi V8-Y4",
        "technology_baseline": TECHNOLOGY_BASELINE,
        "source_commit": source_commit,
        "mode": mode,
        "phase": "DESIGN",
        "jobs": jobs,
        "training_jobs": trained_jobs,
        "historical_jobs": hist_jobs,
        "confirm_eligible_jobs": confirm_eligible,
        "studies": studies,
        "frozen_lineages": [f"Y1-t1_factorial-S3-{arm}-s{s}" for arm in ("F00", "F11") for s in SEEDS_3],
        "rng_manifest": {"algorithm": "numpy.random.PCG64+python.Random+sha256", "python": sys.version.split()[0], "streams": STREAM_SEEDS},
        "defect_register": list(DEFECT_REGISTER),
        "y1_discovery_links": Y1_DISCOVERY_LINKS,
        "platform_uses": [{"use_id": u, "name": n, "owning_study": st, "status": "PLANNED"} for u, n, st in PLATFORM_USES],
        "donor_pins": DONOR_PINS,
        "expected_y1_checkpoints": Y1_EXPECTED_CKPT_SHAS,
        "archive_digests": ARCHIVE_SHAS,
        "qualification_gates": QUALIFICATION_GATES,
        "update_totals": {"total_jobs": len(jobs), "trained_jobs": len(trained_jobs), "historical_jobs": len(hist_jobs), "total_updates": sum(j["updates"] for j in trained_jobs)},
    }


def validate_plan_dag(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate 126 distinct jobs, 120 trained + 6 historical, 50 studies, 25 discoveries, 44 uses, and DAG acyclicity."""
    jobs = plan["jobs"]
    by_id = {j["job_id"]: j for j in jobs}
    assert len(jobs) == 126 and len(by_id) == 126, f"Expected 126 distinct jobs, got {len(jobs)}"
    assert len(plan["training_jobs"]) == 120 and len(plan["historical_jobs"]) == 6
    assert [s["study_id"] for s in plan["studies"]] == [f"U{n:02}" for n in range(1, 51)]
    assert len(plan["y1_discovery_links"]) == 25 and len(plan["platform_uses"]) == 44
    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(jid: str) -> None:
        if jid in visited:
            return
        if jid in visiting:
            raise ValueError(f"CYCLE_IN_DAG:{jid}")
        visiting.add(jid)
        for dep in by_id[jid].get("depends_on", ()):
            if dep not in by_id:
                raise ValueError(f"MISSING_PARENT:{dep}")
            _dfs(dep)
        visiting.remove(jid)
        visited.add(jid)

    for jid in by_id:
        _dfs(jid)
    return {"valid": True, "jobs": len(jobs), "trained": len(plan["training_jobs"]), "historical": len(plan["historical_jobs"]), "studies": 50, "platform_uses": 44}


def build_y4_notebook(source_commit: str) -> dict[str, Any]:
    """Generate the source-pinned Avendi V8-Y4 Kaggle 2xT4 notebook (`OWNER_AUTHORIZED=False` by default)."""
    from anra_v5.palantiri import build_pinned_kaggle_notebook

    globs = ("*.zip", "*.sha256", "FINAL_STATUS.json", "PENDING_QUEUE.json", "BLOCKED_JOBS.json", "RESULTS.md", "METRICS.json", "CONFIRM_LOCK.json", "VERIFIED_AUDIT.json", "SESSION_TELEMETRY.json", "PLATFORM_COVERAGE.json")
    nb = build_pinned_kaggle_notebook(
        source_commit=source_commit,
        title="Avendi V8-Y4 · Can a Learned Lookup Rule Compose? (Kaggle 2×T4)",
        launch_command="python -m experiments.avendi_y4.module --mode FULL --session-minutes 330",
        inventory_command="python -m experiments.avendi_y4.module --mode FULL --inventory-only",
        owner_authorized=False,
        repo_url=REPO_URL,
        campaign_mode="FULL",
        session_minutes=330,
        discover_kaggle_inputs=("ELENDRIL_V8_Y1_FULL.zip", "ELENDRIL_V8_Y1_EVIDENCE.zip", "AVENDI_V8_Y2_EVIDENCE.zip", "AVENDI_Y3_EVIDENCE (1).zip", "AVENDI_Y3_FULL.zip"),
        mirror_outputs_to_working=True,
        preflight_canary_backend="avendi_y4",
        require_dual_gpu=True,
        emit_file_links=True,
        working_export_globs=globs,
        backend_modules=("experiments.avendi_y4.module", "experiments.avendi_y4.verify"),
    )
    nb["cells"][0]["source"] = [
        "# Avendi V8-Y4 · Can a Learned Lookup Rule Compose? (Kaggle 2×T4)\n",
        f"- Pinned commit: `{source_commit}` · Baseline: `{TECHNOLOGY_BASELINE}`\n",
        "- Defaults: `OWNER_AUTHORIZED = False` (set `OWNER_AUTHORIZED = True` to run FULL 126-job campaign on 2×T4), `SESSION_MINUTES = 330`, `ONE_GO = True`.\n",
    ]
    nb["cells"][1]["source"] = [ln + "\n" for ln in [
        f"SOURCE_SHA = {source_commit!r}", "OWNER_AUTHORIZED = False", "CAMPAIGN_MODE = 'FULL'", "SESSION_MINUTES = 330", "ONE_GO = True", "RESUME_INPUT = None", f"REPO_URL = {REPO_URL!r}",
        "DISCOVER_INPUTS = ['ELENDRIL_V8_Y1_FULL.zip', 'ELENDRIL_V8_Y1_EVIDENCE.zip', 'AVENDI_V8_Y2_EVIDENCE.zip', 'AVENDI_Y3_EVIDENCE (1).zip', 'AVENDI_Y3_FULL.zip']",
        "MIRROR_WORKING = True", f"EXPORT_GLOBS = {list(globs)!r}",
        "import os, shutil, subprocess, sys", "from pathlib import Path",
        "repo = Path('/kaggle/working/anra_avendi_y4_src') if Path('/kaggle/working').is_dir() else Path('.')",
        "if repo != Path('.') and not (repo / '.git').exists():",
        "    if repo.exists(): shutil.rmtree(repo)",
        "    subprocess.check_call(['git', 'clone', '--no-checkout', REPO_URL, str(repo)])",
        "if repo != Path('.'):",
        "    if subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', SOURCE_SHA + '^{commit}']).returncode != 0:",
        "        subprocess.check_call(['git', '-C', str(repo), 'fetch', 'origin', '--tags'])",
        "    subprocess.check_call(['git', '-C', str(repo), 'checkout', '--detach', SOURCE_SHA])",
        "    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()",
        "    assert head == SOURCE_SHA, f'HEAD mismatch: {head} != {SOURCE_SHA}'",
        "    os.chdir(str(repo))",
        "if str(Path.cwd()) not in sys.path: sys.path.insert(0, str(Path.cwd()))",
        "os.environ['PLANTHIR_BACKEND_MODULES'] = 'experiments.avendi_y4.module'",
        "if not OWNER_AUTHORIZED:",
        "    print('NOT LAUNCHED: set OWNER_AUTHORIZED = True before running GPU launch.')",
        "    subprocess.check_call([sys.executable, '-m', 'experiments.avendi_y4.module', '--mode', CAMPAIGN_MODE, '--session-minutes', str(SESSION_MINUTES), '--inventory-only'])",
        "else:",
        "    import torch",
        "    assert torch.cuda.device_count() >= 2, f'Expected >=2 GPUs, found {torch.cuda.device_count()}'",
        "    free_gb = shutil.disk_usage('/kaggle/working' if Path('/kaggle/working').is_dir() else '.').free / 2**30",
        "    assert free_gb >= 4.0, f'Low disk: {free_gb:.2f} GiB'",
        "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'doctor'])",
        "    cdir = Path('work/preflight_y4'); cdir.mkdir(parents=True, exist_ok=True); spec_p = cdir / 'spec.json'",
        "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'init', '--backend', 'avendi_y4', '--out', str(spec_p)])",
        "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'validate', '--spec', str(spec_p), '--canary'])",
        "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'run', '--spec', str(spec_p), '--work-root', str(cdir / 'cpu'), '--canary', '--device', 'cpu'])",
        "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'verify-bundle', '--bundle-dir', str(cdir / 'cpu' / 'result_bundle')])",
        "    subprocess.check_call([sys.executable, '-m', 'anra_v5.palantiri', 'replay-bundle', '--bundle-dir', str(cdir / 'cpu' / 'result_bundle'), '--output', str(cdir / 'REPLAY_RECEIPT.json')])",
    ]]
    nb["cells"][2]["source"] = [ln + "\n" for ln in [
        "if not globals().get('OWNER_AUTHORIZED', False):",
        "    print('Launch skipped while OWNER_AUTHORIZED = False.')",
        "else:",
        "    import os, shutil, subprocess, sys",
        "    from pathlib import Path",
        "    roots = []",
        "    if Path('/kaggle/input').is_dir():",
        "        for n in DISCOVER_INPUTS: roots.extend(str(p) for p in sorted(Path('/kaggle/input').rglob(n)))",
        "        for ck in sorted(Path('/kaggle/input').rglob('Y1-t1_factorial-S3-F00-s7')):",
        "            if ck.is_dir() and str(ck.parent) not in roots: roots.append(str(ck.parent))",
        "    if RESUME_INPUT is None and Path('/kaggle/input').is_dir():",
        "        rc = [str(p) for p in sorted(Path('/kaggle/input').rglob('AVENDI_Y4_FULL.zip')) if p.is_file()]",
        "        if rc: RESUME_INPUT = rc[-1]",
        "    work = Path('/kaggle/working/avendi_y4_run') if Path('/kaggle/working').is_dir() else Path('runs/avendi_y4_run')",
        "    cmd = [sys.executable, '-m', 'experiments.avendi_y4.module', '--mode', CAMPAIGN_MODE, '--session-minutes', str(SESSION_MINUTES), '--work-root', str(work)]",
        "    if not globals().get('ONE_GO', True): cmd.append('--no-one-go')",
        "    if RESUME_INPUT: cmd.extend(['--resume-input', str(RESUME_INPUT)])",
        "    if roots: cmd.extend(['--input-roots', ','.join(roots)])",
        "    print('Launching:', ' '.join(cmd))",
        "    subprocess.check_call(cmd)",
        "    if MIRROR_WORKING and Path('/kaggle/working').is_dir():",
        "        top_dir = Path('/kaggle/working')",
        "        mp = sorted(p for p in (work / 'archives').iterdir() if p.is_file()) if (work / 'archives').is_dir() else []",
        "        for pat in EXPORT_GLOBS: mp.extend(p for p in sorted(work.glob(pat)) if p.is_file())",
        "        for src in mp:",
        "            dst = top_dir / src.name",
        "            if dst.resolve() != src.resolve():",
        "                if dst.exists(): dst.unlink()",
        "                try: os.link(src, dst)",
        "                except OSError: shutil.copy2(src, dst)",
        "        print('Mirrored:', sorted(x.name for x in mp))",
    ]]
    return nb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Avendi V8-Y4 plan and pinned notebook generator")
    ap.add_argument("--write-notebook", type=Path, default=None)
    ap.add_argument("--check-notebook", type=Path, default=None)
    ap.add_argument("--source-commit", default=Y4_BASE_PIN)
    args = ap.parse_args(argv)
    if args.write_notebook:
        args.write_notebook.parent.mkdir(parents=True, exist_ok=True)
        args.write_notebook.write_text(json.dumps(build_y4_notebook(args.source_commit), indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"written": str(args.write_notebook), "source_commit": args.source_commit}))
        return 0
    if args.check_notebook:
        exp = json.dumps(build_y4_notebook(args.source_commit), indent=2) + "\n"
        ok = args.check_notebook.is_file() and args.check_notebook.read_text(encoding="utf-8") == exp
        print(json.dumps({"match": ok, "path": str(args.check_notebook), "source_commit": args.source_commit}))
        return 0 if ok else 1
    plan = build_plan(args.source_commit)
    print(json.dumps(validate_plan_dag(plan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
