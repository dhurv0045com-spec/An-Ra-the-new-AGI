"""Expand the Avendi Y3 preregistration, identity pinouts, and Kaggle notebook."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

SEEDS = (7, 19, 41)
ARMS = {
    "P2": ("ce", "margin"),
    "P3": ("refresh", "curriculum", "address", "address_sham", "rename", "rename_sham", "low_lr"),
    "P4": ("categorical", "continuous", "dense"),
    "P5": ("direct_plus_neutral", "direct_plus_info", "replace_direct"),
    "P6": ("diagnostic_policy",),
    "P7": ("s20_ce", "s20_low_lr"),
}
STUDIES = {"P0": range(1, 7), "P1": range(7, 15), "P2": range(15, 21), "P3": range(21, 28), "P4": range(28, 36), "P5": range(36, 41), "P6": range(41, 47), "P7": range(47, 51)}
BASE_REVISION = "ba1c0e978f890021a10e2365b61d5403bc486ce5"
REPO_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
STREAM_SEEDS = {"corpus": 3301, "dev": 3302, "confirm": 3303, "transfer": 3304, "reference": 3300, "bootstrap": 3317}
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
}
DONOR_PINS = {"cymek": "28bf57a0d299a2c13a99fe0046616c00a1b8530c", "arkenstone": "4ae9e3b6dc806d6ada7b7b2f05d32271d072d543", "elendril": "cb9a91598c1d831fecb8b7d483f4ceeba1d063a8"}
Y1_DISCOVERY_LINKS = {
    "01": ("U03", "U12"), "02": ("U37",), "03": ("U04",), "04": ("U02",), "05": ("U01", "U03", "U08"),
    "06": ("U11",), "07": ("U09", "U15", "U31"), "08": ("U10",), "09": ("U15", "U19"), "10": ("U07", "U08"),
    "11": ("U26",), "12": ("U36", "U39"), "13": ("U21",), "14": ("U14", "U37", "U49"), "15": ("U25", "U47"),
    "16": ("U16",), "17": ("U18",), "18": ("U15", "U27"), "19": ("U35", "U50"), "20": ("U06", "U07"),
    "21": ("U05", "U06"), "22": ("U02", "U05", "U06", "U50"), "23": ("U40",), "24": ("U08", "U14", "U25", "U37"), "25": tuple(f"U{i:02}" for i in range(41, 47)),
}
BASELINE_DEFECT_REGISTER = (
    {"classification": "Confirmed and unresolved for historical Y2", "evidence_now": "Omitted generated tokens, P0-inherited replay, suppressed source checks, rank-inferred EOS, confirm in train, seed intersection", "required_disposition": "Record raw tokens/continuations, fail source drift, independent replay, confirmation lock, full seed set", "repro": "y2-audit-legacy-gaps", "fix_sha": "pending_y3", "regression_cmd": "pytest tests/test_avendi_y3.py -q"},
    {"classification": "Confirmed and repaired within tested scope", "evidence_now": "Platform release repaired stateful/multi-step replay, resume rejection, dual-clock, peer custody, Y2 tokenized P0/Y1 restore", "required_disposition": "Pass platform contracts on native Y3 data incl. stateful P6 policy; report repaired_in_platform vs qualified_in_y3 separately", "repro": "tests/test_planthir_v2_audit_upgrades.py", "fix_sha": BASE_REVISION, "regression_cmd": "pytest tests/test_planthir_v2_audit_upgrades.py -q"},
    {"classification": "Suspected causes", "evidence_now": "Null distractors, literal wording, output policy, positional bias, dose, initial corpus, optimization, interference", "required_disposition": "Discriminate via U08-U12, U16, U21, U25, U26, U40 with causal counterfactuals", "repro": "historical_shortcuts_and_abstention", "fix_sha": "open_hypothesis", "regression_cmd": "python -m experiments.avendi_y3.module --mode CANARY"},
    {"classification": "Scientific unknown", "evidence_now": "Reusable state, causal donor tracking, relevance-directed querying, unseen correction, donor/scale benefit", "required_disposition": "Test U28-U49 and assign learned/engineered/failed/inconclusive outcomes", "repro": "y3_full_campaign", "fix_sha": "open_hypothesis", "regression_cmd": "python -m experiments.avendi_y3.module --mode FULL"},
)
PLATFORM_USES = (
    ("01", "doctor", "P0"), ("02", "calibrate", "P0"), ("03", "plugin_registration", "P0"), ("04", "init", "P0"), ("05", "validate", "P0"), ("06", "cartridge_construction", "P0"),
    ("07", "GenericDatasetBundle", "P0"), ("08", "UniversalObservation", "P0"), ("09", "environment_kernel", "P0"), ("10", "stateful_policy_reset", "P6"), ("11", "batched_rollout", "P1"),
    ("12", "sampled_strata_audit", "P3"), ("13", "locate_checkpoint_root", "P1"), ("14", "load_legacy_checkpoint", "P1"), ("15", "LegacyByteObservablePolicy", "P1"), ("16", "evaluate_frozen_checkpoints", "P1"),
    ("17", "byte_framing", "P2"), ("18", "byte_trainer", "P2"), ("19", "ce_and_pair_margin", "P2"), ("20", "gradient_sign_control", "P2"), ("21", "GenericExperimentTrainer", "P4"),
    ("22", "backbone_overhead_verification", "P7"), ("23", "surgical_ablation", "P4"), ("24", "crossover_fixture_verification", "P0"), ("25", "crossover_scoring", "P2"), ("26", "family_bootstrap", "P2"),
    ("27", "seed_pair_builder", "P3"), ("28", "paired_seed_analysis", "P3"), ("29", "paired_comparison_preflight", "P0"), ("30", "paired_episode_comparison", "P3"), ("31", "stratify_intervention_worlds", "P1"),
    ("32", "diagnostic_coverage", "P1"), ("33", "stage_dag", "P7"), ("34", "dual_clock_admission", "P7"), ("35", "verified_resume_reconciliation", "P7"), ("36", "independent_worker_waves", "P7"),
    ("37", "session_archive_export", "P7"), ("38", "notebook_generator", "P7"), ("39", "verify_bundle", "P7"), ("40", "replay_bundle", "P7"), ("41", "compact_archive_rescoring", "P7"),
    ("42", "verification_matrix", "P7"), ("43", "analyst_first_page", "P7"), ("44", "markdown_rendering", "P7"),
)
QUALIFICATION_GATES = {
    "P0_P1": {"logit_max_abs_diff": 1e-5, "oracle_required": 1.0},
    "P2": {"fam4_confirm_min": 0.80, "fam8_confirm_min": 0.80, "literal_confirm_min": 0.95, "margin_dev_gain_min": 0.10, "max_literal_drop": 0.05},
    "P3": {"dev_gain_min": 0.10, "fam8_confirm_min": 0.80, "max_old_drop": 0.05},
    "P4": {"hop1_min": 0.80, "hop34_min": 0.70, "interchange_min": 0.75, "sham_gain_min": 0.20, "max_identity_drop": 0.05, "rewire_both_min": 0.70},
    "P5": {"macro_opt_min": 0.80, "max_false_abstain": 0.05, "obtainable_traj_min": 0.75, "max_old_drop": 0.05, "dp_regret_tol": 1e-8},
    "P6": {"unseen_gain_min": 0.10, "post_acc_min": 0.75, "max_irrel_shift": 0.05, "max_retention_drop": 0.05, "weight_rel_norm_cap": 0.001},
    "P7": {"gain_min": 0.10, "max_old_drop": 0.05},
}
STUDY_STAGES = {f"U{n:02}": st for st, nums in STUDIES.items() for n in nums}


def _canon_sha(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_plan(source_commit: str = BASE_REVISION, *, mode: str = "FULL") -> dict[str, Any]:
    """Expand the 54 main jobs, 24 short forks, child/P6 jobs, and 50 studies."""
    backbone_map, parent_map = {"P2": "S3", "P3": "S3", "P4": "P4-matched", "P5": "S3", "P6": "S3", "P7": "S20"}, {"P5": "P2-ce", "P7": "frozen-s3-and-donors"}
    jobs = [
        {"job_id": f"{st}-{arm}-s{seed}", "stage": st, "arm": arm, "init_seed": seed, "order_seed": seed, "updates": 1024, "backbone": backbone_map[st], "model_identity": f"y3-{st}-{arm}-s{seed}", "data_identity": f"y3-corpus-{STREAM_SEEDS['corpus']}", "intervention": f"{st}:{arm}", "parent_job_id": f"{parent_map[st]}-s{seed}" if st == "P5" else parent_map.get(st), "resource_profile": "gpu_t4_s20" if st == "P7" else "gpu_t4_s3", "spec_sha256": _canon_sha({"stage": st, "arm": arm, "seed": seed, "updates": 1024, "mode": mode}), "status": "PLANNED", "receipt": f"jobs/{st}-{arm}-s{seed}/receipt.json"}
        for st, arms in ARMS.items() for arm in arms for seed in SEEDS
    ]
    short_forks = (
        [{"job_id": f"U19-{arm}-s{s}", "stage": "P2", "arm": f"u19_{arm}", "init_seed": s, "order_seed": s, "updates": 128, "model_identity": f"y3-U19-{arm}-s{s}", "spec_sha256": _canon_sha({"stage": "P2", "arm": f"u19_{arm}", "seed": s, "updates": 128}), "status": "PLANNED"} for arm in ("ce", "margin") for s in SEEDS]
        + [{"job_id": f"P4-{a}_endonly-s{s}", "stage": "P4", "arm": f"{a}_endpoint_only", "init_seed": s, "order_seed": s, "updates": 256, "model_identity": f"y3-P4-{a}_endonly-s{s}", "spec_sha256": _canon_sha({"stage": "P4", "arm": f"{a}_endpoint_only", "seed": s, "updates": 256}), "status": "PLANNED"} for a in ARMS["P4"] for s in SEEDS]
        + [{"job_id": f"P7-{c}-s{s}", "stage": "P7", "arm": c, "init_seed": s, "order_seed": s, "updates": 256, "model_identity": f"y3-P7-{c}-s{s}", "spec_sha256": _canon_sha({"stage": "P7", "arm": c, "seed": s, "updates": 256}), "status": "PLANNED"} for c in ("ark_capped", "ark_uncapped", "ark_expaug") for s in SEEDS]
    )
    p6_adapt = [{"job_id": f"P6-adapt-{r}-s{s}", "stage": "P6", "arm": r, "init_seed": s, "order_seed": s, "updates": 256, "model_identity": f"y3-P6-diagnostic_policy-s{s}", "spec_sha256": _canon_sha({"stage": "P6", "arm": r, "seed": s, "updates": 256}), "status": "PLANNED"} for s in SEEDS for r in ("context", "memory", "weight_update")]
    child_jobs = [{"job_id": f"{sid}-eval-cpu" if st in ("P0", "P1") else f"{sid}-eval", "study_id": sid, "stage": st, "model_identity": "sentinel:model_free" if st == "P0" else f"stage:{st}", "status": "PLANNED"} for sid, st in STUDY_STAGES.items()]
    expanded_rows = [{"study_id": c["study_id"], "job_id": c["job_id"], "model_identity": c["model_identity"], "seed": STREAM_SEEDS["dev"], "split": "dev", "cohort": f"y3-{c['stage'].lower()}", "intervention": c["study_id"]} for c in child_jobs]
    studies = [{"study_id": f"U{n:02}", "home_stage": STUDY_STAGES[f"U{n:02}"], "status": "PLANNED", "receipt_pattern": f"studies/U{n:02}/<job>/<split>/receipt.json"} for n in range(1, 51)]
    return {
        "campaign": "Avendi V8-Y3", "technology_baseline": BASE_REVISION, "source_commit": source_commit, "mode": mode, "phase": "DESIGN",
        "training_jobs": jobs, "short_fork_jobs": short_forks, "p6_adaptation_jobs": p6_adapt, "child_jobs": child_jobs, "expanded_rows": expanded_rows, "studies": studies,
        "frozen_lineages": [f"Y1-t1_factorial-S3-{arm}-s{s}" for arm in ("F00", "F11") for s in SEEDS],
        "reused_scale_controls": [f"{st}-{arm}-s{s}" for st, arm in (("P2", "ce"), ("P3", "low_lr")) for s in SEEDS],
        "update_totals": {"main": 55296, "u19": 768, "p4_endpoint": 2304, "arkenstone": 2304, "short_forks": 5376},
        "rng_manifest": {"algorithm": "numpy.random.PCG64+sha256", "python": sys.version.split()[0], "streams": STREAM_SEEDS},
        "defect_register": list(BASELINE_DEFECT_REGISTER), "y1_discovery_links": Y1_DISCOVERY_LINKS,
        "platform_uses": [{"use_id": u, "name": n, "owning_stage": s, "status": "PLANNED"} for u, n, s in PLATFORM_USES],
        "donor_pins": DONOR_PINS, "expected_y1_checkpoints": Y1_EXPECTED_CKPT_SHAS, "archive_digests": ARCHIVE_SHAS, "qualification_gates": QUALIFICATION_GATES,
    }


def build_y3_notebook(source_commit: str) -> dict[str, Any]:
    """Generate the source-pinned Kaggle 2xT4 notebook via anra_v5.palantiri."""
    from anra_v5.palantiri import build_pinned_kaggle_notebook

    nb = build_pinned_kaggle_notebook(source_commit=source_commit, title="Avendi V8-Y3 · Full Campaign (Kaggle 2×T4)", launch_command="python -m experiments.avendi_y3.module --mode FULL --session-minutes 300", owner_authorized=False, discover_kaggle_inputs=("ELENDRIL_V8_Y1_FULL.zip", "ELENDRIL_V8_Y1_EVIDENCE.zip", "AVENDI_V8_Y2_EVIDENCE.zip"), mirror_outputs_to_working=True, preflight_canary_backend="avendi_y3", require_dual_gpu=True, emit_file_links=True, working_export_globs=("*.zip", "*.sha256", "FINAL_STATUS.json", "PENDING_QUEUE.json", "RESULTS.md", "METRICS.json"), backend_modules=("experiments.avendi_y3.module",))
    nb["cells"][1]["source"] = [ln + "\n" for ln in [
        f"SOURCE_SHA = {source_commit!r}", "OWNER_AUTHORIZED = False", "CAMPAIGN_MODE = 'FULL'", "SESSION_MINUTES = 300", "RESUME_INPUT = None", f"REPO_URL = {REPO_URL!r}",
        "DISCOVER_INPUTS = ['ELENDRIL_V8_Y1_FULL.zip', 'ELENDRIL_V8_Y1_EVIDENCE.zip', 'AVENDI_V8_Y2_EVIDENCE.zip']", "MIRROR_WORKING = True", "EXPORT_GLOBS = ['*.zip', '*.sha256', 'FINAL_STATUS.json', 'PENDING_QUEUE.json', 'RESULTS.md', 'METRICS.json']",
        "import os, shutil, subprocess, sys", "from pathlib import Path",
        "repo = Path('/kaggle/working/anra_avendi_y3_src') if Path('/kaggle/working').is_dir() else Path('.')",
        "if repo != Path('.') and not (repo / 'experiments' / 'avendi_y3' / 'module.py').is_file():",
        "    if repo.exists(): shutil.rmtree(repo)",
        "    subprocess.check_call(['git', 'clone', '--no-checkout', REPO_URL, str(repo)])",
        "    subprocess.check_call(['git', '-C', str(repo), 'fetch', 'origin', '--tags'])",
        "    subprocess.check_call(['git', '-C', str(repo), 'checkout', SOURCE_SHA])",
        "os.chdir(str(repo)); os.environ['PLANTHIR_BACKEND_MODULES'] = 'experiments.avendi_y3.module'",
        "if not OWNER_AUTHORIZED:",
        "    print('NOT LAUNCHED: set OWNER_AUTHORIZED = True before running GPU launch.')",
        "    subprocess.check_call([sys.executable, '-m', 'experiments.avendi_y3.module', '--mode', CAMPAIGN_MODE, '--session-minutes', str(SESSION_MINUTES), '--inventory-only'])",
        "else:",
        "    roots = [str(p) for n in DISCOVER_INPUTS if Path('/kaggle/input').is_dir() for p in sorted(Path('/kaggle/input').rglob(n))]",
        "    work = Path('/kaggle/working/avendi_y3_run') if Path('/kaggle/working').is_dir() else Path('runs/avendi_y3_run')",
        "    cmd = [sys.executable, '-m', 'experiments.avendi_y3.module', '--mode', CAMPAIGN_MODE, '--session-minutes', str(SESSION_MINUTES), '--work-root', str(work)]",
        "    if RESUME_INPUT: cmd.extend(['--resume-input', str(RESUME_INPUT)])",
        "    if roots: cmd.extend(['--input-roots', ','.join(roots)])",
        "    subprocess.check_call(cmd)",
        "    if MIRROR_WORKING and Path('/kaggle/working').is_dir():",
        "        for pat in EXPORT_GLOBS:",
        "            for p in sorted(work.rglob(pat)):",
        "                if p.is_file(): shutil.copy2(p, Path('/kaggle/working') / p.name)",
    ]]
    nb["cells"][2]["source"] = ["if not globals().get('OWNER_AUTHORIZED', False):\n", "    print('Preflight GPU skipped while OWNER_AUTHORIZED = False.')\n", "else:\n"] + ["    " + l for l in nb["cells"][2]["source"]]
    return nb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Avendi V8-Y3 plan and Kaggle notebook CLI")
    ap.add_argument("--write-notebook", type=Path, default=None); ap.add_argument("--check-notebook", type=Path, default=None); ap.add_argument("--source-commit", type=str, default=BASE_REVISION)
    args = ap.parse_args(argv)
    if args.write_notebook is not None:
        args.write_notebook.parent.mkdir(parents=True, exist_ok=True)
        args.write_notebook.write_text(json.dumps(build_y3_notebook(args.source_commit), indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"written": str(args.write_notebook), "source_commit": args.source_commit})); return 0
    if args.check_notebook is not None:
        ok = (args.check_notebook.read_text(encoding="utf-8") if args.check_notebook.is_file() else "") == (json.dumps(build_y3_notebook(args.source_commit), indent=2) + "\n")
        print(json.dumps({"match": ok, "path": str(args.check_notebook), "source_commit": args.source_commit})); return 0 if ok else 1
    print(json.dumps(build_plan(source_commit=args.source_commit), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
