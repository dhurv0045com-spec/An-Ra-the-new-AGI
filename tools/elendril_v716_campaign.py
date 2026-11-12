"""Owner-run, custody-gated V7.16 falsification campaign.

Phase A (custody gate) runs before any data generation or training update: a
missing or mismatched parent archive yields NO_GO with zero optimizer steps.
The execution binding is written once, after data and profile canaries but
before the first registered update. Registered phases: mechanism (H1..H5 at
S3), continual (registered v7.15 control arms plus native ark017-sparse) and
promotion (a single winning S3 arm rerun at S8, admitted only against the
preregistered thresholds). Imports never launch training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
from queue import Queue, Empty
from threading import Lock
import re
import shutil
import subprocess
import time
import traceback
from uuid import uuid4

from anra_v5.elendril import custody_gate, v716_protocol
from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.v716_protocol import (PROTOCOL_SHA, SEEDS,
                                            jobs as v716_jobs)
from anra_v5.elendril.v71_protocol import (PROTOCOL_SHA as V71_PROTOCOL_SHA,
                                           jobs as v71_jobs)
from anra_v5.elendril.v6_evaluation import ControlPolicy
from anra_v5.elendril.v71_data import suite_manifest
from anra_v5.elendril.v71_evaluation import acquisition_gate, evaluate_suite, retention, verify_suite
from tools.elendril_cloud import check_source, check_torch
from tools.elendril_v6_campaign import _gpu_records
from tools.elendril_v6_export import _write_derived, append_event, immutable_json
from tools.elendril_v716_export import export_results, finalize as finalize_delivery
from tools.elendril_v716_report import write_summary
from tools.elendril_v71_campaign import (coverage, invoke, old_skill, prepare_data,
                                         require_pass, rescore)

REPO = Path(__file__).resolve().parents[1]
CONTINUOUS_STAGE_CONTROLS = ("b-only", "replay", "fresh-b")


def verify_preregistration():
    """The committed preregistration must round-trip the protocol module exactly."""
    path = REPO / "research" / "v7.16" / "PREREGISTRATION.json"
    if not path.is_file():
        raise ValueError("committed research/v7.16/PREREGISTRATION.json is missing")
    raw = path.read_bytes()
    committed = json.loads(raw)
    if committed != v716_protocol.preregistration():
        raise ValueError("committed preregistration drifted from the protocol module")
    if canonical(committed) + b"\n" != raw:
        raise ValueError("committed preregistration is not canonical bytes plus a newline")
    binding = json.loads((REPO / "research" / "v7.16" / "EXECUTION_BINDING.json").read_bytes())
    if binding.get("preregistration_sha256") != digest(v716_protocol.preregistration()) \
            or binding.get("protocol_sha256") != PROTOCOL_SHA:
        raise ValueError("committed execution binding pins drifted from the protocol")
    return {"preregistration_sha256": binding["preregistration_sha256"],
            "protocol_sha256": PROTOCOL_SHA}


def selected_jobs(phase, *, arm_subset=None, seed_subset=None, stage_subset=None,
                  promotion_arm=None):
    """Predeclared, balanced phase slices; protocol membership stays exact."""
    registered = v716_jobs(phase)
    if phase == "mechanism" and arm_subset is not None:
        if arm_subset not in v716_protocol.ARMS:
            raise ValueError("unknown registered mechanism arm")
        registered = [job for job in registered if job["arm"] == arm_subset]
    if phase == "continual" and stage_subset is not None:
        if stage_subset not in v716_protocol.CONTINUAL_ARMS:
            raise ValueError("unknown registered continual stage")
        registered = [job for job in registered if job["stage"] == stage_subset]
    if phase == "promotion":
        if promotion_arm not in v716_protocol.ARMS:
            raise ValueError("promotion requires one registered winning arm")
        registered = [job for job in registered if job["arm"] == promotion_arm]
    if seed_subset is not None:
        if seed_subset not in SEEDS:
            raise ValueError("unknown registered seed")
        registered = [job for job in registered if job["seed"] == seed_subset]
    if not registered:
        raise ValueError("empty registered phase slice")
    return registered


def promotion_gate(arm, mechanism_output, evidence_path):
    """Admit one S8 promotion arm only against the preregistered thresholds."""
    rules = v716_protocol.DECISION_RULES["promotion_requires"]
    path = Path(mechanism_output) / "analysis" / "summary.json"
    if not path.is_file():
        raise ValueError("promotion requires the mechanism-phase summary")
    summary = json.loads(path.read_bytes())
    # A grammar-constrained evaluator is not a substitute for learned
    # free-decoding. Old or partial summaries have no promotion authority.
    if summary.get("status") != "AVAILABLE" or summary.get("primary_decoder") != "raw-neural":
        raise ValueError("promotion requires a complete raw-neural primary summary")
    entry = (summary.get("arm_aggregates") or {}).get(arm)
    if not entry or "jobs" not in entry:
        raise ValueError(f"mechanism summary carries no final evaluation for {arm}")
    seeds = {}
    for job_id, cohorts in entry["jobs"].items():
        match = re.search(r"-s(\d+)$", job_id)
        if not match:
            raise ValueError(f"cannot parse the registered seed from {job_id}")
        seed = int(match.group(1))
        base = cohorts.get("base")
        if base:
            if seed in seeds:
                raise ValueError("promotion summary duplicates a registered seed")
            seeds[seed] = base
    if sorted(seeds) != sorted(SEEDS):
        raise ValueError("promotion gate requires all three registered seeds")
    if any(type(row.get("correct")) is not int
           or type(row.get("episodes")) is not int
           or row["episodes"] != 112
           or not 0 <= row["correct"] <= row["episodes"]
           for row in seeds.values()):
        raise ValueError("promotion requires exactly 112 valid raw observations per seed")
    aggregate = sum(row["correct"] for row in seeds.values())
    observations = sum(row["episodes"] for row in seeds.values())
    if observations != 336:
        raise ValueError("confirmation observations must be exactly 336")
    checks = {"aggregate_confirmation": f"{aggregate}/{observations}",
              "aggregate_pass": aggregate >= 279,
              "per_seed": {str(seed): f"{row['correct']}/{row['episodes']}"
                           for seed, row in sorted(seeds.items())},
              "per_seed_pass": all(row["correct"] >= 79 for row in seeds.values())}
    # Registered rule "raw <= 861 across 6,720 records; grammar == 0": the
    # invalid-action ceiling scales with the evaluated record count so the
    # V7.15 blocked-CE invalid rate (861/6720) is reproduced exactly at the
    # registered scope and applied at the same rate to the arm's own final
    # family. Grammar-constrained decoding must remain exactly zero-invalid.
    final_free, grammar_rows = {}, []
    job_ids = set(entry["jobs"])
    for row in summary.get("rows") or []:
        if row.get("arm") != arm or row.get("job_id") not in job_ids:
            continue
        if row.get("mode") == "grammar":
            grammar_rows.append(row)
        elif row.get("mode") == "free":
            job_id, epoch = row.get("job_id"), row.get("epoch")
            if type(epoch) is not int:
                raise ValueError(f"raw row for {job_id} lacks an integer epoch")
            if job_id not in final_free or epoch > final_free[job_id]:
                final_free[job_id] = epoch
    if not final_free:
        raise ValueError(f"promotion summary carries no raw evaluation rows for {arm}")
    records, raw_invalid = 0, 0
    for row in summary.get("rows") or []:
        if row.get("arm") != arm or row.get("job_id") not in job_ids:
            continue
        if row.get("mode") != "free" or row.get("epoch") != final_free[row["job_id"]]:
            continue
        episodes, invalid = row.get("episodes"), row.get("invalid")
        if type(episodes) is not int or type(invalid) is not int \
                or not 0 <= invalid <= episodes:
            raise ValueError(f"invalid-action accounting unusable for "
                             f"{row.get('job_id')}/{row.get('cohort')}")
        records += episodes
        raw_invalid += invalid
    grammar_invalid = 0
    for row in grammar_rows:
        invalid = row.get("invalid")
        if type(invalid) is not int or invalid < 0:
            raise ValueError(f"grammar invalid accounting unusable for "
                             f"{row.get('job_id')}/{row.get('cohort')}")
        grammar_invalid += invalid
    ceiling = records * 861 // 6720
    checks["invalid_actions"] = {"raw": raw_invalid, "records": records,
                                 "ceiling": ceiling, "grammar": grammar_invalid,
                                 "rule": "raw <= floor(861*records/6720); grammar == 0",
                                 "pass": raw_invalid <= ceiling and grammar_invalid == 0}
    stale, consistency = None, None
    if evidence_path is not None:
        evidence = json.loads(Path(evidence_path).read_bytes())
        if evidence.get("schema") != "elendril-v716-promotion-evidence-v1":
            raise ValueError("promotion evidence schema mismatch")
        stale = evidence.get("stale_memory_corrected")
        consistency = evidence.get("row_reversal_consistency")
        if type(stale) is not int or not 0 <= stale <= 48:
            raise ValueError("promotion evidence stale-memory score must be X/48")
        if type(consistency) not in (int, float) or not math.isfinite(consistency) or not 0 <= consistency <= 1:
            raise ValueError("promotion evidence row-reversal consistency must be in [0,1]")
        checks["stale_memory"] = f"{stale}/48"
        checks["stale_memory_pass"] = stale >= 28
        checks["row_reversal_consistency"] = consistency
        checks["row_reversal_pass"] = consistency >= 0.5
    checks["admitted"] = (checks["aggregate_pass"] and checks["per_seed_pass"]
                          and checks["invalid_actions"]["pass"]
                          and stale is not None and consistency is not None
                          and checks["stale_memory_pass"] and checks["row_reversal_pass"])
    if not checks["admitted"]:
        if not checks["invalid_actions"]["pass"]:
            reason = ("PROMOTION_BLOCKED: invalid-action rule failed "
                      f"(raw {raw_invalid}/{records} vs ceiling {ceiling}; "
                      f"grammar {grammar_invalid})")
        elif stale is not None:
            reason = "PROMOTION_BLOCKED: preregistered thresholds not met"
        else:
            reason = ("PROMOTION_BLOCKED: stale-memory and row-reversal diagnostic "
                      "evidence not supplied (--promotion-evidence)")
        raise ValueError(reason + "; no S8 run may start")
    return {"schema": "elendril-v716-promotion-gate-v1", "arm": arm,
            "thresholds": rules, "checks": checks,
            "mechanism_summary_sha256": sha256(path.read_bytes()).hexdigest()}


def verify_parent_output(parent_output, source):
    """A previously extracted parent must carry the verified V7.1 core identity."""
    parent_output = Path(parent_output).resolve()
    identity = json.loads((parent_output / "analysis" / "identity.json").read_bytes())
    admitted = ({"source": source, "protocol_sha256": V71_PROTOCOL_SHA, "phase": "core"},
                {"source": v716_protocol.V71_CORE_SOURCE_SHA,
                 "protocol_sha256": V71_PROTOCOL_SHA, "phase": "core"})
    if identity not in admitted:
        raise ValueError("parent core identity mismatch")
    return {"status": "PASS", "via": "extracted_parent_output", "identity": identity,
            "path": str(parent_output)}


def bind_execution(output, source, *, gate, parent_evidence, data, suite, profiles):
    """Write-once execution binding; every null stays null until actually measured."""
    science_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                             text=True).strip()
    if science_commit != source:
        raise ValueError("execution commit is not the pinned source checkout")
    binding = v716_protocol.binding_template()
    binding.update({
        "status": "BOUND",
        "science_commit": science_commit,
        "execution_commit": science_commit,
        "public_surface_sha256": digest(suite_manifest(suite)),
        "data_split_sha256": digest([data["train"].identity, data["validation"].identity,
                                     data["confirmation"].identity]),
        "matched_seed_initialization": {
            str(seed): next((row["init_weights_sha256"] for row in profiles
                             if row.get("init_weights_sha256")
                             and row["job"]["seed"] == seed), None) for seed in SEEDS},
        "custody": {
            "parent_full_sha256": v716_protocol.PARENT_FULL_SHA256,
            "parent_full_verified": parent_evidence["status"] == "PASS",
            "parent_evidence": parent_evidence,
            "s8_analysis_sha256": binding["custody"]["s8_analysis_sha256"],
            "s8_analysis_verified": gate["s8_analysis_archive"]["match"] is True},
    })
    custody_gate.write_once(Path(output) / "analysis" / "EXECUTION_BINDING.json", binding)
    return binding


def _profile_source(phase, registered):
    """Jobs whose shapes the phase's profiles must measure."""
    return v716_jobs("mechanism") if phase == "continual" else registered


def _profile_conditions(phase, registered):
    """Production-shaped profile pairs for the phase's capacity forecast.

    Continual arms are fresh-only and are never profiled; the whole phase is
    forecast from ONE measured S3 mechanism shape plus the protocol's 1.25
    margin, not from fourteen profile-lane runs.
    """
    source = _profile_source(phase, registered)
    conditions = sorted({(job["profile"], job["mechanism"]) for job in source
                         if job["mechanism"] != "continual"})
    if phase == "continual":
        conditions = [("S3", v716_protocol.ARM_MECHANISMS[v716_protocol.ARMS[0]])]
    return conditions


def forecast(registered, profiles, data, suite):
    """Two-lane forecast over the whole registered slice with measured profiles."""
    work = []
    for job in registered:
        if job["mechanism"] == "continual":
            s3 = [row for row in profiles if row.get("profile") == "S3"]
            if not s3:
                raise ValueError("continual forecast requires a measured S3 mechanism profile")
            step = 1.25 * max(row["seconds_per_update"] for row in s3)
            dataset = data["replay" if job.get("stage") == "replay" else "new-b"]
        else:
            measurements = [row for row in profiles if row.get("profile") == job["profile"]
                            and row.get("mechanism") == job["mechanism"]]
            if not measurements:
                raise ValueError("missing production profile")
            step = max(row["seconds_per_update"] for row in measurements)
            dataset = data["train"]
        episode = max(row["evaluation_seconds_per_episode"] for row in profiles)
        updates = math.ceil(len(dataset.samples) / 16) * job["epochs"]
        intermediate = sum(len(suite[name]) for name in ("base", "hop-3"))
        episodes = job["epochs"] // 2 * intermediate + intermediate \
            + 2 * sum(len(cases) for cases in suite.values())
        work.append(2 * (updates * step + episodes * episode))
    lanes = [0., 0.]
    for seconds in work:
        lanes[lanes.index(min(lanes))] += seconds
    return {"seconds": max(lanes) + 600, "job_seconds": work,
            "policy": "whole registered slice, 2x measured headroom plus a 25% "
                      "continual overhead margin, ten-minute export reserve; "
                      "no score-based pruning"}


def _continual_parent(parent_output, seed):
    parentjob = f"S3-blocked-ce-s{seed}"
    parent = json.loads((Path(parent_output) / "analysis" / "jobs" / parentjob
                         / "epoch-4.json").read_bytes())
    return parentjob, parent


def _job_request(job, base, output, parent_output=None):
    request = {**base, "job": job,
               "output": str(Path(output) / "analysis" / "jobs" / job["job_id"]),
               "checkpoint_root": str(Path(output) / "checkpoints" / "jobs" / job["job_id"])}
    if job["mechanism"] == "continual":
        request["data_file"] = str(Path(output) / "data" / "new-b.json")
        if job["stage"] == "ark017-sparse":
            request["replay_file"] = str(Path(output) / "data" / "replay.json")
        if job["stage"] != "fresh-b":
            parentjob, parent = _continual_parent(parent_output, job["seed"])
            request["parent"] = {"root": str(Path(parent_output) / "checkpoints"
                                             / "jobs" / parentjob),
                                 "lineage": parentjob, "checkpoint": parent["checkpoint"]}
    else:
        request["data_file"] = str(Path(output) / "data" / "train.json")
    return request


def run(*, output, source, phase="mechanism", owner_enabled=False, minutes=330,
        parent_archive=None, s8_analysis_zip=None, parent_output=None,
        arm_subset=None, seed_subset=None, stage_subset=None, promotion_arm=None,
        promotion_evidence=None, mechanism_output=None):
    if owner_enabled is not True:
        raise ValueError("explicit owner launch required")
    if phase not in v716_protocol.PHASES or type(minutes) not in (int, float) \
            or not math.isfinite(minutes) or not 1 <= minutes <= 330:
        raise ValueError("invalid phase or owner budget")
    requested = Path(output)
    if requested.is_symlink():
        raise ValueError("output symlink forbidden")
    output = requested.resolve()
    if output.is_relative_to(REPO) or output == Path(output.anchor):
        raise ValueError("dedicated output outside source required")
    prereg = verify_preregistration()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + minutes * 60
    run_id = uuid4().hex
    campaign = {"schema": "elendril-v716-campaign-v1", "run_id": run_id, "phase": phase,
                "source": source, "protocol_sha256": PROTOCOL_SHA,
                "preregistration": prereg, "status": "PREFLIGHT", "results": [],
                "promotion": "NONE", "cause_status": "CAUSE_UNRESOLVED"}
    registered = selected_jobs(phase, arm_subset=arm_subset, seed_subset=seed_subset,
                               stage_subset=stage_subset, promotion_arm=promotion_arm)
    campaign["registered_job_ids"] = [job["job_id"] for job in registered]
    campaign["predeclared_slice"] = {"arm": arm_subset, "seed": seed_subset,
                                     "stage": stage_subset, "promotion_arm": promotion_arm,
                                     "subset_identity": digest(
                                         ["v716-registered-slice-v1", phase,
                                          [job["job_id"] for job in registered]])}
    required = []
    try:
        if phase == "promotion":
            campaign["promotion_gate"] = promotion_gate(promotion_arm, mechanism_output,
                                                        promotion_evidence)
        if phase == "continual" and parent_output is None:
            raise ValueError("continual requires a verified extracted V7.1 core "
                             "output via --parent-output")
        check_source(REPO, source)
        custody_gate.write_once(output / "analysis" / "identity.json",
                                {"source": source, "protocol_sha256": PROTOCOL_SHA,
                                 "phase": phase})
        _write_derived(output / "analysis" / "protocol.json",
                       canonical(v716_protocol.protocol()))
        # ---- Phase A custody gate: no data generation or update before it passes.
        gate = custody_gate.preexecution_gate(
            parent_archive=parent_archive, s8_analysis_zip=s8_analysis_zip,
            require_parent=parent_output is None,
            require_s8_analysis=phase == "promotion")
        campaign["custody_gate"] = gate
        if parent_output is not None:
            parent_evidence = verify_parent_output(parent_output, source)
        elif gate["parent_archive"]["match"] is True:
            parent_evidence = {"status": "PASS", "via": "parent_archive",
                               "sha256": gate["parent_archive"]["sha256"],
                               "byte_size": gate["parent_archive"]["byte_size"]}
        else:
            parent_evidence = {"status": "FAIL", "via": "parent_archive",
                               "reason": gate["CUSTODY_STATUS"]}
        campaign["parent_evidence"] = parent_evidence
        failures = list(gate["failures"])
        if parent_evidence["status"] != "PASS":
            failures.append(parent_evidence.get("reason", "parent custody unresolved"))
        if failures:
            raise ValueError("NO_GO: " + ", ".join(failures))
        free = shutil.disk_usage(output).free
        reserve_gib = {"mechanism": 12, "promotion": 15, "continual": 15}[phase]
        campaign["free_disk_bytes"] = free
        if free < reserve_gib * (1 << 30):
            raise ValueError(f"NO-GO: {reserve_gib} GiB free required; use a fresh "
                             "session and preserve previous outputs")
        devices = _gpu_records()
        immutable_json(output / "analysis" / ("environment-" + run_id + ".json"),
                       check_torch(REPO, devices))
        print("PREFLIGHT PASS: custody verified; generating and auditing data under "
              "the writable output; no input dataset required.", flush=True)
        data, suite = prepare_data(output)
        base = {"source": source, "protocol_sha": PROTOCOL_SHA, "phase": phase,
                "suite_file": str(output / "data" / "suite.json")}
        for method in ("fixed", "always-query", "no-query", "random", "recency",
                       "query-blind"):
            path = output / "analysis" / "controls" / (method + ".json")
            if not path.exists():
                record = evaluate_suite(suite, ControlPolicy(method))
                verify_suite(suite, record)
                identity = next(row["raw"]["model_identity"]
                                for row in record["records"].values() if row)
                immutable_json(path, {"evaluation": record, "checkpoint": identity,
                                      "attribution": "Engineered comparator; no learned "
                                                     "capability evidence."})
        from anra_v5.elendril.v7_protocol import PROTOCOL_SHA as V7_SHA

        def positive(index):
            return invoke("canary", {"source": source, "protocol_sha": V7_SHA,
                                     "output": str(output / "analysis" / "positive"
                                                   / f"{run_id}-lane{index}"),
                                     "checkpoint_root": str(output / "checkpoints"
                                                            / "positive" / f"{run_id}-lane{index}"),
                                     "job": {"job_id": "canary" + str(index),
                                             "method": "uniform", "seed": 7},
                                     "data_file": str(output / "data" / "canary.json"),
                                     "eval_file": str(output / "data" / "monitor.json"),
                                     "production_file": str(output / "data" / "train.json"),
                                     "timecap_seconds": max(1, min(1200, deadline
                                                                   - time.monotonic() - 180))},
                          devices[index], module="tools.elendril_v7_worker")

        with ThreadPoolExecutor(max_workers=2) as pool:
            canaries = list(pool.map(positive, range(2)))
        campaign["positive_controls"] = canaries
        require_pass(canaries, "positive learning/recovery canary")
        profiles = []
        conditions = _profile_conditions(phase, registered)
        for profile_spec, mechanism in conditions:
            job = next(job for job in _profile_source(phase, registered)
                       if job["profile"] == profile_spec and job["mechanism"] == mechanism)
            request_phase = "promotion" if profile_spec == "S8" else "mechanism"

            def qualify(index):
                row = invoke("profile", {**base, "job": job,
                                         "data_file": str(output / "data" / "train.json"),
                                         "output": str(output / "analysis" / "profiles"
                                                       / f"{run_id}-{profile_spec}-{mechanism}-lane{index}"),
                                         "checkpoint_root": str(output / "checkpoints"
                                                                / "profiles"
                                                                / f"{run_id}-{profile_spec}-{mechanism}-lane{index}"),
                                         "phase": request_phase,
                                         "timecap_seconds": max(1, min(1800, deadline
                                                                       - time.monotonic() - 180))},
                              devices[index], module="tools.elendril_v716_worker")
                return {**row, "job": job, "profile": profile_spec,
                        "mechanism": mechanism}

            with ThreadPoolExecutor(max_workers=2) as pool:
                profiles.extend(pool.map(qualify, range(2)))
            campaign["profiles"] = profiles
            require_pass(profiles, "production-shaped profile")
        campaign["profiles"] = profiles
        binding = bind_execution(output, source, gate=gate, parent_evidence=parent_evidence,
                                 data=data, suite=suite, profiles=profiles)
        campaign["execution_binding"] = {"path": str(output / "analysis"
                                                     / "EXECUTION_BINDING.json"),
                                         "status": binding["status"]}
        estimate = forecast(registered, profiles, data, suite)
        campaign["capacity_estimate"] = estimate
        if estimate["seconds"] > deadline - time.monotonic():
            raise ValueError("whole registered slice exceeds measured remaining capacity; "
                             "DEFERRED, no silent matrix reduction")
        # Worker lanes may finish together; V7.16 post-transport renaming is
        # outside the lower V7 export writer lock. Keep all snapshot and
        # readback phases serialized while preserving parallel GPU workers.
        export_lock = Lock()
        pending = Queue()
        for job in registered:
            pending.put(job)
        campaign["status"] = "RUNNING"

        def lane(index):
            rows = []
            while time.monotonic() < deadline - 600:
                try:
                    job = pending.get_nowait()
                except Empty:
                    break
                request = _job_request(job, base, output, parent_output=parent_output)
                request["timecap_seconds"] = max(1, min(21600, deadline - time.monotonic() - 600))
                module = "tools.elendril_v716_worker"
                if job["mechanism"] == "continual":
                    if job["stage"] in CONTINUOUS_STAGE_CONTROLS:
                        v71job = next(item for item in v71_jobs("continual")
                                      if item["job_id"] == job["job_id"])
                        request["job"] = v71job
                        request["data_file"] = str(output / "data" /
                                                   ("replay.json" if job["stage"] == "replay"
                                                    else "new-b.json"))
                        request["phase"] = "continual"
                        request["protocol_sha"] = V71_PROTOCOL_SHA
                        module = "tools.elendril_v71_worker"
                    if job["stage"] != "fresh-b":
                        # ark017-sparse and every replay control compare retention
                        # against the same recorded parent Skill-A; a parent that
                        # never acquired A disqualifies the comparison, not the run.
                        parentjob, parent = _continual_parent(parent_output, job["seed"])
                        prior = old_skill(parent["evaluation"]["metrics"]["base"])
                        gate_row = acquisition_gate(prior)
                        if gate_row["status"] != "ACQUIRED":
                            row = {"status": "NOT_QUALIFIED", "job": job,
                                   "acquisition_gate": gate_row,
                                   "reason": "A not acquired; retention comparison "
                                             "prohibited"}
                            immutable_json(Path(request["output"])
                                           / "acquisition-not-qualified.json", row)
                            rows.append(row)
                            continue
                row = invoke("train", request, devices[index], module=module)
                row["job"] = job
                if job["mechanism"] == "continual" and row["status"] == "COMPLETE":
                    final = json.loads((Path(request["output"]) / "epoch-2.json")
                                       .read_bytes())["evaluation"]["metrics"]
                    # The qualified v7.1 worker writes the pre-training B state to
                    # initial.json for fresh-b (no parent in its request) and to
                    # after-transfer-before-B.json for every parent-carrying arm.
                    initial_file = ("initial.json" if job["stage"] == "fresh-b"
                                    else "after-transfer-before-B.json")
                    initial = json.loads((Path(request["output"]) / initial_file)
                                         .read_bytes())["evaluation"]["metrics"]
                    row["B_before"] = acquisition_gate(initial["hop-3"]["all"])
                    row["B_after"] = acquisition_gate(final["hop-3"]["all"])
                    row["novel_B_acquisition"] = (
                        "NOT_QUALIFIED: B already acquired before training"
                        if row["B_before"]["status"] == "ACQUIRED" else
                        "OBSERVABLE before/after acquisition contrast only")
                    row["retention"] = retention(old_skill(initial["base"]),
                                                 {"new_skill": final["hop-3"]["all"],
                                                  "old_skill": old_skill(final["base"])})
                rows.append(row)
                append_event(output / "analysis",
                             {"event": "JOB_" + row["status"], "job": job})
                print(job["job_id"] + ": " + row["status"], flush=True)
                if row["status"] == "ERROR":
                    print(json.dumps(row, indent=2), flush=True)
                with export_lock:
                    export_results(output, source=source, protocol_sha=PROTOCOL_SHA,
                                   campaign=campaign, full=False, rescore=rescore)
            return rows

        with ThreadPoolExecutor(max_workers=2) as pool:
            campaign["results"] = [row for rows in pool.map(lane, range(2)) for row in rows]
        campaign["completed_jobs"] = sum(row["status"] == "COMPLETE"
                                         for row in campaign["results"])
        campaign["status"] = "EXECUTED" if campaign["completed_jobs"] == len(registered) \
            else "PARTIAL"
        if campaign["status"] == "EXECUTED":
            required = coverage(output, registered, suite)
        campaign["gpu_worker_reservation_seconds"] = sum(
            row.get("attempt_wall_seconds", 0)
            for row in canaries + profiles + campaign["results"])
    except (Exception, KeyboardInterrupt) as exc:
        campaign.update(status="PARTIAL" if isinstance(exc, KeyboardInterrupt) else "NO_GO",
                        error_type=type(exc).__name__, message=str(exc),
                        traceback=traceback.format_exc())
        print("Campaign stopped: " + str(exc), flush=True)
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        try:
            campaign["summary"] = write_summary(output, campaign=campaign)
            campaign["independent_replay"] = rescore(output, campaign)
        except Exception as exc:
            campaign["independent_replay"] = {"status": "FAILED", "message": str(exc)}
            campaign["status"] = "INVALID_RESULTS"
        delivery = finalize_delivery(output, source, PROTOCOL_SHA, campaign,
                                     [job["job_id"] for job in registered], required,
                                     campaign["status"] == "EXECUTED"
                                     and campaign["independent_replay"]["status"] == "PASS"
                                     and campaign.get("summary", {}).get("status")
                                     == "AVAILABLE")
        if campaign["status"] not in ("NO_GO", "INVALID_RESULTS"):
            campaign["status"] = delivery["status"]
        campaign["delivery"] = delivery
        campaign["exports"] = delivery["exports"]
        print(json.dumps({"status": campaign["status"],
                          "exports": campaign.get("exports"),
                          "message": campaign.get("message")}, indent=2), flush=True)
    return campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--phase", choices=v716_protocol.PHASES, default="mechanism")
    parser.add_argument("--owner-enabled", action="store_true")
    parser.add_argument("--minutes", type=float, default=330)
    parser.add_argument("--parent-archive", type=Path)
    parser.add_argument("--s8-analysis-zip", type=Path)
    parser.add_argument("--parent-output", type=Path)
    parser.add_argument("--arm-subset")
    parser.add_argument("--seed-subset", type=int, choices=SEEDS)
    parser.add_argument("--stage-subset")
    parser.add_argument("--promotion-arm")
    parser.add_argument("--promotion-evidence", type=Path)
    parser.add_argument("--mechanism-output", type=Path)
    arguments = parser.parse_args(argv)
    result = run(output=arguments.output, source=arguments.source, phase=arguments.phase,
                 owner_enabled=arguments.owner_enabled, minutes=arguments.minutes,
                 parent_archive=arguments.parent_archive,
                 s8_analysis_zip=arguments.s8_analysis_zip,
                 parent_output=arguments.parent_output,
                 arm_subset=arguments.arm_subset, seed_subset=arguments.seed_subset,
                 stage_subset=arguments.stage_subset,
                 promotion_arm=arguments.promotion_arm,
                 promotion_evidence=arguments.promotion_evidence,
                 mechanism_output=arguments.mechanism_output)
    return 0 if result["status"] in ("DELIVERY_VERIFIED",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
