"""One owner-operated, two-physical-T4 diagnostic campaign. Import is inert.

The source and evaluator are frozen before training. Validation alone chooses
arms; reserved confirmation is generated only after the selection receipt.
Every attempt is journaled and finalization always attempts a verified ZIP.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from tools.elendril_cloud import (GPU, GIB, inventory, check_torch, check_source, device_env,
                                command, safe_id, sha_id, fingerprint)
from tools.elendril_action_cloud import session_identity
from tools.elendril_v6_export import immutable_json, append_event, checkpoint_export, export_bundle

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "elendril-v6-campaign-v1"
EVAL_WORLDS = 112
TRAIN_BYTES = 2_200_000
STAGE_SECONDS = {"screen": 6300, "replicate": 6300, "scale": 4500, "diagnose": 600}


def _protocol(protocol_sha):
    from anra_v5.elendril.v6_registry import REGISTRY_SHA, registry
    if protocol_sha != REGISTRY_SHA:
        raise ValueError("protocol hash mismatch; notebook must pin the implemented amended registry")
    return registry()


def _inside(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("result path escaped campaign output")
    return path


def _write_once(path, value):
    if Path(path).exists():
        if json.loads(Path(path).read_bytes()) != json.loads(json.dumps(value)):
            raise ValueError("existing immutable artifact differs; use a new campaign directory")
        return
    immutable_json(Path(path), value)


def prepare_data(output, *, confirmation=False):
    """Pinned procedural data cache; /kaggle/input is never a write target."""
    from anra_v5.elendril.v6_data import (make_dataset, write_dataset, read_dataset,
        assert_disjoint, _dataset, audit_dataset, panel_cases)
    root = Path(output).resolve(); directory = root / "data"
    directory.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name in (("train", "validation", "confirmation") if confirmation else ("train", "validation")):
        path = directory / (name + ".json")
        if path.exists():
            data = read_dataset(path)
            if (data.split != name or data.manifest["seed"] != 606
                    or (name == "train" and data.manifest["target_bytes"] != TRAIN_BYTES)
                    or (name != "train" and data.manifest["episodes"] != EVAL_WORLDS)):
                raise ValueError("cached dataset recipe mismatch")
        else:
            data = make_dataset(name, target_bytes=TRAIN_BYTES, worlds=None if name == "train" else EVAL_WORLDS)
            write_dataset(data, path)
        datasets[name] = data
    assert_disjoint(*datasets.values())
    if not 2_000_000 <= datasets["train"].manifest["text_bytes"] <= 3_000_000:
        raise ValueError("training text outside registered 2–3MB range; no filler admission")
    # A tiny class-spanning formation control, distinct from held-out validation.
    selected, counts = [], {}
    for case in datasets["train"].cases:
        if counts.get(case.kind, 0) < 2:
            selected.append(case); counts[case.kind] = counts.get(case.kind, 0) + 1
        if len(counts) == 7 and min(counts.values()) == 2:
            break
    canary = _dataset("train", tuple(selected), seed=606, parent=datasets["train"].identity)
    path = directory / "canary.json"
    if path.exists():
        if read_dataset(path).identity != canary.identity:
            raise ValueError("canary fixture mismatch")
    else:
        write_dataset(canary, path)
    datasets["canary"] = canary
    report = {name: {"file": str(directory / (name + ".json")), "sha256": data.identity,
                     "manifest": data.manifest, "file_sha256": fingerprint(directory / (name + ".json"))}
              for name, data in datasets.items()}
    # Validate all public transformations before spending GPU time.
    for panel in ("query-swap", "time-order", "row-order", "alias", "template", "distractor", "combined", "three-hop", "memory"):
        panel_cases(datasets["validation"], panel)
    _write_once(directory / ("manifest-confirmation.json" if confirmation else "manifest.json"),
                {k: {**r, "file": k + ".json"} for k, r in report.items()})
    return report


def _gpu_records(devices=None):
    found = inventory()
    if len(found) != 2 or len({g.uuid for g in found}) != 2:
        raise ValueError("NO-GO: exactly two distinct physical GPU UUIDs required")
    if any("T4" not in g.name.upper() or g.free_mib < 11000 or g.total_mib < 14000 for g in found):
        raise ValueError("NO-GO: two full T4s, each with >=11000 MiB currently free")
    if devices is not None:
        supplied = {row["uuid"] for row in devices}
        if supplied != {g.uuid for g in found}:
            raise ValueError("notebook and worker physical inventory disagree")
    return found


def restore_output(prior, output, *, source, protocol_sha):
    """Reattach complete saved raw output or the verified one-file ZIP.

    Inputs stay read-only. Existing different files are never overwritten.
    This is custody transfer, not a fresh-session continuation proof.
    """
    from tools.elendril_v6_export import inventory as file_inventory
    prior, output = Path(prior).resolve(strict=True), Path(output).resolve()
    if prior.is_file():
        from tools.elendril_v6_export import extract_verified
        if output.exists():
            if any(output.iterdir()):
                raise ValueError('ZIP reattachment requires a new or empty output directory')
            output.rmdir()  # Empty, explicitly named output only.
        extract_verified(prior, output)
    else:
        origin = json.loads((prior/'source.json').read_bytes())
        if origin['source'] != source or origin['protocol_sha256'] != protocol_sha:
            raise ValueError('saved output source/protocol mismatch')
        output.mkdir(parents=True,exist_ok=True)
        records = file_inventory(prior)
        if shutil.disk_usage(output).free < sum(r['byte_size'] for r in records) + GIB:
            raise ValueError('insufficient disk for verified saved-output materialization')
        for record in records:
            src, dst = prior/record['name'], output/record['name']
            if dst.exists():
                if fingerprint(src) != fingerprint(dst):
                    raise ValueError('reattachment would overwrite different evidence: '+record['name'])
                continue
            dst.parent.mkdir(parents=True,exist_ok=True)
            temp = dst.with_name('.pending-'+uuid4().hex)
            shutil.copyfile(src,temp)
            if fingerprint(temp) != fingerprint(src): raise ValueError('reattached output readback mismatch')
            os.replace(temp,dst)
    origin = json.loads((output/'source.json').read_bytes())
    if origin['source'] != source or origin['protocol_sha256'] != protocol_sha:
        raise ValueError('materialized output is from another scientific source/protocol')
    return sum(p.stat().st_size for p in output.rglob('*') if p.is_file())


def preflight(*, source, protocol_sha, devices=None, run_id, output=None, available_minutes=360, prior_output=None):
    protocol = _protocol(protocol_sha); sha_id(source, length=40); safe_id(run_id)
    check_source(REPO, source)
    if type(available_minutes) is not int or not 15 <= available_minutes <= 360:
        raise ValueError("owner must choose a cap within observed account/session quota, max360 minutes")
    output = Path(output or "/kaggle/working/ELENDRIL_V6").resolve()
    if output.is_relative_to(REPO) or output.is_relative_to(Path("/kaggle/input").resolve()):
        raise ValueError("output must be writable working storage outside immutable source/input")
    devices = _gpu_records(devices)
    environments = check_torch(REPO, devices)
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:
        raise ValueError('NO-GO: native runtime lacks matplotlib required for result charts; no CUDA stack replacement attempted')
    recovered_bytes = restore_output(prior_output,output,source=source,protocol_sha=protocol_sha) if prior_output is not None else 0
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < max(4*GIB,16*GIB-recovered_bytes):
        raise ValueError("NO-GO: require16 GiB free for bounded checkpoints plus a complete ZIP copy")
    data = prepare_data(output)
    _write_once(output / "protocol.json", protocol)
    _write_once(output / "source.json", {"source": source, "protocol_sha256": protocol_sha,
        "run_id":run_id, "source_package": "777232f103312d80810c89f80d40cba94147f648", "promotion": "NONE"})
    snapshot = output / "source-code.tar"
    if not snapshot.exists():
        command(["git", "archive", "--format=tar", "--output", str(snapshot), source], cwd=REPO, timeout=60)
    report = {"schema": SCHEMA + "/preflight", "status": "PASS", "source": source,
        "protocol_sha256": protocol_sha, "run_id": run_id, "output": str(output),
        "session": session_identity(), "devices": [asdict(g) for g in devices],
        "environments": environments, "data": data, "available_seconds": available_minutes * 60,
        "free_disk_bytes": shutil.disk_usage(output).free, "created_monotonic": time.monotonic(),
        "gpu_qualification": "NOT_RUN", "sealed_external_confirmation": False}
    immutable_json(output / "preflight" / (uuid4().hex + ".json"), report)
    return report


def _validate_environment(preflight, source, protocol_sha):
    _protocol(protocol_sha); check_source(REPO, source)
    if (preflight.get("status") != "PASS" or preflight.get("source") != source
            or preflight.get("protocol_sha256") != protocol_sha or preflight.get("session") != session_identity()):
        raise ValueError("NO-GO: stale/source-mismatched preflight")
    devices = _gpu_records()
    if [g.uuid for g in devices] != [g["uuid"] for g in preflight["devices"]]:
        raise ValueError("physical GPU mapping changed after preflight")
    return devices


def _invoke(mode, request, gpu, *, timeout):
    root = Path(request["output"]).resolve(); root.mkdir(parents=True, exist_ok=True)
    attempts = root / "attempts"; attempts.mkdir(exist_ok=True)
    attempt = uuid4().hex
    request_file, receipt = attempts / (attempt + "-request.json"), attempts / (attempt + "-receipt.json")
    immutable_json(request_file, request)
    error = None
    invocation_started = time.monotonic()
    try:
        command([sys.executable, "-m", "tools.elendril_v6_worker", mode, "--owner-enabled",
                 "--request", str(request_file), "--receipt", str(receipt)],
                cwd=REPO, env=device_env(gpu), timeout=max(1, int(timeout)), log=attempts / (attempt + ".log"))
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    if receipt.exists():
        report = json.loads(receipt.read_bytes())
    else:
        report = {"schema": SCHEMA + "/missing-worker", "status": "FAILED",
                  "reason": "worker exited without a terminal receipt", "error": error,
                  "source": request["source"], "protocol_sha256":request['protocol_sha'],
                  "job": request.get("job"), "mode":mode, "request_file": str(request_file)}
        immutable_json(receipt, report)
    report["receipt_file"] = str(receipt)
    report["receipt_sha256"] = fingerprint(receipt)
    report['invocation_wall_seconds'] = time.monotonic()-invocation_started
    return report


def run_canaries(*, source, protocol_sha, preflight, output, per_device=True):
    if per_device is not True:
        raise ValueError("each physical GPU must qualify independently")
    devices = _validate_environment(preflight, source, protocol_sha)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    from anra_v5.elendril.v6_training import Job
    qualification_remaining = min(1500.,1800.-(time.monotonic()-preflight['created_monotonic']))
    if qualification_remaining < 60: raise ValueError('S0 qualification time budget exhausted')
    def lane(index):
        request = {"source": source, "protocol_sha": protocol_sha,
            "output": str(output / ("lane-" + str(index) + "-" + uuid4().hex)),
            "data_file": preflight["data"]["train"]["file"], "eval_file": preflight["data"]["validation"]["file"],
            "canary_file": preflight["data"]["canary"]["file"],
            "job": asdict(Job("canary-" + str(index), "qualification", "A1", 7)),
            "updates": 4, "positive": True, "timecap_seconds": int(qualification_remaining)}
        return _invoke("benchmark", request, devices[index], timeout=qualification_remaining+30)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lane, range(2)))
    report = {"schema": SCHEMA + "/canaries", "source": source, "protocol_sha256": protocol_sha,
        "status": "PASS" if all(r.get("status") == "PASS" for r in results) else "FAIL",
        "session": session_identity(), "results": results,
        "off_session_recovery": "OWNER_REATTACHMENT_REQUIRED"}
    immutable_json(output / ("canaries-" + uuid4().hex + ".json"), report)
    qualification = {'schema':SCHEMA, 'source':source, 'protocol_sha256':protocol_sha,
                     'output_root':preflight['output'], 'status':'QUALIFICATION_ONLY' if report['status']=='PASS' else 'FAILED_WITH_RECEIPTS',
                     'results':[], 'diagnostics':[], 'canary':report, 'promotion':'NONE'}
    immutable_json(Path(preflight['output'])/'campaigns'/('qualification-'+uuid4().hex+'.json'),qualification)
    report['export'] = checkpoint_export(Path(preflight['output']),source=source,protocol_sha=protocol_sha,
                                        campaign=qualification,rescore=_rescore_callback)
    return report


def verify_off_session_recovery(*, source, prior_dir, current_environment, output, must_resume_optimizer=True):
    if must_resume_optimizer is not True:
        raise ValueError("weights-only recovery cannot qualify training")
    protocol_sha = current_environment["protocol_sha256"]
    devices = _validate_environment(current_environment, source, protocol_sha)
    prior = Path(prior_dir).resolve(strict=True)
    if prior.is_file():
        from tools.elendril_v6_export import extract_verified
        target = Path(output).resolve().parent / ('reattached-'+uuid4().hex)
        extract_verified(prior,target)
        prior = target
    candidates = []
    for p in prior.rglob("*challenge*.json"):
        if p.is_symlink(): raise ValueError("linked recovery challenge")
        value = json.loads(p.read_bytes())
        if value.get("source") == source and value.get("protocol_sha256", value.get("protocol_sha")) == protocol_sha:
            candidates.append((p, value))
    candidates = [(p, c) for p, c in candidates if c.get("job", {}).get("profile") == "S3"]
    if not candidates:
        raise ValueError("NO-GO: reattach complete prior qualification output containing source-bound recovery challenges")
    remaining = min(600.,1800.-(time.monotonic()-current_environment['created_monotonic']))
    if remaining < 30: raise ValueError('S0 qualification budget exhausted before fresh-session recovery')
    def lane(index):
        gpu = devices[index]
        path, challenge = candidates[index % len(candidates)]
        previous_session = challenge.get("session", challenge.get("environment", {}).get("session"))
        if previous_session == session_identity():
            raise ValueError("NO-GO: kernel restart on the same VM is not off-session recovery")
        request = {"source": source, "protocol_sha": protocol_sha,
            "output": str(Path(output).resolve() / ("lane-" + str(index) + "-" + uuid4().hex)),
            "challenge": str(path), "prior_root": str(prior), "fresh_session": True,
            "timecap_seconds": int(remaining)}
        return _invoke("recover", request, gpu, timeout=remaining+20)
    with ThreadPoolExecutor(max_workers=2) as pool:
        selected = list(pool.map(lane,range(2)))
    report = {"schema": SCHEMA + "/recovery", "source": source, "protocol_sha256": protocol_sha,
        "session": session_identity(), "status": "PASS" if all(r.get("status") == "PASS" for r in selected) else "FAIL",
        "results": selected, "physical_uuids": [g.uuid for g in devices]}
    immutable_json(Path(output) / ("recovery-" + uuid4().hex + ".json"), report)
    return report


def estimate_job(job, measurements, examples, *, eval_multiplier=1.):
    """Forecast includes checkpoint/CPU overhead and conservative evaluation margin."""
    measurement = measurements[job.profile]
    step = measurement["seconds_per_update"]
    evaluation = measurement["evaluation_seconds_per_episode"]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (step, evaluation, eval_multiplier)):
        raise ValueError("nonfinite or invalid measured forecast")
    return 1.75 * step * math.ceil(examples / 16) * job.epochs + 2 * evaluation * EVAL_WORLDS * eval_multiplier + 60


def _measurements(canary):
    merged = {}
    for result in canary["results"]:
        for name, measure in result["measurements"].items():
            row = merged.setdefault(name, {})
            for key in ("seconds_per_update", "evaluation_seconds_per_episode", "peak_cuda_allocated_bytes"):
                value = measure[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid measured canary budget")
                row[key] = max(row.get(key, 0), value)
    return merged


def select_candidates(results):
    """Validation-only, both-seed formation; deterministic tie order, unchanged CE retained."""
    from anra_v5.elendril.v6_registry import formation_gate
    grouped = {}
    for result in results:
        if result.get("status") != "COMPLETE": continue
        job = result["job"]
        if job["stage"] != "screen" or job["representation"] != "utf8-raw": continue
        grouped.setdefault(job["arm"], {})[job["seed"]] = result
    qualified = []
    if set(grouped.get('A1', {})) != {7,19}:
        return {'qualified':[], 'nominated':[], 'winner':None, 'alternative_signal':[],
                'reason':'unchanged CE control lacks both complete seeds; failures remain in ledger',
                'source_of_selection':'validation only'}
    for arm, seeds in grouped.items():
        if set(seeds) != {7, 19}: continue
        okay = True; utility = []
        for result in seeds.values():
            report = result["evaluation"]
            if report["split"] != "validation": raise ValueError("confirmation cannot select an arm")
            if not formation_gate(report): okay = False
            total = report["metrics"]["all"]
            fixed = result["controls"]["fixed"]["metrics"]["all"]
            if total["requests"] / total["episodes"] > fixed["requests"] / fixed["episodes"] + .05: okay = False
            utility.append(total["utility"] / total["episodes"])
        if okay: qualified.append((sum(utility) / 2, arm))
    qualified.sort(key=lambda row: (-row[0], row[1]))
    nominated = [arm for _, arm in qualified[:2]]
    signals = []
    for arm in grouped:
        if arm == 'A1' or set(grouped[arm]) != {7,19}: continue
        signal = True
        for seed in (7,19):
            cm = grouped[arm][seed]['evaluation']['metrics']
            bm = grouped['A1'][seed]['evaluation']['metrics']
            c, b = cm['all'], bm['all']
            if c['utility']/c['episodes'] - b['utility']/b['episodes'] < .05: signal = False
            for kind, baseline in bm['by_kind'].items():
                candidate = cm['by_kind'].get(kind)
                if not candidate or candidate['correct']/candidate['episodes'] < baseline['correct']/baseline['episodes']-.05:
                    signal = False
        if signal: signals.append(arm)
    return {"qualified": [a for _, a in qualified], "nominated": nominated,
        "alternative_signal": sorted(signals),
        "winner": nominated[0] if nominated else None,
        "source_of_selection": "validation only; no confirmation outcomes consulted",
        "selection_rule": "both-seed90/80/invalid0/requestcost floor; deterministic meanutility, lexical tie"}


def _receipt_gate(receipt, source, protocol_sha, kind, devices=None):
    if (receipt.get("status") != "PASS" or receipt.get("source") != source
            or receipt.get("protocol_sha256") != protocol_sha or receipt.get("session") != session_identity()
            or len(receipt.get("results", [])) != 2):
        raise ValueError("NO-GO: current source/session-bound " + kind + " proof required")
    if devices is not None and {r.get("gpu_uuid") for r in receipt["results"]} != {g.uuid for g in devices}:
        raise ValueError("NO-GO: proof does not cover both current physical UUIDs")
    for row in receipt["results"]:
        if row.get("status") != "PASS" or row.get("source") != source or row.get("session") != session_identity():
            raise ValueError("NO-GO: unsuccessful, stale or source-mismatched physical-lane proof")
        if kind == "recovery" and (row.get("fresh_session") is not True or row.get("original_session") == session_identity()
                                   or not row.get("challenge_sha256")):
            raise ValueError("NO-GO: actual new-VM challenge continuation required")


def run_campaign(*, source, protocol_sha, preflight, canary, recovery, run_id, output,
                 wall_limit_seconds=21600, export_reserve_seconds=2100, preserve_failed_and_partial=True):
    from anra_v5.elendril.v6_training import Job
    from anra_v5.elendril.v6_data import read_dataset
    if (type(wall_limit_seconds) is not int or not 5700 <= wall_limit_seconds <= 21600
            or type(export_reserve_seconds) is not int or not 2100 <= export_reserve_seconds < wall_limit_seconds
            or preserve_failed_and_partial is not True):
        raise ValueError("unsafe cap/export reserve or failure-preservation policy")
    devices = _validate_environment(preflight, source, protocol_sha)
    _receipt_gate(canary, source, protocol_sha, "canary", devices); _receipt_gate(recovery, source, protocol_sha, "recovery", devices)
    if wall_limit_seconds > preflight["available_seconds"]:
        raise ValueError("campaign exceeds owner-reported current quota/session cap")
    if recovery.get("physical_uuids") != [g.uuid for g in devices]:
        raise ValueError("each current physical lane must pass off-session recovery")
    safe_id(run_id); output = Path(output).resolve()
    if output != Path(preflight["output"]): raise ValueError("campaign output differs from admitted output")
    started = time.monotonic()
    qualification_elapsed = max(0., started - preflight["created_monotonic"])
    available = wall_limit_seconds - qualification_elapsed
    if available <= export_reserve_seconds + 60:
        raise ValueError("current-session qualification consumed available budget; preserve proofs and choose a new session")
    deadline = started + available - export_reserve_seconds
    spent = {name: 0. for name in STAGE_SECONDS}
    measures = _measurements(canary)
    dataset = read_dataset(preflight["data"]["train"]["file"])
    results, skipped, diagnostics = [], [], []
    campaign = {"schema": SCHEMA, "source": source, "protocol_sha256": protocol_sha,
        "run_id": run_id, "status": "PARTIAL", "results": results, "skipped": skipped,
        "diagnostics": diagnostics, "canary": canary, "recovery": recovery,
        "started_session": session_identity(), "output_root": str(output), "promotion": "NONE"}
    def snapshot(event):
        append_event(output, {"source": source, "protocol_sha256": protocol_sha, **event})
        checkpoint_export(output, source=source, protocol_sha=protocol_sha, campaign=campaign,
                          rescore=_rescore_callback, reserve_bytes=512 * (1 << 20))
    def stage(stage_name, jobs, data_report, *, eval_key="validation"):
        if not jobs: return []
        cap = min(STAGE_SECONDS[stage_name] - spent[stage_name], max(0, deadline - time.monotonic()))
        plan_key = stage_name + '-' + jobs[0].profile
        plan_path = output / 'plans' / (plan_key + '.json')
        if plan_path.exists():
            planned = json.loads(plan_path.read_bytes())
            if planned['source'] != source or planned['protocol_sha256'] != protocol_sha:
                raise ValueError('resume plan identity mismatch')
            jobs = [Job(**r) for r in planned['jobs']]
        def forecasts(jobs):
            return [estimate_job(j, measures, dataset.manifest["examples"] * (1.4 if j.arm in ("A4", "A5") else 1.)) for j in jobs]
        estimates = forecasts(jobs)
        if stage_name == 'screen' and not plan_path.exists() and max(sum(estimates[i] for i in range(lane,len(jobs),2)) for lane in range(2)) > cap:
            # Uniform prospective shortening before any score, never selective
            # outcome-driven budget changes. Keep complete semantic epochs.
            from dataclasses import replace
            jobs = [replace(j, epochs=1) for j in jobs]
            estimates = forecasts(jobs)
        _write_once(plan_path, {'source':source,'protocol_sha256':protocol_sha,'jobs':[asdict(j) for j in jobs],
                               'selection_time':'before any scores for this stage'})
        lanes = [sum(estimates[i] for i in range(lane, len(jobs), 2)) for lane in range(2)]
        if max(lanes) > cap:
            skipped.extend({"job": asdict(j), "status": "SKIPPED", "reason": "complete matched stage forecast exceeds remaining capped budget", "predicted_stage_seconds": max(lanes)} for j in jobs)
            snapshot({"stage": stage_name, "status": "SKIPPED", "reason": "forecast"}); return []
        from anra_v5.elendril.v6_registry import MODEL_PROFILES
        new_payloads = sum(2 * MODEL_PROFILES[j.profile].parameter_receipt().total * 12 for j in jobs)
        existing = sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and 'exports' not in p.parts)
        if shutil.disk_usage(output).free < 2 * new_payloads + existing + GIB:
            skipped.extend({"job": asdict(j), "status": "SKIPPED", "reason": "checkpoint plus ZIP disk forecast fails"} for j in jobs)
            snapshot({"stage": stage_name, "status": "SKIPPED", "reason": "disk forecast"}); return []
        stage_started = time.monotonic()
        stage_deadline = min(deadline, stage_started + cap)
        def lane(index):
            local = []
            for position in range(index, len(jobs), 2):
                job = jobs[position]
                attempts = output / 'jobs' / job.job_id / 'attempts'
                previous = sorted(attempts.glob('*-receipt.json'), key=lambda p:p.stat().st_mtime_ns) if attempts.exists() else []
                recovered = None
                for path in reversed(previous):
                    value = json.loads(path.read_bytes())
                    if (value.get('status') == 'COMPLETE' and value.get('source') == source and value.get('job') == asdict(job)):
                        value.update(receipt_file=str(path),receipt_sha256=fingerprint(path))
                        from tools.verify_elendril_v6 import verify_result
                        verify_result(output,value,campaign)
                        recovered=value;break
                if recovered is not None:
                    local.append(recovered)
                    append_event(output,{'status':'COMPLETE_REATTACHED','job':asdict(job),'receipt':recovered['receipt_file']})
                    continue
                remaining = stage_deadline - time.monotonic()
                if remaining <= 60:
                    local.append({"job": asdict(job), "source":source, "protocol_sha256":protocol_sha,
                                  "status": "SKIPPED", "reason": "stage deadline"}); continue
                request = {"source": source, "protocol_sha": protocol_sha, "job": asdict(job),
                    "output": str(output / "jobs" / job.job_id), "data_file": data_report["train"]["file"],
                    "eval_file": data_report[eval_key]["file"], "timecap_seconds": max(1, int(remaining - 30)),
                    "evaluation_reserve_seconds":min(600.,max(45.,measures[job.profile]['evaluation_seconds_per_episode']*EVAL_WORLDS*(4 if eval_key=='confirmation' else 2)+30))}
                report = _invoke("train", request, devices[index], timeout=remaining)
                if report.get("status") == "COMPLETE":
                    from tools.verify_elendril_v6 import verify_result
                    verify_result(output, report, campaign)
                local.append(report)
                append_event(output, {"job": asdict(job), "status": report["status"],
                                      "receipt": report["receipt_file"], "receipt_sha256": report["receipt_sha256"]})
                # The individual raw receipt is already durable in /working.
                # A racing ZIP snapshot may defer, preserving the previous ZIP.
                checkpoint_export(output, source=source, protocol_sha=protocol_sha, campaign=campaign,
                                  rescore=_rescore_callback, reserve_bytes=512 * (1 << 20))
            return local
        with ThreadPoolExecutor(max_workers=2) as pool:
            completed = sum(list(pool.map(lane, range(2))), [])
        spent[stage_name] += time.monotonic() - stage_started
        results.extend(completed)
        snapshot({"stage": stage_name, "status": "FINISHED", "attempts": len(completed)})
        return completed
    try:
        # Preregistered broad S3 screen; all 15 diagnostics share these weights.
        jobs = [Job(f"{run_id}-screen-{arm}-{seed}", "screen", arm, seed)
                for seed in (7, 19) for arm in ("A1", "A2", "A3", "A4", "A5")]
        jobs += [Job(f"{run_id}-repr-{arm}-{seed}", "screen", arm, seed,
                     representation="legacy-ascii" if arm == "R0" else "utf8-ascii")
                 for seed in (7, 19) for arm in ("R0", "R1")]
        screened = stage("screen", jobs, preflight["data"])
        selection = select_candidates(screened)
        screen_epochs = next((r['job']['epochs'] for r in screened if r.get('status') == 'COMPLETE'), 2)
        selection['matched_epochs'] = screen_epochs
        _write_once(output / "selection.json", {"source": source, "protocol_sha256": protocol_sha, **selection})
        campaign["selection"] = selection
        if selection["nominated"]:
            # Materialize the reserved generator only AFTER immutable selection.
            data_report = prepare_data(output, confirmation=True)
            replicas = [Job(f"{run_id}-confirm-{arm}-{seed}", "replicate", arm, seed, epochs=screen_epochs)
                        for seed in (41, 61) for arm in selection["nominated"]]
            replicated = stage("replicate", replicas, data_report, eval_key="confirmation")
            for profile in ("S8", "S20"):
                if profile == "S20" and not campaign.get("s8_signal"):
                    skipped.append({"profile": profile, "status": "SKIPPED", "reason": "S8 did not improve validation utility by .02 at both seeds"}); break
                arms = list(dict.fromkeys(("A1", selection["winner"])))
                scaled = stage("scale", [Job(f"{run_id}-{profile}-{arm}-{seed}", "scale", arm, seed, profile, epochs=screen_epochs)
                                         for seed in (7, 19) for arm in arms], preflight["data"])
                if profile == "S8":
                    base_by_seed = {r["job"]["seed"]: r for r in screened if r.get("status") == "COMPLETE" and r["job"]["arm"] == selection["winner"]}
                    winning = [r for r in scaled if r.get("status") == "COMPLETE" and r["job"]["arm"] == selection["winner"]]
                    def utility(r):
                        m = r["evaluation"]["metrics"]["all"]; return m["utility"] / m["episodes"]
                    campaign["s8_signal"] = len(winning) == 2 and all(utility(r) >= utility(base_by_seed[r["job"]["seed"]]) + .02 for r in winning)
        else:
            skipped.append({"stage": "replicate/scale", "status": "NOT_QUALIFIED", "reason": "both-seed S3 formation floor failed; no larger-model fishing"})
        # Diagnostic candidates are selected on validation, including failed CE
        # when nothing forms. No confirmation score chooses which model to inspect.
        candidates = [r for r in screened if r.get("status") == "COMPLETE" and r["job"]["representation"] == "utf8-raw"]
        wanted = selection["nominated"] or ["A1"]
        candidates = [r for r in candidates if r["job"]["arm"] in wanted and r["job"]["seed"] == 7][:2]
        confirmation_diagnostics = [r for r in locals().get('replicated',[]) if r.get('status')=='COMPLETE'
                                    and r['job']['arm'] in wanted and r['job']['seed']==41][:2]
        if confirmation_diagnostics:
            candidates = confirmation_diagnostics
        diagnostic_deadline = min(deadline, time.monotonic() + STAGE_SECONDS["diagnose"])
        for index, result in enumerate(candidates):
            remaining = diagnostic_deadline - time.monotonic()
            if remaining <= 60:
                skipped.append({"stage": "diagnose", "status": "SKIPPED", "reason": "export reserve protected"}); break
            request = {"source": source, "protocol_sha": protocol_sha, "job": result["job"],
                "output": str(output / "diagnostics" / result["job"]["job_id"]),
                "data_file": preflight["data"]["train"]["file"],
                "eval_file":str(output/'data'/('confirmation.json' if result['job']['stage']=='replicate' else 'validation.json')),
                "parent_result": result["receipt_file"], "positive_control": canary["results"][0].get("positive_control"),
                "supplement_evaluation_seconds_per_episode": measures[result['job']['profile']]['evaluation_seconds_per_episode'],
                "timecap_seconds": int(remaining - 30)}
            diagnostics.append(_invoke("diagnose", request, devices[index % 2], timeout=remaining))
            snapshot({"stage": "diagnose", "status": diagnostics[-1]["status"]})
        campaign["status"] = "COMPLETE" if results and all(r.get("status") == "COMPLETE" for r in results) else "PARTIAL"
    except BaseException as exc:
        campaign.update(status="FAILED_WITH_RECEIPTS", failure={"type": type(exc).__name__, "message": str(exc)})
        append_event(output, {"status": "FAILED", "failure": campaign["failure"]})
    finally:
        known = {r.get('receipt_file') for r in results}
        for p in (output/'jobs').rglob('*-receipt.json') if (output/'jobs').exists() else ():
            if str(p) in known: continue
            value = json.loads(p.read_bytes())
            if value.get('source') == source and value.get('job') and value.get('mode') == 'train':
                value.update(receipt_file=str(p),receipt_sha256=fingerprint(p))
                results.append(value)
        campaign["wall_seconds_before_export"] = time.monotonic() - started
        campaign["skipped"] = skipped
        campaign["remaining_export_seconds"] = max(0, started + available - time.monotonic())
        campaign["current_session_qualification_seconds"] = qualification_elapsed
        campaign["stage_seconds"] = spent
        summary = output / "campaigns" / (uuid4().hex + ".json")
        immutable_json(summary, campaign)
        try:
            _render_outputs(output, campaign)
            campaign["export"] = independent_rescore_and_export(source=source, protocol_sha=protocol_sha,
                campaign=campaign, output=output)
        except Exception as exc:
            campaign["export"] = {"status": "EXPORT_FAILED", "error": type(exc).__name__, "message": str(exc),
                                  "raw_evidence": str(output), "retry": "RUN_EXPORT=True after resolving disk/output problem"}
            append_event(output, campaign["export"])
    return campaign


def _rescore_callback(root, campaign):
    from tools.verify_elendril_v6 import verify_campaign
    return verify_campaign(root, campaign)


def independent_rescore_and_export(*, source, protocol_sha, campaign, output,
                                  zip_name="ELENDRIL_V6_RESULTS.zip", include_all_failures=True):
    if include_all_failures is not True: raise ValueError("failure receipts cannot be dropped")
    _protocol(protocol_sha); check_source(REPO, source)
    output = Path(output).resolve(strict=True)
    if campaign is None:
        paths = sorted((output / "campaigns").glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
        if not paths: raise ValueError("no actual campaign metadata exists; no fabricated results ZIP")
        campaign = json.loads(paths[-1].read_bytes())
    if campaign["source"] != source or campaign["protocol_sha256"] != protocol_sha:
        raise ValueError("cannot relabel another source/protocol campaign")
    _render_outputs(output, campaign)
    return export_bundle(output, source=source, protocol_sha=protocol_sha, campaign=campaign,
                         rescore=_rescore_callback, zip_name=zip_name, reserve_bytes=256 * (1 << 20))


def _render_outputs(output, campaign):
    """Derived summaries never replace immutable raw receipts."""
    def derived(name, value):
        data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        temp = output / (".pending-" + uuid4().hex); temp.write_bytes(data); os.replace(temp, output / name)
    panels = {f"P{i:02d}": {"status": "NOT_QUALIFIED", "reason": "no completed compatible diagnostic checkpoint", "samples": 0} for i in range(1, 16)}
    for result in campaign.get("diagnostics", []):
        diagnostics = result.get("diagnostics", {})
        if diagnostics.get("panels"): panels = diagnostics["panels"]; break
    derived("diagnostics_15panels.json", panels)
    derived('diagnostics_by_checkpoint.json', {r.get('checkpoint',str(i)):r.get('diagnostics',{}).get('panels',{})
            for i,r in enumerate(campaign.get('diagnostics',[]))})
    from anra_v5.elendril.v6_supplement import SPECS
    supplemental = {r.get('checkpoint', str(i)): r.get('diagnostics', {}).get('supplemental', {})
                    for i,r in enumerate(campaign.get('diagnostics', []))}
    derived('supplemental_experiments.json', {'registered': list(SPECS), 'by_checkpoint': supplemental,
        'unattempted': {s['id']: {'status': 'NOT_QUALIFIED', 'reason': 'no completed diagnostic checkpoint'}
                        for s in SPECS} if not any(supplemental.values()) else {},
        'qualification': 'exploratory frozen-checkpoint contrasts; no automatic promotion'})
    derived("all_arm_results.json", campaign.get("results", []))
    manifests = []
    for p in sorted(output.rglob('manifest.json')):
        if 'objects' in p.parts:
            manifests.append({'path':str(p.relative_to(output)), 'sha256':fingerprint(p), 'manifest':json.loads(p.read_bytes())})
    derived('checkpoint_manifest.json', {'generations':manifests,
        'retention':'two verified latest generations per new job; immutable completed stage checkpoints retained; intermediate scalar history retained in logs'})
    data_manifest = output/'data/manifest.json'
    if data_manifest.exists(): derived('data_manifest.json',json.loads(data_manifest.read_bytes()))
    pref = sorted((output/'preflight').glob('*.json'),key=lambda p:p.stat().st_mtime_ns)
    if pref: derived('device_inventory.json',json.loads(pref[-1].read_bytes())['devices'])
    events = [json.loads(p.read_bytes()) for p in sorted((output/'journal').glob('*.json'))]
    ledger = b''.join(json.dumps(r,sort_keys=True,allow_nan=False).encode()+b'\n' for r in events)
    temp = output/('.pending-'+uuid4().hex);temp.write_bytes(ledger);os.replace(temp,output/'experiment_ledger.jsonl')
    from tools.verify_elendril_v6 import decision_matrix, comparisons
    derived("decision_matrix.json", decision_matrix(campaign, panels))
    derived("comparisons.json", comparisons(campaign))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        append_event(output, {"status": "PLOT_UNAVAILABLE", "reason": "matplotlib not installed; raw results preserved"}); return
    figures = (("plot_learning.png", "Training loss (raw attempted updates)", "loss"),
               ("plot_utility.png", "Full-episode utility by arm and seed", "utility"),
               ("plot_ood.png", "Diagnostic status (not independent discoveries)", "panels"))
    for filename, title, kind in figures:
        fig, ax = plt.subplots(figsize=(10, 4))
        if kind == "loss":
            for r in campaign.get("results", []):
                rows = r.get("updates", [])
                if rows: ax.plot([x["update"] for x in rows], [x["loss"] for x in rows], label=r.get("job", {}).get("job_id", "attempt"))
        elif kind == "utility":
            valid = [r for r in campaign.get("results", []) if r.get("evaluation")]
            for i, r in enumerate(valid):
                m = r["evaluation"]["metrics"]["all"]
                ax.scatter(i, m["utility"] / m["episodes"])
            ax.set_xticks(range(len(valid)), [r["job"]["arm"] + "/" + str(r["job"]["seed"]) for r in valid], rotation=45)
        else:
            colors = {"PASS": "#447755", "FAIL": "#aa5555", "NOT_QUALIFIED": "#aaaaaa", "SKIPPED": "#dddddd"}
            ax.bar(list(panels), [1] * 15, color=[colors[p["status"]] for p in panels.values()])
            ax.set_yticks([])
        ax.set_title(title); fig.tight_layout(); fig.savefig(output / filename, dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for checkpoint, bundle in supplemental.items():
        extra = bundle.get('experiments', {})
        label = checkpoint[:12]
        effort = extra.get('S16', {})
        if effort.get('status') == 'COMPLETE':
            records = effort['records']
            axes[0,0].plot([1,2,4], [records[str(n)]['metrics']['all']['correct'] /
                          records[str(n)]['metrics']['all']['episodes'] for n in (1,2,4)], marker='o', label=label)
        permutation = extra.get('S17', {})
        if permutation.get('status') == 'COMPLETE':
            summary = permutation['summary']
            pair = summary['paired']['baseline_vs_permuted']
            axes[0,1].bar([label+' both-correct', label+' equivariance'],
                          [pair['both_correct']/pair['pairs'], summary['output_equivariance']])
        memory = extra.get('S18', {})
        if memory.get('status') == 'COMPLETE':
            pair = memory['summary']['paired']['baseline_vs_stale']
            axes[1,0].plot(['empty prefetch','stale prefetch'],
                          [pair['first']['accuracy'], pair['second']['accuracy']], marker='o', label=label)
        cost = extra.get('S19', {})
        if cost.get('status') == 'COMPLETE':
            for name, values in cost['summary']['mean_utility_by_query_cost'].items():
                xs = sorted(values, key=float)
                axes[1,1].plot([float(x) for x in xs], [values[x] for x in xs],
                               label=label+'/'+name)
    for ax, title in zip(axes.flat, ('S16: accuracy vs decision budget', 'S17: binding and equivariance',
                                    'S18: stale-memory interference', 'S19: fixed behavior vs query price')):
        ax.set_title(title)
        if ax.lines: ax.legend(fontsize=6)
        if not ax.has_data(): ax.text(.5,.5,'Not completed / no evidence',ha='center',transform=ax.transAxes)
    for ax in (axes[0,0], axes[0,1], axes[1,0]):
        ax.set_ylim(0, 1.05); ax.set_ylabel('Fraction')
    axes[0,0].set_xlabel('Decision ceiling')
    axes[1,1].set_xlabel('Price per verified query'); axes[1,1].set_ylabel('Mean utility')
    fig.suptitle('Exploratory frozen-checkpoint probes; no capability promotion')
    fig.tight_layout(); fig.savefig(output/'plot_supplemental.png', dpi=140); plt.close(fig)
