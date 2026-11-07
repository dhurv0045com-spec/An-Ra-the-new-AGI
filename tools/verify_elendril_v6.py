"""Independent public-world action replay and checkpoint custody for V6 ZIPs."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path


def resolve_recorded(root, recorded, campaign):
    root = Path(root).resolve()
    original = Path(campaign.get("output_root", root)).resolve()
    candidate = Path(recorded)
    if candidate.is_absolute():
        try: candidate = candidate.resolve().relative_to(original)
        except ValueError as exc: raise ValueError("recorded artifact outside original campaign root") from exc
    target = root / candidate
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError("linked or escaping result path")
    return target


def verify_result(root, report, campaign):
    from anra_v5.elendril.v6_data import read_dataset
    from anra_v5.elendril.v6_training import Job
    from anra_v5.elendril.v6_evaluation import independent_rescore
    from anra_v5.elendril.checkpoint_tools import verify_checkpoint
    from anra_v5.elendril.v6_registry import REGISTRY_SHA
    root = Path(root).resolve()
    if report.get("source") != campaign["source"] or report.get("protocol_sha256", report.get("protocol_sha")) != REGISTRY_SHA:
        raise ValueError("result source/protocol mismatch")
    if report.get("status") not in ("COMPLETE", "PARTIAL", "FAILED", "PASS", "SKIPPED"):
        raise ValueError("unknown result completion status")
    if report.get("status") in ("FAILED", "SKIPPED"):
        return {"status": "FAILURE_PRESERVED", "reason": report.get("failure", report.get("reason"))}
    job = Job(**report["job"]); job.validate()
    if report.get("evaluation"):
        split = report["evaluation"]["split"]
        if split not in ("train", "validation", "confirmation"):
            raise ValueError("unregistered evaluation split")
        if report["evaluation"]["decoder_schema"] != job.representation:
            raise ValueError("registered representation and action decoder disagree")
        data = read_dataset(root / "data" / (split + ".json"))
        independent_rescore(data, report["evaluation"])
        for name, control in report.get("controls", {}).items():
            if control["model_identity"] is not None:
                raise ValueError("fixed control cannot claim a model identity")
            independent_rescore(data, control)
    checkpoint = report.get("checkpoint")
    if checkpoint:
        checkpoint_root = report.get("checkpoint_root")
        if checkpoint_root is None:
            checkpoint_root = str(Path(report.get("output", root / "jobs" / job.job_id)) / "checkpoints")
        location = resolve_recorded(root, checkpoint_root, campaign)
        checked = verify_checkpoint(location, job.job_id, checkpoint)
        state = checked["state"]
        if (state["identities"]["source_commit"] != campaign["source"]
                or state["global_update"] != report["global_update"]
                or (report["status"] == "COMPLETE" and state["cumulative_tokens"] != state["token_budget"])):
            raise ValueError("completed result and verified checkpoint state disagree")
        if report.get("evaluation") and report["evaluation"]["model_identity"] != checkpoint:
            raise ValueError("evaluation does not bind the actual committed checkpoint")
    elif report.get("status") == "COMPLETE":
        raise ValueError("complete training result has no resumable checkpoint")
    return {"status": "RESCORED", "job": asdict(job), "checkpoint": checkpoint,
            "evaluation": report.get("evaluation", {}).get("metrics"), "completion": report["status"]}


def verify_campaign(root, campaign):
    from anra_v5.elendril.v6_registry import REGISTRY_SHA
    root = Path(root).resolve()
    if campaign.get("protocol_sha256") != REGISTRY_SHA:
        raise ValueError("campaign evaluator registry mismatch")
    audits = []
    for report in campaign.get("results", []):
        if report.get("receipt_file"):
            path = resolve_recorded(root, report["receipt_file"], campaign)
            if sha256(path.read_bytes()).hexdigest() != report["receipt_sha256"]:
                raise ValueError("raw worker receipt hash mismatch")
            original = json.loads(path.read_bytes())
            if any(original.get(k) != v for k, v in report.items() if k not in ("receipt_file", "receipt_sha256", "invocation_wall_seconds")):
                raise ValueError("campaign altered a raw worker receipt")
        audits.append(verify_result(root, report, campaign))
    # Diagnostic receipts include reconstructable cases for each frozen panel.
    from anra_v5.elendril.v6_data import read_dataset
    from anra_v5.elendril.v6_evaluation import independent_rescore, diagnostic_cases
    for report in campaign.get("diagnostics", []):
        if not report.get("diagnostics"): continue
        if report.get('receipt_file'):
            raw = resolve_recorded(root, report['receipt_file'], campaign)
            if sha256(raw.read_bytes()).hexdigest() != report['receipt_sha256']:
                raise ValueError('diagnostic raw receipt hash mismatch')
            original = json.loads(raw.read_bytes())
            if any(original.get(k) != v for k,v in report.items()
                   if k not in ('receipt_file','receipt_sha256','invocation_wall_seconds')):
                raise ValueError('campaign altered a raw diagnostic receipt')
        diagnosis = report["diagnostics"]
        split=diagnosis['base']['split']
        if split not in ('validation','confirmation'):raise ValueError('unregistered diagnostic split')
        data = read_dataset(root / 'data' / (split+'.json'))
        independent_rescore(data, diagnosis["base"])
        parents = [r for r in campaign.get('results', []) if r.get('status') == 'COMPLETE'
                   and r.get('checkpoint') == report.get('checkpoint') and r.get('evaluation') == diagnosis['base']]
        if not parents:
            raise ValueError('diagnostics have no verified frozen parent result')
        if report.get('weight_sha256_before') != report.get('weight_sha256_after') or not report.get('weight_sha256_before'):
            raise ValueError('diagnostic frozen-weight evidence missing or changed')
        for name, record in diagnosis.get("variants", {}).items():
            if record['model_identity'] != report['checkpoint']:
                raise ValueError('diagnostic variant changed checkpoints')
            independent_rescore(diagnostic_cases(data, name), record)
        if 'supplemental' in diagnosis:
            verify_supplements(data, diagnosis['supplemental'], parents[0].get('controls', {}), report['checkpoint'])
        audits.append({"status": "DIAGNOSTIC_RESCORED", "panels": {p: r["status"] for p, r in diagnosis["panels"].items()}})
    return {"status": "PASS" if audits else "NO_COMPLETED_EVIDENCE", "audits": audits,
            "failed_and_partial_retained": True, "scientific_promotion": "NONE"}


def verify_supplements(data, bundle, controls, checkpoint):
    """Rebuild cohorts, replay raw conditions, and recompute every extra endpoint."""
    from anra_v5.elendril.v6_supplement import SPECS, condition_specs, verify_experiment
    from anra_v5.elendril.v6_evaluation import independent_rescore
    if (bundle.get('schema') != 'elendril-v6-supplement-bundle-v1'
            or set(bundle.get('experiments', {})) != {s['id'] for s in SPECS}
            or bundle.get('limit_seconds') != 90):
        raise ValueError('supplemental bundle protocol mismatch')
    for experiment_id, report in bundle['experiments'].items():
        for record in report.get('records', {}).values():
            if record.get('model_identity') != checkpoint:
                raise ValueError('supplemental checkpoint mismatch')
        if report.get('status') in ('COMPLETE', 'NOT_QUALIFIED'):
            verify_experiment(data, experiment_id, report, controls)
        elif report.get('status') in ('PARTIAL', 'SKIPPED') and experiment_id != 'S19':
            specs = condition_specs(data, experiment_id)
            records = report.get('records', {})
            if (report.get('schema') != 'elendril-v6-supplement-partial-v1'
                    or report.get('experiment_id') != experiment_id
                    or not set(records) < set(specs)
                    or report.get('missing_conditions') != sorted(set(specs) - set(records))
                    or report['status'] != ('PARTIAL' if records else 'SKIPPED')
                    or report.get('reason') != 'supplemental time allowance or campaign deadline'):
                raise ValueError('partial supplemental condition mismatch')
            for name, record in records.items():
                spec = specs[name]
                if (record['budget'] != asdict(spec['budget']) or record['memory_mode'] != spec['memory_mode']
                        or record['split'] != data.split):
                    raise ValueError('partial supplemental budget mismatch')
                independent_rescore(spec['cases'], record)
        else:
            raise ValueError('unknown supplemental completion status')
    return True


def comparisons(campaign):
    results = {}
    for record in campaign.get("results", []):
        if record.get("status") != "COMPLETE" or not record.get("evaluation"): continue
        job, metric = record["job"], record["evaluation"]["metrics"]["all"]
        results[(job["stage"], job["profile"], job["seed"], job["arm"])] = (record, metric)
    pairs = []
    for key, (record, metric) in sorted(results.items()):
        stage, profile, seed, arm = key
        if arm in ("A1", "R0"): continue
        control = results.get((stage, profile, seed, "R0" if arm == "R1" else "A1"))
        if control is None: continue
        baseline, bm = control
        delta = metric["utility"] / metric["episodes"] - bm["utility"] / bm["episodes"]
        pairs.append({"stage": stage, "profile": profile, "seed": seed, "candidate": arm,
            "control": "R0" if arm == "R1" else "A1", "utility_delta_per_episode": delta,
            "same_training_worlds": record.get("training_worlds_sha256") == baseline.get("training_worlds_sha256"),
            "candidate_input_positions": record.get("input_positions"), "control_input_positions": baseline.get("input_positions"),
            "qualification": "matched seed/world development contrast; A4/A5/representation token/work differences explicit"})
    return {"schema": "elendril-v6-comparisons-v1", "pairs": pairs,
            "multiplicity": "all tested alternatives retained; heldout selection frozen before confirmation"}


def decision_matrix(campaign, panels):
    mapping = {
        "P01": "Debug tokenizer, gradients, objective and EOS; do not scale.",
        "P02": "Inspect evidence binding and value copying on unseen worlds.",
        "P03": "Inspect action-formation signal and missing-evidence coverage.",
        "P04": "Inspect timestamp/source conflict representation and query choice.",
        "P05": "Separate null from unavailable evidence and unsafe answers.",
        "P06": "Inspect whether new public observations causally change answers.",
        "P07": "Compare cost-aware training with matched fixed/always-query controls.",
        "P08": "Fit calibration on separate validation; no awareness claim.",
        "P09": "Repair action grammar, EOS or bounded-controller failures.",
        "P10": "Test query blindness and paired counterfactual information flow.",
        "P11": "Test timestamp authority versus serialization recency shortcuts.",
        "P12": "Target representation invariance and independently authored surfaces.",
        "P13": "Dissect fixed-depth information flow before increasing size.",
        "P14": "Distinguish learned retrieval choice from engineered prefetch.",
        "P15": "Qualify A and B separately, then retention and truly novel C at equal total work."}
    return {"schema": "elendril-v6-decision-matrix-v1", "campaign_status": campaign["status"],
            "panels": {p: {"observed_status": panels[p]["status"], "next": mapping[p]} for p in mapping},
            "scale_rule": "S20 slower/worse or no S8 validation gain: stop scaling; diagnose data/model/compute bottleneck",
            "promotion": "No automatic promotion; reserved generator is not an externally sealed evaluator"}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args(argv)
    candidates = sorted((args.output / "campaigns").glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    path = args.campaign or (candidates[-1] if candidates else None)
    if path is None: parser.error("no real campaign receipt found")
    result = verify_campaign(args.output, json.loads(path.read_bytes()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__": main()
