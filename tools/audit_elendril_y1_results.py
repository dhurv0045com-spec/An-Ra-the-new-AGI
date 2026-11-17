"""Audit Y1 exported evidence by replaying recorded raw actions, without inference."""
from __future__ import annotations
import argparse, hashlib, io, json, math, time, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from anra_v5.elendril.cognition import OperationCandidate
from anra_v5.elendril.v6_evaluation import parse_action
from anra_v5.elendril.y1_data import load_y1_suite
from anra_v5.elendril.y1_evaluation import evaluate_y1_cases, summarize_y1_predictions, evaluate_all_heuristics, evaluate_t1_or_t2_gate
from anra_v5.elendril.y1_protocol import PROTOCOL_SHA
from tools.elendril_v6_export import verify_zip


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def replay(cases, doc):
    rows = doc['predictions']
    indexed = {r['case_id']: r for r in rows}
    if len(rows) != len(cases) or len(indexed) != len(rows) or set(indexed) != {c.case_id for c in cases}:
        raise ValueError('prediction inventory mismatch')
    class Factory:
        def for_case(self, case, *, memory_available=None):
            steps = indexed[case.case_id]['decisions']
            class Actor:
                origin = 'model'
                model_identity = None
                def __init__(self): self.index = 0
                def propose(self, state, legal, channels, positions):
                    if self.index >= len(steps): return None
                    step = steps[self.index]; self.index += 1
                    op, arg = parse_action(tuple(step['raw_tokens']), channels, 'utf8-raw')
                    return OperationCandidate(op, arg, None, step['model_positions'], step['model_calls'], tuple(step['raw_tokens']))
            return Actor()
    checked = evaluate_y1_cases(cases, Factory(), job_id=doc['job_id'], decoder_condition=doc['decoder_condition'], checkpoint_sha256=doc['checkpoint_sha256'])
    mismatches = []
    for actual in checked['predictions']:
        original = indexed[actual['case_id']]
        differences = {k: {'recorded': original.get(k), 'replayed': v} for k,v in actual.items() if original.get(k) != v}
        if differences: mismatches.append({'case_id': actual['case_id'], 'differences': differences})
    return checked, mismatches


def audit(evidence, summary):
    start = time.perf_counter()
    archives = {}
    for name, path in [('evidence', evidence), ('summary', summary)]:
        archives[name] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'verification': verify_zip(path)}
    with zipfile.ZipFile(evidence) as z, zipfile.ZipFile(summary) as s:
        zn, sn = set(z.namelist()), set(s.namelist())
        common = sorted(zn & sn)
        common_diff = [n for n in common if z.read(n) != s.read(n)]
        ident = json.loads(z.read('analysis/identity.json'))
        if ident['protocol_sha256'] != PROTOCOL_SHA: raise ValueError('protocol mismatch')
        suites = {name: load_y1_suite(json.loads(z.read('prepared/'+name+'_suite.json'))) for name in ('development','confirmation','followup')}
        heuristic = evaluate_all_heuristics(suites['development']['cases'], historical_cases=suites['development']['historical_continuity'])
        recorded_heuristic = json.loads(z.read('analysis/heuristics/development.json'))
        jobs, gate_inputs = {}, defaultdict(dict)
        total = 0
        names = sorted(n for n in zn if n.startswith('evidence/predictions/') and n.endswith('.json') and not n.endswith('_hist.json'))
        for n in names:
            jid = Path(n).stem
            doc = json.loads(z.read(n)); hist = json.loads(z.read(n[:-5]+'_hist.json'))
            suite_name = 'confirmation' if 't2_confirm' in jid else 'followup' if 't3_' in jid else 'development'
            suite = suites[suite_name]
            checked, diffs = replay(suite['cases'], doc)
            checked_hist, hd = replay(suite['historical_continuity'], hist)
            total += len(checked['predictions']) + len(checked_hist['predictions'])
            result = summarize_y1_predictions(suite['cases'], checked, historical_continuity_cases=suite['historical_continuity'], historical_pred_doc=checked_hist)
            receipt_name = f'analysis/jobs/{jid}/run_receipt.json'
            receipt = json.loads(z.read(receipt_name))
            stored_name = f'analysis/jobs/{jid}/metrics_summary.json'
            stored = json.loads(z.read(stored_name))
            summary_diffs = {k:{'recorded':stored.get(k),'replayed':v} for k,v in result.items() if k in stored and stored[k] != v}
            clusters_name = f'analysis/jobs/{jid}/family_clusters.json'
            cluster_diffs = {}
            if clusters_name in zn:
                clusters = json.loads(z.read(clusters_name))
                cluster_diffs = {k:{'recorded':v,'replayed':result.get(k)} for k,v in clusters.items() if result.get(k) != v}
            jobs[jid] = {'suite':suite_name, 'episodes':len(checked['predictions']), 'historical_episodes':len(checked_hist['predictions']), 'primary_endpoint':result['primary_endpoint'], 'per_hop_primary':result['per_hop_primary'], 'guardrails':result['guardrails'], 'by_panel':result['by_panel'], 'preservation_suite':result['preservation_suite'], 'edge_change_panel':result['edge_change_panel'], 'missing_bridge_panel':result['missing_bridge_panel'], 'dual_chain_query_panel':result['dual_chain_query_panel'], 'serialization_holdout':result['serialization_holdout'], 'failed_count':result['failed_count'], 'first_divergence_counts':result['first_divergence_counts'], 'historical_continuity':result['historical_continuity'], 'row_mismatches':diffs, 'historical_row_mismatches':hd, 'summary_mismatches':summary_diffs, 'cluster_mismatches':cluster_diffs}
            if jid.startswith('Y1-t1_factorial-S3-'):
                gate_inputs[receipt['arm']][int(receipt['seed'])] = result
            print(jid, 'episodes', len(checked['predictions'])+len(checked_hist['predictions']), 'mismatches',len(diffs)+len(hd), flush=True)
        gate = evaluate_t1_or_t2_gate(stage='t1_factorial', arms_metrics=gate_inputs, heuristics=heuristic, require_ci_lower_gt_zero=False)
        gate = json.loads(json.dumps(gate))
        stored_gate = json.loads(z.read('analysis/gates/t1_gate.json'))
        logs = {}
        for n in sorted(n for n in zn if n.startswith('evidence/update_logs/') and n.endswith('.jsonl')):
            lines = [json.loads(x) for x in z.read(n).splitlines() if x]
            jid = Path(n).stem; expected = 256 if '-T0-' in jid else 2368
            nonfinite = [{'update':r.get('update'),'field':k,'value':str(v)} for r in lines for k,v in r.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and not math.isfinite(v)]
            updates = [r.get('update') for r in lines]
            logs[jid] = {'count':len(lines), 'expected_count':expected, 'contiguous':updates == list(range(1, expected+1)), 'nonfinite':nonfinite, 'first_loss':lines[0]['loss'], 'last_loss':lines[-1]['loss']}
        sidecars = {}
        for n in sorted(n for n in zn if n.startswith('evidence/logits/') and n.endswith('.npz')):
            jid = Path(n).stem; doc = json.loads(z.read(f'evidence/predictions/{jid}.json'))
            indexed = {r['case_id']:r for r in doc['predictions']}
            with np.load(io.BytesIO(z.read(n)), allow_pickle=False) as f:
                arrays = {k:f[k] for k in f.files}
                base = arrays['base_logits']; ids = arrays['base_case_ids']
                checks, mismatches, missing = 0, [], []
                for cid, row in zip(ids.tolist(), base):
                    pred = indexed.get(cid)
                    if not pred or not pred['decisions'] or not pred['decisions'][0]['raw_tokens']:
                        missing.append(cid); continue
                    token = pred['decisions'][0]['raw_tokens'][0]; argmax = int(np.argmax(row)); checks += 1
                    if token != argmax: mismatches.append({'case_id':cid,'recorded_first_token':token,'logits_argmax':argmax})
                sidecars[jid] = {'inventory':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all()) if np.issubdtype(v.dtype,np.number) else None} for k,v in arrays.items()}, 'base_shape_expected':list(base.shape)==[1024,260], 'argmax_checks':checks, 'argmax_mismatches':mismatches, 'argmax_missing':missing}
        return {'schema':'anra-v8-y1-raw-action-audit-v1','identity':ident,'archives':archives,'cross_archive':{'common_content_members_compared':len(common),'differing_members':common_diff,'packaging_specific_members':['BUNDLE_MANIFEST.json','SHA256SUMS']},'verification':{'prediction_jobs':len(jobs),'episodes_replayed':total,'row_mismatch_count':sum(len(j['row_mismatches'])+len(j['historical_row_mismatches']) for j in jobs.values()),'heuristics_equal':heuristic == recorded_heuristic,'t1_gate_equal':gate == stored_gate,'elapsed_seconds':time.perf_counter()-start,'limitations':['Recorded raw actions replayed through source parser and public environment; no neural inference or checkpoint restoration.','Model call counts and positions are recorded telemetry, not independently reconstructed.','Archive hashes establish internal integrity, not independent provenance.','T0 raw predictions and full checkpoints are not exported in these archives.','Sidecar argmax comparison is limited to first-token base logits; fail logits are inventoried and checked finite.']},'heuristics_development':heuristic,'t1_gate':gate,'recorded_t1_gate_if_different':None if gate==stored_gate else stored_gate,'jobs':jobs,'training_logs':logs,'logit_sidecars':sidecars}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=audit(a.evidence,a.summary)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    print('Saved',a.output,flush=True)
if __name__=='__main__': main()
