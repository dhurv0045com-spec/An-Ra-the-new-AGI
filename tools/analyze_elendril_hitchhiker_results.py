"""Offline, bounded Hitchhiker Stage1 owner-result verification; no model execution."""
import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
import math
import re
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

from anra_v5.elendril.hitchhiker_data import build_counterfactual_suite, build_health_probes
from anra_v5.elendril.hitchhiker_evaluation import independent_rescore_hitchhiker, evaluate_screening_gate
from anra_v5.elendril.hitchhiker_protocol import PROTOCOL_SHA, stage_jobs
from anra_v5.elendril.v6_data import make_dataset
from tools.elendril_v6_export import verify_zip

SOURCE = '6859904a02c8083c4a0ab320c9c4a511899f73e4'
RUN = 'hh-stage1-4bfe3a806e3b'


def audit_c7_row_order(doc):
    """Check the positional hypothesis directly on recorded, matched four-hop worlds."""
    records = doc['suite_eval']['records']
    worlds = {e['world_id'] for e in records['hop-4']['episodes']}
    pattern = r'\b[A-P]=([A-P])@\d+/[\w-]+'
    panels = defaultdict(Counter)
    seen = defaultdict(set)
    for row in records['factor-row-order']['episodes']:
        if row['world_id'] not in worlds:
            continue
        assert len(row['prompts']) == 1
        facts = re.findall(pattern, row['prompts'][0])
        assert len(facts) == 4
        panel = row['panel']
        assert row['world_id'] not in seen[panel]
        seen[panel].add(row['world_id'])
        panels[panel]['episodes'] += 1
        panels[panel]['correct'] += int(row['outcome']['correct'])
        panels[panel]['answer_equals_last_fact_object'] += int(row['trace']['answer'] == facts[-1])
    assert set(panels) == {'order-original', 'order-reverse', 'order-random'}
    assert all(ids == worlds for ids in seen.values())
    return {
        'method': 'regex ' + pattern + ' on prompt; require four facts',
        'panels': dict(panels),
        'scope': 'C seed 7, 256 shared four-hop worlds in factor-row-order; parse last serialized fact object from each recorded prompt and compare to recorded answer',
    }


def compact(m):
    keys = ('episodes', 'correct', 'accuracy', 'invalid', 'invalid_rate', 'abstained',
            'queries', 'retrievals', 'complete_pairs', 'both_correct_pairs', 'pair_accuracy', 'unique_worlds')
    out = {k: m[k] for k in keys if k in m}
    for group in ('by_panel', 'by_kind'):
        if group in m:
            out[group] = {k: {f: v[f] for f in ('episodes', 'correct', 'invalid') if f in v}
                          for k, v in m[group].items()}
    return out


def analyze(path):
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(['git', 'diff', '--exit-code', SOURCE, 'HEAD', '--', 'anra_v5', 'v5_model',
                    'v5_contracts', 'v5_training', 'tools/elendril_v6_export.py'], cwd=root,
                   check=True, capture_output=True)
    with path.open('rb') as f:
        archive_sha = file_digest(f, 'sha256').hexdigest()
    verification = verify_zip(path)
    print('Archive inventory and hashes verified', flush=True)
    with ZipFile(path) as z:
        names = z.namelist()
        identity = json.loads(z.read('identity.json'))
        assert identity['source'] == SOURCE and identity['protocol_sha256'] == PROTOCOL_SHA
        markers = [n for n in names if n.startswith('campaign-') and n.endswith('.json')]
        assert len(markers) == 1
        marker = json.loads(z.read(markers[0])); campaign = marker['campaign']
        assert campaign['run_id'] == RUN and campaign['phase'] == 'stage1'
        expected = {j['job_id']: j for j in stage_jobs('stage1_s3_screen')}
        actual = {n.split('/')[1] for n in names if n.endswith('/final_eval.json')}
        assert actual == set(expected) == set(marker['required_jobs'])
        prep = campaign['prepared_manifest']
        suite = build_counterfactual_suite(seed=17071, worlds_per_hop=prep['val_worlds'], split='validation')
        probes = build_health_probes(seed=27173, worlds_per_probe=prep['probe_worlds'], split='validation')
        original = make_dataset('validation', target_bytes=60_000, seed=707)
        assert len(original.cases) == prep['val_orig_cases']
        assert {k:len(v) for k,v in suite['cohorts'].items()} == prep['val_suite_cohort_counts']
        print('Exact evaluation suites reconstructed', flush=True)
        results = {}; gate_inputs = defaultdict(dict); replayed = episodes = 0
        primary = 'factor-entity-permutation'
        examples = []
        shortcut_audit = None
        for jid, job in expected.items():
            prefix = 'jobs/' + jid + '/'
            assert json.loads(z.read(prefix+'job_spec.json')) == job
            run = json.loads(z.read(prefix+'run_receipt.json'))
            doc = json.loads(z.read(prefix+'final_eval.json'))
            status = json.loads(z.read(prefix+'status.json'))
            assert status['status'] == 'PASS'
            assert run['checkpoint_sha256'] == doc['checkpoint_sha256'] == status['checkpoint_sha256']
            for rec in (run, doc):
                assert rec['source_commit'] == SOURCE and rec['protocol_sha256'] == PROTOCOL_SHA
                assert (rec['job_id'], rec['arm'], rec['seed'], rec['profile']) == (jid,job['arm'],job['seed'],'S3')
            assert doc['split'] == 'validation' and doc['decoder_schema'] == 'utf8-raw'
            assert doc['structural_eval'] is None
            logs = [json.loads(line) for line in z.read(prefix+'update_logs.jsonl').splitlines() if line]
            assert [l['update'] for l in logs] == list(range(1,2369))
            assert run['global_update'] == run['target_updates'] == 2368
            assert all(math.isfinite(l['loss']) for l in logs)
            for key, rk in [('examples','cumulative_examples_processed'),('input_positions','cumulative_input_positions'),
                            ('supervised_tokens','cumulative_supervised_tokens')]:
                assert sum(l[key] for l in logs) == run[rk]
            records = [(cases, doc['suite_eval']['records'][name]) for name,cases in suite['cohorts'].items()]
            records += [(original.cases,doc['original_eval']['record'])]
            records += [(cases,doc['probes_eval']['records'][name]) for name,cases in probes.items() if cases]
            for cases, rec in records:
                independent_rescore_hitchhiker(cases, rec)
                assert rec['model_identity'] == doc['model_identity']
                replayed += 1; episodes += len(cases)
            assert doc['cohort_metrics'] == {k:v['metrics'] for k,v in doc['suite_eval']['records'].items()}
            inputs = dict(doc['suite_eval']); inputs['records'] = {primary: inputs['records'][primary]}
            gate_inputs[job['arm']][job['seed']] = inputs
            actions = {name:dict(Counter(e['first_action']['operation'] for e in rec['episodes']))
                       for name,rec in doc['suite_eval']['records'].items() if name.startswith('hop-')}
            byhop = {}
            for name,rec in doc['suite_eval']['records'].items():
                counters = defaultdict(lambda:Counter())
                casemap = {c.case_id:c for c in suite['cohorts'][name]}
                for e in rec['episodes']:
                    key = f"{e['panel']}/hop-{casemap[e['case_id']].surface.hops}"
                    counters[key]['episodes'] += 1
                    counters[key]['correct'] += int(e['outcome']['correct'])
                byhop[name] = dict(counters)
            results[jid] = {
                'arm':job['arm'],'seed':job['seed'],'checkpoint_sha256':run['checkpoint_sha256'],
                'model_identity':doc['model_identity'], 'cohorts':{k:compact(v) for k,v in doc['cohort_metrics'].items()},
                'panels_by_hop':byhop,'first_actions':actions,'original':compact(doc['original_eval']['record']['metrics']),
                'health':doc['probes_scores'],'equivariance':doc['equivariance_report'],
                'run':run,'updates_verified':len(logs),
                'loss_first':logs[0]['loss'],'loss_last':logs[-1]['loss'],
            }
            if job['arm']=='C' and job['seed']==7:
                shortcut_audit = audit_c7_row_order(doc)
                e=doc['suite_eval']['records']['hop-4']['episodes'][0]; wid=e['world_id']
                for name in ('hop-4','factor-row-order','factor-distractors','factor-source-renaming','factor-timestamps'):
                    for row in doc['suite_eval']['records'][name]['episodes']:
                        if row['world_id']==wid:
                            examples.append({'case_id':row['case_id'],'panel':row['panel'],'prompt':row['prompts'],
                                             'answer':row['trace']['answer'],'correct':row['outcome']['correct'],
                                             'first_action':row['first_action']})
            print(f'{jid}: replayed; primary={doc["cohort_metrics"][primary]["accuracy"]:.4f}',flush=True)
        recorded_gate = json.loads(z.read('gate_receipt.json'))
        recomputed = {arm:evaluate_screening_gate(gate_inputs[arm],gate_inputs['B0']) for arm in ('A','B','AB','C')}
        # JSON roundtrip converts integer seed keys to the serialized string representation.
        assert json.loads(json.dumps(recomputed)) == recorded_gate['arms']
        sizes = z.infolist()
        finals = [r for r in sizes if r.filename.endswith('/final_eval.json')]
        return {
            'schema':'hitchhiker-owner-analysis-v1','source_commit':SOURCE,'run_id':RUN,'protocol_sha256':PROTOCOL_SHA,
            'archive':{'path':str(path),'sha256':archive_sha,'bytes':path.stat().st_size,'members':len(sizes),
                       'expanded_bytes':sum(r.file_size for r in sizes),'inventory_verification':verification,
                       'final_eval_compressed_bytes':sum(r.compress_size for r in finals),
                       'final_eval_expanded_bytes':sum(r.file_size for r in finals)},
            'verification':{'evaluation_reports_replayed':replayed,'episode_evaluations_replayed':episodes,
                            'training_updates_audited':sum(r['updates_verified'] for r in results.values()),
                            'gates_recomputed_exactly':True,'elapsed_seconds':time.perf_counter()-start,
                            'not_performed':['model inference','checkpoint restoration','final dual-export certification']},
            'prepared':prep,'jobs':results,'gate':recorded_gate,'selection':json.loads(z.read('selection_receipt.json')),
            'forecast':json.loads(z.read('forecast_receipt.json')),'profiles':json.loads(z.read('profile_receipt.json')),
            'illustrative_c7_same_world':examples,
            'c_seed7_four_hop_row_order_audit':shortcut_audit,
        }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); result=analyze(a.zip)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
    print('Saved',a.output,'bytes',a.output.stat().st_size,flush=True)


if __name__=='__main__':
    main()
