"""Rescore owner X1 A0 recorded actions against regenerated development and depth worlds."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
import zipfile

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8x.analysis import evaluate_x1_development_selection, summarize_x1_episodes
from anra_v5.elendril.v8x.data import build_x1_d1h_training_dataset, build_x1_eval_splits, verify_x1_episode
from anra_v5.elendril.v8x.protocol import session_jobs
from tools.elendril_v8x_export import verify_single_x1_zip


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def audit(path):
    source = '8e62ca0f2046c0f07910fdd20abee40508631227'
    verified = verify_single_x1_zip(path, expected_source_commit=source)
    train = build_x1_d1h_training_dataset(quota_per_family=512)
    splits = build_x1_eval_splits(train_worlds=train.worlds)
    out = {}
    replayed = 0
    with zipfile.ZipFile(path) as archive:
        def read(name):
            return json.loads(archive.read(name))

        def lines(name):
            return [json.loads(line) for line in archive.read(name).splitlines()]

        manifest = read('data/manifest.json')
        require(train.data_identity == manifest['d1h_identity'], 'training identity mismatch')
        registry = read('registry.json')['jobs']
        expected = {job['job_id'] for job in session_jobs('A0')}
        status = read('FINAL_STATUS.json')
        require(set(status['completed_jobs']) == expected == {job['job_id'] for job in registry}
                and len(registry) == 6 and status['status'] == 'COMPLETE', 'incomplete registry')
        for spec in registry:
            prefix = 'jobs/' + spec['job_id'] + '/'
            job = read(prefix + 'job.json')
            updates = lines(prefix + 'updates.jsonl')
            require([r['update'] for r in updates] == list(range(1,2369)), 'update sequence')
            record = {'job_seconds':job['wall_seconds'], 'checkpoint_sha256':job['latest_checkpoint_sha256'],
                      'parameters':job['total_parameters'], 'mean_loss_first_last_100':[mean(r['loss'] for r in block) for block in (updates[:100],updates[-100:])], 'trajectory':{}}
            for step, suite, filename in [(s,'milestone_views' if s < 2368 else 'dev_views','episodes') for s in [0,592,1184,1472,2368]] + [(2368,'depth_stress_views','depth_episodes')]:
                root = prefix + f'milestones/u{step:04d}/'
                rows = lines(root + filename + '.jsonl')
                worlds = splits[suite]
                ids = [w['world_id'] for w in worlds]
                require(digest(ids) == manifest[suite+'_sha256'] and ids == [r['world_id'] for r in rows], 'cohort mismatch')
                for w,r in zip(worlds,rows):
                    verdict = verify_x1_episode(w, [(s['executed_op'],s['executed_arg']) for s in r['steps']])
                    require(all(r[k] == v for k,v in asdict(verdict).items() if k != 'verifier_identity'), r['world_id'])
                replayed += len(rows)
                summary = summarize_x1_episodes(rows)
                archived = read(root+'summary.json') if filename == 'episodes' else read(root+'diagnostics.json')['depth_stress']
                # Python versions differ in floating-point summation of family rates.
                macro = 'macro_12_family_accuracy'
                require(abs(summary[macro] - archived[macro]) < 1e-14, 'macro mismatch')
                require(all(archived[k] == v for k,v in summary.items() if k != macro), 'summary mismatch')
                subset = set(w['world_id'] for w in splits['milestone_views'])
                if filename == 'depth_episodes':
                    record['depth_by_hop'] = summary['by_hop']
                else:
                    record['trajectory'][str(step)] = mean(r['correct'] for r in rows if r['world_id'] in subset)
                    if step == 2368:
                        record['final'] = {k:summary[k] for k in ('overall_accuracy','by_form','by_family','r_mechanistic_cf_both_correct','relevant_cf_both_correct','d2_view_conjunction_accuracy','first_divergence_histogram')}
                        record['lesions'] = read(root+'diagnostics.json')['lesions']
                        record['renderer_wrong_first_op'] = {f:sum(r['steps'][0]['executed_op'] != r['steps'][0]['ideal_op'] for r in rows if r['form']==f) for f in ('canonical','alt_renderer')}
            out[spec['job_id']] = record
        pairs = []
        for seed in (7, 19, 41):
            def score(arm):
                return out[f'X1-A0-S20-{arm}-standard-s{seed}']['final']['overall_accuracy']
            pairs.append(dict(init_seed=seed, order_seed=seed, ctrl_macro_12=score('P'), trt_macro_12=score('R')))
        return {'corrected_gate':evaluate_x1_development_selection(pairs),'archive_sha256':verified['zip_sha256'],'source':source,'rescored_episodes':replayed,'jobs':out,'archived_comparison':read('analysis/comparisons.json')['comparisons'][0]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.archive), indent=2, sort_keys=True)+'\n', encoding='utf-8')
