"""Summarize the exported V6 analysis archive without importing archive code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(path, *, replay=False):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        results = json.loads(archive.read('all_arm_results.json'))
        panels = json.loads(archive.read('diagnostics_15panels.json'))
        supplement = json.loads(archive.read('supplemental_experiments.json'))
    jobs = []
    for result in results:
        record = result.get('evaluation')
        oracle = result.get('controls', {}).get('oracle')
        if not record or not oracle: continue
        truth = {r['case_id']: r['trace']['answer'] for r in oracle['episodes']}
        if len(truth) != len(oracle['episodes']) or set(truth) != {r['case_id'] for r in record['episodes']}:
            raise ValueError('oracle cohort mismatch')
        nonnull = [r for r in record['episodes'] if truth[r['case_id']] is not None]
        answers = Counter(str(r['trace']['answer']) if r['trace']['answer'] is not None else '<abstain>' for r in record['episodes'])
        jobs.append({'job_id': result['job']['job_id'], 'status': result['status'],
                     'episodes': len(record['episodes']), 'terminal_answers': dict(sorted(answers.items())),
                     'oracle_non_null': len(nonnull),
                     'non_null_correct': sum(r['outcome']['correct'] for r in nonnull),
                     'non_null_abstained': sum(r['trace']['status'] == 'abstained' for r in nonnull),
                     'metrics': record['metrics']['all'],
                     'resources': {k: result.get(k) for k in ('wall_seconds', 'process_cpu_seconds', 'gpu_wall_reservation_seconds', 'input_positions', 'supervised_tokens', 'optimizer_updates', 'peak_cuda_allocated_bytes')}})
    report = {'schema': 'elendril-v7-offline-audit-v1', 'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'qualification': 'Descriptive archive analysis; oracle answers joined offline by case ID, no training or scientific promotion.',
            'jobs': jobs, 'panel_statuses': {k: v.get('status') for k, v in panels.items()},
            'supplement_statuses': {checkpoint: {k: v.get('status') for k, v in info.get('experiments', {}).items()} for checkpoint, info in supplement.get('by_checkpoint', {}).items()}}
    if replay:
        from anra_v5.elendril.v6_data import make_dataset
        from anra_v5.elendril.v6_evaluation import independent_rescore
        from anra_v5.elendril.v6_supplement import condition_specs
        from anra_v5.elendril.v7_measurement import verify_legacy
        data = make_dataset('validation', seed=606, worlds=112)
        counts = Counter()
        def check(cohort, record, kind):
            verify_legacy(cohort, record)
            counts[kind] += 1
            try:
                independent_rescore(cohort, record)
                counts['strict_pass'] += 1
            except ValueError:
                # V7 replay has already verified every other field and bounded
                # both aggregate utilities against exact integer counts.
                counts['strict_fail_utility_reduction'] += 1
        for job in results:
            if job.get('evaluation'): check(data, job['evaluation'], 'model')
            for control in job.get('controls', {}).values(): check(data, control, 'control')
        for bundle in supplement.get('by_checkpoint', {}).values():
            for experiment, item in bundle.get('experiments', {}).items():
                specs = condition_specs(data, experiment)
                for name, record in item.get('records', {}).items():
                    check(data if experiment == 'S19' else specs[name]['cases'], record, 'supplement')
        report['replay'] = {'status': 'PASS', 'counts': dict(counts),
                            'validation_identity': data.identity,
                            'qualification': 'Independent recorded-action replay; no checkpoint inference.'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--replay', action='store_true', help='reconstruct and replay archived evaluation records')
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.archive, replay=args.replay), indent=2, sort_keys=True) + '\n', encoding='utf-8')

if __name__ == '__main__': main()
