"""Owner-run V7.17 optimization and representation campaign controller.

Importing this module never launches training. Every job is bound to a pinned
source/protocol, admitted against two physical T4s, and exported with its raw
evidence even when the phase cannot complete.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import time
import traceback
from uuid import uuid4

from anra_v5.elendril.v717_protocol import PROTOCOL_SHA, PHASES, protocol, jobs
from tools.elendril_cloud import check_source, check_torch, fingerprint
from tools.elendril_v6_campaign import _gpu_records
from tools.elendril_v6_export import immutable_json
from tools.elendril_v7_campaign import write_once
from tools.elendril_v71_campaign import prepare_data, invoke, require_pass, rescore

REPO = Path(__file__).resolve().parents[1]
WORKER = 'tools.elendril_v717_worker'
MEASUREMENTS = ('initial', 'epoch-1', 'epoch-2', 'epoch-4')


def storage_admission(output, registered):
    """Reserve raw generations plus one staged full ZIP; account for resume files."""
    from anra_v5.elendril.v6_registry import MODEL_PROFILES
    root=Path(output); existing=0; reusable_jobs=0
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('linked output member')
        if not path.is_file() or 'exports' in path.relative_to(root).parts: continue
        size=path.stat().st_size; existing+=size
        if path.is_relative_to(root/'checkpoints'/'jobs'): reusable_jobs+=size
    # FP32 model + two Adam moments, metadata allowance, three retained epoch
    # generations/job, two concurrently live extra generations, four profiles
    # retaining generations 2/4. Old profile evidence is never discarded.
    generation=MODEL_PROFILES['S20'].parameter_receipt().total*12+(8<<20)
    job_goal=(3*len(registered)+2)*generation
    additional=max(0,job_goal-reusable_jobs)+8*generation+(512<<20)
    raw_bound=existing+additional
    required=additional+raw_bound+(256<<20)
    free=shutil.disk_usage(root).free
    if raw_bound>16*(1<<30): raise ValueError('projected full archive exceeds verified restore bound')
    if free<required: raise ValueError(f'NO-GO: storage needs {required} additional free bytes; have {free}')
    return dict(free_bytes=free,existing_raw_bytes=existing,projected_raw_bytes=raw_bound,
                required_additional_bytes=required,qualification='Conservative uncompressed staging bound; not measured archive size.')


def _measurement_coverage(output, registered):
    """Validate required auditable measurement sidecars and checkpoint links."""
    required = []
    for job in registered:
        folder = Path(output) / 'analysis' / 'jobs' / job['job_id']
        checkpoint_by_epoch = {}
        for epoch, name in ((0, 'initial'), (1, 'epoch-1'), (2, 'epoch-2'), (4, 'epoch-4')):
            path = folder / 'measurements' / (name + '.json')
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 << 20:
                raise ValueError('missing, linked, or oversized required measurement: ' + str(path))
            from tools.elendril_v717_worker import verify_measurement
            row = verify_measurement(path, job)
            if (not isinstance(row.get('checkpoint'), str) or len(row['checkpoint']) != 64
                    or any(c not in '0123456789abcdef' for c in row['checkpoint'])):
                raise ValueError('measurement identity/update mismatch: ' + str(path))
            for key in ('train', 'heldout'):
                metric = row.get(key)
                if (not isinstance(metric, dict) or not isinstance(metric.get('teacher'), dict)
                        or not isinstance(metric.get('decoded'), dict)
                        or metric['decoded'].get('decoded', {}).get('status') != 'MEASURED'):
                    raise ValueError('incomplete training/heldout measurement: ' + str(path))
            if epoch == 0:
                probe = row.get('gradient_probe')
                if not isinstance(probe, dict) or probe.get('status') != 'PASS':
                    raise ValueError('initial gradient probe missing or failed: ' + str(path))
            checkpoint_by_epoch[epoch] = row['checkpoint']
        # The ordinary V7.1 report remains useful for the evaluation record;
        # require all milestones and tie its checkpoint IDs to the sidecars.
        for epoch in (0, 1, 2, 4):
            name = 'initial' if epoch == 0 else f'epoch-{epoch}'
            report_path = folder / (name + '.json')
            if report_path.is_symlink() or not report_path.is_file():
                raise ValueError('missing required evaluation milestone: ' + str(report_path))
            report = json.loads(report_path.read_bytes())
            measurement_path = folder / 'measurements' / (name + '.json')
            measured = json.loads(measurement_path.read_bytes())
            if (report.get('checkpoint') != checkpoint_by_epoch[epoch]
                    or report.get('update') != measured.get('update')
                    or (epoch == 0 and report.get('update') != 0)):
                raise ValueError('evaluation and measurement checkpoint/update differ: ' + str(report_path))
            if epoch:
                required.append(str(Path('checkpoints') / 'jobs' / job['job_id'] /
                                    job['job_id'] / 'objects' / report['checkpoint'] / 'manifest.json'))
        required.append(str(Path('checkpoints') / 'jobs' / job['job_id'] /
                            job['job_id'] / 'objects' / checkpoint_by_epoch[4] / 'manifest.json'))
    return sorted(set(required))


def _profile_forecast(registered, profiles, data, suite):
    """Extend the legacy measured-update forecast with V7.17 audit work."""
    profile_rows = [r for r in profiles if r.get('status') == 'PASS']
    if len(profile_rows) != 4:
        raise ValueError('expected two production profile jobs on each of two T4 lanes')
    audits = [r.get('measurement_seconds_per_audit') for r in profile_rows]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in audits):
        raise ValueError('profile omitted finite full-audit timing')
    from tools.elendril_v71_campaign import forecast as v71_forecast
    estimate = v71_forecast(registered, profile_rows, data, suite)
    # Profile-worker startup has already paid for its eight audit steps. Add
    # four milestone audits per registered job with the requested 2x margin.
    per_job = max(audits) * 4 * 2
    work = [seconds + per_job for seconds in estimate['job_seconds']]
    estimate['seconds'] = max(sum(work[0::2]), sum(work[1::2])) + 600
    extra_audits = per_job * len(registered)
    estimate['v717_audit_reserve_seconds'] = extra_audits
    estimate['measurement_seconds_per_audit_max'] = max(audits)
    estimate['policy'] += '; V7.17 adds four diagnostic milestones/job at 2x measured audit time'
    return estimate


def run(*, output, source, phase='optimization', owner_enabled=False, minutes=330):
    if owner_enabled is not True:
        raise ValueError('explicit owner launch required')
    if phase not in PHASES or type(minutes) not in (int, float) or isinstance(minutes, bool) or not math.isfinite(minutes) or not 1 <= minutes <= 330:
        raise ValueError('invalid phase or session budget')
    registered = jobs(phase)
    if len(registered) not in (6, 8):
        raise ValueError('protocol phase must register six or eight jobs')
    requested = Path(output)
    if requested.is_symlink():
        raise ValueError('output symlink forbidden')
    output = requested.resolve()
    if output.is_relative_to(REPO) or output == Path(output.anchor):
        raise ValueError('dedicated output outside source required')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + minutes * 60
    run_id = uuid4().hex
    campaign = {'schema': 'elendril-v717-campaign-v1', 'run_id': run_id,
                'phase': phase, 'source': source, 'protocol_sha256': PROTOCOL_SHA,
                'status': 'PREFLIGHT', 'results': [], 'promotion': 'NONE',
                'cause_status': 'CAUSE_UNRESOLVED'}
    required_checkpoints = []
    profiles = []
    try:
        check_source(REPO, source)
        write_once(output / 'analysis' / 'identity.json',
                   {'source': source, 'protocol_sha256': PROTOCOL_SHA, 'phase': phase})
        write_once(output / 'analysis' / 'protocol.json', protocol())
        campaign['storage_admission'] = storage_admission(output,registered)
        devices = _gpu_records()
        runtimes = check_torch(REPO, devices)
        immutable_json(output / 'analysis' / ('environment-' + run_id + '.json'), runtimes)
        data, suite = prepare_data(output)
        from anra_v5.elendril.v717_training import prepare_variant
        from anra_v5.elendril.v6_data import read_dataset, write_dataset
        if any(j['arm'] == 'row-order' for j in registered):
            path = output / 'data' / 'train-row-order.json'
            variant = prepare_variant(data['train'])
            if not path.exists():
                write_dataset(variant, path)
            data['train-row-order'] = read_dataset(path)
            if data['train-row-order'].identity != variant.identity:
                raise ValueError('cached row-order variant identity mismatch')
            write_once(output/'analysis'/'row-order-manifest.json',
                       {'identity':variant.identity,'parent':data['train'].identity,
                        'file_sha256':fingerprint(path),'manifest':variant.manifest})
        base = {'source': source, 'protocol_sha': PROTOCOL_SHA, 'phase': phase,
                'suite_file': str(output / 'data' / 'suite.json')}
        # Baseline and alternate registered profile arms are run once on each
        # physical GPU before admitting the full phase.
        alternate = 'low-lr' if phase == 'optimization' else 'row-order'
        profile_jobs = [next(j for j in registered if j['arm'] == arm)
                        for arm in ('baseline', alternate)]
        if len(profile_jobs) != 2:
            raise ValueError('profile protocol must expose baseline and alternate arms')
        requests = []
        for idx, job in enumerate(profile_jobs):
            for lane in range(2):
                variant = 'row-order' if job['arm'] == 'row-order' else 'base'
                requests.append((job, lane, variant))
        def profile(item):
            job, lane, variant = item
            data_file = output / 'data' / ('train-' + variant + '.json' if variant != 'base' else 'train.json')
            row = invoke('profile', {**base, 'job': job, 'data_file': str(data_file),
                'output': str(output / 'analysis' / 'profiles' / (run_id + '-' + job['job_id'] + '-lane' + str(lane))),
                'checkpoint_root': str(output / 'checkpoints' / 'profiles' / (run_id + '-' + job['job_id'] + '-lane' + str(lane))),
                'timecap_seconds': max(1, min(1800, deadline - time.monotonic() - 180))},
                devices[lane], module=WORKER)
            return {**row, 'job': job, 'lane': lane}
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Keep one worker per GPU at a time; two sequential profile rounds.
            for start in (0, 2):
                profiles.extend(pool.map(profile, requests[start:start + 2]))
                campaign['profiles'] = profiles
                require_pass(profiles[-2:], 'V7.17 production profile and recovery')
        campaign['profiles'] = profiles
        forecast = _profile_forecast(registered, profiles, data, suite)
        campaign['capacity_estimate'] = forecast
        if forecast['seconds'] > deadline - time.monotonic():
            raise ValueError('whole phase exceeds measured remaining capacity; NO_GO')
        campaign['status'] = 'RUNNING'
        # Four audit milestones for each job; fresh checkpoint roots and
        # worker-owned safe recovery. Never convert a missing initial audit to
        # a synthetic PASS.
        def lane(index):
            from tools.elendril_v71_export import export_results
            rows = []
            for job in registered[index::2]:
                if time.monotonic() >= deadline - 600: break
                row = invoke('train', {**base, 'job': job,
                'data_file': str(output / 'data' / (('train-row-order' if job.get('arm') == 'row-order' else 'train') + '.json')),
                'output': str(output / 'analysis' / 'jobs' / job['job_id']),
                'checkpoint_root': str(output / 'checkpoints' / 'jobs' / job['job_id']),
                'timecap_seconds': max(1, min(21600, deadline - time.monotonic() - 600))},
                    devices[index], module=WORKER) | {'job': job}
                rows.append(row)
                print(job['job_id'] + ': ' + row.get('status', 'UNKNOWN'), flush=True)
                if row.get('status') not in ('COMPLETE', 'PARTIAL'):
                    print(json.dumps(row,indent=2),flush=True)
                export_results(output,source=source,protocol_sha=PROTOCOL_SHA,
                               campaign=campaign,full=False,rescore=rescore)
            return rows
        with ThreadPoolExecutor(max_workers=2) as pool:
            campaign['results'] = [row for rows in pool.map(lane, range(2)) for row in rows]
        campaign['completed_jobs'] = sum(r.get('status') == 'COMPLETE' for r in campaign['results'])
        if campaign['completed_jobs'] == len(registered):
            from tools.elendril_v71_campaign import coverage
            coverage(output,registered,suite)
            required_checkpoints = _measurement_coverage(output, registered)
            campaign['status'] = 'EXECUTED'
        else:
            campaign['status'] = 'PARTIAL'
        campaign['gpu_worker_reservation_seconds'] = sum(r.get('attempt_wall_seconds', 0)
            for r in profiles + campaign['results'])
    except (Exception, KeyboardInterrupt) as exc:
        campaign.update(status='PARTIAL' if isinstance(exc, KeyboardInterrupt) else 'NO_GO',
                        error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        print('Campaign stopped: ' + str(exc), flush=True)
    finally:
        campaign['wall_seconds'] = time.monotonic() - started
        try:
            from tools.elendril_v717_report import write_diagnosis
            campaign['diagnosis'] = write_diagnosis(output,registered)
            from tools.elendril_v71_report import write_report
            campaign['summary'] = write_report(output, campaign=campaign)
            campaign['independent_replay'] = rescore(output, campaign)
        except Exception as exc:
            campaign['independent_replay'] = {'status': 'FAILED', 'message': str(exc)}
            campaign['status'] = 'INVALID_RESULTS'
        from tools.elendril_v71_export import finalize
        delivery = finalize(output, source, PROTOCOL_SHA, campaign,
            [j['job_id'] for j in registered], required_checkpoints,
            campaign['status'] == 'EXECUTED'
            and campaign.get('independent_replay', {}).get('status') == 'PASS'
            and campaign.get('summary', {}).get('status') == 'AVAILABLE')
        campaign.update(status=delivery['status'], exports=delivery['exports'], delivery=delivery)
        print(json.dumps({'status': campaign['status'], 'exports': campaign.get('exports'),
                          'message': campaign.get('message')}, indent=2), flush=True)
    return campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--phase', choices=PHASES, default='optimization')
    parser.add_argument('--owner-enabled', action='store_true')
    parser.add_argument('--minutes', type=float, default=330)
    args = parser.parse_args(argv)
    result = run(output=args.output, source=args.source, phase=args.phase,
                 owner_enabled=args.owner_enabled, minutes=args.minutes)
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
