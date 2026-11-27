"""Run development, seal campaign-wide confirmation custody once, then evaluate frozen models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from anra_v5.palantiri import (
    CampaignOrchestrator,
    ConfirmLock,
    ExperimentModule,
    SessionJob,
    TypedByteSequenceKernel,
    export_session_archives,
    sanitize_for_json,
)
from anra_v5.planthir.archives import safe_extract_planthir_archive, verify_session_archive
from . import composition, information, reader
from .runtime import Context, digest, prune_job_checkpoints, publish, read, verify_and_load_weights

MODULES = {'AB': reader, 'F': reader, 'C': composition, 'G': composition, 'D': information, 'E': information}
HERE = Path(__file__).resolve().parent
BASELINE_RECEIPT = HERE.parents[1] / 'research' / 'avendi' / 'Y5_BASELINE_RECEIPT.json'
SCORER_VERSION = 'avendi_y5_scorer_v2'
MAX_WORKING_BYTES = int(19.5 * (1024**3))


def registry():
    rows = reader.jobs() + composition.jobs() + information.jobs()
    out = {j['id']: j for j in rows}
    if len(out) != len(rows) or any(p not in out for j in rows for p in j['depends']):
        raise ValueError('invalid experiment dependencies')
    return out


def diff_against_baseline(jobs):
    baseline_ids = []
    baseline_commit = '25841d24'
    if BASELINE_RECEIPT.is_file():
        try:
            doc = json.loads(BASELINE_RECEIPT.read_text(encoding='utf-8'))
            baseline_commit = doc.get('commit', baseline_commit)
            for item in doc.get('registry', []):
                if isinstance(item, dict) and 'id' in item:
                    baseline_ids.append(item['id'])
        except (OSError, json.JSONDecodeError):
            baseline_ids = []
    cur = set(jobs)
    base = set(baseline_ids)
    return {
        'baseline_commit': baseline_commit,
        'current_count': len(cur),
        'baseline_count': len(base) if base else None,
        'added': sorted(cur - base) if base else [],
        'removed': sorted(base - cur) if base else [],
    }


def registry_manifest(jobs):
    return {
        jid: {
            'id': j['id'],
            'stage': j['stage'],
            'arm': j['arm'],
            'seed': j['seed'],
            'role': j.get('role', 'primary'),
            'depends': list(j['depends']),
            'updates': j['updates'],
            'split_counts': j.get('split_counts', {}),
            'profile_group': j.get('profile_group', j['stage']),
            'objective': j.get('objective', j['arm']),
            'checkpoint_parent': j.get('checkpoint_parent'),
            'allowed_exclusions': list(j.get('allowed_exclusions', ())),
        }
        for jid, j in sorted(jobs.items())
    }


def split_design_hashes(jobs):
    """Freeze split definitions and seeds canonically before training without loading confirmation panels."""
    spec = {
        jid: {
            'seed': j['seed'],
            'stage': j['stage'],
            'split_counts': j.get('split_counts', {}),
            'profile_group': j.get('profile_group', j['stage']),
        }
        for jid, j in sorted(jobs.items())
    }
    return {
        'scorer_version': SCORER_VERSION,
        'confirm_panels_loaded_at_design': False,
        'design_split_sha256': digest(spec),
        'stage_splits': {
            st: digest({k: v for k, v in spec.items() if v['stage'] == st})
            for st in sorted({j['stage'] for j in jobs.values()})
        },
    }


def benchmark_hardware(canary=True):
    """Measure exact model constructor parameter counts and single-step canary timing without hardcoding."""
    from experiments.avendi_y4.models import AvendiY4ByteModel, P6LearnedMemoryPolicy, Y4TypedCompositionModel, get_y4_byte_spec

    specs = [
        ('AB_F_S3', lambda: AvendiY4ByteModel(get_y4_byte_spec('S3', cpu_canary=canary), seed=7, use_aux=True)),
        ('C_categorical', lambda: Y4TypedCompositionModel('categorical', 128, seed=7)),
        ('DE_P6', lambda: P6LearnedMemoryPolicy(width=32, seed=7)),
    ]
    out = {}
    for name, ctor in specs:
        m = ctor()
        params = sum(int(p.numel()) for p in m.parameters())
        out[name] = {'params': params, 'approx_state_bytes': params * 4}
        del m
    return {'canary': bool(canary), 'max_working_bytes': MAX_WORKING_BYTES, 'models': out}


def identity():
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}
    return source, digest([files, registry()])


def checks():
    res = {m.__name__: m.preflight() for m in dict.fromkeys(MODULES.values())}
    if not all(r.get('passed', r.get('ok', False)) for r in res.values()):
        raise ValueError(f'engineering fixture failed: {res}')
    return res


def receipt(root, split, jid):
    return Path(root) / split / (jid + '.json')


def _confirm_lock_path(root):
    return Path(root) / 'CONFIRM_LOCK.json'


def _planthir_lock_path(root):
    return Path(root) / 'CONFIRM_LOCK.planthir.json'


def verify_global_confirmation_lock(root, job_id, jobs, source, protocol):
    """Verify campaign-wide CONFIRM_LOCK.json and Planthir ConfirmLock before touching any confirm panel."""
    lp = _confirm_lock_path(root)
    if not lp.exists():
        raise ValueError('global confirmation embargo active: CONFIRM_LOCK.json missing')
    lock = read(lp)
    if not lock.get('authorized'):
        raise ValueError('confirmation lock is not authorized')
    if lock.get('source') != source or lock.get('protocol') != protocol or lock.get('scorer_version') != SCORER_VERSION:
        raise ValueError('confirmation lock design/scorer identity mismatch')
    eligible = lock.get('eligible', {})
    if job_id not in eligible:
        raise ValueError(f'job {job_id} is not eligible in CONFIRM_LOCK.json')
    pl_rep = ConfirmLock(_planthir_lock_path(root)).inspect(
        completed_job_ids=sorted(eligible),
        expected_lock_sha256=lock.get('planthir_lock_sha256'),
    )
    if pl_rep.get('defects'):
        raise ValueError(f'ConfirmLock defects detected: {pl_rep["defects"]}')
    dev_rec = read(receipt(root, 'dev', job_id))
    expected_entry = eligible[job_id]
    if digest(dev_rec) != expected_entry['dev_sha256']:
        raise ValueError('development selection changed after global seal')
    ckpt_sha = (dev_rec.get('weights') or {}).get('checkpoint_sha256')
    if ckpt_sha != expected_entry.get('checkpoint_sha256'):
        raise ValueError('checkpoint SHA changed after global seal')
    return lock


def worker(args, jobs, source, protocol):
    job = jobs[args.worker]
    if args.device.startswith('cuda') and torch.cuda.device_count() > 1:
        args.device = os.environ.get('PLANTHIR_ASSIGNED_DEVICE', args.device)
    design = read(Path(args.work_root) / 'DESIGN.json')
    if job['id'] not in design['jobs']:
        raise ValueError('job outside sealed design')
    ctx = Context(args.work_root, job, args.split, args.device, args.canary, args.full, args.inputs, source)
    if design['protocol'] != protocol or design['source'] != source or design['canary'] != args.canary:
        raise ValueError('worker design mismatch')
    if args.split == 'confirm':
        verify_global_confirmation_lock(ctx.root, job['id'], jobs, source, protocol)
    torch.set_num_threads(2)
    torch.manual_seed(job['seed'])
    start = time.monotonic()
    result = MODULES[job['stage']].run(job, ctx)
    status = result.get('status', 'COMPLETED')
    if status not in ('COMPLETE', 'COMPLETED', 'GATED', 'BLOCKED'):
        raise ValueError(f'unrecognized job status: {status}')
    if args.split == 'dev' and job['updates'] and status in ('COMPLETE', 'COMPLETED') and not ctx.trained:
        raise ValueError('missing final checkpoint')
    publish(
        receipt(ctx.root, args.split, job['id']),
        dict(
            job=job,
            source=source,
            protocol=protocol,
            split=args.split,
            result=result,
            status=status,
            weights=ctx.trained,
            seconds=time.monotonic() - start,
            canary=args.canary,
        ),
    )


def restored(root, split, job, source, protocol, canary):
    path = receipt(root, split, job['id'])
    if not path.exists():
        return None
    row = read(path)
    if (row['job'], row['source'], row['protocol'], row['split'], row['canary']) != (job, source, protocol, split, canary):
        raise ValueError(f'incompatible receipt: {path}')
    if row['weights']:
        upd, src, _ = verify_and_load_weights(
            Path(root) / 'weights',
            job['id'],
            row['weights']['checkpoint_sha256'],
        )
        if upd != row['weights']['updates'] or src != source:
            raise ValueError('inconsistent completed checkpoint')
    return row


def cohort(root, stage, group, dev):
    """Legacy per-stage cohort helper retained solely for unit-test compatibility; never used by execute()."""
    if any(j not in dev for j in group):
        return None
    eligible = {j: digest(dev[j]) for j in group if dev[j]['weights']}
    lock = dict(stage=stage, eligible=eligible, dispositions={j: dev[j]['status'] for j in group}, dev_hashes={j: digest(dev[j]) for j in group})
    publish(Path(root) / 'locks' / (stage + '.json'), lock)
    return eligible


def _verify_dev_weights(root, jid, row, source, canary):
    weights = row.get('weights')
    if not weights or not weights.get('checkpoint_sha256'):
        raise ValueError(f'missing checkpoint metadata for eligible job {jid}')
    verify_and_load_weights(
        Path(root) / 'weights',
        jid,
        weights['checkpoint_sha256'],
        expected_source=source,
        expected_updates=weights['updates'],
    )
    if not canary and weights.get('updates', 0) <= 0:
        raise ValueError(f'zero registered updates in FULL for {jid}')


def seal_global_confirmation(root, jobs, dev_rows, source, protocol, split_hashes, canary=False, scorer_version=SCORER_VERSION):
    """Seal ONE global CONFIRM_LOCK.json strictly after all dev jobs resolve; refuse otherwise."""
    if set(dev_rows) != set(jobs):
        return None
    eligible = {}
    exclusions = {}
    diagnostic_dispositions = {}
    checkpoint_shas = {}
    for jid in sorted(jobs):
        job = jobs[jid]
        row = dev_rows[jid]
        status = row.get('status', 'COMPLETE')
        role = job.get('role', 'diagnostic' if (jid.startswith(('G-', 'C-baseline-', 'C-scout-'))) else 'primary')
        if role == 'diagnostic':
            diagnostic_dispositions[jid] = {'status': status, 'dev_sha256': digest(row), 'role': role}
            continue
        if status in ('COMPLETE', 'COMPLETED') and row.get('weights'):
            _verify_dev_weights(root, jid, row, source, canary)
            sha = row['weights']['checkpoint_sha256']
            eligible[jid] = {'dev_sha256': digest(row), 'checkpoint_sha256': sha, 'updates': row['weights']['updates']}
            checkpoint_shas[jid] = sha
        elif status in ('GATED', 'BLOCKED'):
            allowed = set(job.get('allowed_exclusions', ()))
            reason = (row.get('result') or {}).get('verification') or (row.get('result') or {}).get('reason') or status
            exclusions[jid] = {
                'status': status,
                'reason': reason,
                'witness_sha256': digest(row),
                'allowed_rule_declared': bool(allowed),
            }
        else:
            return None

    pl_lock = ConfirmLock(_planthir_lock_path(root))
    pl_doc = pl_lock.seal(
        sorted(eligible),
        protocol_sha256=protocol,
        source_commit=source,
        metadata={'scorer_version': scorer_version, 'split_hashes': split_hashes, 'exclusions': sorted(exclusions)},
    )
    lock_payload = {
        'schema': 'avendi-y5-global-confirm-lock-v2',
        'authorized': True,
        'source': source,
        'protocol': protocol,
        'scorer_version': scorer_version,
        'split_hashes': split_hashes,
        'planthir_lock_sha256': pl_doc['lock_sha256'],
        'eligible': eligible,
        'exclusions': exclusions,
        'diagnostic_dispositions': diagnostic_dispositions,
        'checkpoint_shas': checkpoint_shas,
    }
    lp = _confirm_lock_path(root)
    if lp.exists():
        existing = read(lp)
        if (
            existing.get('eligible') != lock_payload['eligible']
            or existing.get('checkpoint_shas') != lock_payload['checkpoint_shas']
            or existing.get('scorer_version') != lock_payload['scorer_version']
            or existing.get('source') != lock_payload['source']
            or existing.get('protocol') != lock_payload['protocol']
        ):
            raise ValueError('CONFIRM_LOCK_SELECTION_OR_WEIGHTS_MISMATCH')
        return existing
    publish(lp, lock_payload)
    return lock_payload


def _compact_ready_leaf_jobs(root, jobs, dev_rows, canary):
    """Prune intermediate objects and compact leaf optimizers once downstream dependents have resolved."""
    results = {}
    for jid, row in sorted(dev_rows.items()):
        if not row.get('weights'):
            continue
        unresolved_children = [cid for cid, cv in jobs.items() if jid in cv['depends'] and cid not in dev_rows]
        all_children = [cid for cid, cv in jobs.items() if jid in cv['depends']]
        if unresolved_children:
            continue
        # Only strip final optimizer when the job is a true terminal leaf with no dependents.
        is_terminal_leaf = (len(all_children) == 0)
        res = prune_job_checkpoints(
            root,
            jid,
            completed_updates=row['weights']['updates'],
            full_job_updates=1 if canary else max(1, jobs[jid]['updates']),
            downstream_children=all_children,
            strip_optimizer=(is_terminal_leaf and not canary),
        )
        results[jid] = res
    return results


def _admit_wave_nonstarving(coordinator, queue, jobs, rows, devices, root, output_dir, canary, start, total_minutes):
    """Admit over all ready queue items first so oversized jobs defer without starving smaller ready jobs."""
    specs_all = [
        SessionJob(
            jid,
            jobs[jid]['stage'],
            jobs[jid]['seed'],
            jobs[jid]['seed'],
            max(1, jobs[jid]['updates'] if split == 'dev' and not canary else 1),
        )
        for split, jid in queue
    ]
    remaining = total_minutes - (time.monotonic() - start) / 60
    estimates = {}
    for split, jid in queue:
        samples = [
            r['seconds'] / max(1, r['job']['updates'])
            for r in rows['dev'].values()
            if r.get('weights') and r['job']['arm'] == jobs[jid]['arm']
        ]
        estimates[jid] = max(samples, default=0.2) if split == 'dev' else 5.0
    weight_bytes = sum(p.stat().st_size for p in (Path(root) / 'weights').rglob('*') if p.is_file()) if (Path(root) / 'weights').exists() else 0
    avail_free = min(shutil.disk_usage(root).free, shutil.disk_usage(output_dir).free)
    plan = coordinator.admit_slice(
        specs_all,
        measured_seconds_per_update=0.2,
        per_job_seconds_per_update=estimates,
        per_job_disk_bytes=100_000 if canary else 210_000_000,
        parallel_workers=len(devices),
        outer_remaining_minutes=remaining,
        startup_overhead_seconds=5,
        export_reserve_bytes=weight_bytes,
        available_disk_bytes=avail_free,
    )
    allowed = set(plan['admitted_job_ids'])
    admitted_queue = [x for x in queue if x[1] in allowed]
    if not admitted_queue:
        return [], [], plan
    wave = admitted_queue[: len(devices)]
    if len(wave) == 2 and jobs[wave[0][1]]['stage'] == jobs[wave[1][1]]['stage']:
        alt = next((x for x in admitted_queue[1:] if jobs[x[1]]['stage'] != jobs[wave[0][1]]['stage']), None)
        if alt is not None:
            wave[1] = alt
    wave_ids = {j for _, j in wave}
    spec_map = {s.job_id: s for s in specs_all}
    wave_specs = []
    for idx, (_, j) in enumerate(wave):
        if j in spec_map and j in wave_ids:
            s = spec_map[j]
            assigned = 'cpu' if canary else devices[idx % max(1, len(devices))]
            wave_specs.append(SessionJob(s.job_id, s.architecture, s.init_seed, s.order_seed, s.updates, session=s.session, device=assigned, extra=s.extra))
    return wave, wave_specs, plan


def execute(args, jobs, source, protocol):
    root = Path(args.work_root)
    output_dir = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if (root / 'DESIGN.json').exists():
            raise ValueError('use an empty work root to import FULL, or omit --resume for in-place continuation')
        verify_session_archive(
            args.resume,
            expected_source_commit=source,
            require_sidecar_sha256=False,
            required_members=['DESIGN.json', 'CAMPAIGN.json'],
        )
        safe_extract_planthir_archive(args.resume, root)
    if not args.canary:
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=HERE, text=True)
        if dirty.strip():
            raise ValueError('production requires clean tracked sources')
        if torch.cuda.device_count() < 2:
            raise ValueError('owner campaign requires two GPUs')
    full_hash = None
    if args.full is not None:
        with open(args.full, 'rb') as f:
            sha = hashlib.sha256()
            for block in iter(lambda: f.read(4 * 1024**2), b''):
                sha.update(block)
        full_hash = sha.hexdigest()
        if full_hash != '7d006b2e72c18ed5a763d873cb23c5bba053e6ce8426c79ad13cb1857c2bb4a1':
            raise ValueError('Y4 FULL differs from the verified owner baseline')

    split_hashes = split_design_hashes(jobs)
    design = dict(
        y4_full_sha256=full_hash,
        source=source,
        protocol=protocol,
        scorer_version=SCORER_VERSION,
        split_hashes=split_hashes,
        canary=args.canary,
        jobs=jobs,
        ledger=registry_manifest(jobs),
        baseline_diff=diff_against_baseline(jobs),
    )
    publish(root / 'DESIGN.json', design)
    preflight_report = checks()
    preflight_report['hardware_budget'] = benchmark_hardware(canary=args.canary)
    publish(root / 'PREFLIGHT.json', preflight_report)

    devices = ['cpu'] if args.canary else ['cuda:0', 'cuda:1']
    start = time.monotonic()
    reserve = min(30.0, float(args.minutes) * 0.15)
    os.environ['AVENDI_Y5_STOP_EPOCH'] = str(time.time() + 60.0 * (float(args.minutes) - reserve))
    coordinator = CampaignOrchestrator(
        campaign_id='avendi_y5',
        source_commit=source,
        work_dir=root / 'coordinator',
        max_minutes=args.minutes,
        reserve_minutes=reserve,
        min_disk_bytes=0 if args.canary else 1024**3,
    )
    rows = {s: {} for s in ('dev', 'confirm')}
    for split, bucket in rows.items():
        for jid, job in jobs.items():
            r = restored(root, split, job, source, protocol, args.canary)
            if r:
                bucket[jid] = r

    errors = []
    admission = []
    attempted_dev = set()
    lock_doc = None

    # --- PASS 1: DEVELOPMENT ONLY (global confirmation embargo strictly active) ---
    if _confirm_lock_path(root).exists():
        # Verify existing lock consistency against dev receipts on resume after Pass 1.
        lock_doc = seal_global_confirmation(root, jobs, rows['dev'], source, protocol, split_hashes, canary=args.canary)
    else:
        while True:
            queue = [
                ('dev', j)
                for j, v in jobs.items()
                if j not in rows['dev'] and all(p in rows['dev'] for p in v['depends']) and ('dev', j) not in attempted_dev
            ]
            if not queue:
                break
            queue.sort(key=lambda x: (bool(jobs[x[1]]['updates']), sum(jobs[k]['stage'] == jobs[x[1]]['stage'] for k in rows['dev']), x[1]))
            if time.monotonic() - start >= (float(args.minutes) - reserve) * 60.0:
                break
            wave, specs, plan = _admit_wave_nonstarving(coordinator, queue, jobs, rows, devices, root, output_dir, args.canary, start, float(args.minutes))
            admission.append(plan)
            if not specs:
                break
            attempted_dev.update(wave)
            def command_dev(s, _wroot):
                flags = ['--canary'] if args.canary else []
                full = ['--full', str(args.full)] if args.full else []
                return (
                    sys.executable, '-m', 'experiments.avendi_y5.campaign',
                    '--worker', s.job_id, '--split', 'dev', '--device', str(s.device),
                    '--work-root', str(root), *flags, *full, '--inputs', *[str(p) for p in args.inputs],
                )
            wave_id = f'dev-{time.time_ns()}'
            runner = CampaignOrchestrator(campaign_id='avendi_y5', source_commit=source, work_dir=root / 'waves' / wave_id)
            try:
                runner.run_waves(specs, command_builder=command_dev, devices=devices)
            except Exception as exc:
                errors.append(dict(split='dev', wave=wave_id, error=f'{type(exc).__name__}: {exc}'))
            for _, jid in wave:
                row = restored(root, 'dev', jobs[jid], source, protocol, args.canary)
                if row is None:
                    errors.append(dict(split='dev', job=jid, wave=wave_id))
                else:
                    rows['dev'][jid] = row
            _compact_ready_leaf_jobs(root, jobs, rows['dev'], args.canary)

        # Seal global confirmation ONLY when 100% of dev jobs resolved cleanly without pending/crashed jobs.
        if set(rows['dev']) == set(jobs) and not errors:
            lock_doc = seal_global_confirmation(root, jobs, rows['dev'], source, protocol, split_hashes, canary=args.canary)

    # --- PASS 2: CONFIRMATION ONLY (strictly gated behind sealed CONFIRM_LOCK.json) ---
    attempted_confirm = set()
    if lock_doc is not None:
        eligible_ids = sorted(lock_doc.get('eligible', {}))
        if any(j not in lock_doc['eligible'] for j in rows['confirm']):
            raise ValueError('confirm receipt outside frozen global CONFIRM_LOCK')
        while True:
            c_queue = [
                ('confirm', j)
                for j in eligible_ids
                if j not in rows['confirm'] and ('confirm', j) not in attempted_confirm
            ]
            if not c_queue:
                break
            if time.monotonic() - start >= (float(args.minutes) - reserve) * 60.0:
                break
            wave, specs, plan = _admit_wave_nonstarving(coordinator, c_queue, jobs, rows, devices, root, output_dir, args.canary, start, float(args.minutes))
            admission.append(plan)
            if not specs:
                break
            attempted_confirm.update(wave)
            def command_confirm(s, _wroot):
                flags = ['--canary'] if args.canary else []
                full = ['--full', str(args.full)] if args.full else []
                return (
                    sys.executable, '-m', 'experiments.avendi_y5.campaign',
                    '--worker', s.job_id, '--split', 'confirm', '--device', str(s.device),
                    '--work-root', str(root), *flags, *full, '--inputs', *[str(p) for p in args.inputs],
                )
            wave_id = f'confirm-{time.time_ns()}'
            runner = CampaignOrchestrator(campaign_id='avendi_y5', source_commit=source, work_dir=root / 'waves' / wave_id)
            try:
                runner.run_waves(specs, command_builder=command_confirm, devices=devices)
            except Exception as exc:
                errors.append(dict(split='confirm', wave=wave_id, error=f'{type(exc).__name__}: {exc}'))
            for _, jid in wave:
                row = restored(root, 'confirm', jobs[jid], source, protocol, args.canary)
                if row is None:
                    errors.append(dict(split='confirm', job=jid, wave=wave_id))
                else:
                    rows['confirm'][jid] = row

    pending = sorted(set(jobs) - set(rows['dev']))
    if lock_doc is not None:
        confirm_pending = sorted(j for j in lock_doc['eligible'] if j not in rows['confirm'])
    else:
        confirm_pending = sorted(j for j in jobs if jobs[j].get('role', 'primary') != 'diagnostic' and j not in rows['confirm'])

    pending_queue = {
        'schema': 'avendi-y5-pending-queue-v1',
        'confirm_locked': lock_doc is not None,
        'confirm_embargo_status': 'SEALED_GLOBAL_COHORT' if lock_doc is not None else 'EMBARGOED_PENDING_DEV',
        'pending_dev': pending,
        'pending_confirm': confirm_pending,
        'errors': errors,
    }
    (root / 'PENDING_QUEUE.json').write_text(json.dumps(sanitize_for_json(pending_queue), indent=2, allow_nan=False), encoding='utf-8')

    from .analysis import rescore_evidence, summarize
    analysis = summarize({
        s: {
            j: dict(r['result'], engineering_canary=args.canary, stage=jobs[j]['stage'], arm=jobs[j]['arm'], seed=jobs[j]['seed'])
            for j, r in bucket.items()
        }
        for s, bucket in rows.items()
    })
    dispositions = {j: r['status'] for j, r in rows['dev'].items()}
    status = (
        'PARTIAL'
        if pending or confirm_pending or errors or lock_doc is None
        else ('FINISHED_WITH_BLOCKS' if 'BLOCKED' in dispositions.values() else 'FINISHED')
    )
    data = dict(
        execution=status,
        scientific_status='SEE_PER_CONTRAST',
        engineering_canary=args.canary,
        global_confirm_lock_sealed=(lock_doc is not None),
        scorer_version=SCORER_VERSION,
        jobs=len(jobs),
        dev=len(rows['dev']),
        confirm=len(rows['confirm']),
        dispositions=dispositions,
        pending=pending,
        confirm_pending=confirm_pending,
        errors=errors,
        admission=admission,
    )
    for name, value in [('CAMPAIGN', data), ('ANALYSIS', analysis)]:
        (root / (name + '.json')).write_text(json.dumps(sanitize_for_json(value), indent=2, allow_nan=False), encoding='utf-8')

    rescore_doc = rescore_evidence(root, strict=False)
    summary = {name: root / name for name in ('DESIGN.json', 'PREFLIGHT.json', 'CAMPAIGN.json', 'ANALYSIS.json')}
    if (root / 'CONFIRM_LOCK.json').exists():
        summary['CONFIRM_LOCK.json'] = root / 'CONFIRM_LOCK.json'
    if (root / 'PENDING_QUEUE.json').exists():
        summary['PENDING_QUEUE.json'] = root / 'PENDING_QUEUE.json'
    evidence = dict(summary)
    for folder in ('dev', 'confirm', 'locks', 'exposure', 'waves'):
        if (root / folder).exists():
            for p in (root / folder).rglob('*'):
                if p.is_file() and (p.suffix in ('.json', '.log') or p.name.endswith('.txt')):
                    evidence[p.relative_to(root).as_posix()] = p
    for extra in ('CONFIRM_LOCK.planthir.json',):
        if (root / extra).exists():
            evidence[extra] = root / extra
    full = dict(evidence)
    if (root / 'weights').exists():
        full.update({p.relative_to(root).as_posix(): p for p in (root / 'weights').rglob('*') if p.is_file()})
    export_session_archives(
        root,
        output_dir,
        prefix='AVENDI_Y5',
        summary_files=summary,
        evidence_files=evidence,
        full_files=full,
        status=status,
        source_commit=source,
        protocol_sha256=protocol,
        rescore=rescore_doc,
    )
    print(json.dumps({k: data[k] for k in ('execution', 'jobs', 'dev', 'confirm', 'errors')}))
    return int(bool(errors))


def _plugin_model_factory(arch, model_cfg, seed):
    from experiments.avendi_y4.models import AvendiY4ByteModel, get_y4_byte_spec
    spec = get_y4_byte_spec('S3', cpu_canary=bool(model_cfg.get('cpu_canary', True)))
    return AvendiY4ByteModel(spec, seed=int(seed), use_aux=True)


def _plugin_dataset_factory(cfg, cpu_canary):
    seed = int(cfg.get('seed', 7))
    n = 1 if cpu_canary else int(cfg.get('num_families', 4))
    train_items = reader._make_items(seed, n, 'answer', 'fixed', 'train')[:4] if cpu_canary else reader._make_items(seed, n, 'answer', 'fixed', 'train')
    eval_items = reader._make_items(seed, 1 if cpu_canary else 2, 'answer', 'fixed', 'dev')[:4]
    conf_items = reader._make_items(seed, 1 if cpu_canary else 2, 'answer', 'fixed', 'confirm')[:4]
    probe_items = reader._make_items(seed, 1, 'answer', 'varied', 'transfer')[:4]
    def _wrap_worlds(items):
        return [
            {
                'world_id': x['world_id'],
                'family_id': x['family_id'],
                'canon_sig': x['canon_sig'],
                'public_prompt': x['public_prompt'],
                'hidden': {'expected': x['gold'], 'expected_operation': 'ANSWER'},
            }
            for x in items
        ]
    ds_id = digest([x['canon_sig'] for x in train_items])
    train_data = SimpleNamespace(
        items=tuple(train_items),
        data_identity=ds_id,
        dataset_name=str(cfg.get('dataset_name', 'avendi_y5_reader')),
        manifest={'data_identity': ds_id, 'count': len(train_items)},
    )
    return {
        'train_data': train_data,
        'train_worlds': tuple(_wrap_worlds(train_items)),
        'eval_worlds': tuple(_wrap_worlds(eval_items)),
        'multihop_worlds': tuple(_wrap_worlds(conf_items)),
        'probe_train_worlds': tuple(_wrap_worlds(probe_items)),
    }


def _plugin_loss_fn(model, batch, recipe):
    obj = getattr(recipe, 'objective', 'ce')
    return reader.loss(model, batch, objective=str(obj))


def register_planthir_backends():
    return ExperimentModule(
        backend_id='avendi_y5',
        backend_version='5.1',
        program='AVENDI-Y5',
        architectures=('avendi_byte_s3', 'avendi_byte_s3_pointer'),
        objectives=('ce', 'pointer', 'sham'),
        model_factory=_plugin_model_factory,
        dataset_factory=_plugin_dataset_factory,
        loss_fn=_plugin_loss_fn,
        kernel_factory=lambda: TypedByteSequenceKernel(structural_hash_fn=lambda w: w.get('canon_sig')),
        default_dataset_config={'dataset_name': 'avendi_y5_reader', 'num_families': 4, 'seed': 7},
        default_model_config={'cpu_canary': True, 'use_aux': True},
        treatment_architecture='avendi_byte_s3_pointer',
        treatment_model_config={'cpu_canary': True, 'use_aux': True},
        supports_fp16=True,
    ).register()


AVENDI_Y5_BACKEND = register_planthir_backends()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--split', choices=['dev', 'confirm'], default='dev')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--work-root', type=Path, default=Path('work/avendi_y5'))
    parser.add_argument('--output', type=Path, default=Path('.'))
    parser.add_argument('--inputs', nargs='*', type=Path, default=[Path('/kaggle/input')])
    parser.add_argument('--full', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='exercise representative paths with tiny models and 1-2 updates')
    parser.add_argument('--minutes', type=float, default=float(os.environ.get('PLANTHIR_SESSION_MINUTES', 480)))
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    args.work_root = args.work_root.resolve()
    if args.preflight:
        print(json.dumps(sanitize_for_json({**checks(), 'hardware_budget': benchmark_hardware(canary=True)})))
        return 0
    jobs = registry()
    if args.smoke:
        args.canary = True
        wanted = {
            'AB-reader-fixed-ce-s7', 'AB-answer-fixed-ce-s7', 'F-new_cap-s7', 'D-informative-s7', 'E-relevant-s7',
            'C-baseline-s7', 'C-scout-s7', 'C-iit-s7', next(k for k, v in jobs.items() if v['stage'] == 'G'),
        }
        jobs = {k: v for k, v in jobs.items() if k in wanted}
    if args.inventory:
        print(json.dumps(registry_manifest(jobs), indent=2))
        return 0
    source, protocol = identity()
    if not args.full:
        matches = {
            p.resolve()
            for root in args.inputs
            if root.exists()
            for p in ([root] if root.is_file() and root.name == 'AVENDI_Y4_FULL.zip' else root.rglob('AVENDI_Y4_FULL.zip') if root.is_dir() else [])
        }
        if len(matches) > 1:
            raise ValueError('multiple Y4 archives, select --full')
        args.full = next(iter(matches), None)
    if args.worker:
        worker(args, jobs, source, protocol)
        return 0
    if os.environ.get('PLANTHIR_RESUME_INPUT') and not args.resume:
        args.resume = Path(os.environ['PLANTHIR_RESUME_INPUT'])
    return execute(args, jobs, source, protocol)


if __name__ == '__main__':
    raise SystemExit(main())
