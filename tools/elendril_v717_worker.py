"""Process-local V7.17 adapter to the qualified V7.1 checkpoint worker.

No historical trainer semantics are changed. Recovery children use this same
adapter, registry and factory, including independent order-seed provenance.
"""
import json
from pathlib import Path
import time

from anra_v5.elendril.v717_protocol import PROTOCOL_SHA, validate
from anra_v5.elendril.v717_training import create_trainer
from anra_v5.elendril.v6_data import read_dataset
from tools import elendril_v71_worker as inherited
from tools.elendril_v6_export import immutable_json

_evaluation = inherited.evaluation
_profile = inherited.profile


def create(data, job, *, source, root, resume=False, transition=None, parent_store=None):
    if transition is not None or parent_store is not None:
        raise ValueError('V7.17 requires fresh weights or exact same-job continuation')
    return create_trainer(data, job, source=source, root=root, resume=resume)


def measure(trainer, validation, job, *, initial=False):
    from anra_v5.elendril.v717_measurement import diagnostic, representative_indices, gradient_probe
    before = inherited._weights(trainer.model)
    started = time.monotonic()
    result = dict(checkpoint=trainer.backend.binding.checkpoint_sha256,
                  weight_sha256=before, update=trainer.state.global_update, job=job)
    for key, data in (('train', trainer.diagnostic_data), ('heldout', validation)):
        result[key] = dict(dataset_identity=data.identity,
            teacher=diagnostic(trainer, data, decode=False),
            decoded=diagnostic(trainer, data, indices=representative_indices(data, per_kind=16), decode=True))
    if initial:
        result['gradient_probe'] = gradient_probe(trainer)
        if result['gradient_probe'].get('status') != 'PASS':
            raise ValueError('independent gradient accumulation probe failed')
    if inherited._weights(trainer.model) != before:
        raise ValueError('diagnostic changed model weights')
    result['wall_seconds'] = time.monotonic() - started
    return result


def verify_measurement(path, job, *, checkpoint=None, update=None):
    import math
    row = json.loads(Path(path).read_bytes())
    if (row.get('job') != job or not isinstance(row.get('checkpoint'), str)
            or (checkpoint is not None and row['checkpoint'] != checkpoint)
            or (update is not None and row['update'] != update)):
        raise ValueError('measurement identity mismatch')
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite diagnostic')
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        if isinstance(value, list):
            for v in value: finite(v)
    finite(row)
    for key in ('train', 'heldout'):
        if not all(k in row[key] for k in ('dataset_identity', 'teacher', 'decoded')):
            raise ValueError('incomplete diagnostic')
        for part in ('teacher', 'decoded'):
            metric = row[key][part]
            if (metric.get('schema') != 'elendril-v717-measurement-v1'
                    or type(metric.get('examples')) is not int or metric['examples'] <= 0
                    or metric.get('audit', {}).get('conflicting_targets') != 0
                    or not metric.get('teacher_forced', {}).get('all', {}).get('tokens', 0)):
                raise ValueError('incomplete diagnostic metrics')
        if row[key]['decoded'].get('decoded', {}).get('status') != 'MEASURED':
            raise ValueError('missing decoded measurement')
    if row['update'] == 0 and row.get('gradient_probe', {}).get('status') != 'PASS':
        raise ValueError('missing initial gradient control')
    return row


def evaluation(trainer, suite, path, *, conditions=None, grammar=False):
    result = _evaluation(trainer, suite, path, conditions=conditions, grammar=grammar)
    if not grammar:
        measurement = path.parent / 'measurements' / path.name
        job = trainer.metadata['provenance']['v717_job']
        if not measurement.exists():
            validation = read_dataset(path.parents[3] / 'data' / 'validation.json')
            immutable_json(measurement, measure(trainer, validation, job,
                                               initial=trainer.state.global_update == 0))
        verify_measurement(measurement, job, checkpoint=result['checkpoint'], update=result['update'])
    return result


def profile(req, deadline):
    result = _profile(req, deadline, worker_module='tools.elendril_v717_worker')
    if time.monotonic() >= deadline:
        raise TimeoutError('diagnostic profile deadline')
    trainer = create(read_dataset(req['data_file']), req['job'], source=req['source'],
                     root=Path(req['checkpoint_root']) / 'measurement-only')
    inherited.bound(trainer)
    row = measure(trainer, read_dataset(Path(req['suite_file']).parent / 'validation.json'), req['job'], initial=True)
    immutable_json(Path(req['output']) / 'measurement-profile.json', row)
    result['measurement_seconds_per_audit'] = row['wall_seconds']
    result['gradient_probe'] = row['gradient_probe']
    result['qualification'] += '; independent gradient parity and full diagnostic timing; not a positive skill acquisition result'
    return result


def main(argv=None):
    inherited.PROTOCOL_SHA = PROTOCOL_SHA
    inherited.validate = validate
    inherited.create = create
    inherited.evaluation = evaluation
    inherited.profile = profile
    return inherited.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
