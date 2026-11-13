"""S20 failure investigation. Arms are fixed before observing new GPU results."""
from .codec import digest

SEEDS = (7, 19, 41)
PHASES = ('optimization', 'representation')


def _job(arm, seed, order=None):
    order = seed if order is None else order
    return dict(job_id=f'S20-{arm}-i{seed}-o{order}', profile='S20',
                method='blocked', mechanism='ce', epochs=4, seed=seed,
                order_seed=order, lr=1e-4 if arm == 'low-lr' else 3e-4, arm=arm)


def jobs(phase):
    if phase not in PHASES:
        raise ValueError('unknown S20 diagnosis phase')
    rows = [_job('baseline', s) for s in SEEDS]
    if phase == 'optimization':
        rows += [_job('low-lr', s) for s in SEEDS]
        rows += [_job('crossed', 7, 19), _job('crossed', 19, 7)]
    else:
        rows += [_job('row-order', s) for s in SEEDS]
    return rows


def validate(job, phase):
    if job not in jobs(phase):
        raise ValueError('job outside preregistered S20 diagnosis phase')


def protocol():
    return dict(schema='elendril-v717-s20-diagnosis-v1',
        jobs={p: jobs(p) for p in PHASES}, train_seed=606,
        validation_seed=707, confirmation_seed=909, train_bytes=2200000,
        maximum_session_minutes=330, epochs=4, batch_size=16, micro_batch_size=2,
        model='S20, fresh weights, UTF8, fp32, unchanged neural architecture',
        hypotheses={
            'optimization': 'Lower LR improves train fitting and held-out behavior across paired initializations.',
            'seed-coupling': 'Crossed initial-weight/order seeds localize sensitivity to initialization versus sample order; exploratory 2x2 only.',
            'representation': 'Deterministically reversing evidence rows in half of training worlds improves held-out reordered evidence without changing targets.'},
        falsifiers='No consistent paired effect, or train-only gains without held-out improvement, do not support a generalization repair.',
        measurements='Initial/epochs 1,2,4: full training and validation teacher-forced CE/accuracy by action, value, EOS and kind; bounded per-kind autoregressive decoding; initial independent gradient accumulation parity.',
        evaluation='Inherited raw interactive base+hop-3 at initial/epochs1,2; full unchanged stress suite and separate grammar decoder at epoch4. Confirmation final only.',
        interpretation='Teacher-forced success is not interactive success. Gradient parity is an engineering control, not evidence of cognition. Repeated baseline across phases is not independent replication.',
        admission='Both isolated T4s must pass S20 production-shaped update/recovery and full diagnostic timing. Whole declared phase fits with 2x margin or NO_GO; no silent pruning.',
        limitation='Fixed four-epoch exposure; neither larger-data nor longer-training hypothesis is tested. Shared procedural authorship, no external sealed evaluation. No S3 research runs.',
        scientific_status='CAUSE UNRESOLVED; no automatic promotion')


PROTOCOL_SHA = digest(protocol())
