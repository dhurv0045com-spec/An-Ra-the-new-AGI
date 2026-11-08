"""Frozen, outcome-independent V7 diagnostic study. Import never launches work."""
import math
from .codec import digest

METHODS = ('uniform', 'blocked', 'interleaved')
SEEDS = (7, 19, 41)
EPOCHS = 4


def protocol():
    return {'schema': 'elendril-v7-protocol-v1', 'methods': list(METHODS), 'seeds': list(SEEDS),
        'epochs': EPOCHS, 'profile': 'S3', 'representation': 'utf8-raw',
        'train_seed': 606, 'train_bytes': 2_200_000,
        'development_seed': 707, 'confirmation_seed': 909, 'evaluation_worlds': 112,
        'monitor_worlds_per_kind': 4, 'train_monitor_worlds_per_kind': 4,
        'monitor_selection':'first per kind sorted by world_id,case_id',
        'training_monitor_epochs':[1,2,4], 'development_epochs':[1,2,4],
        'untrained_monitor':True, 'batch_size': 16, 'micro_batch_size': 2,
        'lr': .0003, 'schedule': 'cosine', 'objective': 'ce', 'precision': 'fp32',
        'monitor_epoch_fractions': [[1,8],[1,4],[3,8],[1,2],[1,1],[2,1],[4,1]],
        'checkpoint_interval': 64, 'maximum_minutes': 330,
        'selection': 'All nine predeclared runs; no outcome-based selection or scaling.',
        'primary': ['nonnull_correct', 'needed_queried', 'both_correct_payload_pairs',
                    'answer_entropy_bits', 'correct_abstentions'],
        'promotion': 'NONE: development study; no intelligence or causal mechanism claim.'}


PROTOCOL_SHA = digest(protocol())


def milestones(examples):
    if type(examples) is not int or examples < 1:
        raise ValueError('nonempty sample count required')
    per_epoch = math.ceil(examples / 16)
    return tuple(sorted({math.ceil(per_epoch * a / b)
                         for a,b in protocol()['monitor_epoch_fractions']}))


def jobs():
    # Paired seed blocks first: budget truncation must not preferentially finish
    # every seed of a single method. Report incomplete blocks explicitly.
    return [{'job_id': f'{method}-s{seed}', 'method': method, 'seed': seed}
            for seed in SEEDS for method in METHODS]


def validate_job(job):
    if job not in jobs():
        raise ValueError('job outside frozen V7 design')
