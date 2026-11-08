"""V7.1 source-bound CE and intermediate-supervision trainer factory.

Importing this module does not run training. Pointer/copy is not implemented;
it must not be attributed to the ordinary language-model architecture.
"""
from collections import Counter
import math
from pathlib import Path
import re

from .foundation_training import Recipe
from .representation import UTF8
from .training import configure_device
from .v6_data import transform_dataset
from .v6_registry import MODEL_PROFILES
from .v7_training import V7Trainer
from .v71_protocol import PROTOCOL_SHA


def create_trainer(data, job, *, source, root, device='cuda:0', resume=False,
                   cpu_canary=False, spec_override=None, updates_override=None,
                   transition=None, parent_store=None):
    required = {'job_id', 'profile', 'method', 'seed', 'epochs', 'mechanism'}
    if (not isinstance(job, dict) or set(job) != required
            or not isinstance(job['job_id'], str)
            or re.fullmatch(r'[A-Za-z0-9_-]{1,90}', job['job_id']) is None
            or job['profile'] not in MODEL_PROFILES or job['method'] not in ('uniform', 'blocked')
            or type(job['seed']) is not int or job['seed'] not in (7, 19, 41)
            or type(job['epochs']) is not int or job['epochs'] not in (2, 4)
            or job['mechanism'] not in ('ce', 'intermediate', 'pointer')):
        raise ValueError('invalid V7.1 job')
    if job['mechanism'] == 'pointer':
        raise NotImplementedError('pointer architecture is not implemented or checkpoint-qualified')
    if data.split != 'train' or data.manifest.get('scratch') or data.manifest.get('augmentation'):
        raise ValueError('V7.1 requires untransformed training data')
    if device != 'cuda:0' and not (device == 'cpu' and cpu_canary is True):
        raise ValueError('V7.1 production requires isolated CUDA; no CPU fallback')
    if device != 'cpu' and (cpu_canary or spec_override is not None or updates_override is not None):
        raise ValueError('overrides are reserved for bounded CPU canaries')
    transformed = transform_dataset(data, scratch=True) if job['mechanism'] == 'intermediate' else data
    samples = transformed.samples
    spec = spec_override or MODEL_PROFILES[job['profile']]
    batch = min(16, len(samples))
    total_updates = math.ceil(len(samples) / batch) * job['epochs']
    if device == 'cpu':
        if len(samples) > 32 or spec.parameter_receipt().total > 100_000:
            raise ValueError('CPU canary exceeds sample or parameter bound')
        if type(updates_override) is not int or not 1 <= updates_override <= 8:
            raise ValueError('CPU canary requires 1..8 updates')
        total_updates = updates_override
    if total_updates > 9216:
        raise ValueError('training plan exceeds engine update bound')
    counts = Counter(e.target for e in transformed.examples)
    recipe = Recipe(seed=job['seed'], total_updates=total_updates, batch_size=batch,
                    micro_batch_size=min(2, batch), lr=3e-4, objective='ce',
                    sampling='uniform', schedule='cosine', precision='fp32',
                    activation_checkpointing=False)
    provenance = {**transformed.manifest, 'v71_job': dict(job), 'v71_protocol_sha256':PROTOCOL_SHA,
                  'v71_parent_dataset': data.identity,
                  'v71_supervision': {'mechanism': job['mechanism'],
                    'worlds': len(transformed.cases), 'samples': len(samples),
                    'target_actions': dict(sorted(counts.items())),
                    'supervised_tokens_per_epoch': sum(sum(s.eligible) for s in samples),
                    'input_positions_per_epoch': sum(len(s.tokens) for s in samples),
                    'ce_parent_samples': len(data.samples),
                    'ce_parent_supervised_tokens': sum(sum(s.eligible) for s in data.samples),
                    'compute_matched_to_ce': job['mechanism'] == 'ce',
                    'qualification': 'intermediate adds supervised route actions and extra compute' if job['mechanism'] == 'intermediate' else 'ordinary language-model cross entropy'}}
    configure_device(device)
    return V7Trainer(samples, spec=spec, recipe=recipe, source_commit=source,
                     root=Path(root), lineage=job['job_id'], data_provenance=provenance,
                     tokenizer_manifest=UTF8.manifest.record(), device=device,
                     method=job['method'], resume=resume, transition=transition,
                     parent_store=parent_store)
