"""Separate initialization/order seeds without changing the qualified update engine."""
from dataclasses import replace
import math
from pathlib import Path

from .codec import digest
from .foundation_training import Recipe
from .representation import UTF8
from .training import configure_device
from .v6_data import _dataset
from .v6_registry import MODEL_PROFILES
from .v7_training import V7Trainer, ordered_indices
from .v717_protocol import PROTOCOL_SHA, PHASES, jobs


def prepare_variant(data):
    cases = tuple(replace(c, surface=replace(c.surface, order='reverse'))
                  if int(digest(['v717-row-order', c.world_id]), 16) % 2 else c
                  for c in data.cases)
    result = _dataset('train', cases, seed=data.manifest['seed'], parent=data.identity)
    if (len(result.samples) != len(data.samples)
            or [e.target for e in result.examples] != [e.target for e in data.examples]
            or [s.world_id for s in result.samples] != [s.world_id for s in data.samples]):
        raise ValueError('row-order intervention changed worlds, exposure or targets')
    return result


class IndependentOrderTrainer(V7Trainer):
    def __init__(self, *args, order_seed, **kwargs):
        self.order_seed = order_seed
        super().__init__(*args, **kwargs)

    def indices(self, update):
        per_epoch = math.ceil(len(self.samples) / self.recipe.batch_size)
        epoch, batch = divmod(update, per_epoch)
        if epoch not in self._epoch_orders:
            self._epoch_orders[epoch] = ordered_indices(self.samples, self.order_seed, epoch, self.method)
        start = batch * self.recipe.batch_size
        return self._epoch_orders[epoch][start:start + self.recipe.batch_size]


def create_trainer(data, job, *, source, root, resume=False, device='cuda:0',
                   cpu_canary=False, spec_override=None, updates_override=None):
    if not any(job in jobs(p) for p in PHASES):
        raise ValueError('unregistered S20 diagnosis job')
    if data.split != 'train' or data.manifest.get('scratch') or data.manifest.get('augmentation'):
        raise ValueError('unqualified data transformation')
    if device != 'cuda:0' and not (device == 'cpu' and cpu_canary is True):
        raise ValueError('production requires isolated CUDA')
    if device != 'cpu' and (cpu_canary or spec_override is not None or updates_override is not None):
        raise ValueError('CPU-only overrides')
    spec = spec_override or MODEL_PROFILES['S20']
    batch = min(16, len(data.samples))
    updates = math.ceil(len(data.samples) / batch) * 4
    if device == 'cpu':
        if (len(data.samples) > 32 or spec.parameter_receipt().total > 100000
                or type(updates_override) is not int or not 1 <= updates_override <= 8):
            raise ValueError('CPU mechanical canary exceeds bounds')
        updates = updates_override
    recipe = Recipe(seed=job['seed'], total_updates=updates, batch_size=batch,
        micro_batch_size=min(2, batch), lr=job['lr'], objective='ce',
        sampling='uniform', schedule='cosine', precision='fp32')
    configure_device(device)
    trainer = IndependentOrderTrainer(data.samples, spec=spec, recipe=recipe,
        source_commit=source, root=Path(root), lineage=job['job_id'],
        data_provenance={**data.manifest, 'v717_job': job,
                         'v717_protocol_sha256': PROTOCOL_SHA,
                         'v717_data_identity': data.identity},
        tokenizer_manifest=UTF8.manifest.record(), device=device,
        method=job['method'], order_seed=job['order_seed'], resume=resume)
    trainer.diagnostic_data = data
    return trainer
