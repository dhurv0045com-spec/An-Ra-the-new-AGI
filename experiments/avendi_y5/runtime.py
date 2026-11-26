"""Checkpoint custody, paired exposure and immutable result receipts for Y5."""
import hashlib
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from anra_v5.palantiri import GenericExperimentTrainer, GenericTrainRecipe
from v5_training.checkpoint import CheckpointStore


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def publish(path, value):
    """Publish identical bytes on retry. Reject a second value at the same address."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encode(dict(content=value, sha256=digest(value)))
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_bytes(raw)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if raw != path.read_bytes():
                raise ValueError(f'conflicting immutable record: {path}')
    finally:
        tmp.unlink(missing_ok=True)


def read(path):
    doc = json.loads(Path(path).read_bytes())
    if digest(doc['content']) != doc['sha256']:
        raise ValueError(f'damaged receipt: {path}')
    return doc['content']


class Context:
    def __init__(self, root, job, split, device, canary, full, inputs, source):
        self.root, self.job, self.split = Path(root), job, split
        self.device, self.canary, self.full = device, canary, full
        self.inputs, self.source = inputs, source
        self.trained = None

    def previous(self, job_id):
        return read(self.root / 'dev' / f'{job_id}.json')['result']

    def load_parent(self, job_id, model):
        receipt = read(self.root / 'dev' / f'{job_id}.json')
        info = receipt['weights']
        if receipt['source'] != self.source or not info:
            raise ValueError(f'missing qualified source weights for {job_id}')
        state, payloads = CheckpointStore(self.root / 'weights', job_id).restore(info['checkpoint_sha256'])
        if state.global_update != info['updates'] or state.identities.source_commit != self.source:
            raise ValueError('restored update or source identity mismatch')
        weights = torch.load(io.BytesIO(payloads['model.bin']), map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.eval()
        return info

    def train(self, model, items, batches, loss_fn, lr, *, on_step=None, post_step=None):
        job = self.job
        if self.split == 'confirm':
            return dict(self.load_parent(job['id'], model), mode='restore_only')
        if not self.canary and len(batches) != job['updates']:
            raise ValueError('training exposure differs from registered updates')
        if not batches or any(len(b) != len(batches[0]) for b in batches):
            raise ValueError('invalid batch schedule')
        flat = [i for b in batches for i in b]
        if any(i < 0 or i >= len(items) for i in flat):
            raise ValueError('invalid exposure index')
        batch_size, count = len(batches[0]), len(batches)
        provenance = dict(items=digest(items), actual_batch_order=digest(batches), split='train',
                          parent=[self.previous(p).get('training', {}).get('checkpoint_sha256') for p in job['depends']])
        recipe = GenericTrainRecipe(seed=job['seed'], total_updates=count, batch_size=batch_size,
                                    micro_batch_size=batch_size, lr=lr, objective=job['arm'])
        store = CheckpointStore(self.root / 'weights', job['id'])
        # Each virtual slot appears exactly once in the native shuffled epoch.
        # Bind its contents to the registered paired exposure without changing the native sampler.
        trainer = GenericExperimentTrainer(backend_id='avendi_y5', model=model,
            model_spec=SimpleNamespace(architecture=type(model).__name__), recipe=recipe,
            loss_fn=lambda m, b, _: loss_fn(m, b), items=range(len(flat)), order_seed=job['seed'],
            data_provenance=provenance, source_commit=self.source, root=self.root / 'weights',
            lineage=job['id'], device=self.device, cpu_canary=self.canary,
            max_cpu_samples=4096, max_cpu_params=3000000, max_cpu_updates=4,
            resume=store.latest_sha256() is not None)
        slots = [None] * len(flat)
        for indices, selected in zip(trainer.plan, batches):
            for slot, index in zip(indices, selected):
                slots[slot] = items[index]
        trainer.items = slots
        milestones = {0, max(1, count//4), max(1, count//2), count}
        checkpoints = self.root / 'exposure' / job['id']
        def observe(u):
            if on_step is not None:
                model.eval()
                with torch.no_grad():
                    target = checkpoints / f'observation-{u}.json'
                    if not target.exists():
                        observation = on_step(u, model)
                        publish(target, dict(update=u, observation=observation))
        if trainer.state.global_update == 0:
            observe(0)
        while trainer.state.global_update < count:
            u = trainer.state.global_update + 1
            torch.manual_seed(job['seed'] * 100000 + u)
            meta = trainer.step(post_step_hook=(lambda tr, rec: post_step(rec['update'], tr.model)) if post_step else None)
            stop = time.time() >= float(os.environ.get('AVENDI_Y5_STOP_EPOCH','inf'))
            if u in milestones:
                observe(u)
            if u in milestones or stop:
                sha = trainer.save(prune_old=False)
                publish(checkpoints / f'{u}.json', dict(update=u, checkpoint_sha256=sha, loss=meta['loss']))
            if stop and u < count:
                raise TimeoutError('session reserve reached; checkpoint saved for continuation')
        state, payloads = store.restore(trainer.parent)
        weights = torch.load(io.BytesIO(payloads['model.bin']), map_location='cpu', weights_only=True)
        if state.global_update != count or any(not torch.equal(t.cpu(), weights[k]) for k,t in model.state_dict().items()):
            raise ValueError('evaluated weights differ from final checkpoint')
        model.eval()
        self.trained = dict(checkpoint_sha256=trainer.parent, updates=count, exposure=len(flat),
                            unique_items=len(set(flat)), dataset_items=len(items), **provenance)
        observations = [read(checkpoints / f'observation-{u}.json')['observation'] for u in sorted(milestones)] if on_step else []
        return dict(self.trained, milestones=observations)
