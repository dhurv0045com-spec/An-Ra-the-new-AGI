"""Checkpoint custody, paired exposure and immutable result receipts for Y5."""
import hashlib
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from anra_v5.palantiri import GenericExperimentTrainer, GenericTrainRecipe, prune_completed_job_artifacts
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
    if not isinstance(doc, dict) or 'content' not in doc or 'sha256' not in doc or digest(doc['content']) != doc['sha256']:
        raise ValueError(f'damaged receipt: {path}')
    return doc['content']


def verify_and_load_weights(weights_root, job_id, sha, *, expected_source=None, expected_updates=None):
    """Verify full CheckpointStore object OR leaf-compacted (weights-only) object strictly."""
    weights_root = Path(weights_root)
    obj_dir = weights_root / job_id / 'objects' / sha
    comp_rec_path = obj_dir / 'COMPACTION_RECEIPT.json'
    if not (comp_rec_path.is_file() and not (obj_dir / 'optimizer.bin').exists()):
        state, payloads = CheckpointStore(weights_root, job_id).restore(sha)
        if expected_updates is not None and state.global_update != expected_updates:
            raise ValueError('restored update or source identity mismatch')
        if expected_source is not None and state.identities.source_commit != expected_source:
            raise ValueError('restored update or source identity mismatch')
        return state.global_update, state.identities.source_commit, payloads['model.bin']
    comp = json.loads(comp_rec_path.read_text(encoding='utf-8'))
    if comp.get('object_sha') != sha or comp.get('stripped_components') != ['optimizer.bin']:
        raise ValueError('invalid compaction receipt')
    manifest_bytes = (obj_dir / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != sha:
        raise ValueError('checkpoint manifest hash mismatch')
    manifest = json.loads(manifest_bytes)
    if manifest.get('lineage_id') != job_id:
        raise ValueError('checkpoint lineage mismatch')
    st = manifest['state']
    upd = int(st['global_update'])
    src = str(st['identities']['source_commit'])
    if expected_updates is not None and upd != expected_updates:
        raise ValueError('restored update or source identity mismatch')
    if expected_source is not None and src != expected_source:
        raise ValueError('restored update or source identity mismatch')
    payloads = {}
    for item in manifest['components']:
        name = item['name']
        if name == 'optimizer.bin':
            continue
        raw = (obj_dir / name).read_bytes()
        if len(raw) != int(item['byte_size']) or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'checkpoint component corrupt: {name}')
        payloads[name] = raw
    if 'model.bin' not in payloads:
        raise ValueError('missing model.bin in compacted checkpoint')
    return upd, src, payloads['model.bin']


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
        _, _, model_bytes = verify_and_load_weights(
            self.root / 'weights',
            job_id,
            info['checkpoint_sha256'],
            expected_source=self.source,
            expected_updates=info['updates'],
        )
        weights = torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=True)
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
            crash_spec = os.environ.get('AVENDI_Y5_CRASH_POINT', '').strip()
            if crash_spec and crash_spec in (f"{job['id']}:{u}", f"stage:{job['stage']}:{u}", job['id']):
                sha = trainer.save(prune_old=False)
                publish(checkpoints / f'{u}.json', dict(update=u, checkpoint_sha256=sha, loss=meta['loss']))
                raise RuntimeError(f'injected crash at {crash_spec}')
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


def prune_job_checkpoints(
    root,
    job_id,
    *,
    completed_updates,
    full_job_updates,
    downstream_children=(),
    strip_optimizer=False,
):
    """Dependency-aware compaction of finished lineage objects after verifying final checkpoint restore."""
    lin_dir = Path(root) / 'weights' / job_id
    store = CheckpointStore(Path(root) / 'weights', job_id)
    latest = store.latest_sha256()
    if latest is not None:
        comp_rec = lin_dir / 'objects' / latest / 'COMPACTION_RECEIPT.json'
        if comp_rec.is_file() and not (lin_dir / 'objects' / latest / 'optimizer.bin').exists():
            return {'job_root': str(lin_dir), 'already_compacted': True, 'optimizer_stripped': True}
        verify_and_load_weights(
            Path(root) / 'weights',
            job_id,
            latest,
            expected_updates=int(completed_updates),
        )
    return prune_completed_job_artifacts(
        lin_dir,
        keep_latest=True,
        keep_milestones=True,
        strip_final_optimizer=bool(strip_optimizer and not downstream_children),
        preserve_final_optimizer=not bool(strip_optimizer and not downstream_children),
        child_job_ids=list(downstream_children),
        completed_updates=int(completed_updates),
        full_job_updates=int(full_job_updates),
    )
