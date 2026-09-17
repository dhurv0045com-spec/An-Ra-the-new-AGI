"""Bounded single-update engineering check. Run explicitly; no import side effects."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
FROZEN = Path('C:/Users/ankit/cyr012-frozen')
PARENT = Path('C:/Users/ankit/cyr012-evidence/full01/compact/checkpoints/FINAL/model.bin')
OUT = Path('C:/Users/ankit/cyr012-evidence/one-step01')
PARENT_SHA = '94e82920eaf03089630b85b9061bd0fa680b629c0bcd526e58bf301712127a99'


def main():
    started = time.monotonic()
    sys.path.insert(0, str(FROZEN))
    import psutil
    import torch
    from anra_v5 import cyr_gpu011_run as runner
    from anra_v5 import cyr_gpu006_run as legacy
    from v5_experiments import cyr_gpu011 as core
    from v5_model.core import packed_layout

    helper_path = REPO / 'v5_experiments/one_step_pilot.py'
    spec = importlib.util.spec_from_file_location('pilot_fixture', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(PARENT) == PARENT_SHA, 'parent hash mismatch'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(8121)
    device = torch.device('cuda')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.5, 3 * 2**30 / total))
    resources = []

    def guard(preflight=False):
        temperature = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            text=True, timeout=10).strip().splitlines()[0])
        ram = psutil.virtual_memory().available / 2**30
        free = torch.cuda.mem_get_info()[0] / 2**30
        elapsed = time.monotonic() - started
        resources.append(dict(seconds=elapsed, temperature=temperature, ram_gib=ram, gpu_free_gib=free))
        if temperature >= 85 or ram < (3 if preflight else 2) or elapsed > 120:
            raise RuntimeError('resource/time guard')
        if preflight and free < 4:
            raise RuntimeError('insufficient GPU headroom')

    guard(True)
    manifest = FROZEN / 'docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json'
    data = core.load_ark002b_manifest(manifest)
    fixture = helper.make_fixture(data)
    assert len(fixture['train']) == 64 and len(fixture['holdout']) == 100
    assert not ({tuple(r['canonical_pair']) for r in fixture['train']} &
                {tuple(r['canonical_pair']) for r in fixture['holdout']})
    OUT.mkdir(exist_ok=False)

    def save(name, value):
        (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    save('FIXTURE.json', fixture)
    sources = {str(p): sha(p) for p in (Path(__file__), helper_path,
        REPO / 'experiments/CYR-GPU-012/ONE_STEP_PLAN.md', manifest)}
    for module in list(sys.modules.values()):
        source = getattr(module, '__file__', None)
        if source and Path(source).suffix == '.py' and Path(source).resolve().is_relative_to(FROZEN):
            sources[str(Path(source).resolve())] = sha(source)
    provenance = dict(purpose='ONE_STEP_ENGINEERING_NOT_AGI', parent_sha256=PARENT_SHA,
        source_file_sha256=sources, fixture_file_sha256=sha(OUT / 'FIXTURE.json'),
        torch_version=torch.__version__, device=torch.cuda.get_device_name(0),
        optimizer=dict(name='Adam', lr=1e-5, betas=[.9, .999], eps=1e-8, weight_decay=0),
        metric='TEACHER_FORCED_WHOLE_ANSWER_WITH_EOS',
        frozen_commit=subprocess.check_output(['git', '-C', str(FROZEN), 'rev-parse', 'HEAD'], text=True).strip(),
        repo_dirty=bool(subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True)))
    save('PREREGISTRATION.json', provenance)
    model = optimizer = state = before_weights = None
    try:
        model = runner._build_model(core.research_small_spec(19), 3301, torch=torch, device=device)
        state = torch.load(PARENT, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        assert all(torch.equal(v.detach().cpu(), state[k]) for k, v in model.state_dict().items())
        before_weights = torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])
        tokenizer = core.CompactCharTokenizer()

        def forward(rows):
            tokens, segments, eligible, _ = legacy.render_batch(tokenizer, rows,
                torch=torch, device=device, special=tokenizer.special)
            positions, mask = packed_layout(segments, torch_module=torch)
            logits = model(tokens, positions, mask)
            return logits[:, :-1], tokens[:, 1:], eligible[:, 1:]

        def evaluate():
            guard()
            model.eval()
            with torch.no_grad():
                logits, targets, eligible = forward(fixture['holdout'])
                predictions = logits.argmax(-1)
                predicted = [p[m].tolist() for p, m in zip(predictions, eligible)]
                expected = [t[m].tolist() for t, m in zip(targets, eligible)]
            flags = helper.exact_flags(predicted, expected)
            return dict(correct=sum(flags), total=len(flags), accuracy=sum(flags) / len(flags),
                rows=[dict(world_id=r['world_id'], prompt=r['prompt'], answer=r['answer'],
                           predicted_answer_token_ids=p, expected_answer_token_ids=e, exact=ok)
                      for r, p, e, ok in zip(fixture['holdout'], predicted, expected, flags)])

        before = evaluate()
        save('BEFORE.json', before)
        guard()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        logits, targets, eligible = forward(fixture['train'])
        loss = torch.nn.functional.cross_entropy(logits[eligible], targets[eligible])
        assert torch.isfinite(loss).item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(int(s['step'].item()) == 1 for s in optimizer.state.values())
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        optimizer.zero_grad(set_to_none=True)
        loss_value, grad_norm_value = float(loss.detach()), float(grad_norm)
        del logits, targets, eligible, loss, grad_norm
        after = evaluate()
        save('AFTER.json', after)
        after_weights = torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])
        changed = int((after_weights != before_weights).sum().item())
        assert changed > 0
        assert sha(PARENT) == PARENT_SHA
        assert all(sha(path) == digest for path, digest in sources.items())
        delta = after['correct'] - before['correct']
        report = dict(status='COMPLETE', updates=1, training_rows=64,
            verdict='DESCRIPTIVE_GAIN_ONLY' if delta > 0 else 'NO_ONE_STEP_ACCURACY_GAIN',
            before_correct=before['correct'], after_correct=after['correct'], total=100,
            delta_correct=delta, train_loss=loss_value, grad_norm=grad_norm_value,
            changed_parameters=changed, parent_unchanged=True, production_promotion=False,
            agi_claim=False, wall_seconds=time.monotonic()-started,
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            raw_file_sha256={n: sha(OUT/n) for n in ('FIXTURE.json', 'BEFORE.json', 'AFTER.json', 'PREREGISTRATION.json')},
            limitations=['Single step and fixture, not a dose test', 'Teacher-forced metric',
                        'Prior diagnostic overlap not excluded', 'No replication or mechanism claim'])
        save('RESULT.json', report)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        save('FAILURE.json', dict(status='INVALID_OR_INCOMPLETE', error=repr(exc)))
        raise
    finally:
        model = optimizer = state = before_weights = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        save('RESOURCES.json', dict(samples=resources,
            free_gpu_gib_after_release=torch.cuda.mem_get_info()[0]/2**30))


if __name__ == '__main__':
    main()
