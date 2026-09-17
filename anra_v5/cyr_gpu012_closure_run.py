"""Bounded local operator: V11 compact only, full semantic dose, corrected probes."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

from anra_v5 import cyr_gpu011_run as inherited
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat
from v5_experiments import cyr_gpu011 as base
from v5_experiments import cyr_gpu012_closure as core

PAIRED = ('COMMUTATION_MATCHED_BAND', 'COMMUTATION_OOD_BAND')


def available_ram():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
    state = Status()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise RuntimeError('RAM inspection failed')
    return state.available / 2**30


def resources(torch, *, launch=False):
    free, total = torch.cuda.mem_get_info()
    temperature = float(subprocess.check_output([
        'nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0])
    ram = available_ram()
    if ram < (3 if launch else 2) or temperature >= 85 or (launch and free < 4 * 2**30):
        raise RuntimeError(f'RESOURCE_GUARD: RAM={ram}, GPU_free={free / 2**30}, C={temperature}')
    return dict(free_ram_gib=ram, gpu_free_gib=free / 2**30, temperature_c=temperature)


def corrected_battery(original, model, tokenizer, battery, **kwargs):
    result = original(model, tokenizer, battery, **kwargs)
    result['structural_flags'].pop('COMMUTATION_INVARIANCE', None)
    result['legacy_commuted_note'] = 'ORDER_ASYMMETRY_ONLY; not clean invariance'
    for name in PAIRED:
        rows = inherited._generate_texts(model, tokenizer, battery[name], **{
            key: kwargs[key] for key in ('torch', 'device', 'special')})
        result[name] = {**inherited._score_texts(rows), 'paired': core.score_pairs(rows)}
        result[name]['per_band'] = {}
        for band in sorted({r['band'] for r in battery[name]}):
            ids = {r['world_id'] for r in battery[name] if r['band'] == band}
            result[name]['per_band'][str(band)] = core.score_pairs([r for r in rows if r['world_id'] in ids])
        result['structural_flags'][name + '_BOTH_EXACT_G90'] = result[name]['paired']['both_exact_with_eos'] >= .9
        if kwargs.get('include_predictions'):
            result['candidate_free_predictions'][name] = inherited._prediction_receipt(rows)
    return result


def compact_summary(acquisition):
    battery = acquisition['reasoning_battery_final']
    out = {key: acquisition[key] for key in ('actual_real_tokens', 'ark_exposure_fraction',
           'batch_rows', 'g50_confirm_update', 'g90_confirm_update', 'm99_confirm_update',
           'row_presentations', 'status', 'updates')}
    out.update(parameters=987392,
               dev_controller_final_exact=acquisition['dev_controller_final']['complete_exact_with_valid_stop'],
               dev_measurement_standard_exact=battery['STANDARD']['complete_exact_with_valid_stop'],
               standard_digit_accuracy=battery['STANDARD']['digit_accuracy'],
               locality_both_exact=battery['LOCALITY']['structural']['both_exact'],
               locality_relation_consistency=battery['LOCALITY']['structural']['counterfactual_relation_consistency'])
    for key, name in (('carry_exact', 'CARRY'), ('commuted_exact', 'COMMUTED'),
                      ('three_digit_exact', 'THREE_DIGIT'), ('triple_add_exact', 'TRIPLE_ADD')):
        out[key] = battery[name]['complete_exact_with_valid_stop']
    out['corrected_commutation'] = {name: battery[name] for name in PAIRED}
    return out


def run(out, *, smoke=False):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU scientific fallback')
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('Frozen execution worktree is dirty')
    frozen = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(3 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    environment = inherited._environment(torch, device)
    environment.update(resources(torch, launch=True))
    manifest = 'docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json'
    original_blob = subprocess.check_output(['git', 'show', '0a97257:' + manifest], cwd=root)
    local_bytes = (root / manifest).read_bytes()
    if local_bytes.replace(b'\r\n', b'\n') != original_blob.replace(b'\r\n', b'\n'):
        raise RuntimeError('Actual ARK manifest bytes drifted')
    data = base.load_ark002b_manifest(root / manifest)
    battery = core.make_battery(data)
    tok = base.CompactCharTokenizer()
    spec = base.research_small_spec(19)
    prereg = dict(frozen_executable_sha=frozen, purpose='ENGINEERING_SMOKE' if smoke else 'SCIENTIFIC_FULL_EXPOSURE',
                 plan_sha256=inherited._sha256(root / 'experiments/CYR-GPU-012/PLAN.md'),
                 data_file_sha256=hashlib.sha256(local_bytes).hexdigest(),
                 model_seed=9911 if smoke else 3301, order_seed=4701,
                 batch_rows=64, target_updates=10 if smoke else 18000,
                 production_promotion_authorized=False)
    for name, value in [('ENVIRONMENT', environment), ('PREREGISTRATION', prereg), ('BATTERY', battery)]:
        inherited.write_json(out / (name + '.json'), value)
    start = time.monotonic()
    budget = 5 * 60 if smoke else 90 * 60
    original_update = inherited.legacy._one_update
    original_battery = inherited.reasoning_battery
    checks = []

    def guarded_update(**kwargs):
        if kwargs['update'] % 200 == 1:
            checks.append(dict(update=kwargs['update'], **resources(torch)))
        return original_update(**kwargs)

    def battery_call(*args, **kwargs):
        return corrected_battery(original_battery, *args, **kwargs)

    try:
        with canonical_optimizer_compat(), patch.object(inherited, 'reasoning_battery', battery_call), \
             patch.object(inherited.legacy, '_one_update', guarded_update), \
             patch.object(base, 'CYR11_MAX_UPDATES', 10 if smoke else 18000):
            acquisition = inherited.run_acquisition(
                label='COMPACT_BRIDGE', model_seed=prereg['model_seed'], order_seed=4701,
                spec=spec, tokenizer=tok, special=tok.special, batch_rows=64,
                data=data, battery=battery, torch=torch, device=device,
                deadline=start + budget, out=out / 'compact', include_verbal=False,
                stop_on_g90=False, progress=lambda s: print(s, flush=True))
            verdict = 'ENGINEERING_SMOKE_ONLY' if smoke else core.decide(acquisition)
            inherited.write_json(out / 'DECISION.json', dict(verdict=verdict, production_promotion_authorized=False))
            sealed = inherited._measure_sealed(
                checkpoint=acquisition['final_checkpoint'], spec=spec, seed=prereg['model_seed'],
                tokenizer=tok, special=tok.special,
                rows=data['sealed_reserved'][:2] if smoke else data['sealed_reserved'],
                torch=torch, device=device)
        wall = time.monotonic() - start
        environment.update(wall_seconds=wall, wall_minutes=wall / 60,
                           peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
        inherited.write_json(out / 'ENVIRONMENT.json', environment)
        inherited.write_json(out / 'SEALED.json', sealed)
        inherited.write_json(out / 'RESOURCES.json', checks)
        bundle = out / 'CYR_GPU_012_RESULTS.zip'
        paths = sorted(p for p in out.rglob('*.json') if 'checkpoints' not in p.parts)
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                z.write(p, p.relative_to(out).as_posix())
        with zipfile.ZipFile(bundle) as z:
            for name in z.namelist():
                json.loads(z.read(name))
        receipt = dict(schema='anra-cyr-gpu012-result-receipt/v1', experiment='CYR-GPU-012',
                       frozen_executable_sha=frozen, compact_bridge=compact_summary(acquisition),
                       execution=dict(cuda_available=True, gpu=environment['gpu_name'],
                                      vram_gib=environment['vram_gib'], torch=environment['torch'],
                                      wall_seconds=wall, wall_minutes=wall / 60,
                                      status='COMPLETE' if acquisition['updates'] == 18000 else 'INCOMPLETE'),
                       bundle=dict(filename=bundle.name, bytes=bundle.stat().st_size,
                                   sha256=inherited._sha256(bundle), all_json_parsed=True,
                                   json_files=[p.relative_to(out).as_posix() for p in paths]),
                       official_decision=dict(verdict=verdict, production_promotion_authorized=False,
                                              pre500m_authorized=False, training_500m_authorized=False,
                                              broad_reasoning_claim_authorized=False),
                       sealed=dict(exact=sealed['score']['complete_exact_with_valid_stop'],
                                   n=sealed['score']['n'], status='MEASURED_AFTER_DECISION'),
                       preregistration=prereg,
                       postrun_audit=dict(raw_commutation_flag_accepted=False,
                                          production_promotion_authorized=False))
        inherited.write_json(out / 'cyr_gpu_012_result_receipt.json', receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        return receipt
    except Exception as exc:
        inherited.write_json(out / 'FAILURE.json', dict(error=repr(exc), elapsed_seconds=time.monotonic() - start,
                                                       frozen_executable_sha=frozen))
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        print('GPU free GiB after release:', torch.cuda.mem_get_info()[0] / 2**30, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.out, smoke=args.smoke)
