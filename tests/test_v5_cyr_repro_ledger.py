"""Synthetic CPU fixtures, not historical experiment evidence."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from anra_v5.cyr_repro_ledger import compare_records, load_run


def records():
    return {
        'receipt': {'frozen_executable_sha': 'a' * 40},
        'environment': {'torch': 'test-only', 'gpu_name': 'fixture'},
        'acquisition': {'model_seed': 1, 'order_seed': 2, 'batch_rows': 64,
                        'semantic_stream_sha256': 'b' * 64,
                        'relative_displacement_final': {'initial_l2': 3.0}},
    }


def test_missing_identity_is_unknown_not_equal():
    rows = {r['field']: r for r in compare_records(records(), records())}
    assert rows['initial_tensor_sha256']['classification'] == 'unknown'
    assert rows['model_seed']['classification'] == 'proven_identical'
    assert rows['initial_l2']['classification'] == 'proven_identical'
    assert 'tensor' in rows['initial_l2']['limit']
    assert rows['semantic_stream_sha256']['classification'] == 'proven_identical'
    generated = compare_records(records(), records())
    assert len(generated) == len({r['field'] for r in generated})


def test_differences_and_nulls_do_not_imply_causality():
    left, right = records(), records()
    right['environment']['gpu_name'] = 'other fixture'
    left['acquisition']['g50_confirm_update'] = None
    right['acquisition']['g50_confirm_update'] = 200
    rows = {r['field']: r for r in compare_records(left, right)}
    assert rows['gpu_name']['classification'] == 'proven_different'
    assert rows['g50_confirm_update']['classification'] == 'proven_different'
    assert rows['g90_confirm_update']['classification'] == 'unknown'


def bundle(tmp_path):
    archive = tmp_path / 'fixture.zip'
    members = {
        'ENVIRONMENT.json': records()['environment'],
        'COMPACT_BRIDGE.json': records()['acquisition'],
        'PREREGISTRATION.json': {},
    }
    with ZipFile(archive, 'w') as z:
        for name, value in members.items():
            z.writestr(name, json.dumps(value))
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({
        'experiment': 'CYR-GPU-011', 'frozen_executable_sha': 'a' * 40,
        'bundle': {'bytes': archive.stat().st_size,
                   'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                   'json_files': list(members)},
    }))
    return receipt, archive


def test_verified_loader_repeatable_and_tamper_rejected(tmp_path):
    receipt, archive = bundle(tmp_path)
    assert load_run(receipt, archive, '011') == load_run(receipt, archive, '011')
    archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='byte count mismatch'):
        load_run(receipt, archive, '011')


def test_wrong_experiment_rejected(tmp_path):
    receipt, archive = bundle(tmp_path)
    with pytest.raises(ValueError, match='experiment'):
        load_run(receipt, archive, '012')


def test_missing_bundle_fails_instead_of_receipt_only_comparison(tmp_path):
    receipt, archive = bundle(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_run(receipt, tmp_path / 'absent.zip', '011')


def test_cli_writes_provenance_bound_ledger(tmp_path, capsys):
    receipt, archive = bundle(tmp_path)
    out = tmp_path / 'LEDGER.json'
    from anra_v5 import cyr_repro_ledger as module
    module.main(['--left-receipt', str(receipt), '--left-bundle', str(archive),
                 '--left-experiment', '011',
                 '--right-receipt', str(receipt), '--right-bundle', str(archive),
                 '--right-experiment', '011', '--out', str(out)])
    data = json.loads(out.read_text())
    assert data['schema'] == 'anra-cyr-repro-ledger/v1'
    assert data['left']['bundle_sha256'] == data['right']['bundle_sha256']
    assert data['left']['receipt_sha256']
    assert any(row['field'] == 'initial_tensor_sha256'
               and row['classification'] == 'unknown' for row in data['rows'])
