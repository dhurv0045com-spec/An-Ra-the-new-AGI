"""Post-run evidence verifier; reads receipts only, never trains."""
from __future__ import annotations

import hashlib
import json
import zipfile


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(path, declared):
    if path.stat().st_size != int(declared['bytes']):
        raise ValueError('bundle byte count mismatch')
    if _sha(path) != declared['sha256']:
        raise ValueError('bundle SHA mismatch')
    expected = declared.get('json_files')
    if not isinstance(expected, list) or not expected or not all(isinstance(name, str) for name in expected):
        raise ValueError('missing or invalid declared JSON file list')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if sorted(names) != sorted(expected):
            raise ValueError('bundle file list mismatch')
        for name in names:
            json.loads(z.read(name))
    return True


def summarize(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    compact = receipt['compact_bridge']
    return {
        'experiment': receipt.get('experiment'),
        'frozen_executable_sha': receipt.get('frozen_executable_sha'),
        'bundle_sha256': receipt['bundle']['sha256'],
        'updates': compact['updates'],
        'row_presentations': compact['row_presentations'],
        'ark_exposure_fraction': compact['ark_exposure_fraction'],
        'status': compact['status'],
        'm99_confirm_update': compact['m99_confirm_update'],
        'g50_confirm_update': compact['g50_confirm_update'],
        'g90_confirm_update': compact['g90_confirm_update'],
        'dev_controller_final_exact': compact['dev_controller_final_exact'],
        'dev_measurement_standard_exact': compact['dev_measurement_standard_exact'],
        'standard_digit_accuracy': compact['standard_digit_accuracy'],
        'verdict': receipt['official_decision']['verdict'],
    }


def compare_bundle(receipt_path):
    """Verify that the bundle in the receipt still hashes to the recorded value."""
    receipt = json.loads(receipt_path.read_text())
    bundle = receipt_path.parent / receipt['bundle']['filename']
    verify_bundle(bundle, receipt['bundle'])
    return True


__all__ = ['verify_bundle', 'summarize', 'compare_bundle']
