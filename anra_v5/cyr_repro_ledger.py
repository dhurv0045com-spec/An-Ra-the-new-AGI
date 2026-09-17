"""CYR-011 vs CYR-012 reproducibility ledger; receipts only, no training."""
from __future__ import annotations

import json
from pathlib import Path

from anra_v5.cyr_gpu012_evidence import verify_bundle

UNKNOWN = 'unknown'
IDENTICAL = 'proven_identical'
DIFFERENT = 'proven_different'
_MISSING = object()


def load_run(receipt_path, bundle_path, experiment):
    """Load a verified CYR run bundle; fail closed before any comparison."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get('experiment') != f'CYR-GPU-{experiment}':
        raise ValueError('receipt experiment mismatch')
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f'missing bundle: {bundle_path}')
    verify_bundle(bundle_path, receipt['bundle'])
    with __import__('zipfile').ZipFile(bundle_path) as z:
        names = z.namelist()
        environment = json.loads(z.read('ENVIRONMENT.json'))
        acquisition = json.loads(z.read(
            'COMPACT_BRIDGE.json' if 'COMPACT_BRIDGE.json' in names
            else 'compact/acquisition.json'))
    return {
        'receipt': receipt,
        'environment': environment,
        'acquisition': acquisition,
    }


def _sha(text):
    import hashlib
    return hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()


def _row(field, left, right, classification, limit=''):
    sentinel = _MISSING
    return {'field': field,
            'left': None if left is sentinel else left,
            'right': None if right is sentinel else right,
            'classification': classification, 'limit': limit}


def compare_records(left, right):
    """Classify bounded fields; absence stays unknown; identity needs tensors."""
    rows = []

    def pick(record, path):
        value = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return _MISSING
            value = value[key]
        return value

    fields = [
        ('frozen_executable_sha', ('receipt', 'frozen_executable_sha'),
         ('receipt', 'frozen_executable_sha')),
        ('torch', ('environment', 'torch'), ['environment', 'torch']),
        ('gpu_name', ('environment', 'gpu_name'), ['environment', 'gpu_name']),
        ('model_seed', ('acquisition', 'model_seed'), ['acquisition', 'model_seed']),
        ('order_seed', ('acquisition', 'order_seed'), ['acquisition', 'order_seed']),
        ('batch_rows', ('acquisition', 'batch_rows'), ['acquisition', 'batch_rows']),
        ('semantic_stream_sha256',
         ('acquisition', 'semantic_stream_sha256'),
         ('acquisition', 'semantic_stream_sha256')),
        ('updates', ('acquisition', 'updates'), ['acquisition', 'updates']),
        ('row_presentations', ('acquisition', 'row_presentations'), ['acquisition', 'row_presentations']),
        ('g50_confirm_update', ('acquisition', 'g50_confirm_update'), ['acquisition', 'g50_confirm_update']),
        ('g90_confirm_update', ('acquisition', 'g90_confirm_update'), ['acquisition', 'g90_confirm_update']),
        ('initial_l2', ('acquisition', 'relative_displacement_final', 'initial_l2'),
         ['acquisition', 'relative_displacement_final', 'initial_l2']),
    ]

    def extract(record, field, paths):
        value = _MISSING
        for path in paths:
            candidate = pick(record, path)
            if candidate is not _MISSING:
                return candidate
        return value

    for field, left_path, right_path in fields:
        lvalue = extract(left, field, [left_path])
        rvalue = extract(right, field, [right_path])
        if lvalue is _MISSING or rvalue is _MISSING:
            classification = UNKNOWN
            limit = 'absent from at least one verified bundle'
        elif lvalue == rvalue:
            classification = IDENTICAL
            limit = ''
            if field == 'initial_l2':
                limit = 'matching scalar norm does not prove tensor identity'
            elif field == 'semantic_stream_sha256':
                limit = 'digest identity does not independently prove row contents'
        else:
            classification = DIFFERENT
            limit = ''
            if field == 'g50_confirm_update':
                limit = 'outcome difference; not itself causal evidence'
        rows.append(_row(field, lvalue, rvalue, classification, limit))

    lvalue = pick(left['acquisition'], ('final_checkpoint', 'model_sha256'))
    rvalue = pick(right['acquisition'], ('final_checkpoint', 'model_sha256'))
    if lvalue is _MISSING or rvalue is _MISSING:
        classification = UNKNOWN
    else:
        classification = DIFFERENT if lvalue != rvalue else IDENTICAL
    rows.append(_row('final_model_sha256', lvalue, rvalue, classification,
                     'final state identity; not initial tensors'))

    rows.append(_row('initial_tensor_sha256', _MISSING, _MISSING, UNKNOWN,
                     'not recorded in either bundle; requires initial-tensor replay'))
    rows.append(_row('kernel_selection', None, None, UNKNOWN,
                     'no kernel trace recorded in either bundle'))
    return rows


def compare_runs(left, right):
    return {
        'schema': 'anra-cyr-repro-ledger/v1',
        'left': left['receipt'].get('experiment'),
        'right': right['receipt'].get('experiment'),
        'rows': compare_records(left, right),
        'conclusion': ('classification only; no causal ranking without a '
                       'controlled intervention'),
    }


def _file_sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    """Write the provenance-bound ledger; no training, receipts only."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ('left', 'right'):
        parser.add_argument(f'--{side}-receipt', required=True)
        parser.add_argument(f'--{side}-bundle', required=True)
        parser.add_argument(f'--{side}-experiment', required=True,
                            choices=('011', '012'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    left = load_run(args.left_receipt, args.left_bundle, args.left_experiment)
    right = load_run(args.right_receipt, args.right_bundle,
                     args.right_experiment)
    ledger = compare_runs(left, right)
    ledger['left'] = {
        'experiment': left['receipt'].get('experiment'),
        'receipt_sha256': _file_sha(args.left_receipt),
        'bundle_sha256': left['receipt']['bundle']['sha256'],
    }
    ledger['right'] = {
        'experiment': right['receipt'].get('experiment'),
        'receipt_sha256': _file_sha(args.right_receipt),
        'bundle_sha256': right['receipt']['bundle']['sha256'],
    }
    out = Path(args.out)
    out.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    print(f'wrote {out} with {len(ledger["rows"])} classified fields')
    return 0


__all__ = ['load_run', 'compare_runs', 'compare_records', 'main']


if __name__ == '__main__':
    raise SystemExit(main())
