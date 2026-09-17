"""Deterministic initial-tensor replay vs recorded anchors; CPU only, no training.

Loads the frozen CYR sources (isolated module namespace), rebuilds the compact
initialization from seed 3301 twice, and classifies the computed initial L2
against the value recorded in both verified run bundles.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FROZEN_ROOT = Path('C:/Users/ankit/cyr012-frozen')
RECORDED_INITIAL_L2 = 46.88997268676758
COMPACT_PARAMETER_COUNT = 987392
_MODEL_SEED = 3301
_FROZEN_PREFIXES = ('v5_model', 'v5_experiments', 'v5_contracts',
                    'v5_registry', 'v5_data', 'v5_training')

MATCH = 'INITIALIZATION_REPLAYS_MATCHING_NORM'
NEAR = 'NORM_NEAR_INCONCLUSIVE'
MISMATCH = 'NORM_MISMATCH'
SPEC = 'SPEC_MISMATCH'


def _frozen_state():
    return {name: module for name, module in sys.modules.items()
            if name.startswith(_FROZEN_PREFIXES)}


def _purge_frozen(saved):
    for name in [n for n in sys.modules if n.startswith(_FROZEN_PREFIXES)]:
        del sys.modules[name]
    sys.path.insert(0, str(FROZEN_ROOT))


def _restore_frozen(saved):
    sys.path.remove(str(FROZEN_ROOT))
    for name in [n for n in sys.modules if n.startswith(_FROZEN_PREFIXES)]:
        del sys.modules[name]
    sys.modules.update(saved)


def build_initial_flat(seed: int = _MODEL_SEED):
    """Rebuild the compact initial flat parameter vector from frozen sources."""
    for relative in ('v5_model/core.py', 'v5_model/initialize.py',
                     'v5_experiments/cyr_gpu011.py'):
        source = FROZEN_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f'missing frozen source: {relative}')
    torch = __import__('torch')
    saved = _frozen_state()
    try:
        _purge_frozen(saved)
        core = __import__('importlib').import_module('v5_model.core')
        experiments = __import__('importlib').import_module(
            'v5_experiments.cyr_gpu011')
        spec = experiments.research_small_spec(
            len(experiments.COMPACT_TOKENS))
        model = core.initialize(spec, seed, torch_module=torch)
    finally:
        _restore_frozen(saved)
    return torch.cat([p.detach().float().cpu().reshape(-1)
                      for p in model.parameters()])


def tensor_digest(flat) -> str:
    return hashlib.sha256(flat.contiguous().numpy().tobytes()).hexdigest()


def classify(computed_norm: float, recorded_norm: float,
             first_digest: str, repeat_digest: str) -> dict:
    delta = abs(computed_norm - recorded_norm)
    if delta <= 1e-6:
        verdict = MATCH
    elif delta <= 1e-3:
        verdict = NEAR
    else:
        verdict = MISMATCH
    return {'verdict': verdict, 'norm_delta': delta,
            'in_process_deterministic': first_digest == repeat_digest}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ('left', 'right'):
        parser.add_argument(f'--{side}-receipt', required=True)
        parser.add_argument(f'--{side}-bundle', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    from anra_v5.cyr_repro_ledger import _file_sha, load_run
    left = load_run(args.left_receipt, args.left_bundle, '011')
    right = load_run(args.right_receipt, args.right_bundle, '012')

    def recorded(run):
        value = run['acquisition'].get('relative_displacement_final', {})
        return value.get('initial_l2')

    recorded_left, recorded_right = recorded(left), recorded(right)
    if recorded_left != recorded_right:
        raise ValueError('bundles disagree on recorded initial_l2')

    torch = __import__('torch')
    flat = build_initial_flat(_MODEL_SEED)
    flat_repeat = build_initial_flat(_MODEL_SEED)
    digest, digest_repeat = tensor_digest(flat), tensor_digest(flat_repeat)
    computed_norm = float(flat.norm().item())
    element_count = int(flat.numel())

    verdict = classify(computed_norm, recorded_left, digest, digest_repeat)
    if element_count != COMPACT_PARAMETER_COUNT:
        verdict = {'verdict': SPEC, 'norm_delta': None,
                   'in_process_deterministic': digest == digest_repeat}

    out = {
        'schema': 'anra-cyr-initial-replay/v1',
        'left': {'experiment': left['receipt'].get('experiment'),
                 'recorded_initial_l2': recorded_left,
                 'receipt_sha256': _file_sha(args.left_receipt),
                 'bundle_sha256': left['receipt']['bundle']['sha256']},
        'right': {'experiment': right['receipt'].get('experiment'),
                  'recorded_initial_l2': recorded_right,
                  'receipt_sha256': _file_sha(args.right_receipt),
                  'bundle_sha256': right['receipt']['bundle']['sha256']},
        'replay': {'seed': _MODEL_SEED, 'device': 'cpu',
                   'torch_version': torch.__version__,
                   'element_count': element_count,
                   'computed_initial_l2': computed_norm,
                   'computed_tensor_sha256': digest,
                   'repeat_tensor_sha256': digest_repeat,
                   'frozen_root': str(FROZEN_ROOT)},
        'verdict': verdict,
        'limits': ('matching initial norm with identical in-process replay '
                   'supports but does not prove historical tensor identity; '
                   'a mismatch rejects this replay, not seed determinism'),
    }
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {path}: {verdict['verdict']} "
          f"(computed {computed_norm:.8f} vs recorded {recorded_left:.8f})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
