"""ARK-019 V3.1 runtime-only amendment.

The original V3 science stays frozen. An operator T4 calibration projected the
complete campaign at ~209.68 minutes under the preregistered 1.30 safety
factor, so the original 175-minute wall correctly failed before comparative
continuation arms. V3.1 changes only the execution wall to 240 minutes.
"""
from __future__ import annotations

import json
from pathlib import Path

import run_ark019_v3 as R

EXPECTED_ORIGINAL_WALL_MINUTES = 175
AMENDED_WALL_MINUTES = 240
EXPECTED_PACKAGING_RESERVE_MINUTES = 5
EXPECTED_RUNTIME_SAFETY_FACTOR = 1.30


def _archive_expected_preexecution_failure() -> None:
    """Preserve the known 175-minute gate failure without hiding other failures."""
    failure = R.OUT / 'ARK-019_V3_FAILURE.json'
    if not failure.exists():
        return
    body = json.loads(failure.read_text(encoding='utf-8'))
    message = str(body.get('message', ''))
    completed = list(R.OUT.glob('matched_sets/*/*/RESULT.json'))
    if completed:
        # A generic failure after scientific continuation outcomes is evidence and
        # must never be relabelled as the preexecution runtime failure.
        raise RuntimeError(
            'R3 V3.1 found a prior generic failure plus continuation-arm results; '
            'manual audit required before resume'
        )
    if not message.startswith('full R3 does not fit 175-minute wall'):
        raise RuntimeError(
            'R3 V3.1 refuses to overwrite an unexpected prior failure: ' + message
        )
    archive = R.OUT / 'PREEXECUTION_FAILURE_ARCHIVE' / 'RUNTIME_GATE_175M.json'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        old = json.loads(archive.read_text(encoding='utf-8'))
        if old != body:
            raise RuntimeError('preexecution failure archive identity mismatch')
        failure.unlink()
        return
    failure.replace(archive)


def apply_runtime_amendment() -> None:
    if int(R.C.WALL_MINUTES) != EXPECTED_ORIGINAL_WALL_MINUTES:
        raise RuntimeError(
            f'ARK-019 V3 wall drift: expected {EXPECTED_ORIGINAL_WALL_MINUTES}, '
            f'found {R.C.WALL_MINUTES}'
        )
    if int(R.C.PACKAGING_RESERVE_MINUTES) != EXPECTED_PACKAGING_RESERVE_MINUTES:
        raise RuntimeError('ARK-019 V3 packaging reserve drift')
    if abs(float(R.C.RUNTIME_SAFETY_FACTOR) - EXPECTED_RUNTIME_SAFETY_FACTOR) > 1e-12:
        raise RuntimeError('ARK-019 V3 runtime safety factor drift')
    # The sole executable amendment.
    R.C.WALL_MINUTES = AMENDED_WALL_MINUTES


def main() -> int:
    _archive_expected_preexecution_failure()
    apply_runtime_amendment()
    print(
        'ARK-019 V3.1 RUNTIME AMENDMENT: wall 175 -> 240 minutes; '
        'science/arms/horizon/thresholds unchanged',
        flush=True,
    )
    return int(R.main())


if __name__ == '__main__':
    raise SystemExit(main())
