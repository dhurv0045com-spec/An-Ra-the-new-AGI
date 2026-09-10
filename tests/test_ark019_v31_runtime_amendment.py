from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'experiments' / 'ARK-019' / 'ark019_v3_core.py'
WRAP = ROOT / 'experiments' / 'ARK-019' / 'run_ark019_v31.py'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_original_scientific_core_remains_175_minute_v3():
    c = _load(CORE, 'ark019_v3_core_runtime_test')
    assert c.WALL_MINUTES == 175
    assert c.PACKAGING_RESERVE_MINUTES == 5
    assert c.RUNTIME_SAFETY_FACTOR == 1.30
    assert c.HORIZON == 1000
    assert len(c.PRETRAIN_SEEDS) * len(c.SKILL_B_SEEDS) * len(c.ARMS) == 16


def test_v31_wrapper_declares_only_wall_extension():
    text = WRAP.read_text(encoding='utf-8')
    assert 'AMENDED_WALL_MINUTES = 240' in text
    assert 'EXPECTED_ORIGINAL_WALL_MINUTES = 175' in text
    assert 'R.C.WALL_MINUTES = AMENDED_WALL_MINUTES' in text
    for frozen in ('HORIZON', 'ARMS', 'PRETRAIN_SEEDS', 'SKILL_B_SEEDS', 'EVAL_EVERY', 'RUNTIME_SAFETY_FACTOR'):
        assert f'R.C.{frozen} =' not in text
