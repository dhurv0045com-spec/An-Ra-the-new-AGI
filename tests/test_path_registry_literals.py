from __future__ import annotations

from pathlib import Path

BANNED_SUBSTRINGS = (
    '/content/drive',
    'training_data/',
    'tokenizer/',
    'workspace/',
)


def test_no_path_literals_outside_registry() -> None:
    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    allowed_files = {
        'anra_paths.py',
        'anra/anra_paths.py',
        'anra/core/config.py',
        'tests/test_path_registry_literals.py',
    }

    for file in root.rglob('*.py'):
        rel = file.relative_to(root)
        rel_posix = rel.as_posix()
        if rel_posix.startswith('An-Ra-the-new-AGI/') or rel_posix in allowed_files:
            continue
        text = file.read_text(encoding='utf-8', errors='replace')
        for idx, line in enumerate(text.splitlines(), start=1):
            if any(token in line for token in BANNED_SUBSTRINGS):
                offenders.append(f'{rel_posix}:{idx}: {line.strip()}')

    assert not offenders, 'Path-like literals found outside anra_paths.py:\n' + '\n'.join(offenders)
