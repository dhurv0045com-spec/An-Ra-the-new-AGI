"""Read-only declared-byte custody inventory; never validates scientific results."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA = 'anra-horm-custody/v1'
RAW_MATCH = 'RAW_MATCH'
MISSING = 'MISSING_AT_EXPECTED_PATH'
EOL_RECONSTRUCTED = 'EOL_RECONSTRUCTION_MATCH_ONLY'
HASH_MISMATCH = 'HASH_MISMATCH'
INTEGRITY_OK = 'DECLARED_BYTES_MATCH_ONLY'
ARTIFACT_INTEGRITY_BLOCKED = 'ARTIFACT_INTEGRITY_BLOCKED'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def _declarations(manifest):
    declared = manifest.get('artifact_sha256') if isinstance(manifest, dict) else None
    if not isinstance(declared, dict) or not declared:
        raise ValueError('nonempty artifact_sha256 mapping required')
    seen = set()
    for name, expected in declared.items():
        parts = name.split('/')
        if (not all(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', p)
                    and not p.endswith('.') for p in parts)
                or name.casefold() in seen):
            raise ValueError(f'unsafe or ambiguous artifact path: {name}')
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError(f'invalid SHA-256 for {name}')
        seen.add(name.casefold())
    return declared


def audit_line(line, directory):
    """Inventory one experiment directory without loading any checkpoint."""
    results = Path(directory) / 'results'
    manifest_bytes = (results / 'RESULT.json').read_bytes()
    manifest = _json(manifest_bytes)
    artifacts = {}
    for name, expected in sorted(_declarations(manifest).items()):
        path = results / name
        if not path.resolve().is_relative_to(results.resolve()):
            raise ValueError(f'artifact escapes results directory: {name}')
        entry = {'declared_sha256': expected, 'raw_sha256': None,
                 'observed_bytes': None, 'eol_reconstructed_sha256': None}
        if not path.is_file():
            entry['classification'] = MISSING
        else:
            raw = path.read_bytes()
            if name.endswith('.json'):
                _json(raw)
            entry.update(raw_sha256=_sha(raw), observed_bytes=len(raw))
            entry['classification'] = RAW_MATCH if _sha(raw) == expected else HASH_MISMATCH
            # Diagnostic only: pure LF JSON to CRLF. Never write transformed bytes.
            if name.endswith('.json') and b'\r' not in raw and b'\n' in raw:
                reconstructed = _sha(raw.replace(b'\n', b'\r\n'))
                entry['eol_reconstructed_sha256'] = reconstructed
                if entry['classification'] != RAW_MATCH and reconstructed == expected:
                    entry['classification'] = EOL_RECONSTRUCTED
        if entry['classification'] != RAW_MATCH:
            entry['recovery'] = {
                'expected_path': 'results/' + name,
                'declared_sha256': expected,
                'owner_action': 'Locate original bytes in original run storage or backup; verify raw SHA-256 before use. Do not retrain as recovery.',
                'storage_location': 'UNKNOWN',
            }
        artifacts[name] = entry
    return {
        'experiment': line,
        'manifest_sha256': _sha(manifest_bytes),
        'manifest_bytes': len(manifest_bytes),
        'recorded_verdict_unvalidated': manifest.get('verdict'),
        'status': INTEGRITY_OK if all(a['classification'] == RAW_MATCH for a in artifacts.values()) else ARTIFACT_INTEGRITY_BLOCKED,
        'scientific_validation': 'NOT_PERFORMED',
        'artifacts': artifacts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    lines = {line: audit_line(line, args.root / 'experiments' / line)
             for line in ('HORM-001', 'HORM-002')}
    report = {
        'schema': SCHEMA,
        'diagnostic_sha256': _sha(Path(__file__).read_bytes()),
        'lines': lines,
        'fail_closed_notes': [
            'EOL reconstruction is diagnostic, never raw-byte verification.',
            'Manifest hashes bind the observed manifests, not their historical authenticity.',
            'Only declared artifact paths are inventoried; extra files, source identity, schema semantics, result validity and checkpoints are not validated.',
            'Missing means absent at the expected checkout path; other storage was not searched.',
            'Recorded verdicts are historical labels, not endorsements. No sealed data or model deserialization.',
        ],
    }
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    blocked = any(r['status'] != INTEGRITY_OK for r in lines.values())
    print(f'custody report written; blocked={blocked}')
    return 2 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
