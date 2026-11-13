"""V7 dual snapshots and a bounded audit of the V6 analysis ledger."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import codecs
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4
import zipfile

from tools import elendril_v6_export as v6
from v5_training.checkpoint_lock import checkpoint_writer

_ALLOWED = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.md', '.yaml', '.yml', '.toml', '.py'}
_MAX_TEXT_FILE = 64 << 20
_ANALYSIS_LIMIT = 1 << 30


def _validate_analysis(root: Path) -> None:
    """Only bounded UTF-8 text artifacts are eligible for the analysis bundle."""
    for row in v6.inventory(root, max_bytes=_ANALYSIS_LIMIT):
        name = row['name']
        if (Path(name).suffix.lower() not in _ALLOWED and name != 'SHA256SUMS') or row['byte_size'] > _MAX_TEXT_FILE:
            raise ValueError('analysis refuses non-text or oversized artifact: ' + name)
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        with (root / name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(v6.CHUNK), b''):
                if b'\0' in chunk:
                    raise ValueError('analysis refuses binary artifact: ' + name)
                decoder.decode(chunk)
            decoder.decode(b'', final=True)


def _publish_analysis(root: Path, *, source: str, protocol_sha: str, campaign: dict, score: dict) -> dict:
    analysis = root / 'analysis'
    if analysis.is_symlink():
        raise ValueError('analysis directory cannot be a link')
    analysis.mkdir(exist_ok=True)
    _validate_analysis(analysis)
    v6._write_derived(analysis / 'README_V7.md', (
        '# Elendril V7 analysis snapshot\n\n'
        'This bounded text-only bundle excludes model and optimizer payloads. '
        'Checkpoint manifests and hashes are reports, not checkpoint bytes. '
        'Archive integrity and independent scientific rescore are separate results.\n'
    ).encode())
    native = None
    temporary = None
    try:
        report = v6.export_bundle(analysis, source=source, protocol_sha=protocol_sha,
                                  campaign=campaign, rescore=lambda _root, _campaign: score,
                                  zip_name='ELENDRIL_V7_ANALYSIS.zip', max_bytes=_ANALYSIS_LIMIT)
        native = Path(report['zip_path'])
        # Validate the finished archive too: files may have appeared after the initial inventory.
        with zipfile.ZipFile(native) as archive:
            for info in archive.infolist():
                if info.filename == v6.MANIFEST_NAME:
                    continue
                if ((Path(info.filename).suffix.lower() not in _ALLOWED and info.filename != 'SHA256SUMS')
                        or info.file_size > _MAX_TEXT_FILE):
                    raise ValueError('analysis archive contains disallowed artifact: ' + info.filename)
                decoder = codecs.getincrementaldecoder('utf-8')('strict')
                with archive.open(info) as handle:
                    for chunk in iter(lambda: handle.read(v6.CHUNK), b''):
                        if b'\0' in chunk:
                            raise ValueError('analysis archive contains binary artifact: ' + info.filename)
                        decoder.decode(chunk)
                    decoder.decode(b'', final=True)
        exports = root / 'exports'
        if exports.is_symlink():
            raise ValueError('exports directory cannot be a link')
        exports.mkdir(exist_ok=True)
        temporary = exports / ('.pending-' + uuid4().hex + '.zip')
        with native.open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst, v6.CHUNK)
            dst.flush()
            os.fsync(dst.fileno())
        verified = v6.verify_zip(temporary, max_bytes=_ANALYSIS_LIMIT)
        copied_size, copied_hash = v6._file_sha(temporary)
        if (copied_hash != report['zip_sha256'] or copied_size != report['zip_bytes']
                or verified['manifest_sha256'] != report['manifest_sha256']):
            raise ValueError('analysis archive copy failed readback verification')
        destination = exports / 'ELENDRIL_V7_ANALYSIS.zip'
        os.replace(temporary, destination)
        return {**report, 'zip_path': str(destination), 'zip_sha256': copied_hash,
                'zip_bytes': copied_size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if native is not None:
            native.unlink(missing_ok=True)


def export_results(output, *, source, protocol_sha, campaign, full=True, rescore=None):
    """Return independently verified archive statuses; never promote scientific results.

    A deferred snapshot leaves its previous archive and all raw evidence intact.
    """
    requested = Path(output)
    if requested.is_symlink():
        raise ValueError('output directory cannot be a link')
    root = requested.resolve(strict=True)
    if rescore is None:
        score = {'status': 'NOT_RUN', 'reason': 'No independent rescorer supplied'}
    else:
        try:
            score = rescore(root, campaign)
            if not isinstance(score, dict) or not isinstance(score.get('status'), str):
                raise ValueError('independent rescorer must return an explicit status')
        except Exception as exc:
            score = {'status': 'FAILED', 'error_type': type(exc).__name__, 'message': str(exc)}

    def attempt(operation):
        try:
            return operation()
        except (OSError, ValueError, UnicodeError, v6.SnapshotChanged, zipfile.BadZipFile) as exc:
            return {'status': 'EXPORT_DEFERRED', 'error_type': type(exc).__name__,
                    'message': str(exc), 'scientific_rescore': score, 'raw_evidence': str(root)}

    def make_full():
        # V6 excludes only the root exports directory. A stale nested staged
        # archive must never be folded into a subsequent full snapshot.
        nested = root / 'analysis' / 'exports'
        if nested.is_symlink() or (nested.exists() and any(nested.iterdir())):
            raise ValueError('analysis staging directory is not empty; retain and inspect it before retrying full export')
        return v6.export_bundle(root, source=source, protocol_sha=protocol_sha,
                                campaign=campaign, rescore=lambda _r, _c: score,
                                zip_name='ELENDRIL_V7_FULL.zip')

    try:
        with checkpoint_writer(root, filename='.pending-v7-export-writer'):
            analysis_report = attempt(lambda: _publish_analysis(root, source=source, protocol_sha=protocol_sha,
                                                                 campaign=campaign, score=score))
            full_report = attempt(make_full) if full else {'status': 'SKIPPED'}
    except (OSError, ValueError) as exc:
        deferred = {'status': 'EXPORT_DEFERRED', 'error_type': type(exc).__name__,
                    'message': str(exc), 'scientific_rescore': score, 'raw_evidence': str(root)}
        return {'analysis': deferred, 'full': dict(deferred) if full else {'status': 'SKIPPED'}}
    return {'analysis': analysis_report, 'full': full_report}


def audit_journal(archivepath):
    """Summarize the archived V6 JSONL ledger; do not infer causes from messages."""
    with zipfile.ZipFile(archivepath) as archive:
        info = archive.getinfo('experiment_ledger.jsonl')
        if info.file_size > 16 << 20:
            raise ValueError('ledger exceeds audit bound')
        lines = archive.read(info).decode('utf-8').splitlines()
    counts, reasons = Counter(), Counter()
    for line in lines:
        event = json.loads(line)['event']
        kind = event.get('kind') or event.get('status', 'UNKNOWN')
        counts[kind] += 1
        if kind == 'EXPORT_DEFERRED':
            error, message = event.get('error_type', 'UNKNOWN'), event.get('message', '')
            if error == 'FileNotFoundError':
                category = 'missing_file'
            elif error == 'SnapshotChanged':
                category = 'snapshot_changed'
            elif 'lock unavailable' in message:
                category = 'lock_unavailable'
            elif 'SHA256SUMS' in message:
                category = 'checksum_inventory_mismatch'
            else:
                category = 'other'
            reasons[category] += 1
    return {'events': len(lines), 'counts': dict(counts), 'deferred_reasons': dict(reasons),
            'limitations': 'Ledger records reported errors and later verified exports; it does not establish root cause, archive contents, or raw evidence survival for every attempt.'}
