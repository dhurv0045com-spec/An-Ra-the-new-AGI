# ============================================================
# FILE: evaluation/tracker.py
# Model version tracking, quality history, changelog, rollback.
#
# Every version is registered here when saved.
# Tracks: loss, quality scores, what changed, when.
# Rollback to any previous version in one command.
# ============================================================

import json, os, shutil
from datetime import datetime


REGISTRY_PATH = 'data/version_registry.json'


class VersionTracker:
    """
    Tracks every saved model version with metadata and quality scores.

    Registry format (JSON):
    {
      "versions": [
        {
          "version_id":    "v1",
          "timestamp":     "2025-...",
          "checkpoint_dir":"checkpoints/finetuned_epoch1_...",
          "method":        "lora",
          "train_loss":    0.812,
          "val_loss":      0.889,
          "quality_score": 0.62,
          "is_active":     false,
          "changelog":     "Initial LoRA fine-tune on 200 examples",
          "tags":          ["lora", "v1"]
        },
        ...
      ],
      "active": "v2"
    }
    """

    def __init__(self, registry_path=REGISTRY_PATH):
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(registry_path)
                    if os.path.dirname(registry_path) else '.', exist_ok=True)
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                return json.load(f)
        return {'versions': [], 'active': None}

    def _save(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self._data, f, indent=2)

    def _next_version_id(self):
        existing = [v['version_id'] for v in self._data['versions']]
        i = len(existing) + 1
        while f'v{i}' in existing:
            i += 1
        return f'v{i}'

    def register(self, checkpoint_dir, method='lora', train_loss=None,
                 val_loss=None, quality_score=None, changelog='', tags=None):
        """
        Register a new model version.

        Args:
            checkpoint_dir (str):   Path to the saved checkpoint directory.
            method         (str):   'full' or 'lora'.
            train_loss     (float): Final training loss.
            val_loss       (float): Validation loss.
            quality_score  (float): Eval suite overall score.
            changelog      (str):   What changed in this version.
            tags           (list):  Free-form labels.

        Returns:
            str: The assigned version ID (e.g. 'v3').
        """
        version_id = self._next_version_id()
        entry = {
            'version_id':     version_id,
            'timestamp':      datetime.now().isoformat(),
            'checkpoint_dir': checkpoint_dir,
            'method':         method,
            'train_loss':     train_loss,
            'val_loss':       val_loss,
            'quality_score':  quality_score,
            'is_active':      False,
            'changelog':      changelog,
            'tags':           tags or [],
        }
        self._data['versions'].append(entry)
        self._save()
        print(f"[Tracker] Registered {version_id}: {changelog[:60]}")
        return version_id

    def set_active(self, version_id):
        """Mark a version as the currently deployed model."""
        found = False
        for v in self._data['versions']:
            v['is_active'] = (v['version_id'] == version_id)
            if v['version_id'] == version_id:
                found = True
        if not found:
            raise ValueError(f"Version '{version_id}' not found in registry.")
        self._data['active'] = version_id
        self._save()
        print(f"[Tracker] Active model → {version_id}")

    def get_active(self):
        """Return the metadata dict for the currently active version."""
        aid = self._data.get('active')
        if not aid:
            return None
        return next((v for v in self._data['versions']
                     if v['version_id'] == aid), None)

    def get(self, version_id):
        return next((v for v in self._data['versions']
                     if v['version_id'] == version_id), None)

    def rollback(self, version_id):
        """
        Set the active model to a previous version.
        Optionally copies checkpoint to a 'current/' symlink dir.
        """
        target = self.get(version_id)
        if not target:
            raise ValueError(f"Version '{version_id}' not in registry.")
        self.set_active(version_id)
        print(f"[Tracker] ↩ Rolled back to {version_id} "
              f"(checkpoint: {target['checkpoint_dir']})")
        return target

    def update_quality(self, version_id, quality_score):
        """Update the quality score for a version after evaluation."""
        v = self.get(version_id)
        if v:
            v['quality_score'] = quality_score
            self._save()
            print(f"[Tracker] {version_id} quality_score → {quality_score:.4f}")

    def alert_regression(self, version_id, threshold=0.05):
        """
        Warn if a new version is worse than the previous one.

        Args:
            version_id (str):   The newly registered version.
            threshold  (float): How much worse before alerting.

        Returns:
            bool: True if regression detected.
        """
        versions = self._data['versions']
        current  = self.get(version_id)
        if not current or current.get('quality_score') is None:
            return False

        # Find the version registered just before this one
        idx = next((i for i, v in enumerate(versions)
                    if v['version_id'] == version_id), -1)
        if idx <= 0:
            return False

        prev = None
        for v in reversed(versions[:idx]):
            if v.get('quality_score') is not None:
                prev = v
                break

        if not prev:
            return False

        delta = current['quality_score'] - prev['quality_score']
        if delta < -threshold:
            print(f"\n⚠️  REGRESSION ALERT: {version_id} quality "
                  f"({current['quality_score']:.4f}) is {abs(delta):.4f} "
                  f"lower than {prev['version_id']} "
                  f"({prev['quality_score']:.4f}).\n"
                  f"   Consider rolling back: tracker.rollback('{prev['version_id']}')")
            return True
        return False

    def changelog(self, n=10):
        """Print the last N version changelogs."""
        print(f"\n{'=' * 55}")
        print(f"  MODEL CHANGELOG (last {n} versions)")
        print(f"{'=' * 55}")
        for v in self._data['versions'][-n:]:
            active_str = " ◀ ACTIVE" if v['is_active'] else ""
            q_str = f"  quality={v['quality_score']:.4f}" \
                    if v['quality_score'] is not None else ""
            print(f"\n  {v['version_id']}{active_str}  [{v['timestamp'][:16]}]")
            print(f"    Method     : {v['method']}")
            if v['train_loss'] is not None:
                print(f"    train_loss : {v['train_loss']:.4f}"
                      f"  val_loss : {v['val_loss']:.4f}" if v['val_loss'] else "")
            if q_str:
                print(f"   {q_str}")
            print(f"    Change     : {v['changelog'] or '(no notes)'}")
        print(f"\n{'=' * 55}")

    def list_versions(self):
        return [v['version_id'] for v in self._data['versions']]


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':
    import tempfile

    print("=" * 55)
    print("  VERSION TRACKER SELF TEST")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmp:
        tracker = VersionTracker(
            registry_path=os.path.join(tmp, 'data', 'registry.json'))

        v1 = tracker.register('/tmp/ckpt_v1', method='lora',
                               train_loss=1.42, val_loss=1.51, quality_score=0.40,
                               changelog='Initial LoRA fine-tune on 100 examples')
        v2 = tracker.register('/tmp/ckpt_v2', method='lora',
                               train_loss=1.18, val_loss=1.23, quality_score=0.58,
                               changelog='Added 200 more examples, lr decay')
        v3 = tracker.register('/tmp/ckpt_v3', method='lora',
                               train_loss=1.21, val_loss=1.28, quality_score=0.52,
                               changelog='Experimental: doubled LoRA rank')

        tracker.set_active(v2)
        assert tracker.get_active()['version_id'] == 'v2'

        # Regression alert for v3
        regressed = tracker.alert_regression(v3)
        assert regressed, "Should detect regression"

        # Rollback
        tracker.rollback(v2)
        assert tracker.get_active()['version_id'] == 'v2'

        tracker.changelog()

    print("\n✓ Version tracker all checks passed.")
    print("=" * 55)
