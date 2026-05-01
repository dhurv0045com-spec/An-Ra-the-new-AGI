from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    source: Path
    ext: str


class DriveSessionManager:
    EMERGENCY_SAVE_ORDER = ["ghost", "graph", "lora", "replay", "training_state", "goals", "logs"]

    def __init__(self, drive_dir: Path, session_id: str = "default", autosave_minutes: int = 30) -> None:
        self.drive_dir = Path(drive_dir)
        self.session_id = session_id
        self.session_dir = self.drive_dir / "sessions" / session_id
        self.manifest_path = self.session_dir / "manifest.json"
        self.autosave_seconds = autosave_minutes * 60
        self._autosave_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._autosave_callback: Callable[[], None] | None = None
        self._sigterm_registered = False
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"families": {}}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"families": {}}

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def save_family(self, family: str, source: Path) -> Path:
        if not source.exists():
            raise FileNotFoundError(source)
        family_dir = self.session_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        ext = "".join(source.suffixes) or ".bin"
        ts = time.time_ns()
        target = family_dir / f"{family}_v1_{ts}{ext}"
        shutil.copy2(source, target)

        versions = sorted(family_dir.glob(f"{family}_v1_*{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in versions[3:]:
            old.unlink(missing_ok=True)

        manifest = self._load_manifest()
        family_manifest = manifest.setdefault("families", {}).setdefault(family, {"versions": []})
        family_manifest["versions"].insert(0, {
            "path": str(target.relative_to(self.session_dir)),
            "sha256": self._sha256(target),
            "size": target.stat().st_size,
            "mtime": int(target.stat().st_mtime),
        })
        family_manifest["versions"] = family_manifest["versions"][:3]
        self._save_manifest(manifest)
        return target

    def load_family(self, family: str, destination: Path) -> bool:
        manifest = self._load_manifest()
        entries = manifest.get("families", {}).get(family, {}).get("versions", [])
        destination.parent.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            candidate = self.session_dir / entry["path"]
            if not candidate.exists():
                continue
            if self._sha256(candidate) != entry.get("sha256"):
                continue
            shutil.copy2(candidate, destination)
            return True
        return False

    def emergency_save(self, mapping: dict[str, Path]) -> list[str]:
        saved: list[str] = []
        for family in self.EMERGENCY_SAVE_ORDER:
            source = mapping.get(family)
            if source and source.exists():
                self.save_family(family, source)
                saved.append(family)
        return saved

    def start_autosave(self, callback: Callable[[], None]) -> None:
        self._autosave_callback = callback
        if self._autosave_thread is not None:
            return

        def _loop() -> None:
            while not self._stop.wait(self.autosave_seconds):
                if self._autosave_callback:
                    self._autosave_callback()

        self._autosave_thread = threading.Thread(target=_loop, daemon=True, name="drive-autosave")
        self._autosave_thread.start()
        atexit.register(self.stop_autosave)

    def stop_autosave(self) -> None:
        self._stop.set()
        if self._autosave_thread and self._autosave_thread.is_alive():
            self._autosave_thread.join(timeout=1)

    def register_sigterm_hook(self, callback: Callable[[], None]) -> None:
        if self._sigterm_registered:
            return

        def _handler(signum, frame):
            del signum, frame
            callback()

        signal.signal(signal.SIGTERM, _handler)
        self._sigterm_registered = True
