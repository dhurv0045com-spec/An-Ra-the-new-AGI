"""Process-owned local checkpoint writer exclusion."""

from contextlib import contextmanager
from pathlib import Path
import os


def _win_long_path(path: Path | str) -> Path:
    """Prefix Windows paths >=240 chars with ``\\\\?\\`` so deep checkpoint objects never hit MAX_PATH."""

    s = str(path)
    if os.name == "nt" and s.startswith("\\\\?\\"):
        return Path(s)
    raw = os.path.abspath(s)
    if os.name == "nt" and not raw.startswith("\\\\?\\") and len(raw) >= 240:
        if raw.startswith("\\\\"):
            return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
        return Path("\\\\?\\" + raw)
    return Path(path) if not s.startswith("\\\\?\\") else Path(s)


@contextmanager
def checkpoint_writer(lineage: Path, *, filename: str = ".checkpoint-writer"):
    """Hold an exclusive nonblocking lock until close or process exit."""
    with _win_long_path(lineage / filename).open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("checkpoint writer lock unavailable for this lineage") from exc
        yield
