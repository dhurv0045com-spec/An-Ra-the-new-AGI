"""Process-owned local checkpoint writer exclusion."""

from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def checkpoint_writer(lineage: Path, *, filename: str = ".checkpoint-writer"):
    """Hold an exclusive nonblocking lock until close or process exit."""
    with (lineage / filename).open("a+b") as handle:
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
