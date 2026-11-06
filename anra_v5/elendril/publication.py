"""Serialize writers around the inherited local checkpoint publication fence."""

from contextlib import contextmanager
from pathlib import Path
import os
import uuid


@contextmanager
def writer_lock(lineage_root: Path):
    # OS-owned lock releases when a process dies. Never unlink the lock inode.
    with (lineage_root / ".elendril-writer").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def quarantine_interrupted_publication(lineage_root: Path, generation: int) -> None:
    # Under the writer lock, only uncommitted paths for the retried generation move.
    for name in (f".staging-{generation}", f".LATEST-{generation}"):
        path = lineage_root / name
        if path.exists():
            path.rename(lineage_root / f".interrupted-{generation}-{uuid.uuid4().hex}")
