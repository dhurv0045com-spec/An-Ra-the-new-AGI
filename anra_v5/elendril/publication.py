"""Serialize writers around the inherited local checkpoint publication fence."""

from pathlib import Path
import uuid

from v5_training.checkpoint_lock import checkpoint_writer


def writer_lock(lineage_root: Path):
    return checkpoint_writer(lineage_root, filename=".elendril-writer")


def quarantine_interrupted_publication(lineage_root: Path, generation: int) -> None:
    # Under the writer lock, only uncommitted paths for the retried generation move.
    for name in (f".staging-{generation}", f".LATEST-{generation}"):
        path = lineage_root / name
        if path.exists():
            path.rename(lineage_root / f".interrupted-{generation}-{uuid.uuid4().hex}")
