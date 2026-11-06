"""Verify frozen ELENDRIL receipts and optional local checkpoint custody without training."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import file_digest, sha256
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from anra_v5.elendril.__main__ import main as elendril_main


ARCHIVE = "930ca3bb723e0b5b191700eec3d5f450875a6e25"
ARTIFACT_ROOT = PurePosixPath("artifacts/elendril")
ARTIFACT_MAPS = ("receipts_sha256", "figures_sha256", "receipts_and_figures_sha256")
CHECKPOINT_MAPS = (
    "local_checkpoint_manifests_sha256",
    "checkpoint_manifests_sha256",
    "checkpoints_sha256",
)
OPERATOR_HASHES = {
    2: ("inversion_operator_sha256", "docs/elendril/reproduce_inversion.py"),
    4: ("diagnostic_operator_sha256", "tools/elendril_join_diagnostic.py"),
}


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or ".." in path.parts or ":" in value
            or "\\" in value or any(ord(char) < 32 for char in value)):
        raise ValueError(f"unsafe archived path: {value}")
    return path


def archive_blob(process, archive: str, path: PurePosixPath) -> bytes:
    process.stdin.write(f"{archive}:{path.as_posix()}\n".encode())
    process.stdin.flush()
    header = process.stdout.readline().split()
    if len(header) != 3 or header[1] != b"blob":
        raise ValueError(f"archive is missing blob {archive}:{path}")
    size = int(header[2])
    data = process.stdout.read(size)
    if len(data) != size or process.stdout.read(1) != b"\n":
        raise ValueError(f"incomplete archive blob {archive}:{path}")
    return data


def check_file(path: Path, expected: str, *, alternate: str | None = None) -> None:
    with path.open("rb") as source:
        actual = file_digest(source, "sha256").hexdigest()
    if actual not in (expected, alternate):
        raise ValueError(f"checksum mismatch: {path}")


def checkpoint_path(base: Path, number: int, value: str) -> Path:
    path = relative_path(value)
    if number == 4:
        prefix, *parts = path.parts
        roots = {"003-replay": "Elendril-V2-replay-data", "004": "Elendril-ELD-DEV-004"}
        return base / roots[prefix] / Path(*parts)
    return base / f"Elendril-ELD-DEV-{number:03}" / Path(*path.parts)


def verify(archive: str, checkpoint_base: Path | None) -> dict:
    counts = Counter()
    manifests = {}
    with subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as process:
        try:
            for number in range(1, 5):
                name = f"eld-dev-{number:03}"
                folder = ARTIFACT_ROOT / name
                archived = archive_blob(process, archive, folder / "manifest.json")
                manifest_sha256 = sha256(archived).hexdigest()
                manifests[name] = manifest_sha256
                check_file(Path(folder / "manifest.json"), manifest_sha256)
                manifest = json.loads(archived)
                for key in ARTIFACT_MAPS:
                    for artifact, expected in manifest.get(key, {}).items():
                        archive_path = folder / relative_path(artifact)
                        data = archive_blob(process, archive, archive_path)
                        if sha256(data).hexdigest() != expected:
                            raise ValueError(f"archived artifact checksum mismatch: {archive_path}")
                        path = Path(archive_path)
                        alternate = manifest.get("original_local_crlf_receipts_sha256", {}).get(artifact)
                        check_file(path, expected, alternate=alternate)
                        counts["receipt_figure_resource_hashes"] += 1
                        if path.suffix == ".json" and json.loads(data).get("schema") == "elendril-development-pilot/v1":
                            with redirect_stdout(StringIO()) as output:
                                elendril_main(["rescore", "--receipt", str(path)])
                            counts["pilot_receipts"] += 1
                            counts["scoring_snapshots"] += int(output.getvalue().split()[1])
                if number in OPERATOR_HASHES:
                    key, operator = OPERATOR_HASHES[number]
                    data = archive_blob(process, archive, PurePosixPath(operator))
                    if sha256(data).hexdigest() != manifest[key]:
                        raise ValueError(f"archived operator checksum mismatch: {operator}")
                    counts["operator_hashes"] += 1
                for key in CHECKPOINT_MAPS:
                    for entry, expected in manifest.get(key, {}).items():
                        counts["checkpoint_manifests_indexed"] += 1
                        if checkpoint_base is None:
                            continue
                        path = checkpoint_path(checkpoint_base, number, entry)
                        check_file(path, expected)
                        counts["checkpoint_manifests_verified"] += 1
                        for component in json.loads(path.read_bytes())["components"]:
                            payload = path.parent / relative_path(component["name"])
                            if payload.stat().st_size != component["byte_size"]:
                                raise ValueError(f"component byte size mismatch: {payload}")
                            check_file(payload, component["sha256"])
                            counts["checkpoint_components_verified"] += 1
        finally:
            process.stdin.close()
    return {"archive": archive, "manifests_sha256": manifests, "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=ARCHIVE)
    parser.add_argument("--checkpoint-base", type=Path, help="Parent of the Elendril checkpoint sibling directories")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.archive):
        parser.error("--archive must be a full lowercase Git commit SHA")
    print(json.dumps(verify(args.archive, args.checkpoint_base), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
