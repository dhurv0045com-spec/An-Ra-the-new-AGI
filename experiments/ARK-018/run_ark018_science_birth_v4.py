from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "run_ark018_science_birth_v3.py"
FAST_BINDING_PATH = HERE / "ark018_v3_binding_fast.py"
WRAPPER_PATH = Path(__file__)
CLARIFICATION_COMMIT = "a3077d513fc77f6563ff3a8c186112b40699a31f"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


BASE = load_module("ark018_v3_base", BASE_PATH)
FAST = load_module("ark018_v3_binding_fast_impl", FAST_BINDING_PATH)

_original_compact = BASE.compact_receipt


def compact_receipt_hardened(obj: dict) -> dict:
    x = _original_compact(obj)
    x.pop("receipt_sha256", None)
    x["execution_wrapper_sha256"] = BASE.sha256_file(WRAPPER_PATH)
    x["base_runner_sha256"] = BASE.sha256_file(BASE_PATH)
    x["binding_probe_impl_sha256"] = BASE.sha256_file(FAST_BINDING_PATH)
    x["score_clarification_commit"] = CLARIFICATION_COMMIT
    x["receipt_sha256"] = BASE.sha_json(x)
    return x


BASE.compact_receipt = compact_receipt_hardened


def fast_binding(seed: int, arm: str, prepared, tok, device):
    return FAST.run_binding_probe_from_checkpoint(BASE, seed, arm, prepared, tok, device)


BASE.run_binding_probe_from_checkpoint = fast_binding


def package_results_fixed() -> Path:
    rd = BASE.result_dir()
    out = rd / "ARKENSTONE_ARK018_SCIENCE_BIRTH_RESULTS.zip"
    content_manifest_path = rd / "ARK-018_ZIP_CONTENT_MANIFEST.json"
    members = {}
    for p in sorted(rd.glob("*.json")):
        if p.name == content_manifest_path.name:
            continue
        members[p.name] = BASE.sha256_file(p)
    BASE.safe_save_json(content_manifest_path, {
        "schema": "arkenstone-ark018-zip-content/v4",
        "members_sha256": members,
        "note": "Final ZIP SHA256 is written as a sidecar after ZIP creation to avoid a self-referential hash.",
    })
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(rd.glob("*.json")):
            zf.write(p, p.name)
        for p in [
            HERE / "EXECUTION_V3_ADDENDUM.md",
            HERE / "EXECUTION_V3_CLARIFICATIONS.md",
            HERE / "SCIENCE_BIRTH_PERIODIC_MIXTURE_ADDENDUM.md",
            HERE / "BIRTH_BOOK_PROBES.json",
        ]:
            if p.exists():
                zf.write(p, p.name)
    final_sha = BASE.sha256_file(out)
    (rd / "ARKENSTONE_ARK018_SCIENCE_BIRTH_RESULTS.zip.sha256").write_text(final_sha + "  " + out.name + "\n")
    print("RESULT ZIP:", out, out.stat().st_size, "bytes | SHA256", final_sha, flush=True)
    return out


BASE.package_results = package_results_fixed


if __name__ == "__main__":
    raise SystemExit(BASE.main())
