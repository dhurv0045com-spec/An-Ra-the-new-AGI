"""Self-verifying ledgers, redesigned (ARK-004A-R mission section 10).

Design (option B -- immutable content hashes; no self-reference problem):
- LEDGER_VERIFICATION.json stores sha256 of every governed file at check time.
- A later verification recomputes those hashes: any mismatch = DRIFTED file
  (mutation detection without needing commit SHAs).
- The README stamp is past-tense ("stamped-at-commit") and classification is
  drift-based on stored content hashes -- never a self-claim about the commit
  that contains the stamp.
- Receipts: legacy ARK-001 receipts hash with default json separators; newer
  receipts use compact separators. The verifier tries both and records which
  scheme matched; a receipt matching neither is flagged. History is never
  rewritten to satisfy the verifier.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOVERNED = [
    "docs/arkenstone/EXPERIMENT_LOG.md",
    "docs/arkenstone/MECHANISM_TOURNAMENT.md",
    "docs/arkenstone/NEGATIVE_RESULTS.md",
    "docs/arkenstone/AGI_FEATURE_LEDGER.md",
    "docs/arkenstone/NOVELTY_REGISTER.md",
    "docs/arkenstone/README.md",
]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_variants(payload: dict) -> list:
    stripped = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    return [
        json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        json.dumps(stripped, sort_keys=True).encode("utf-8"),
    ]


def check_receipt(path: Path) -> dict:
    """Dual-scheme receipt verification. Never mutates the receipt."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if "receipt_sha256" not in receipt:
        return {"file": str(path), "status": "NO_SELF_HASH", "scheme": None}
    stored = receipt["receipt_sha256"]
    for scheme, canonical in zip(("compact", "default-separators"), _canonical_variants(receipt)):
        if hashlib.sha256(canonical).hexdigest() == stored:
            return {"file": str(path), "status": "VERIFIED", "scheme": scheme}
    return {"file": str(path), "status": "HASH_MISMATCH", "scheme": None}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def verify(root: Path = REPO) -> dict:
    problems: list[str] = []
    verification_path = Path(root) / "docs/arkenstone/LEDGER_VERIFICATION.json"
    previous: dict = {}
    if verification_path.exists():
        try:
            previous = json.loads(verification_path.read_text(encoding="utf-8"))
        except ValueError:
            problems.append("LEDGER_VERIFICATION.json is unparseable")
    drifted = []
    if previous.get("governed_file_hashes"):
        for rel, stored in previous["governed_file_hashes"].items():
            path = root / rel
            if not path.exists():
                drifted.append(rel)
            elif _sha_file(path) != stored:
                drifted.append(rel)

    log_text = (Path(root) / "docs/arkenstone/EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    for ref in sorted(set(re.findall(r"(experiments/[\w\-./]+)", log_text))):
        if not (root / ref).exists():
            problems.append(f"referenced artifact missing: {ref}")

    receipts = []
    for path in sorted((Path(root) / "experiments").rglob("*.json")):
        if path.name in {"LEDGER_VERIFICATION.json", "TASK_MANIFEST.json"}:
            continue
        try:
            receipts.append(check_receipt(path))
        except ValueError as exc:
            receipts.append({"file": str(path), "status": "UNPARSEABLE", "scheme": str(exc)})
    for entry in receipts:
        if entry["status"] in {"HASH_MISMATCH", "UNPARSEABLE"}:
            problems.append(f"{entry['file']}: {entry['status']}")

    verification = {
        "schema": "arkenstone-ledger-verification/v2",
        "stamped_at_commit": _head(),
        "governed_file_hashes": {rel: _sha_file(Path(root) / rel) for rel in GOVERNED
                                 if (Path(root) / rel).exists()},
        "receipt_checks": receipts,
        "drifted_since_last_check": drifted,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


def stamp_readme() -> None:
    """Past-tense stamp: names the commit whose tree the check ran against."""
    readme_path = REPO / "docs/arkenstone/README.md"
    readme = readme_path.read_text(encoding="utf-8")
    block = f"stamped-at-commit: {_head()}\n"
    if "stamped-at-commit:" in readme:
        readme = re.sub(r"stamped-at-commit:\s*[0-9a-f]{40}\n", block, readme)
    elif "verified-at-commit:" in readme:
        readme = re.sub(r"verified-at-commit:\s*[0-9a-f]{40}\n", block, readme)
    else:
        readme = readme.rstrip("\n") + "\n\n---\n\n" + block
    readme_path.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    result = verify()
    stamp_readme()
    print(json.dumps({k: result[k] for k in ("status", "drifted_since_last_check", "problems")}, indent=1))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
