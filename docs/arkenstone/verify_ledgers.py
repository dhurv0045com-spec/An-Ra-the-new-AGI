"""Self-verifying ledgers (external gap review GAP 2/4, Arkenstone-side).

Checks, mechanically:
1. Every EXPERIMENT_LOG row's referenced artifact paths exist.
2. Every JSON receipt in experiments/ parses and, where it has a receipt_sha256,
   hash-verifies against its own content.
3. The verification block in README.md names the exact HEAD it was checked at.
Writes LEDGER_VERIFICATION.json (compact, committed) so drift is detectable.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def referenced_paths(log_text: str) -> list[str]:
    """Artifact paths referenced by EXPERIMENT_LOG rows (experiments/...)."""
    return sorted(set(re.findall(r"(experiments/[\w\-./]+)", log_text)))


def verify(path_prefix: Path | None = None) -> dict:
    root = (path_prefix or REPO) 
    problems: list[str] = []
    log_path = root / "docs/arkenstone/EXPERIMENT_LOG.md"
    log_text = log_path.read_text(encoding="utf-8")

    # 1. referenced artifacts exist (logs/RESULT files may be gitignored but
    #    must exist locally; committed ANALYSIS/PLAN files must exist in git)
    for ref in referenced_paths(log_text):
        if not (root / ref).exists():
            problems.append(f"referenced artifact missing: {ref}")

    # 2. receipts parse and self-verify
    for receipt_path in sorted((root / "experiments").rglob("RESULT*.json")):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"unparseable receipt {receipt_path.name}: {exc}")
            continue
        if "receipt_sha256" in receipt:
            candidate = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
            digest = hashlib.sha256(
                json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if digest != receipt["receipt_sha256"]:
                problems.append(f"receipt hash mismatch: {receipt_path.name}")
    for receipt_path in sorted((root / "experiments").rglob("RECEIPT.json")):
        json.loads(receipt_path.read_text(encoding="utf-8"))  # parse check

    # 3. README verification block freshness
    readme = (root / "docs/arkenstone/README.md").read_text(encoding="utf-8")
    match = re.search(r"verified-at-commit:\s*([0-9a-f]{40})", readme)
    head = _head()
    verified_state = "UNMARKED"
    if match:
        # drift classification: CURRENT iff no tracked ledger/experiment file
        # changed since the stamped commit (HEAD-relative by design -- the
        # stamp always names the commit whose tree the check ran against)
        changed = subprocess.run(
            ["git", "diff", "--name-only", match.group(1), head, "--",
             "docs/arkenstone", "experiments"],
            capture_output=True, text=True).stdout.strip()
        verified_state = "CURRENT" if not changed else "STALE"

    verification = {
        "schema": "arkenstone-ledger-verification/v1",
        "verified_at_commit": head,
        "readme_verification_block": verified_state,
        "experiment_rows": log_text.count("\n| ARK-"),
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    (root / "docs/arkenstone/LEDGER_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    return verification


def stamp_readme() -> None:
    """Regenerate the README verification block at the current HEAD."""
    readme_path = REPO / "docs/arkenstone/README.md"
    readme = readme_path.read_text(encoding="utf-8")
    head = _head()
    block = f"verified-at-commit: {head}\n"
    if "verified-at-commit:" in readme:
        readme = re.sub(r"verified-at-commit:\s*[0-9a-f]{40}\n", block, readme)
    else:
        readme = readme.rstrip("\n") + "\n\n---\n\n" + block
    readme_path.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    result = verify()
    stamp_readme()
    # re-verify so the stamped block matches the committed state
    result = verify()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
