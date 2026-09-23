"""Build a non-destructive Kaggle launcher for the frozen Formation-MUX campaign."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "notebooks" / "CYMEK_FORMATION_MUX_001_KAGGLE_T4X2.ipynb"
OUTPUT = ROOT / "notebooks" / "CYMEK-BETA-ALL-EXPERIMENTS-KAGGLE-T4X2.ipynb"
OPERATOR_COMMIT = "4ee05f6e386f15d34f9dfa7bd7f3300a496b9896"
OPERATOR_BLOB = "e9e1f701b0d4edc509194da55fe1ba37ed62ef86"


def build() -> dict[str, object]:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    metadata = notebook["metadata"]
    assert metadata["operator_commit"] == OPERATOR_COMMIT
    assert metadata["operator_blob"] == OPERATOR_BLOB
    assert len(notebook["cells"]) == 4

    markdown = "".join(notebook["cells"][0]["source"])
    markdown = markdown.replace(
        "# CYMEK FORMATION-MUX-001 + TIE-ROLE FRONTIER — Kaggle T4 x2",
        "# Cymek Beta: full preregistered experiment campaign — Kaggle T4 ×2",
    )
    markdown = markdown.replace(
        "The two T4s are independent matched-seed arm workers; there is no DDP.",
        "This is the useful long-run GPU campaign: both T4s run independent matched-seed arm workers concurrently (no DDP and no artificial idle-GPU loop). The full frozen campaign may take more than one Kaggle session; its operator saves exact-resume state and never shortens the science to fit a session. Actual wall time depends on Kaggle hardware/runtime and is not guaranteed to be exactly 5–6 hours.",
    )
    markdown = markdown.replace(
        "Stage 1 is frozen **Science S5**:",
        "The training surface is deterministic, development-scale synthetic data: 60,000 training rows across six families, 480 development rows, and 720 sealed rows. It is for controlled mechanism/representation research, not production pretraining and not evidence of model capability by itself.\n\nThe campaign comprises four preregistered training experiments (48 official matched arms total) plus two development-only diagnostics. Stage 1 is frozen **Science S5**:",
    )
    markdown = markdown.replace(
        "If either state reports `PARTIAL_SESSION`, save the Kaggle version/output, attach that output to the next run, and rerun this exact notebook.",
        "If either state reports `PARTIAL_SESSION`, save the Kaggle version/output and attach the complete saved `/kaggle/working/FORMATION_MUX_001` tree to the next run of this exact notebook. Do not attach only `FORMATION_MUX_001_RESULTS.zip`: that evidence bundle intentionally omits `resume.pt` and is not resumable. Run `tools/formation_mux_001_recovery_preflight.py` and require PASS before launching the operator. The result ZIP and campaign state stay under `/kaggle/working/FORMATION_MUX_001`; source code is cloned into `/kaggle/temp` so rerunning this notebook does not reset, clean, or remove user files in `/kaggle/working`.",
    )
    notebook["cells"][0]["source"] = markdown.splitlines(keepends=True)

    setup = "".join(notebook["cells"][1]["source"])
    old_setup = '''REPO = pathlib.Path('/kaggle/working/An-Ra-the-new-AGI')
REMOTE = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'
OPERATOR_COMMIT = '4ee05f6e386f15d34f9dfa7bd7f3300a496b9896'
OPERATOR_PATH = 'tools/formation_mux_001_kaggle_operator_v12.py'
OPERATOR_BLOB = 'e9e1f701b0d4edc509194da55fe1ba37ed62ef86'
SCIENCE_COMMIT_S5 = 'c15ad8beb409537db42d075684ea54847a074ebd'
if REPO.exists() and not (REPO / '.git').exists():
    shutil.rmtree(REPO)
if not REPO.exists():
    subprocess.run(['git', 'clone', REMOTE, str(REPO)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO), 'reset', '--hard'], check=True)
    subprocess.run(['git', '-C', str(REPO), 'clean', '-fdx'], check=True)
subprocess.run(['git', '-C', str(REPO), 'fetch', 'origin'], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '-f', '-q', OPERATOR_COMMIT], check=True)
'''
    safe_setup = '''OPERATOR_COMMIT = '4ee05f6e386f15d34f9dfa7bd7f3300a496b9896'
REPO = pathlib.Path('/kaggle/temp/formation-mux-source-' + OPERATOR_COMMIT[:12])
REMOTE = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'
OPERATOR_PATH = 'tools/formation_mux_001_kaggle_operator_v12.py'
OPERATOR_BLOB = 'e9e1f701b0d4edc509194da55fe1ba37ed62ef86'
SCIENCE_COMMIT_S5 = 'c15ad8beb409537db42d075684ea54847a074ebd'
if REPO.exists():
    if not (REPO / '.git').exists():
        raise RuntimeError(f'Refusing to remove or overwrite non-Git path: {REPO}')
    status = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'],
                            capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise RuntimeError(f'Refusing to modify dirty source checkout; preserve it and choose a new /kaggle/temp path: {REPO}')
    existing_head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                   capture_output=True, text=True, check=True).stdout.strip()
    if existing_head != OPERATOR_COMMIT:
        raise RuntimeError(f'Refusing to switch an existing checkout from {existing_head}; choose a new /kaggle/temp path')
else:
    subprocess.run(['git', 'clone', '--no-checkout', REMOTE, str(REPO)], check=True)
    subprocess.run(['git', '-C', str(REPO), 'fetch', 'origin'], check=True)
    subprocess.run(['git', '-C', str(REPO), 'switch', '--detach', OPERATOR_COMMIT], check=True)
'''
    assert old_setup in setup
    setup = setup.replace(old_setup, safe_setup)
    notebook["cells"][1]["source"] = setup.splitlines(keepends=True)

    runner = "".join(notebook["cells"][2]["source"])
    runner = runner.replace(
        "'--repo', '/kaggle/working/An-Ra-the-new-AGI',",
        "'--repo', str(REPO),",
    )
    runner = runner.replace(
        "cwd='/kaggle/working/An-Ra-the-new-AGI'",
        "cwd=str(REPO)",
    )
    assert "'/kaggle/working/An-Ra-the-new-AGI'" not in runner
    notebook["cells"][2]["source"] = runner.splitlines(keepends=True)

    metadata.update(
        {
            "experiment": "FORMATION-MUX-001 + TIE-ROLE-FRONTIER-001",
            "schema": "anra.formation-mux-kaggle-safe-wrapper/v1",
            "source_checkout": "/kaggle/temp/formation-mux-source-<operator-commit>",
            "output_directory": "/kaggle/working/FORMATION_MUX_001",
            "science_and_operator_pins_unchanged": True,
        }
    )
    return notebook


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
