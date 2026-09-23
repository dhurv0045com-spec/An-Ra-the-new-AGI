# FORMATION-MUX-001 recovery runbook

Status: **BLOCKED ON THE ORIGINAL KAGGLE SAVED OUTPUT**  
Campaign snapshot: 24/24 S5 development arms; 2/24 TIE-ROLE frontier arms; no sealed evaluation  
Recovery rule: no retraining, no new seeds, no protocol changes, and no sealed access until custody passes

## What the attached 6,119 KB item is

`FORMATION_MUX_001_RESULTS (2)` is consistent by displayed size with the preserved evidence archive:

- preserved file: `FORMATION_MUX_001_RESULTS.partial.zip`
- bytes: `6,265,814`
- KiB: `6,118.959`, displayed by Kaggle as `6,119 KB`
- ZIP members: 755
- actual `resume.pt` checkpoint payloads: 0

The `(1)` versus `(2)` filename identity is not independently proven, but downloading `(2)` alone is not expected to recover checkpoint state. It is a result bundle, not the complete saved Output tree.

## Required user action

1. Open the Kaggle notebook version that produced the 2026-09-23 dual-T4 session.
2. Open that version's **Output** view. Do not delete or overwrite the version.
3. Preserve the complete Output tree beginning with:

   ```text
   /kaggle/working/FORMATION_MUX_001/
   ```

4. Download or attach the entire saved Output, not only the generated `FORMATION_MUX_001_RESULTS.zip` evidence file.
5. Keep at least these checkpoint paths:

   ```text
   FORMATION_MUX_001/CS-MECH-002/*/S1/S2/S3/S4/resume.pt
   FORMATION_MUX_001/REP-FORM-003A/*/S1/S2/S3/S4/resume.pt
   FORMATION_MUX_001/TIE-ROLE-001/T0_CANONICAL/S1/resume.pt
   FORMATION_MUX_001/TIE-ROLE-001/T0_CANONICAL/S2/resume.pt
   ```

6. Also retain the same-run `CHECKPOINT_RECEIPT.json`, `CAMPAIGN_STATE.json`, `TIE_ROLE_FRONTIER_STATE.json`, `PUBLIC_SURFACE_MANIFEST.json`, and `TOKENIZER_RECEIPT.json` files.

A Kaggle-downloaded archive is acceptable only if it contains the checkpoint tree. The recovery preflight deliberately rejects an evidence-only result ZIP.

## Frozen identities

The recovered state must bind all of the following:

- Science S5 commit: `c15ad8beb409537db42d075684ea54847a074ebd`
- operator v12 commit: `4ee05f6e386f15d34f9dfa7bd7f3300a496b9896`
- operator v12 Git blob: `e9e1f701b0d4edc509194da55fe1ba37ed62ef86`
- public surface identity: `f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c`
- tokenizer artifact identity: `97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc`

## Recovery preflight

Run from a fresh Kaggle notebook version with the complete Output attached under `/kaggle/input`:

```python
import pathlib
import subprocess
import sys

RECOVERY_REPO = pathlib.Path('/kaggle/temp/formation-mux-recovery-preflight')
RECOVERY_COMMIT = '<PIN_AFTER_RECOVERY_COMMIT>'
REMOTE = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'

subprocess.run([
    'git', 'clone', '--no-checkout', REMOTE, str(RECOVERY_REPO)
], check=True)
subprocess.run([
    'git', '-C', str(RECOVERY_REPO), 'fetch', 'origin'
], check=True)
subprocess.run([
    'git', '-C', str(RECOVERY_REPO), 'switch', '--detach', RECOVERY_COMMIT
], check=True)
subprocess.run([
    sys.executable, '-m', 'tools.formation_mux_001_recovery_preflight',
    '--input', '/kaggle/input',
    '--out', '/kaggle/working/FORMATION_MUX_001',
    '--install',
], cwd=str(RECOVERY_REPO), check=True)
```

Expected successful receipt:

- `status: PASS`
- `s5_completed_arms: 24`
- `frontier_completed_arms: 2` or a later exact continuation count
- `official_checkpoint_count: 26` or greater
- `raw_sealed_rows_read: false`

## If preflight fails

Stop. Do not run the operator and do not work around the failure. Preserve the attached Output and the printed error. Typical failures mean:

- `EVIDENCE_ONLY_NOT_RESUMABLE`: the result ZIP was attached instead of the saved Output tree;
- missing checkpoint: the complete Output was not attached;
- hash mismatch: checkpoint and receipt are from different runs or the file is damaged;
- ambiguous recovery input: more than one saved campaign was attached;
- science/operator/surface/tokenizer mismatch: this is not the pinned execution;
- interrupted sealed marker: manual sealed-custody review is required before any continuation.

## Continuation gate

Only after `RECOVERY_PREFLIGHT.json` is present and valid may the exact pinned v12 operator run under the approved Kaggle T4×2 configuration. Completed arms must remain immutable. The scientific endpoint remains the remaining frozen TIE-ROLE frontier work; this recovery action does not authorize a new experiment or any capability/production claim.

## Current evidence interpretation

- S5: development arms complete, but no sealed verdict.
- REP-FORM-003A: development floor-level result with matched exposure; not an equivalence proof.
- TIE-ROLE: two canonical controls are not a treatment contrast; no frontier conclusion is possible.
- Sealed evaluation: not consumed.
- Architecture promotion, production readiness, capability, and AGI claims: unsupported.
