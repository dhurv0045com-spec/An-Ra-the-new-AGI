# FORMATION-MUX-001 Kaggle session snapshot

Status: **PARTIAL CAMPAIGN SNAPSHOT — NOT A FINAL SCIENTIFIC RESULT**

## Provenance

- Source archive supplied by the user: `FORMATION_MUX_001_RESULTS (1).zip`
- Preserved archive: `FORMATION_MUX_001_RESULTS.partial.zip`
- SHA-256: `3e3ad68cd7f80bd242733b61153d4bb8f3fedbb1e4fba8fbc0f3ebc5d904423f`
- Archive integrity: ZIP CRC check passed; 755 unique members; no unsafe paths or duplicate names.
- Frozen Science S5 commit: `c15ad8beb409537db42d075684ea54847a074ebd`
- Recorded operator HEAD: `4ee05f6e386f15d34f9dfa7bd7f3300a496b9896`
- Hardware receipt: official dual-T4 execution (`Tesla T4` ×2).

## What completed

- Science S5 / `CS-MECH-002`: 16/16 official arms complete (4 arms × 4 seeds); 2,000 updates per arm.
- Science S5 / `REP-FORM-003A`: 8/8 official arms complete (2 arms × 4 seeds); each arm reached its 500,000-token exposure target.
- S5 campaign state: `ARMS_COMPLETE`, 24/24 arms; no global failure.
- TIE-role frontier: 2/24 arms complete (canonical T0, seeds S1 and S2); state is `PARTIAL_SESSION` and its wall guard stopped further launches.
- Recorded operator elapsed time for the frontier session: about 9.27 hours.
- The two development-only aggregate files are included in the archive; their endpoints are development measurements only.

## What did not complete

- Sealed evaluations were **not consumed** for either S5 experiment or either frontier experiment.
- No `FINAL_RESULT.json` or architecture gate was produced. There is no campaign verdict and no basis here for changing or promoting an architecture.
- The frontier needs 22 more official arms; the `TIE-ROLE-XFER-001` official arms have not started.
- The ZIP intentionally excludes exact-resume checkpoint files; the frozen package README says those remain in Kaggle Output. This archive alone is not sufficient to resume. To continue, attach the original Kaggle Output/state to the exact pinned notebook run.

## Data handling and claim ceiling

The archive contains the public synthetic surface, development outputs, progress snapshots, qualification receipts, and runtime logs. It contains no raw sealed rows and passed a scan for common private-key/API-token patterns. This is still a partial execution snapshot, not a model-capability result, final experiment result, or authorization for a production training run.

## Recovery audit — 2026-09-24

- The user screenshot's `FORMATION_MUX_001_RESULTS (2)` item is 6,119 KB. The preserved archive is 6,265,814 bytes = 6,118.959 KiB, which rounds to the same displayed size. It is consistent with another download of this evidence result ZIP, but the `(1)` / `(2)` filename provenance is not independently resolved.
- No `resume.pt`, `.pt`, `.pth`, `.bin`, or `.safetensors` checkpoint payload exists among the 755 ZIP members. `CHECKPOINT_RECEIPT.json` files are receipts, not model/optimizer/RNG state.
- No checkpoint-bearing copy of this execution was found in the repository, local Downloads, or an available connected Kaggle account/tool. This proves repository-side absence only; it does not prove the files are absent from the original Kaggle saved Output.
- Exact continuation requires the original saved Output tree for this notebook version, including at least the 24 completed S5 `resume.pt` files and the two completed `TIE-ROLE-001/T0_CANONICAL/S1,S2` checkpoint files. Validate every checkpoint against its same-run `CHECKPOINT_RECEIPT.json` before launching the pinned operator.
- Use `tools/formation_mux_001_recovery_preflight.py` before any resume. Receipt schema v2 at commit `8609ba95f4e978cf3cdf8d20bd8a907eea8f6728`, Git blob `b507c2350a885e5e3432cac6f135fa58df52b69e`, derives the exact 48 official slots from the pinned S5/frontier protocols, validates the real S5 public surface, safely loads official T4 checkpoints with `torch.load(weights_only=True)`, and cross-checks endpoint exposure, model/optimizer/RNG topology, initial-model identity, result/receipt/progress state, trace formation, tokenizer commitments, state counts, failure latches, and sealed custody. It binds checkpoint and completed-result hashes and installs only to a fresh atomically no-replace path.
- The canonical recovery launcher is `notebooks/CYMEK_FORMATION_MUX_001_RECOVERY_T4X2.ipynb`; it pins the recovery and operator commits/blobs, requires a same-kernel PASS nonce, and rechecks completed arm hashes after v12 returns. Pinned v12's historical global-failure latch, sealed transaction edge, and non-atomic result packaging remain known engineering risks; the preflight fails closed rather than clearing or bypassing them.
- The next scientific action remains conditional: exact continuation of the remaining 22 frozen TIE-ROLE frontier development arms after successful recovery. If development reaches 24/24, pinned v12 automatically performs the frozen development-only diagnostics, sealed finalization, and architecture gate. Retraining from this ZIP would be a new execution, not recovery.

Focused CI run `35923273288` failed closed on the first recovery qualification attempt. Its 60 passing tests included the real archive/manifest contracts, while 10 failures exposed one preflight tuple-versus-string control-key defect and one notebook test-order assertion. Commit `8609ba95f4e978cf3cdf8d20bd8a907eea8f6728` corrects both.

Replacement GitHub Actions run `35925460488` passed on head `ee6a5cad4a4f39752b2646a60eeaeef8d2b06019`. It compiled the S5/frontier/recovery surface, passed the focused recovery and existing Formation-Mux tests, validated the preserved real 60k public surface and evidence archive, checked immutable science identities, and compiled/validated notebook pins. Recovery engineering is qualified; execution remains blocked on the original checkpoint-bearing saved Output.

Targeted recovery tests and focused CI coverage were not executed locally because the user prohibited local hardware use. The passing claim above is remote CI evidence only.

No experiment arm, sealed evaluation, training update, or local hardware workload was run during this recovery audit.

