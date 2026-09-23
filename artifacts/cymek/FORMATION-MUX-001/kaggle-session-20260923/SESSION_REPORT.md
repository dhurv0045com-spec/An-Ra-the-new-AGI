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

