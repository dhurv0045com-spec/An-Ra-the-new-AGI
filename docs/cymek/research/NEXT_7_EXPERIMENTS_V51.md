# Current research roadmap — V5.1 (next-core)

Updated: 2026-09-14. One canonical queue; older roadmaps are historical.

| # | Campaign | Question | State |
|---|----------|----------|-------|
| 1 | **FORMATION-MUX-001** (CS-MECH-002 + REP-FORM-003A) | tied-row mechanism dissection (decay / trainability / denominator) + rendering (BPE vs isomorphic) at fixed V24576 | **Science S4 frozen; CPU/static + real-tokenizer surface qualification PASS; KAGGLE_T4X2_NOT_YET_RUN** |
| 2 | REP-FORM-003B | only if Stage A leaves representation insufficiently explained | gated |
| 3 | P35-TRANSFER-004 | does the adjudicated mechanism survive the P35 finalist geometry | gated |
| 4 | COG-MIX-OBJ-005 | cognition-mixture and objective interaction at fixed geometry | gated |
| 5 | OPT-WSD-006 | optimization/WSD schedule at the winning configuration | gated |
| 6 | TPU-SYSTEM-007 | early TPU systems qualification (engineering, parallel track) | parallel |
| 7 | M102-INTEGRATED-008 | integrated mid-scale pilot; NOT authorized until 1-5 earn it | gated |

## FORMATION-MUX-001 current binding

Pre-execution audit history is preserved and must not be launched:

- S1: `36ab16a4950d582fcc26b239dfc8c0ba816911bb` — failed pre-execution audit;
- S2: `534dcccfb8f96a30e80d71a30160e6a16eaa1ede` — repaired S1, then superseded prospectively before outcomes to isolate frozen-row treatment from global clipping;
- S3: `e157835a52a41696ca56512477584fd267391ced` — repaired clip isolation, then superseded prospectively before outcomes because its worker manifest still exposed raw sealed rows.

Current prospective execution authority:

- **Science S4:** `ad25f6944cdd5ac5f47a6bf3a667322cb985314b`;
- **canonical operator:** `tools/formation_mux_001_kaggle_operator_v7.py` at `04278c570aeb1ccba1609ec0e7334b6bfedf894b`, git blob `a0c00cfa61247fc4fd9ca14e0619ac10ae3bc365`;
- **canonical notebook:** `notebooks/CYMEK_FORMATION_MUX_001_KAGGLE_T4X2.ipynb`, pin commit `56bd9c4b9303a41ccac4302e3477b3f4142d4e85`;
- **readiness:** `docs/cymek/experiments/FORMATION-MUX-001/RUN_READINESS_V4.json`;
- official execution requires Kaggle `GPU T4 x2` and Internet ON;
- no FORMATION-MUX scientific outcome and no actual Kaggle T4 x2 runtime-pass claim exists yet.

Science S4 keeps the S3 model/optimizer/representation treatments unchanged and adds strict sealed custody: workers see only training + development rows; before science only cryptographic sealed commitments persist; raw sealed examples are regenerated and verified in coordinator memory only after the corresponding development aggregate is frozen and its sealed marker is STARTED. Raw sealed rows are never written to campaign output or the result ZIP.

Pre-execution qualification on GitHub Actions run `34883240785` passed: S4 sources compiled, **25 targeted tests passed**, all **29 immutable science files** matched S4, the real frozen V24576 tokenizer produced the expected 3600/480 public surface and reproducible 720-row sealed commitments, and the canonical notebook JSON/pins validated. This is engineering evidence only, not GPU runtime or scientific evidence.

Operational note: later campaigns may be multiplexed onto shared sessions when operationally compatible, but each keeps separate preregistrations, data identities, sealed sets, verdicts, and claim ceilings.
