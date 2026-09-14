# Current research roadmap — V5.1 (next-core)

Updated: 2026-09-14. One canonical queue; older roadmaps are historical.

| # | Campaign | Question | State |
|---|----------|----------|-------|
| 1 | **FORMATION-MUX-001** (CS-MECH-002 + REP-FORM-003A) | tied-row mechanism dissection (decay / trainability / denominator) + rendering (BPE vs isomorphic) at fixed V24576 | **Science S3 frozen + canonical Kaggle operator pinned; KAGGLE_T4X2_NOT_YET_RUN** |
| 2 | REP-FORM-003B | only if Stage A leaves representation insufficiently explained | gated |
| 3 | P35-TRANSFER-004 | does the adjudicated mechanism survive the P35 finalist geometry | gated |
| 4 | COG-MIX-OBJ-005 | cognition-mixture and objective interaction at fixed geometry | gated |
| 5 | OPT-WSD-006 | optimization/WSD schedule at the winning configuration | gated |
| 6 | TPU-SYSTEM-007 | early TPU systems qualification (engineering, parallel track) | parallel |
| 7 | M102-INTEGRATED-008 | integrated mid-scale pilot; NOT authorized until 1-5 earn it | gated |

## FORMATION-MUX-001 current binding

Pre-execution audit history is preserved and must not be launched:

- S1: `36ab16a4950d582fcc26b239dfc8c0ba816911bb` — failed pre-execution audit;
- S2: `534dcccfb8f96a30e80d71a30160e6a16eaa1ede` — repaired most S1 defects, then superseded prospectively before outcomes to isolate frozen-row treatment from global clipping.

Current prospective execution authority:

- **Science S3:** `e157835a52a41696ca56512477584fd267391ced`;
- **canonical operator:** `tools/formation_mux_001_kaggle_operator_v4.py` pinned at `88910df06ffca4a467345b3c33dc50e81b414ce8`;
- **canonical notebook:** `notebooks/CYMEK_FORMATION_MUX_001_KAGGLE_T4X2.ipynb` pinned at notebook commit `462e14cf958753c403ab3dc5b073649ca6b620d2`;
- official execution requires Kaggle `GPU T4 x2` and Internet ON;
- no scientific outcome or actual Kaggle runtime-pass claim exists yet.

Science S3 additionally keeps M2 frozen extra-row gradients inside the common global-clip boundary while excluding those rows from optimizer update/state/decay. This prevents the M1→M2 contrast from changing the clip denominator as a second direct intervention.

Operational note: later campaigns may be multiplexed onto shared sessions when operationally compatible, but each keeps separate preregistrations, data identities, sealed sets, verdicts, and claim ceilings.
