# Current research roadmap — V5.1 (next-core)

Updated: 2026-09-15. One canonical queue; older roadmaps are historical.

| # | Campaign | Question | State |
|---|----------|----------|-------|
| 1 | **FORMATION-MUX-001** (CS-MECH-002 + REP-FORM-003A) | tied-row mechanism dissection (decay / trainability / denominator) + rendering (BPE vs isomorphic) at fixed V24576 | **Science S5 frozen; CPU/static + real-tokenizer 60k-surface qualification PASS; KAGGLE_T4X2_NOT_YET_RUN** |
| 2 | REP-FORM-003B | only if Stage A leaves representation insufficiently explained | gated |
| 3 | P35-TRANSFER-004 | does the adjudicated mechanism survive the P35 finalist geometry | gated |
| 4 | COG-MIX-OBJ-005 | cognition-mixture and objective interaction at fixed geometry | gated |
| 5 | OPT-WSD-006 | optimization/WSD schedule at the winning configuration | gated |
| 6 | TPU-SYSTEM-007 | early TPU systems qualification (engineering, parallel track) | parallel |
| 7 | M102-INTEGRATED-008 | integrated mid-scale pilot; NOT authorized until 1-5 earn it | gated |

## FORMATION-MUX-001 current binding

Pre-execution audit history is preserved and must not be launched:

- S1 `36ab16a4950d582fcc26b239dfc8c0ba816911bb`: failed pre-execution audit;
- S2 `534dcccfb8f96a30e80d71a30160e6a16eaa1ede`: repaired S1, then superseded before outcomes for clip isolation;
- S3 `e157835a52a41696ca56512477584fd267391ced`: repaired clip isolation, then superseded before outcomes for sealed custody;
- S4 `ad25f6944cdd5ac5f47a6bf3a667322cb985314b`: repaired sealed custody, then superseded prospectively before outcomes to remove severe small-surface reuse and bind long-run checkpoint durability.

Current prospective execution authority:

- **Science S5:** `c15ad8beb409537db42d075684ea54847a074ebd`;
- **canonical operator:** `tools/formation_mux_001_kaggle_operator_v8.py` at `4e3bb50c683d94a09801703dfb9f469debe64a50`, git blob `b427269f682322c9ce58f1677710c5979c76465c`;
- **canonical notebook:** `notebooks/CYMEK_FORMATION_MUX_001_KAGGLE_T4X2.ipynb`, pin commit `c01970cf6afc5434cc6e48562faa4979fbab969a`;
- **readiness:** `docs/cymek/experiments/FORMATION-MUX-001/RUN_READINESS_V5.json`;
- official execution requires Kaggle `GPU T4 x2` and Internet ON;
- no FORMATION-MUX scientific outcome and no actual Kaggle T4 x2 runtime-pass claim exists yet.

Science S5 keeps the S4 causal questions, arms, seeds, endpoints, thresholds, and sealed firewall. It expands only the synthetic training surface to **60,000 unique rows** (10,000/family) while keeping development at 480 and sealed at 720. CS-MECH consumes 32,000 row presentations per arm, so it does not wrap its training permutation before the endpoint. REP-FORM remains matched by 500,000 actual processed non-padding tokens.

Durability is now explicit: CS-MECH exact-resume checkpoints every **200 updates**; REP-FORM checkpoints every **10,000 processed tokens** because token exposure, not step count, is its causal matching axis. Every checkpoint emits `LATEST_PROGRESS.json` and an immutable `progress/UPDATE_*.json` diagnostic snapshot. The campaign may span as many Kaggle sessions as needed; a session boundary cannot change frozen science.

Pre-execution qualification on GitHub Actions run `34900925973` passed: S5 sources compiled, **28 targeted tests passed**, all **38 immutable science files** matched S5, the real frozen V24576 tokenizer produced the expected 60,000/480 public surface and reproducible 720-row sealed commitments for each experiment, and the canonical notebook pins validated.

Operational note: later campaigns may be multiplexed onto shared sessions when compatible, but each keeps separate preregistrations, data identities, sealed sets, verdicts, and claim ceilings.