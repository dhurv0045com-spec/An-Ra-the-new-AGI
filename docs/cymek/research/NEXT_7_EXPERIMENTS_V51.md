# Current research roadmap — V5.1 (next-core)

Updated: 2026-09-20. One canonical queue; older roadmaps (`doexperiment.md`, `agent.md`, `docs/research/NEXT_3_EXPERIMENTS.md`-era planning) are HISTORICAL planning only and must not be executed without a fresh hash-bound preregistration. This file is the sole authority for next-Core experiment order.

| # | Campaign | Question | State |
|---|---|----------|-------|
| 0 | **METRIC-RES-001** (checkpoint-only identity-resolution audit) | is S5 identity/copy **forming-but-unmeasured** or **genuinely absent**? can the frozen contrasts be read at token resolution? | **NEXT SINGLE ACTION — preregistered, no training, 16 preserved checkpoints, 10–40 min on Colab T4 or Kaggle T4x2. This is the action the S5 returned-bundle audit prescribed verbatim; the tooling did not exist and now does** |
| 0a | **LR/CLIP CHOKE PROBE** (engineering triage, advisory only) | is the S5 floor optimization-choke (clip_fraction 1.0 everywhere)? | `tools/formation_mux_lr_clip_probe_v1.py` (`--lr-probe-only`); 2×300-update M0 lanes, frozen science. **Defer until METRIC-RES-001 reports OPTIMIZATION_CHOKED** — running both in parallel is redundant because METRIC-RES-001 replays clip telemetry for free |
| 0b | **FORMATION-BASELINE-GATE-001 / FORMATION-DIAG-001** (Colab) | can M0 leave the identity floor? | `MIXTURE_OR_DATA_LIMITED` branch points here. Prospective; no sealed reads; S5 verdicts unchanged |
| 1 | **FORMATION-MUX-001** (CS-MECH-002 + REP-FORM-003A) | tied-row mechanism dissection (decay / trainability / denominator) + rendering (BPE vs isomorphic) at fixed V24576 | **S5 EXECUTED 24/24 on Kaggle T4x2; formal NULLs corrected to INCONCLUSIVE_AT_ZERO_BASELINE; DO NOT re-sweep until gate 0 passes** |
| 2 | REP-FORM-003B | only if Stage A leaves representation insufficiently explained | gated |
| 3 | P35-TRANSFER-004 | does the adjudicated mechanism survive the P35 finalist geometry | gated |
| 4 | COG-MIX-OBJ-005 | cognition-mixture and objective interaction at fixed geometry | gated |
| 5 | OPT-WSD-006 | optimization/WSD schedule at the winning configuration | gated |
| 6 | TPU-SYSTEM-007 | early TPU systems qualification (engineering, parallel track) | parallel |
| 7 | M102-INTEGRATED-008 | integrated mid-scale pilot; NOT authorized until 1-5 earn it | gated |

## FORMATION-MUX-001 S5 historical execution record

Pre-execution audit history is preserved and must not be launched:

- S1 `36ab16a4950d582fcc26b239dfc8c0ba816911bb`: failed pre-execution audit;
- S2 `534dcccfb8f96a30e80d71a30160e6a16eaa1ede`: repaired S1, then superseded before outcomes for clip isolation;
- S3 `e157835a52a41696ca56512477584fd267391ced`: repaired clip isolation, then superseded before outcomes for sealed custody;
- S4 `ad25f6944cdd5ac5f47a6bf3a667322cb985314b`: repaired sealed custody, then superseded prospectively before outcomes to remove severe small-surface reuse and bind long-run checkpoint durability.

The following records identify the completed S5 execution; they do not authorize a re-run:

- **Science S5:** `c15ad8beb409537db42d075684ea54847a074ebd`;
- **operator used:** `tools/formation_mux_001_kaggle_operator_v12.py` (storage-hardened + quota lanes);
- **historical operator:** the S5 notebook was used for the completed campaign and removed from the active `notebooks/` launch queue. Its frozen experiment record and preregistered source hash remain under `docs/cymek/experiments/FORMATION-MUX-001/`;
- **readiness record:** `docs/cymek/experiments/FORMATION-MUX-001/RUN_READINESS_V5.json`;
- The original S5 session used Kaggle `GPU T4 x2` with Internet enabled for training; this is historical run configuration;
- S5 EXECUTED on Kaggle T4 x2 (8.14 h, 24/24 official arms COMPLETE, both sealed evaluations COMPLETE, no global failure, no wall-guard truncation).

## FORMATION-MUX-001 S5 observed outcome (post-outcome, does NOT alter frozen preregistration)

- Source bundle SHA-256: `859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5`; science commit `c15ad8beb409537db42d075684ea54847a074ebd`; observed operator `fdc2483fed0bb1a80d4bfad376aac84a66db6055`.
- CS-MECH-002 formal: all three primary identity contrasts `NULL` (extra-row decay mean dev AUC delta +0.001674 / sealed +0.029167; trainability +0.000335 / 0.0; denominator -0.001674 / -0.016667; sign consistency not met).
- REP-FORM-003A formal: production BPE vs isomorphic `NULL` (dev AUC delta 0.0, sealed 0.0; all four paired seeds at floor).
- Corrected interpretation per `docs/cymek/experiments/FORMATION-MUX-001/RETURNED_BUNDLE_AUDIT.md`: **INCONCLUSIVE_AT_ZERO_BASELINE** — identity formation AUC 0.000–0.008 in ALL 24 arms (CS-MECH: only 3/16 runs nonzero endpoint M0/73012=0.15, M1/73011=0.0125, M3/73013=0.0625, M2 all zero, none ≥0.5; REP-FORM: all 8 arms 0.0). Contrasts had no dynamic range; no conclusion about extra-row decay/trainability/denominator in either direction is licensed.
- Structured finding (same grammar/training/arms): termination (1-token counting) 1.00 exact by ~update 300 sustained to 2000; composition (two-hop, 1-token) ~0.29 endpoint still climbing; missing_info abstention ~0.14 flat with eos_rate 1.0 everywhere; identity/copy (multi-token exact) 0.0 flat 2000 updates every arm both renderings; binding/state_order ~0.0–0.05. Rendering exonerated (R1 isomorphic also 0.0). Prime suspect: exact-match metric resolution for multi-token answers (token-level accuracy / LCP / per-position teacher-forced accuracy may reveal formation before exact flips).
- Optimization warning: every CS-MECH official arm `clip_fraction = 1.0`; R1 REP-FORM every update clipped; R0 ~0.88–0.98. Matched contrasts stand, but substrate ran aggressively clipped.
- Exploratory only (NOT promotion): M3−M2 sealed composition mean +0.277083, positive 4/4 seeds — permanently frozen random extra rows in full denominator may be harmful vs masked rows. Must remain `EXPLORATORY_ONLY` until reproduced in a capable regime.
- Full record: `docs/cymek/experiments/FORMATION-MUX-001/observed/S5_KAGGLE_2026-09-15/POSTMORTEM.md`, `ARM_SUMMARY.json`, `BUNDLE_MANIFEST.json`, `RETURNED_BUNDLE_AUDIT.md`. S5 NULLs preserved unchanged as negative evidence.

## Next single action (supersedes the old S5 next_action)

1. Do NOT conclude the tied-row mechanism line is null — it is untested at this baseline. Do NOT start another mechanism sweep, P35 transfer, or 250M run.
2. **Run METRIC-RES-001** (`docs/cymek/experiments/METRIC-RES-001/`): checkpoint-only, no training, 16 preserved S5 checkpoints, decision tree frozen in `PREREGISTRATION.json`. It is the action the S5 returned-bundle audit prescribed; the tooling was missing and has now been built. Either platform works — Kaggle T4 x2 is 2-way sharded and costs ~10–20 min of the weekly pool; Colab T4 costs 20–40 min of the daily allowance. **Prefer Kaggle** to keep Colab free for gate 0b.
3. Follow the branch METRIC-RES-001 returns. `FORMING_BUT_UNMEASURED` → fix the endpoint and re-adjudicate contrasts before any retraining. `OPTIMIZATION_CHOKED` → S6 LR/schedule bracket. `MIXTURE_OR_DATA_LIMITED` → FORMATION-BASELINE-GATE-001. `GENUINELY_ABSENT` → abandon the vocabulary/mechanism line and go to P35-TRANSFER-004.
4. TIE-ROLE pilots / frontier, baseline-gate notebooks, and FORMATION-DIAG-001 are diagnostics under the same ceiling: they cannot alter S5 verdicts, promote Core architecture, change tokenizer/vocabulary, or authorize scale/cognition/AGI claims.
5. S6 optimizer amendment is PROSPECTIVE only (`AMENDMENT_S6_PROSPECTIVE.md` + `PREREGISTRATION_S6_DRAFT.json`): entry requires probe CHOKED/MIXED or a METRIC-RES-001 `OPTIMIZATION_CHOKED` verdict; needs its own science commit + audit before launch. Frozen S5 files stay byte-identical until then.

## Quota architecture (30 GPU-h/week shared, 12 h/session, CPU unlimited)

- GPU sessions train only; sealed scoring + packaging move to free CPU finalize sessions (fails closed unless ARMS_COMPLETE on attached Output).
- Parent processes never hold torch CUDA contexts; workers inherit `expandable_segments` + unbuffered streaming; operator heartbeats every arm defeat silent-cell timeouts.
- Pre-existing uncommitted refactors in frozen-adjacent files (`formation_mux_model_v2.py` param-groups property, `tie_role_train_v1.py` binding refactor, tie-role protocol/tests) are NOT S6 until committed + reviewed; S5 `verify_science` will fail closed on them — stash or commit as S6 before any S5-verify Kaggle run.

Science S5 keeps the S4 causal questions, arms, seeds, endpoints, thresholds, and sealed firewall. It expands only the synthetic training surface to **60,000 unique rows** (10,000/family) while keeping development at 480 and sealed at 720. CS-MECH consumes 32,000 row presentations per arm, so it does not wrap its training permutation before the endpoint. REP-FORM remains matched by 500,000 actual processed non-padding tokens.

Durability is now explicit: CS-MECH exact-resume checkpoints every **200 updates**; REP-FORM checkpoints every **10,000 processed tokens** because token exposure, not step count, is its causal matching axis. Every checkpoint emits `LATEST_PROGRESS.json` and an immutable `progress/UPDATE_*.json` diagnostic snapshot. The campaign may span as many Kaggle sessions as needed; a session boundary cannot change frozen science.

Pre-execution qualification on GitHub Actions run `34900925973` passed: S5 sources compiled, **28 targeted tests passed**, all **38 immutable science files** matched S5, the real frozen V24576 tokenizer produced the expected 60,000/480 public surface and reproducible 720-row sealed commitments for each experiment, and the canonical notebook pins validated.

Operational note: later campaigns may be multiplexed onto shared sessions when compatible, but each keeps separate preregistrations, data identities, sealed sets, verdicts, and claim ceilings.
