# FORMATION-MUX-001 observed Science S5 result — 2026-09-15

This directory preserves the critical text evidence extracted from the user-saved Kaggle bundle `FORMATION_MUX_001_RESULTS.zip` (SHA-256 `859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5`).

## What completed

- Science commit: `c15ad8beb409537db42d075684ea54847a074ebd`
- Observed operator head: `fdc2483fed0bb1a80d4bfad376aac84a66db6055` (`tools/formation_mux_001_kaggle_operator_v8.py`)
- Hardware: Kaggle Tesla T4 x2
- Official arms: **24/24 COMPLETE**
- Sealed: **CS-MECH-002 COMPLETE**, **REP-FORM-003A COMPLETE**
- Global failure: **none**
- Wall guard: **not triggered**
- Wall time: **8.137 h**

## Formal scientific result

Both independent S5 experiments returned **NULL** under their preregistered primary identity criteria. This is a valid completed result, not an execution failure.

## Post-hoc limitation / failure mode

The primary identity capability was strongly floor-limited: 13/16 CS-MECH arms ended at 0 identity, no CS-MECH arm reached the preregistered 0.5 acquisition threshold, and all 8 REP-FORM arms ended at 0 identity. Therefore the campaign had weak discrimination power for the intended primary mechanism questions. The full reasoning and corrective process rule are recorded in `POSTMORTEM.md`.

## Scope

This observed bundle is **S5 only**. It predates execution of the later TIE-ROLE frontier. Do not rewrite history by claiming that the 48-arm expanded campaign ran.
