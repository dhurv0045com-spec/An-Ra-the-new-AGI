# ARK-004A-R REANALYSIS — VERDICT B/C: the precursor claim is INVERTED and is a MARKER, not a precursor

Reconstructed from raw receipts by REANALYSIS.py (no hand-typed statistics).
Direction semantics: positive rho(feature, G90_step) = higher feature predicts
LATER generalization.

## Reconstructed facts
- rho(mean_tens_selectivity, G90_step) = +0.60; rho(., post_mem_delay_90) = +0.60.
  HIGHER early tens-selectivity associates with LATER, not earlier,
  generalization.
- LOO (now explicitly specified: Spearman over the 3 training seeds, target =
  raw G90_step, positive fold = feature-high -> later): 4/4 positive folds —
  i.e. consistently "more selective early -> later transition".
- The original ANALYSIS prose ("seeds whose representations already factorize
  the binding-heavy tens column generalize EARLIER") is therefore
  DIRECTIONALLY WRONG. The statistic was real; the interpretation inverted it.
  The original "LOO 4/4" claim was LOO_RESULT_NOT_REPRODUCIBLY_SPECIFIED as
  written (target, convention, and tie handling unstated).

## Temporal ordering (precursor test)
Selectivity's first material move (+0.10 over its early-window mean):
seed 101 AFTER P10; seed 303 BEFORE P10 (the latest generalizer);
seed 202 insufficient move; seed 404 AFTER P10.
=> 3 of 4 seeds: the selectivity signal does NOT precede OOD emergence.
It is a TRANSITION MARKER, not a precursor.

## Verdict (mission section 7): B + C
- B: predictive but OPPOSITE direction — the ARK-004B causal premise
  ("encourage tens-column factorization -> earlier generalization") is
  unsupported and arguably inverted. ARK-004B is CANCELLED.
- C: co-emergent/marker rather than precursor (ordering evidence above).
- M-008 reclassified: NOT_SUPPORTED as precursor; recorded as a transition
  marker with an inverted association (negative result preserved).

## Mechanistic honesty
No confident mechanism is claimed for the inversion. One coherent (unproven)
reading: early high tens-selectivity may reflect an early solution that is
already differentially sensitive to tens inputs without being correct — a
signature of the lookup phase's structure, not of approaching
factorization. Deciding between readings requires the retention program
(ARK-005), not more correlation.

## Strongest baselines (recomputed)
loss_at_M99 rho +0.40 (3/4 folds) — the best trivial baseline; tens
selectivity's directional consistency does not survive as a PRECURSOR
advantage once ordering and direction are handled correctly.
