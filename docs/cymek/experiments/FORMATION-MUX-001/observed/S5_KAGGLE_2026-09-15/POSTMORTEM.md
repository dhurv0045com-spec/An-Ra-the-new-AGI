# FORMATION-MUX-001 Science S5 — observed result and floor-limited postmortem

Status: **POST-OUTCOME ANALYSIS. Do not treat this document as preregistration.**

Source bundle SHA-256: `859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5`  
Science commit: `c15ad8beb409537db42d075684ea54847a074ebd`  
Observed operator head: `fdc2483fed0bb1a80d4bfad376aac84a66db6055`  
Observed operator: `tools/formation_mux_001_kaggle_operator_v8.py`

## 1. Execution verdict

The Kaggle experiment **executed successfully**. `CAMPAIGN_STATE.json` records 24/24 official arms complete, all four seed bundles complete, both sealed evaluations complete, no global failure, and no wall-guard truncation. Runtime was 29294.9 seconds (8.137 h) on Tesla T4 x2.

This postmortem uses the word **failure** only for the experimental-design failure mode described below. It does not relabel the successful execution as an engineering failure.

## 2. Formal preregistered scientific outcomes

### CS-MECH-002

All three primary identity contrasts were `NULL`:

- extra-row weight decay: mean development formation-AUC delta `+0.001674`; mean sealed identity gap `+0.029167`; sign consistency not met.
- extra-row trainability/update evolution: mean development formation-AUC delta `+0.000335`; mean sealed identity gap `0.0`; sign consistency not met.
- extra-row denominator participation: mean development formation-AUC delta `-0.001674`; mean sealed identity gap `-0.016667`; sign consistency not met.

### REP-FORM-003A

Production BPE versus isomorphic rendering was `NULL` on the primary identity outcome: mean development formation-AUC delta `0.0`, mean sealed identity gap `0.0`; all four paired seeds were at the primary floor.

These formal `NULL` verdicts remain authoritative for the frozen S5 protocol.

## 3. Why the campaign was scientifically underpowered

The primary endpoint did not reliably form in the control regime. Across the 16 CS-MECH arms, only three arm/seed runs had a nonzero development identity endpoint: M0/73012 = 0.15, M1/73011 = 0.0125, M3/73013 = 0.0625. M2 was 0 in all four seeds. No arm reached the preregistered acquisition threshold of 0.5. All eight REP-FORM arms ended at identity 0.0.

That places the main comparison near a hard floor. When both sides of a causal contrast are almost always zero, absence of a threshold-crossing difference cannot distinguish “mechanism does not matter” from “the substrate never entered the capability regime where the mechanism could express itself.”

The training system itself was not globally broken: sealed termination/counting was essentially 1.0 across arms, EOS behavior was healthy, all arms completed, and sealed generalization was nonzero for some harder families. The failure mode is therefore **baseline capability formation / sensitivity**, not campaign execution.

## 4. Process mistake

The full 24-arm campaign should not have been launched before a cheap baseline-capability gate demonstrated that the primary identity metric could leave the floor across multiple seeds. The missing gate allowed ~8.14 wall-hours of matched causal science to run in a regime with low power for its primary question.

This is a research-process error and must be preserved as negative evidence. It is not permissible to erase the NULL results or retroactively change their thresholds.

## 5. Secondary exploratory signals (not promotion evidence)

The strongest secondary signal is composition for M3 versus M2. On sealed composition, M3 - M2 was positive in all four seeds with mean delta `0.277083`. This is consistent with the hypothesis that permanently frozen random extra rows left in the full softmax denominator can be harmful while masked rows avoid that pressure.

This was not the preregistered primary promotion criterion. It must remain `EXPLORATORY_ONLY` until reproduced in a capability regime with adequate primary sensitivity.

REP-FORM also shows that isomorphic one-token rendering did not rescue primary identity. Therefore production BPE segmentation alone is not supported as the sole explanation for the primary failure in this regime.

## 6. Optimization warning

Every CS-MECH official arm recorded `clip_fraction = 1.0`: every update hit the global clipping boundary. R1 REP-FORM also clipped every update; R0 clipped roughly 0.88-0.98 of updates depending on seed. This does not invalidate matched contrasts, but it is evidence that the substrate operated in an aggressively clipped optimization regime and should be part of the next baseline-formation investigation.

## 7. Corrective rule for future expensive mechanism campaigns

Before launching a large matched factorial campaign around a cognitive primary endpoint:

1. Run a cheap control-only capability sweep across multiple seeds.
2. Adjust exposure / scale / optimization only in that preflight stage until the control is demonstrably away from both floor and ceiling.
3. A practical target is a 30-70% primary-performance regime across most seeds; exact thresholds must be prospectively frozen for the next campaign rather than imported retroactively.
4. Only after the capability gate passes should the expensive causal arms be launched.
5. Preserve this S5 NULL result unchanged and use it as prior negative evidence.

## 8. Historical integrity

The uploaded observed bundle is the **24-arm Science S5 run under operator v8**. It does not contain TIE-ROLE frontier outputs. Later branch code may contain the 48-arm v12 expansion, but that expanded campaign was not executed in this bundle.

The source result ZIP did not contain `resume.pt`; its own README states that exact-resume checkpoints stay in Kaggle Output. This affects optional checkpoint-dependent diagnostics, not the validity of the completed S5 final results preserved here.
