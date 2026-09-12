# ARK-012 — RECOVERY-SWITCH THRESHOLD MAP

## Status

**PREREGISTERED BEFORE EXECUTION.**

## Question

ARK-007R and ARK-010 imply a hysteretic training regime: HIGH LR is useful while capability must move/recover, while LOW LR is useful once capability is present. ARK-011 tests one switch point (sustained G90). ARK-012 asks a more mechanistic question:

> Does the quality of LOW-LR consolidation depend on *how far recovery has progressed* when the switch occurs?

## Hypothesis

`H-RECOVERY-THRESHOLD`: switching to LOW too early traps a partially recovered solution, while switching after a stronger recovered state improves subsequent sealed retention. If true, the controller should depend on measured capability state rather than elapsed training time alone.

This is a mechanistic screen, not an independent replication. To increase event yield, historical high-instability source combinations are prospectively frozen here with their selection bias stated explicitly.

## Frozen source combinations

- acquisition seed 909, continuation seed 2702
- acquisition seed 1010, continuation seed 2702
- acquisition seed 1111, continuation seed 2703

These were selected because prior ARK-007R/010 evidence showed HIGH-LR instability. Therefore ARK-012 may estimate schedule behavior conditional on event-producing sources, but it MUST NOT be counted as fresh evidence for the population instability rate.

## Task / model / firewall

Use the canonical T2 manifest SHA256:
`0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`.

Use the ARK-011 deterministic OOD_CONTROL / OOD_SEALED split algorithm. OOD_SEALED is measurement-only and may never trigger a switch.

Model/optimizer are the historical Micro T2 configuration:
- width 128, 4 layers, 4 heads, ffn 512, CompactVocab;
- AdamW betas=(0.9,0.95), eps=1e-8, weight_decay=0.1;
- batch 64, clip 1.0, eval every 200 steps;
- HIGH=1e-3, LOW=1e-5.

## Procedure

For each frozen source combination:
1. Acquire sustained OOD_CONTROL G90 at HIGH LR.
2. Continue on the frozen continuation stream until the first 3 consecutive OOD_CONTROL evals <0.90; snapshot exact state.
3. From the collapse-confirmation snapshot, fork schedules that all consume the same absolute continuation stream:
   - `HIGH_CONTINUE`: HIGH for the entire fixed horizon.
   - `LOW_IMMEDIATE`: LOW from collapse confirmation onward.
   - `SWITCH_75`: HIGH until first 3 consecutive OOD_CONTROL evals >=0.75, then LOW.
   - `SWITCH_85`: HIGH until first 3 consecutive OOD_CONTROL evals >=0.85, then LOW.
   - `SWITCH_90`: HIGH until first 3 consecutive OOD_CONTROL evals >=0.90, then LOW.
   - `SWITCH_95`: HIGH until first 3 consecutive OOD_CONTROL evals >=0.95, then LOW.
4. Fixed post-collapse horizon: 8,000 optimizer steps for every schedule. No arm gets extra steps because it switched later.
5. OOD_SEALED is evaluated every 200 steps but never used for control.

All arms start from the identical collapse snapshot and consume byte-identical minibatches at each absolute step. Only the LR schedule differs.

## Primary readout

For each schedule on OOD_SEALED:
- area under exact trajectory / mean exact;
- RET90;
- final exact;
- first sustained recovery >=0.90;
- recurrent instability after first sealed G90 recovery.

The main analysis is the ordered relationship between switch threshold and sealed retention/recovery. No post-hoc threshold is promoted as a law from this screen alone.

## Mechanistic outcomes

- `STATE_THRESHOLD_SUPPORTED_SCREEN`: later/stronger recovery thresholds consistently improve sealed retention relative to immediate/early switching without simply matching HIGH_CONTINUE instability.
- `TIME_NOT_STATE_SCREEN`: schedules do not order by measured recovery state.
- `LOW_ALWAYS_BEST_SCREEN`: immediate LOW dominates once collapsed, contradicting ARK-010 pattern on these sources.
- `HIGH_ALWAYS_BEST_SCREEN`: switching LOW provides no post-recovery benefit in the tested horizon.
- `INCONCLUSIVE_LOW_EVENT_RATE`: fewer than 2 source combinations produce the required prospective collapse in this rerun.

## Claim limits

This experiment is selected-event mechanistic evidence only. It cannot establish fresh incidence, transfer, scale generality, or production scheduler readiness.