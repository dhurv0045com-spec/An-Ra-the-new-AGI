# CYR-GPU-009 — focused progressive retention replication

## Why 009 exists

The operator ran CYR-GPU-008 Cell 0 on a real free Colab GPU. Calibration passed, but V8 failed closed because its frozen minimum — two acquisition parents times four retention forks — could not fit the 170-minute wall. No scientific training started.

The lesson is execution-design, not model science: the campaign was over-broad for the hardware. CYR-GPU-009 adopts the useful scheduling pattern in live Arkenstone Discovery V7: one fixed wall budget, highest-information matched units first, conservative per-unit launch gates, partial receipts after each unit, and packaging even when later units are budget-blocked.

## Scientific question

> From the exact same candidate-free G90-confirmed Cymek V5 state, does switching to LOW LR preserve T2 capability better than continuing HIGH LR, and does that direction repeat across two independent acquisition parents?

This is intentionally the narrowest direct larger-V5-proxy test of the strongest prior ARK-007R / ARK-011 retention evidence.

## Frozen core

- real Cymek V5 only (`ModelSpec` + `v5_model.core.initialize`);
- frozen 24,576 tokenizer;
- acquisition seeds: 707, 808;
- acquisition LR: inherited HIGH = 1e-3;
- candidate-free G90: DEV_CONTROLLER complete exact with valid EOS >= 0.90 for 3 consecutive evaluations;
- one acquisition per seed;
- fork only from G90-confirmed parent;
- matched arms: `HIGH_CONTINUE`, `LOW_CONTINUE`;
- exact same model bytes, optimizer bytes, continuation stream, and token target per pair;
- acquisition target: 2,000,000 actual real tokens, but G90 confirmation ends acquisition early;
- continuation target: 500,000 actual real tokens/arm;
- acquisition evaluation every 400,000 actual tokens;
- continuation evaluation every 125,000 actual tokens;
- global wall: 165 minutes, 8-minute packaging reserve;
- per-parent launch floor: 45 minutes remaining;
- incomplete/timeboxed matched units are preserved but never count as replicated evidence.

## Progressive scheduling

The resolver never rejects a healthy calibrated CUDA GPU solely because the complete target dose is predicted not to fit. It prefers the largest non-TINY proxy predicted to finish the focused two-parent campaign. If none fits, it chooses the passing proxy with the highest measured useful training throughput and applies a strict development-only claim ceiling.

At runtime, the remaining science wall is divided across the still-unstarted parent units. Each parent unit reserves approximately 68% of its share for acquisition and divides the remaining share across the two matched arms. If a parent does not reach G90 inside its share, no retention fork is run from that parent. The next parent is still attempted if its launch gate is met.

This mirrors Arkenstone Discovery V7's important technique: do not require an entire long campaign to be guaranteed before starting; instead finish valid matched units, save them immediately, and do not start an expensive new unit when the remaining wall is insufficient.

## Verdict

A replicated LOW-retention result requires two contract-valid matched parents. For both, LOW-HIGH RET90 must be at least +0.05 and the mean paired margin at least +0.10. One complete parent is labeled only a single-parent development signal. No complete pair is inconclusive.

## Claim boundary

CYR-GPU-009 is focused development evidence only. It cannot authorize a production LR law, PRE500M, 500M, TPU equivalence, or a production scheduler change. Transfer, fixed-time decay, and hysteretic control are deferred rather than squeezed into an under-dosed session.
