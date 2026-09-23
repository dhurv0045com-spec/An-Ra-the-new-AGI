# CYMEK CURRENT UNIFIED RESEARCH STATE

**Canonical working branch:** `cymek-500m-readiness`  
**Purpose:** one entry point so an operator/agent does not need to inspect Cymek, Citadel, Triquetra and Arkenstone separately for routine decisions.

## Current override — 2026-09-24

The current `cymek-beta` formation-mux state is newer than the historical queue below:

- the 2026-09-23 Kaggle T4×2 snapshot completed **24/24 S5 development arms** and **2/24 TIE-ROLE frontier arms** before the session wall guard stopped further launches;
- sealed evaluation was not consumed, no `FINAL_RESULT.json` or architecture gate exists, and no final scientific verdict or promotion is authorized;
- `artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/FORMATION_MUX_001_RESULTS.partial.zip` is an evidence-only snapshot with zero `resume.pt` members and is not sufficient for exact continuation;
- the next discriminating action is recovery of the original saved Kaggle Output tree, the pinned same-kernel checkpoint/result-hash preflight, and only then continuation of the remaining frozen frontier arms; after 24/24, pinned v12 automatically crosses the frozen sealed-finalization boundary;
- recovery-preflight v2 and its focused CI suite passed in GitHub Actions run `35925460488` at head `ee6a5cad4a4f39752b2646a60eeaeef8d2b06019`; this qualifies repository engineering only, not the absent saved Output or any scientific claim;
- authoritative current details: `docs/cymek/next_core/FORMATION_MUX_001_CURRENT_STATE_OVERRIDE.json` and `docs/cymek/research/NEXT_7_EXPERIMENTS_V51.md`.

The older R1C and formation-mux entries below are retained as historical roadmap context, not current execution status.

Read in this order:

1. `MASTER_AGI_CONSTRUCTION_KNOWLEDGE.md` — cross-branch construction synthesis (refreshed 2026-09-13).
2. `../../research/EXPERIMENT_EVIDENCE_LEDGER.md` — canonical per-experiment evidence ledger (machine copy `.json`).
3. `../experiments/CYR-GPU-011/RESULT.md` — production-vs-compact capability-formation evidence.
4. `../experiments/CYR-GPU-012-R1/RESULT.md` — completed controlled R1 class-space experiment.
5. `../experiments/CYR-GPU-013-R1B/RESULT.md` — **EXECUTED** replicated response curve (`MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`).
6. `../experiments/CYR-GPU-014-R1C/RUN_READINESS_V4.json` — softmax-mechanism dissection, engineering-ready, **scientific NOT_EXECUTED; launcher pins origin-absent commit `6653b4ce` (launch-blocking, see `../../research/EVIDENCE_GAPS.md`)**.
7. Arkenstone: `experiments/ARK-017/RESULT_V2.md` (BOTH_LEVERS_SUFFICIENT), `experiments/ARK-018/FINAL_RESULT_AUDIT.md`, `experiments/ARK-019/FINAL_RESULT_AUDIT_V3.md` (CONTROLLER_NOT_SUPPORTED) and `FINAL_RESULT_AUDIT_V4.md` (transcribed candidate — raw bundle not in repo).

## Current scientific state

- **Engineering / checkpoint machinery:** strong enough for controlled GPU campaigns; not equivalent to scientific authorization for a large run.
- **Production-representation failure:** CYR-GPU-011 produced train M99 but 0% held-out STANDARD and 0/48 SEALED at full ARK semantic exposure with the 24,576-token production representation, while compact V19 reached 56.47% STANDARD at only 44.89% exposure.
- **R1 class-space result:** completed R1 held active arithmetic tokenization and shared initialization fixed while varying only declared tied embedding/output class count. On the executed fresh seed at 512k semantic rows, V19 = 12.94% STANDARD, V4096 = **100%**, V24576 = **0%**. Official preregistered verdict: `MIXED_OR_INTERMEDIATE_REPRESENTATION_EFFECT` because V19 did not meet the preregistered compact-signal threshold. Bundle SHA-256: `a22b538396a3d0957a60a27f39b0cf3dd3b20874585b4c15a03207224f613d29`.
- **Interpretation of R1:** a large non-monotonic class-space/optimization effect is demonstrated in one developmental seed; 4096 is not yet a universal or production-optimal vocabulary. Seed sensitivity remains important because historical V19 behavior was much stronger than the fresh R1 V19 subject.
- **Retention science:** strong Micro evidence that both lower plasticity and continued capability-supporting data can protect an acquired invariant; mechanism attribution remains unresolved. ARK-017 V2 is the active R2 mechanism dissection.
- **Real-text plasticity:** ARK-018 is executed. Heavy 10% specialized Birth rehearsal strongly improved Birth NLL but showed a replicated exploratory slowdown in later temporary-binding acquisition versus token-matched science replay.
- **AGI claim:** none. Broad general intelligence remains `NOT_DEMONSTRATED`.

## Current priorities

**R2 / ARK-017 V2:** EXECUTED (see Arkenstone `experiments/ARK-017/RESULT_V2.md`): `BOTH_LEVERS_SUFFICIENT` — protection can come from lowered applied update magnitude or from sparse invariant-support replay, independently; "small total movement" is falsified. Dose universality and scale transfer remain open.

**R1C / CYR-GPU-014:** the active Cymek experiment. Frozen 6-arm × 4-seed softmax-competition dissection on the fixed 24,576 matrix. Engineering-ready after two repaired failures (optimizer constructor; CLIP_BREACH float32 tolerance). **Before operator launch: rebind the launcher from `6653b4ce` (absent from origin) to the pushed equivalent.**

**Real-text / Guardian line:** ARK-019 V3.1 executed (`CONTROLLER_NOT_SUPPORTED`, formation bottleneck); ARK-019 V4 result exists only as a transcription of an external audit — commit and byte-audit the raw bundle (or rerun) before treating the Guardian question as answered; then ARK-020 V4.

## Readiness interpretation

Approximate maturity remains an evidence-maturity estimate, not “percent to AGI.” R1 increases our understanding of representation/capability formation but does not authorize scale. Experimental integrity ~90%, causal evaluation ~85%, checkpoint/durability ~80%, architecture mechanics ~65%, data governance ~70%, production corpus readiness ~20%, representation understanding now roughly **50%** rather than 35% because a controlled class-space effect has been observed but not replicated, objective ~45%, capability formation ~50%, same-skill retention ~75% Micro / ~20% production-transfer, multi-skill continual learning ~20%, learned self-diagnosis ~15%, real-text cognition/plasticity ~45%, target-scale scientific readiness ~20–25%, demonstrated AGI 0%.
