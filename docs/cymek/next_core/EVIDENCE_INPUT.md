# EVIDENCE INPUT (Task-1 → Task-2)

**Branch:** `cymek-next-core-architecture` (parent: `research/evidence-consolidation-2026-09-13` @ `5879ca6b2954d44b5e5e9881e85edd9b601359f8` — the Task-1 head advanced past `a350c2b` during Phase 2; both validators green at ingest).

## 1. Live evidence deltas incorporated (vs the Task-1 assignment snapshot)

| Branch | Head at Task-2 start | Delta vs Task-1 audit | Architecture relevance |
|---|---|---|---|
| `research/evidence-consolidation-2026-09-13` | `5879ca6` | +7 commits (Phase-2 decision system) | direct input |
| citadel | `e96c9a9` | CITADEL-DATA-001 executed corpus audit | data/evaluation gate (§16/§17), not architecture |
| eval-integrity-001 | `915e23ff` (new) | CITADEL-EVAL-001 per-surface validity | evaluation contract |
| BRAMASTRA | `f25fc96` | B00–B12 integrated build (engineering, **no executed science**) | audited separately below (user-requested) |
| codex/arkenstone-improvements | `516c280` | refactor only | none |
| cymek-500m-readiness | `f2c27a6` | unchanged | the live V5 audited in CURRENT_V5_AUDIT.md |
| Arkenstone / ark020-v4 / astra / triquetra / esoes / core / iterate / main | unchanged since Task-1 ledger | — | ledger entries stand |

## 2. BRAMASTRA deep audit (user-requested)

The B2 integrated build (`d9e4c38`, handoff `engineering/reports/B2/HANDOFF.md`) is a complete **engineering stack with zero new executed science**. Component-by-component evidence verdicts for the next Core:

| BRAMASTRA component | What it is | Verdict for next Core |
|---|---|---|
| B01 profiles 117,312 / 6,493,440 / 38,681,088 params | byte-codec scale ladder | **Reuse concept** in SCALING_PLAN.md (byte rungs); not a Core block |
| B02 byte codec (V=260), EOS, loss masks, packed-segment attention isolation | data contract | **Aligned** with Cymek packing contract; alternative tokenizer regime for byte rungs |
| B04 training-only logit treatments: `participating_mask` / `inactive_offset` with `log((V-\|A\|)/(K-\|A\|))` | **Independent implementation of the R1C treatment family** | **High value**: gives the R1C science an independent-implementation path (R2); ported conceptually into `v5_next` as EXPERIMENT_ONLY modes — never default |
| B04 accumulation-within-tolerance + clip certificate 1e-4 | training contract | Matches Cymek's R1C-derived clip tolerance; no change |
| B05 atomic publish / writer fencing / fresh-process next-update agreement | checkpoint contract | Matches Cymek; keep Cymek implementation |
| B06 plasticity state machine (FORM/STABILIZE/EXPAND/REACQUIRE/HOLD, collapse never lowers LR) | **external** controller | EXPERIMENT_ONLY infrastructure; stays OUTSIDE the Core (§14) |
| B07 hash-chained episode ledger + stratified replay | replay store | External training infrastructure; supports AD20's "replay external" status |
| B08 environments + oracles, B09 recomputed exact+EOS scoring, paired goal metrics | evaluation | Evaluation-contract alignment (§17); Triquetra/Citadel remain authorities |
| B10 planner (imagined-step separation, oracle refusal) | runtime | External; REJECTED for Core internalization |

**Net:** BRAMASTRA changes no Core block. Its architecture-relevant contributions are (1) an independent output-space treatment implementation to cross-check R1C, (2) a byte-codec regime proving small-vocab rungs are buildable, (3) external controller/replay machinery consistent with the formation-first contract.

## 3. Primary evidence constraints the design must respect (verified in Task-1 ledger)

1. **Class-space formation effect** (CYR-GPU-011/012/013): declared tied class-space size moves held-out formation 0%↔100% non-monotonically; mechanism unresolved (R1C frozen, NOT_EXECUTED).
2. **Retention levers** (ARK-007R/010/011/015/017-V2): phase-dependent plasticity law; two independent levers; displacement falsified; doses not universal.
3. **Formation-first continual law** (ARK-013, ARK-019 V3.1): controller verdicts void without reference acquisition; V3.1 `CONTROLLER_NOT_SUPPORTED`; V4 transcribed-only.
4. **EOS contract** (BRM-terminal-EOS, CIT-T1D): answer+EOS supervision EVIDENCE-LOCKED.
5. **Loss ≠ cognition** (ESO-PGE, CIT-T1-series): never select on loss.
6. **Evaluation integrity** (CITADEL-DATA-001/EVAL-001): tiered arithmetic surface retired (shortcut 1.000/tier, 530 leaks, 13.5% dup, 0.004× supply); candidate-free primary + attack screens mandatory; `production_scoring_mode` still null.
7. **Mechanism priors** (ESO-e2, D-024..D-028): QK norm, residual scaling, precision layout, clip invariant LOCKED; zero cognition evidence for any block change.
8. **Data gate** (CIT-500M audit + citadel): production corpus MISSING; entry point MISSING; schedule unexecuted.

## 4. What this input does NOT contain

- Any executed R1C result; any audited ARK-019 V4 / ARK-020 bundle; any qualified 5B corpus; any certified scorer; any scale-vs-formation evidence. Every architecture field depending on these is BLOCKED, not guessed (see NEXT_CORE_SPEC.json).
