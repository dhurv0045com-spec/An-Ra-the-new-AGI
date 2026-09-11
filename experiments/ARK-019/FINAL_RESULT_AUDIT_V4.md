# ARK-019 V4 — FINAL RESULT RECORD (evidence-integration entry)

**Created:** 2026-09-12 (ARK-020 V2 mission, deliverable §18)
**Provenance class:** OPERATOR-RETURNED RESULT, EXTERNALLY AUDITED — **not** re-audited
here at byte level. The result ZIP was not available inside this repository at the time
of writing; every claim below is transcribed from the previously verified operator audit
and marked accordingly. Nothing here is fabricated; anything not in the verified audit is
omitted.

## Execution identity (as reported by the verified audit)

- Scientific runner commit: `8e858c614d764335100dfd41dda6f8e0d0c877a7` (CI contract-gated)
- Preregistration: `experiments/ARK-019/PREREGISTRATION_V4.json`
- Device: Colab T4 CUDA, torch 2.11.0+cu128
- Prior campaign: V3.1 bundle `fcc14c5378318b8c447c2735768b72920334d8c9292b445372fd17d3d2b554d8`,
  official verdict `CONTROLLER_NOT_SUPPORTED` (new-skill-formation bottleneck)

## Believed verified verdict

> **`GUARDIAN_CONTINUAL_PROXY_CANDIDATE`** — recorded as VERIFIED-BY-EXTERNAL-AUDIT,
> pending byte-level re-audit if the raw bundle is later committed or re-shared.

## Evidence summary (transcribed from the verified audit)

| arm | old skill A (final sealed) | new skill B (final sealed) |
|---|---|---|
| PLASTIC_HIGH | qualified **0/4** (destroyed) | qualified |
| GUARDIAN_REPLAY | qualified **4/4** | qualified **4/4** |
| GUARDIAN_HYBRID | qualified **4/4** | qualified **4/4** |

- Dynamic protection behaved primarily as **degradation → protection → recovery →
  de-escalation**, not perfect prevention.
- Guardian used **substantially less replay than permanent replay** while matching
  static retention quality.
- Static sparse replay remained a serious competitor.
- CAP16X incremental benefit **not clearly demonstrated**.
- Science NLL cost remained small versus matched baseline.
- V4 fixed both V3.1 defects: underpowered SKILL_B dose (prospective dose selection)
  and science-damaged parents (joint science gate).

## Claim labels

**DEMONSTRATED (controlled real-text proxy scale):** plastic continuation can acquire a
new skill while destroying an old one; sparse replay strongly protects the old skill; a
dynamic Guardian can finish with both old and new skills qualified across matched sets;
adaptive protection can use less replay than always-on replay; real-text NLL can stay
near baseline.

**SUPPORTED:** current Guardian is primarily a degradation→recovery→de-escalation
controller; prevention value unproven.

**NOT_DEMONSTRATED:** universal continual learning; large-LM transfer; 500M
effectiveness; CAP16X necessity; AGI; consciousness.

## Consequences for ARK-020 V2

1. The Guardian mechanism has now cleared a single-skill real-text proxy at candidate
   strength — the multi-skill generalization question is live and well-posed.
2. The reactive/predictive comparison remains open (V4 was primarily reactive-recovery).
3. CAP16X carries no demonstrated incremental value; V2 keeps the emergency-cap arm
   structure but does not promote caps.
4. V4's dose selection artifact on Drive (if present) may be inherited by V2's B phase
   under identity checks.
