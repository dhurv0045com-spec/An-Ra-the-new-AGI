# ARK-019 V3 — PREEXECUTION RUNTIME FAILURE 001

**Observed:** 2026-09-11 operator Colab/T4 preexecution run  
**Scientific continuation-arm outcomes observed before failure:** **NONE**  
**Failure class:** runtime-fit gate, not a scientific result

The frozen ARK-019 V3 executable `63dc47d9bdcd85e004c03aaf9a738c9682fcd8f0` passed far enough to measure the operator T4 runtime and then intentionally failed closed before entering the 4 matched sets × 4 continuation arms because its conservative full-campaign estimate exceeded the preregistered 175-minute wall.

Operator-captured calibration fields from `ARK-019_V3_FAILURE.json`:

- `update_seconds = 0.5247120965999784`
- `small_eval_seconds ≈ 0.50256029`
- failure message prefix: `full R3 does not fit 175-minute wall`

Using the frozen V3 projector (16,000 continuation updates, 160 evaluation events, parent/cap-shadow overhead, 5-minute packaging reserve, 1.30 safety factor), those measured rates imply a conservative total of approximately **12,580.8 s = 209.68 min**. Therefore the gate behaved correctly: 209.68 > 175.

Control-flow audit: in `run_all()`, the runtime-fit exception is raised immediately after parent qualification, exact-resume smoke, and calibration, and **before** `ars={}` / the matched-set continuation loop. Consequently no `PLASTIC_HIGH`, `STATIC_REPLAY_1OF64`, `GUARDIAN_REPLAY`, or `GUARDIAN_HYBRID` comparative outcome was available when this operational amendment was designed.

Parent qualification and smoke/calibration artifacts are preexecution substrate evidence and may be reused only if their identity checks still pass. They are not Guardian efficacy outcomes.

This failure must remain in the evidence history. It does not support or refute the Guardian hypothesis.