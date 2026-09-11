# ARKENSTONE NEXT RESEARCH PORTFOLIO (ARK-021 → ARK-030)

Ranked by information gain, causal clarity, dependency on current evidence, compute cost,
implementation risk, and bottleneck relevance. **Top-3 implemented; the rest are design
drafts with explicit dependency gates.** No AGI/pre-500M authorization anywhere.

| rank | experiment | objective | dependency | compute | status |
|---|---|---|---|---|---|
| 1 | **ARK-020 V4** | does the multi-skill Guardian hold? | ready now | 9–16 h | READY_FOR_OPERATOR_RUN |
| 2 | **ARK-021** retention vs reacquisition | is replay protection *preservation* or *rapid relearning*? | V4 substrate (exists) | 6–10 h | DEVELOPMENT_READY (core+tests) |
| 3 | **ARK-022** dormant retention | do capabilities survive replay-free gaps? | V4 substrate | 6–12 h | DESIGN_READY |
| 4 | **ARK-028** interference prediction | predict failure 25–100 updates ahead | needs V4 episode data | 8–14 h | DESIGN_READY |
| 5 | **ARK-025** representation × preservation cross | do Cymek class-space regimes interact with Guardian? | **R1C results (pending on cymek)** | 12–20 h | BLOCKED_ON_R1C_EVIDENCE |
| 6 | **ARK-023** memory-cost scaling k=1..8 | protection-cost slope vs number of skills | V4 result informs arms | 15–25 h | DESIGN_READY |
| 7 | **ARK-026** realer-language adaptation | natural-language capability under Guardian | contamination-resistant dataset build | 10–20 h | DESIGN_READY |
| 8 | **ARK-029** gradient-direction intervention | is replay = gradient-direction repair? | V4 diagnostics baseline | 6–10 h | DESIGN_READY |
| 9 | **ARK-024** task-ID-free Guardian | protection without manual skill labels | V4 + probes | 10–16 h | SPECULATIVE |
| 10 | **ARK-030** memory compression | protection per byte of memory | ARK-021+023 results | 8–14 h | SPECULATIVE |

**Why the mission's ordering changes:** ARK-025 is demoted below 021/022 only because its
formation axis is still missing R1C evidence (running it now would confound formation
with preservation — the exact confusion V4's lifecycle accounting was built to prevent).
ARK-021 is promoted to co-first because its question (preserved vs reconstructed)
determines how every later result should be read, and it needs no new substrate.

## ARK-021 — RETENTION VS REACQUISITION (top-3, DEVELOPMENT_READY)

- **OBJECTIVE:** distinguish PRESERVED (old capability present before corrective
  exposure) from RECONSTRUCTED (lost, then replay-relearned).
- **CONSTRAINTS:** reuse V4 parents/tasks/checkpoint machinery unchanged.
- **SIMPLEST ARCHITECTURE:** after A-acquisition, run matched plastic continuation with
  replay withheld; at probe steps, evaluate A immediately (no recovery allowed), then
  either restore replay (recovery arm) or keep withholding (dormancy arm).
- **MOST UNCERTAIN ASSUMPTION:** a short no-replay window is long enough to reveal loss.
- **TEST/METRIC:** probe-time A robust-min trajectory; recovery latency after replay
  resumes; hidden-state similarity to the parent on A-probes.
- **EVIDENCE THAT CHANGES OUR MIND:** high probe-time A accuracy → replay protects
  preservation; zero accuracy with fast post-replay recovery → replay reconstructs.
- **STATUS:** core+tests exist under `experiments/ARK-021/`; full campaign
  BLOCKED_ON_ARK020V4_RESULT (its arms parameterize on V4's verified controller).
- **DESIGN:** `experiments/ARK-021/PLAN.md`; preregistration draft included.

## ARK-022 — DORMANT CAPABILITY RETENTION (top-3, DESIGN_READY)

- **OBJECTIVE:** survival of an untrained, unreplayed capability across gaps of
  250/500/1000/2000/4000 updates, comparing prior protection histories (plastic/static/
  Guardian) under an identical replay-free dormancy.
- **METRIC:** sealed A robust-min at each gap endpoint; no peeking-driven adaptation
  (gap length frozen in advance, SEALED grading only).
- **CHANGES OUR MIND IF:** protection history has zero effect on dormancy survival →
  replay buys nothing durable; strong history effect → protection builds durable
  structure, justifying internalization research (ARK-030).
- **DESIGN:** `experiments/ARK-022/PLAN.md` (draft; run after ARK-021).

## ARK-028 — INTERFERENCE PREDICTION (top-3 by information, DESIGN_READY)

- **OBJECTIVE:** predict capability failure 25–100 updates before it happens, better
  than margin thresholding.
- **DESIGN:** harvest feature/label episodes from ARK-020 V4 telemetry (robust-min,
  slope, gradient cosines, displacement); predictor-train/validation/sealed split BY
  EPISODE — never fit and claim on the same episodes.
- **METRIC:** AUROC at warning horizons; false-protection duty; prevented failures.
- **DEPENDENCY:** V4 outcome data. Run after V4.

## Non-selected drafts (one line each)

- **ARK-023**: k-arm scaling of protection cost — design frozen after V4 (its arms
  depend on V4's measured Guardian behavior). **ARK-024**: probe-based (task-ID-hidden)
  protection — valuable but speculative until 021 separates preserved/reconstructed.
  **ARK-025**: 2×N representation×preservation cross — unblocked the day cymek R1C
  results land; design skeleton in CONTROLLER_SYNTHESIS.md. **ARK-026**: real-language
  adaptation — requires building a contamination-resistant dataset first (owner decision
  on corpus). **ARK-029**: gradient-projection vs replay mechanism dissection — nested
  inside V4's diagnostics; promote only if cosines show destructive-direction signal.
  **ARK-030**: memory compression — only meaningful once replay retention is proven
  durable (ARK-021/022).
