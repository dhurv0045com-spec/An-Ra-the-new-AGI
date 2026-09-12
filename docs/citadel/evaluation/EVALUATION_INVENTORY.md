# EVALUATION INVENTORY — CITADEL-EVAL-001

Date: 2026-09-13. Branch: `eval-integrity-001`.
This inventories every evaluation surface in the An-Ra project and classifies
each by evidence strength and vulnerability.

## Classification vocabulary

| Label | Meaning |
|---|---|
| VALIDATED | Preregistered, shortcut-tested, causally sensitive, sealed or source-disjoint |
| PROVISIONALLY_VALID | Executed with preregistered thresholds but shortcut/causal audit incomplete |
| SHORTCUT_COMPROMISED | A trivial heuristic scores close to or above the model |
| LEAKAGE_COMPROMISED | Train/eval overlap detected |
| UNDERPOWERED | Sample size too small for the claimed precision |
| NOT_EXECUTED | Designed but never run |
| INVALID | Contract flaw makes the measurement meaningless |

---

## 1. Citadel T1D — tiered arithmetic exact-match

- **Claimed capability:** multi-digit arithmetic (add/subtract/multiply/divide)
  with curriculum, teacher, scale, and output-space contrasts
- **Actual observable:** greedy free-generation exact-match against computed target
- **Dataset:** tiered_data generator v1 (5 tiers, 4 ops, 4 templates, ~97 MB)
- **Split:** operand bands per tier (T0/T1 memorization probes, T2+ structural holdout)
- **Gold source:** computed from operands (machine-verified)
- **Reference solver:** direct arithmetic computation
- **Baselines:** latest-position, first-position, most-common-answer, always-zero
- **Model access:** greedy generation, EOS-or-MAX_TOKENS stop
- **Evaluator access:** exact string match after normalization
- **Sample count:** 500 test rows/tier × 5 tiers = 2,500 per arm
- **Aggregation:** per-tier accuracy + Wilson 95% LCB/UCB, pooled tiers 1-4
- **Pass threshold:** LCB > max(null, untrained UCB) + 0.10, all 5 gate rules
- **Preregistered:** YES (PLAN.md before execution)

### Findings
| Issue | Severity |
|---|---|
| EOS never supervised — 15,000/15,000 generations ended MAX_TOKENS | **BLOCKING** |
| Latest-position shortcut scores 1.000 on every tier | **HIGH** |
| Cross-split contamination: 173 dev + 357 test texts appear in train | **HIGH** |
| 13.5% exact duplication in train set | MEDIUM |
| B/D/E budget confound (8M vs 4M/4M) | MEDIUM |
| Data volume: 6.42M rows = ~2.1M BPE tokens = 0.004× of 500M | INFO |

### Verdict: **SHORTCUT_COMPROMISED + LEAKAGE_COMPROMISED**
The termination contract flaw means content and stopping are conflated.
The latest-position shortcut means the model can score without doing arithmetic.
Both must be fixed before T1E can produce meaningful results.

---

## 2. Cymek E0 — cognitive benchmark (T0 certificate)

- **Claimed capability:** binding, state tracking, composition, counterfactual
- **Actual observable:** exact-match on E0-generated cognition cases
- **Dataset:** E0 generator v0.4 (368 cases, 112 pairs, 14 families)
- **Split:** development certificate only (no sealed T2 fixture exists)
- **Gold source:** computed from causal graph
- **Reference solver:** independent solvers per family
- **Baselines:** random, first/last candidate, latest_fact, nearest_position,
  lexical_overlap, bag_of_words, 4 fixed-rule, broken-state, direct-retrieval,
  full-truth-oracle (12 total)
- **Model access:** free generation + candidate scoring
- **Sample count:** 368 cases
- **Aggregation:** per-family worst-case, 23 checks
- **Pass threshold:** all shortcut gates ≤ null + 10pp, oracle = 100%
- **Preregistered:** YES (development certificate, not sealed)

### Findings
| Issue | Severity |
|---|---|
| Development certificate only — no sealed T2 fixture exists | **HIGH** |
| Shortcut repair history: 2 false greens caught (bag_of_words 81.77%, latest_fact 100%) | HIGH |
| Eval generators exist for 9 families but train generators exist for 0 | HIGH |
| All 9 cognition families lack train-side generators | **BLOCKING** |
| EOS never supervised in T1D/T1E training either | HIGH |

### Verdict: **PROVISIONALLY_VALID** (for the development certificate;
NOT valid for production claims — no sealed fixture, no train-side data)

---

## 3. Triquetra — entity×value factorial + query-value evidence

- **Claimed capability:** query-conditioned binding, causal sensitivity
- **Actual observable:** exact-match + query-conditioned contrast
- **Dataset:** synthetic paired binding worlds (BASE + TWIN with one causal change)
- **Split:** dev only (no sealed)
- **Gold source:** computed from generator
- **Reference solver:** direct computation from causal graph
- **Baselines:** copy-first-fact, copy-last-fact, most-frequent-value
- **Sample count:** varies (120-500 per experiment)
- **Preregistered:** YES (Triquetra preregistered each before execution)

### Findings
| Issue | Severity |
|---|---|
| Query-conditioned signal ≈ 0 (raw rank 0.2500 = chance exactly) | HIGH |
| Latest-position shortcut scores 1.000 on every tier | HIGH |
| Self-knowledge probe: 0.0 trained and untrained | — |
| 57/96 self-probe targets exceed the 8-token generation ceiling | HIGH |

### Verdict: **PROVISIONALLY_VALID** (for measuring the absence of capability)
The query-conditioned signal is genuinely absent. The evaluation correctly
detects this absence. But the eval generators don't have train-side analogues.

---

## 4. Cymek PRE50M — production smoke certification

- **Claimed capability:** the production training path works
- **Actual observable:** finite loss, parameter mutation, checkpoint/reload,
  token accounting, writer fence
- **Pass threshold:** all mechanical checks pass
- **Preregistered:** YES (PRE50M_ADDENDUM)

### Findings
| Issue | Severity |
|---|---|
| Data interface cert injected audit rows without registering them (fixed) | FIXED |
| Token budget didn't fund the resume-proof update (fixed) | FIXED |
| `bounded_warmup_schedule` used instead of canonical WSD | MEDIUM |

### Verdict: **VALIDATED** (for mechanical certification; not for cognition)

---

## 5. Citadel PRE500M — production readiness decision

- **Claimed capability:** readiness decision for the 500M campaign
- **Status:** BUILT, FAIL-CLOSED, currently BLOCKED (DATA_NOT_READY + entry point MISSING + tokenizer unfrozen)

### Verdict: **NOT_EXECUTED** (decision builder tested, certification not run)

---

## 6. Citadel data readiness gate (CITADEL-DATA-001)

- **Claimed capability:** mechanical audit of candidate training corpus
- **Actual observable:** 12 independent gates (provenance, splits, dedup, etc.)
- **Tests:** 17/17 passing
- **Verdict:** **PROVISIONALLY_VALID** (tested against synthetic fixtures; not
  yet exercised against a real materialized corpus)

---

## SUMMARY COUNTS

| Classification | Count | Surfaces |
|---|---|---|
| VALIDATED | 1 | PRE50M mechanical smoke |
| PROVISIONALLY_VALID | 3 | E0 dev cert, Triquetra absence detection, data gate |
| SHORTCUT_COMPROMISED | 1 | T1D exact-match |
| LEAKAGE_COMPROMISED | 1 | T1D cross-split overlap |
| UNDERPOWERED | 0 | — |
| NOT_EXECUTED | 1 | PRE500M |
| INVALID | 0 | — |

**Total evaluation surfaces audited: 7**

---

## Strongest shortcut discovered

**Latest-position heuristic scores 1.000 on every tier** of the tiered
arithmetic corpus. The answer is always the last number in the rendered
text. A model that copies the last number achieves perfect score without
doing any arithmetic. This was mechanically verified with real data from
all 5 tiers (300 test docs per tier, latest_position accuracy = 1.000).

## Strongest evaluation currently available

**Cymek E0 development certificate** — 368 cases, 12 heuristic baselines,
causal contrast pairs, shortcut repair history, 148 tests. It is
PROVISIONALLY_VALID because it has never been sealed, but it is the
most rigorously tested evaluation in the project.

## Claims that would disappear under shortcut-resistant, sealed evaluation

| Claim | Would survive? | Why |
|---|---|---|
| T1D "all arms FAIL" | **YES** | The model genuinely can't do arithmetic — even ignoring the EOS and shortcut issues, exact scores are 0-6.6% |
| T1D "curriculum doesn't help" | **MAYBE** | The B-vs-A contrast might change with EOS supervision |
| T1D "teacher doesn't help" | **MAYBE** | C held-out teacher was 51.5% — primitives ARE learnable |
| T1C "scale doesn't help" | **AT RISK** | D-vs-B was budget-confounded (8M vs 4M) |
| E0 "cognitive benchmark PASS" | **YES** | The 368-case dev certificate is legitimate for what it tests |
| Triquetra "query signal absent" | **YES** | The absence is genuinely measured |
| PRE50M "production path works" | **YES** | Mechanical certification is valid |

## Fixes required before next checkpoint promotion

1. Supervise EOS in training targets (T1E PLAN E1 — already designed)
2. Shuffle fact positions in training data (break the latest-position shortcut)
3. Add cross-split text dedup (eliminate the 173+357 contamination docs)
4. Freeze a production tokenizer artifact (B3)
5. Create sealed evaluation fixtures (not just dev certificates)
6. Token-match all contrast arms (D-vs-B at 8M vs 4M is confounded)
