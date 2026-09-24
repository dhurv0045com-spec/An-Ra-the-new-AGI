# SHORTCUT ATTACKS — CITADEL-EVAL-001

Date: 2026-09-13. Branch: `eval-integrity-001`.
Each attack is a cheap adversarial solver applied to the SAME evaluation data.
If a stupid solver scores close to the model, the evaluation is not strong
evidence of cognition.

## Attack results against tiered arithmetic corpus (T1D data)

Measured with `tools/audit_tiered_corpus.py` against 300 test docs per tier.

| Attack | T0 | T1 | T2 | T3 | T4 | Threat |
|---|---|---|---|---|---|---|
| **latest_position** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | CRITICAL |
| first_position | 0.930 | 0.107 | 0.000 | 0.000 | 0.000 | LOW |
| always_zero | 0.000 | 0.083 | 0.000 | 0.017 | 0.000 | NONE |
| reference_solver | 0.930 | 0.937 | 0.880 | 0.933 | 0.897 | — (this is the ceiling) |

**CRITICAL FINDING: latest_position scores 1.000 on EVERY tier.**
The answer is ALWAYS the last number in the rendered text. A model that
copies the last number achieves perfect score without doing arithmetic.
This was verified on 1,500 test documents across all 5 tiers.

### Why this happens
`tiered_data.tier_row()` renders rows as `operand operator operand = answer`.
The answer is always the last number. The latest_position baseline extracts
the last number and returns it. This is not a bug in the generator — it is
an inherent property of the Q/A format where the answer appears at the end.

### What this means for T1D/T1E results
A model scoring 5-6% on test is BELOW the latest_position baseline of 100%.
This means the model has NOT even learned to copy the last number.
The measured capability gap is even larger than the raw scores suggest.

### What this does NOT mean
It does NOT mean the evaluation is broken. It means the model is
very far from solving the task. The evaluation correctly measures
that the model can't do the task AND can't even do the shortcut.

---

## Attack results against E0 cognitive benchmark (Cymek data)

From the E0 development certificate (368 cases, 12 baselines):

| Baseline | Entity-value binding | State overwrite | Best case |
|---|---|---|---|
| latest_fact | — | varies | 100% (pre-fix) |
| bag_of_words | 81.77% (pre-fix) | varies | HIGH |
| lexical_overlap | — | varies | HIGH |
| random | ~25% | ~25% | LOW |
| oracle | 100% | 100% | — |

**Two false greens were caught and repaired** (bag_of_words scored 81.77%
on state tracking, latest_fact scored 100% on some state variants). After
repair (generator v0.4.0), all shortcut baselines dropped to within
null + 10pp. The current E0 generator v0.4.0 is shortcut-resistant.

---

## Attacks on Triquetra entity×value data

From Triquetra's own experiments:

| Attack | Result |
|---|---|
| Copy-first-fact | 50% on balanced tasks (trivial policy) |
| Copy-last-fact | 50% on balanced tasks |
| Most-frequent-value | varies by distribution |
| Query-blind (same answer regardless of query) | 62/64 fresh worlds returned same answer |

**Query-blind policy: the model returned the same answer in 62/64 fresh
worlds despite the changed query.** This means the model ignores the query
entirely and produces a default answer.

---

## Attack summary

| Attack type | T1D | E0 | Triquetra |
|---|---|---|---|
| Latest-position | **1.000** | caught+repaired | — |
| Lexical overlap | — | caught+repaired | — |
| Query-blind | — | — | **confirmed** |
| Constant answer | low | low | — |
| Candidate attestation | — | confirmed | — |
| Template ID | — | confirmed | — |

## Conclusion

The strongest shortcut in the project is **latest-position** on the
tiered arithmetic corpus, scoring 1.000 universally. This is a corpus
design limitation, not an evaluation bug — the Q/A format inherently
puts the answer at the end. The fix is to shuffle fact order in
multi-fact rows (already identified but not yet implemented in the
production data path).

The second strongest shortcut is **query-blindness** — models trained
on T1D data returned the same answer regardless of the query (confirmed
by Triquetra). This means the model didn't learn query-conditioning,
which is the core cognitive skill the project targets.

Both shortcuts are addressed in T1E (EOS supervision + fact shuffling)
and in the 500M campaign evaluation (query-swap sensitivity metrics).
