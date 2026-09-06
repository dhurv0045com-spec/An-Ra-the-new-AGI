# T1E — Preregistration DRAFT: EOS-supervised arithmetic with termination
# and content separated (PENDING OPERATOR APPROVAL — DO NOT EXECUTE)

Status: **PREREGISTERED_PENDING_OPERATOR** (DRAFT — written 2026-09-06, corrects
the T1D limitations; NOT EXECUTED, must not run until the operator approves).

## What T1D demonstrated and what it could not

T1D (all arms SCIENTIFIC_FAIL, INCONCLUSIVE) executed cleanly, but its
generation contract had two structural flaws that make the null ambiguous:

1. **THE MODEL WAS NEVER SUPERVISED TO EMIT EOS.** Training rows encoded to
   literal characters only; the eligible mask covered answer characters only;
   `EOS_ID` was never appended. Generation stops on EOS/PAD/newline/
   non-alphabet **else MAX_TOKENS** — the bundle shows **15,000/15,000
   generation records ended MAX_TOKENS**. Exact-generation therefore
   conflates content failure with termination failure.
2. **Self-probe contract invalid** (57/96 targets exceed the 8-token
   generation ceiling) — carried in a SEPARATE future family, not here.

## T1E fixes (all implementation + contract; hypotheses re-frozen below)

### E1. EOS IS TRAINED (production semantic)
- Row tokens: `prompt chars + answer chars + EOS_ID`, same segment.
- Eligible mask supervises: answer characters **+ EOS**. Prompt unsupervised.
- Cymek `causal_lm_loss` already supports EOS targets when present.
- NOT a literal-newline fake. The generation stop token and the trained
  token are the same `EOS_ID`.

### E2. Generation-limit semantics
- `MAX_CONTENT_TOKENS = 8`, `MAX_GENERATION_STEPS = MAX_CONTENT_TOKENS + 1`
  — a full-length answer still gets one opportunity to emit EOS.
- Report: EOS stop rate, MAX_TOKENS rate, content length, termination
  correctness.

### E3. Classifier split
- `termination_failure_rate = MAX_TOKENS + NON_ALPHABET + premature invalid
  stops`; if > 50% and exact is low → `TERMINATION_FAILURE` (separate from
  `CONTENT_FAILURE`).

### E4. Metrics
- PRIMARY: full exact answer WITH valid termination.
- DIAGNOSTIC: content exact at target length ignoring extra continuation,
  first digit, per-position digits, sign, length, EOS position.
- This separates "computation wrong" from "computation right, stop wrong".

### E5. Teacher diversity
- Arm C evidence: held-out teacher microtasks reached 51.5% (n=200) while
  T2+ arithmetic stayed ~0 → PRIMITIVE_LEARNING_WITHOUT_COMPOSITIONAL_TRANSFER.
- T1E expands deterministic teacher pools substantially (digit add/sub with
  carry/borrow in-out, single-digit multiply, partial products, exact
  division microfacts), targeting LOW replay (T1D replay was 2,509–3,262×).

### E6. Token-matched contrasts
- Scale (B-vs-D) and representation (B-vs-E) contrasts match
  **LOSS_BEARING_TOKENS** (real/capacity tokens reported separately).
- No contrast changes model size + budget together.

### E7. Self-knowledge excluded
- Not carried into T1E (separate hypothesis family; separate generation-
  length contract to be preregistered first).

### E8. Data sizing
- Physical pool comfortably larger than schedulable demand; expansion only
  where diversity matters (teacher tasks, T3/T4 operand diversity, balanced
  answer lengths). Never duplicate bytes.

## Frozen T1E arms (DRAFT — operator may amend before approval)

```text
ARM A — CE, EOS-supervised, flat, 8M     (MID 3.7M)
ARM B — CE, EOS-supervised, curriculum, 8M (MID)
ARM C — B + expanded-diversity teacher, 8M (MID)
ARM D — curriculum @ TOKEN-MATCHED loss-bearing budget vs B (SCALE2 7.4M)
ARM E — masked softmax @ TOKEN-MATCHED budget (MID)
```

Primary metric: held-out TEST exact WITH valid EOS termination.
Contrasts: B−A curriculum, C−B teacher, D−B scale, E−B representation —
all token-matched. Classifiers add TERMINATION_FAILURE vs CONTENT_FAILURE.

## Success condition

Lift-off (any arm ≥ 0.20 LCB on a ≥200-row tier-1+ slice) OR a clean
elimination with termination and content separately measured — either way
the next lever (scale ladder vs objective vs data) is identified without
ambiguity.
