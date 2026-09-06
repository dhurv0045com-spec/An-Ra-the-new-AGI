# EXTERNAL GAP REVIEW — ASSESSMENT (Arkenstone verification pass)

Source: `AN_RA_COMPLETE_GAP_ANALYSIS.md` (external reviewer, reviewed commits
at or before citadel `28ff690` / triquetra `fa44ea3` / cymek `28bf57a`).
This document applies the project's own evidence rule to the review itself:
every claim was re-verified against the live branches on 2026-09-06.

## Claim-by-claim verdicts

| Review claim | Verdict | Evidence |
|---|---|---|
| Citadel t1d/t1c/t1_canary test counts (38/10/6) | CONFIRMED | `grep -c "def test_"` on origin/citadel: 38, 10, 6 |
| bootstrap/cymek_checkpoint are self-contained scripts, not pytest files | CONFIRMED | 0 `def test_` matches in both files |
| Notebook checker BUILTINS list missing `print`, `str` (GAP 5) | **STALE — ALREADY FIXED UPSTREAM** | origin/citadel now uses `BUILTINS = set(dir(__builtins__)) | {"__import__", "get_ipython"}` — exactly the reviewer's recommended permanent fix; the review predates citadel's hardening commits |
| `blueprint/STATUS.md` byte-identical across branches (GAP 2) | CONFIRMED, sharpened | sha `3f46b2ec6447` identical on triquetra, cymek, AND citadel — unchanged across the entire multi-branch history |
| No convergence statement except Citadel<->Cymek (GAP 3) | CONFIRMED | `agent.md` on citadel is the only hit for "validates Cymek"; no other pair has one |
| No doc-vs-test cross-check in CI (GAP 4) | CONFIRMED | citadel's only workflow (e0-research.yml) contains zero agent.md references; cymek's CI verifies contract receipts but no status-doc counts |
| No root-level cross-branch MAP.md (GAP 6) | CONFIRMED | absent on main, cymek, citadel, BRAMASTRA, Arkenstone |
| No experiment-proposing mechanism (GAP 1) | CONFIRMED with context | nothing implements it; BRAMASTRA's RESEARCH_LOOP.md designs a learned discovery policy and explicitly marks it unimplemented |
| Review coverage | INCOMPLETE (not a fault, a scope note) | BRAMASTRA (independent from-scratch program with executed micro-experiments), Arkenstone (this branch), core-vnext, and iterate500/900 were outside its five-branch scope |

## Arkenstone actions taken (this branch only)

1. GAP 3 + GAP 6 -> `docs/arkenstone/BRANCH_RELATIONS.md`: the one-page
   convergence map (per-branch proves/done/connects-to), written from live
   evidence and maintained here since no other branch may be modified.
2. GAP 2 + GAP 4 -> `docs/arkenstone/verify_ledgers.py` +
   `tests/test_arkenstone_ledgers.py`: Arkenstone's ledgers are now
   self-verifying — every EXPERIMENT_LOG row's referenced artifacts must
   exist, every receipt must hash-verify, and the README's verification block
   is regenerated with the exact HEAD it was checked at (citadel's agent.md
   pattern, made mechanical).
3. GAP 5 -> practice recorded in README rules: small gaps are closed in the
   same session they are found (this run's examples: probe rebinding,
   manifest tamper-detection, S1 vacuity).
4. GAP 1 -> recorded as the known ceiling in BRANCH_RELATIONS (BRAMASTRA owns
   the design; Arkenstone owns the causal-measurement loop that would feed a
   learned proposer). Deliberately NOT fake-built.

## Claims Arkenstone explicitly does NOT act on

- Any fix to citadel/triquetra/cymek/blueprint files (branch isolation).
- The 242-ruff-errors observation: not verified (ruff not run here); recorded
  as UNVERIFIED rather than repeated as fact.
- The 9.4/10 ratings: judgment calls by the reviewer; recorded, not adopted
  as evidence.
