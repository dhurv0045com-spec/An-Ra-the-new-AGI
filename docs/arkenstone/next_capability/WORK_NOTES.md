# ARK-014 implementation work notes (packet ARK-NEXT-001)

Date: 2026-09-12. Worktree: `codex/arkenstone-improvements` at
`C:/Users/ankit/AppData/Local/Temp/arkenstone-improvements-20260912`.

## Starting point (recorded before editing)

- Resolved start commit of `codex/arkenstone-improvements`: `9c09d9d682ce3b2cd60f8e5573c84e04922ec8a8`
  ("Improve Arkenstone runtime integrity and define next capability work order").
  Note: the prior handoff's base `933d4f3` is the historical Arkenstone base, not the branch head.
- Branch/worktree status at start: clean working tree, exactly at the published commit.
- Unrelated user work preserved: the `Arkenstone` worktree (branch `Arkenstone`) carries
  modified `tests/test_ark020_v3.py` and untracked `_rebuild_v4.py`; the `BRAMASTRA` checkout
  was not opened for editing. No `.codex-worktrees` paths were touched.
- Environment observed (not assumed): Windows; main Python 3.11.15 with
  PyTorch 2.14.0+cpu (CUDA unusable from Python); the machine also has an
  NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, driver 591.62). An isolated venv
  with `torch==2.14.0+cu126` (same torch version, CUDA build) + numpy was
  created at `C:/Users/ankit/.zcode/workspace/default/ark014-cu126-venv` for the
  GPU run; the main environment was not modified.

## Owned paths created/changed

- `experiments/ARK-014/ark014_binding.py` — frozen task contract (new)
- `experiments/ARK-014/run_ark014.py` — matched runner + controller + retention + verdicts (new)
- `experiments/ARK-014/demo.py` — local before/after report from real checkpoints (new)
- `tests/test_ark014.py`, `tests/test_ark014_demo.py` — focused checks (new)
- `experiments/COLAB/discovery_v6_common.py` — ReceiptWriter gained a backward-compatible
  `extra_source_paths` parameter so ARK-014 receipts bind their own sources
- `experiments/COLAB/run_discovery_v6.py` — `--campaign ARK-014` with the plan's 40-minute entry reserve
- `README.md` — ARK-014 status, experiment command and demo command
- `docs/arkenstone/next_capability/PROTOCOL_CLARIFICATIONS.md`, `WORK_NOTES.md`, `HANDOFF.md`
- `artifacts/arkenstone/ark014/<run-id>/` — run receipts and locally stored checkpoints
  (`*.pt` files are excluded from Git by the existing root ignore; receipts and outcomes are committed)

## Scientific design anchors

- Task construction reproduces the ARK-009 binding universe exactly; the builder
  asserts the historical anchor hashes (train/held-out fact-set signature SHA-256s
  from `experiments/ARK-009/TASK_MANIFEST.json`) at build time, so membership
  drift fails loudly instead of silently reusing a different task.
- The ARK-009 composite diagnostic is repaired into four orthogonal diagnostics
  (CANONICAL / ORDER_ONLY / QUERY_ONLY / QUERY_ORDER); ARK-009's swap rule is
  reused verbatim for the QUERY_ORDER composite.
- Matchedness: both acquisition arms initialize from `torch.manual_seed(2201)`,
  draw identical semantic minibatch streams from a `Generator(2201)`, and differ
  only in presented fact order (augmentation is a pure integer hash, consuming no
  RNG state — asserted by test).
- Controller: pure function of the BIND_CONTROL indicator stream; BIND_SEALED is
  measured only after the qualification decision and checkpoint snapshot, and
  never enters any decision path (behavioral test mutates sealed values across
  runs and requires identical decision streams).

## Engineering notes

- The inherited loss supervises 3 positions per row here (answer-BOS, digit, EOS)
  and the padding concern cannot arise (all rows 18 tokens); see the dated
  protocol clarification before any objective change. The historical objective
  is reused unchanged.
- `greedy_exact` groups rows by prompt length; all ARK-014 prompts share one
  length, so every diagnostic evaluates as a single batch.
- Two runner bugs were caught by the focused tests before any campaign run:
  diagnostic rows passed as dicts where (prompt, answer) pairs were required, and
  `summarize` iterating a dict of matched orders without `.items()`.
- GPU throughput on the RTX 4050: ~27.6 optimizer steps/s for the frozen
  Micro-128/batch-64 configuration; a full 84k-step box (both arms to 24k +
  6×6k retention) is roughly 55 minutes plus evaluations.

## Verification performed locally

- `python -m unittest tests.test_ark014` — 25 checks
- `python -m unittest tests.test_ark014_demo` — 10 checks
- `python -m unittest discover -s tests -p 'test_discovery_v6*.py'` — 23 existing
  runtime/campaign checks still pass after the ReceiptWriter extension
- ARK-014 preflight on CPU and on CUDA (PASS on both; CUDA receipt recorded)
- Tiny diagnostic-scale end-to-end campaign on CPU (30 acquisition steps,
  20 retention steps) exercising receipts, checkpoints, budget-blocked retention
  and the demo against the produced run directory
