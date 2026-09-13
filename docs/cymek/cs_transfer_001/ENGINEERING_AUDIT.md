# CS-TRANSFER-001 pre-execution engineering audit

Status: **no scientific arm has executed**. This document records defects found by static review before the executable freeze so they cannot be mistaken for post-outcome repairs.

## Audit 1 — physical-vocabulary initialization confound

A naïve same-seed construction is invalid for this experiment. V4096 and V24576 embeddings consume different numbers of random draws, so later block tensors diverge despite the same seed. The dedicated matched constructor now derives both arms from one V24576 reference and copies every shared tensor byte-for-byte. The test suite explicitly checks both the naïve failure and matched repair.

## Audit 2 — original answer delimiter impossible under shared `<4096` rule

Initial preregistration used `\nAnswer:` and a leading-space answer prefix. The committed production-tokenizer audit `artifacts/e1/local_tournament/encoding-24576.json`, probe `answer-01`, maps `Answer` to token ID **18224**. Since the primary causal surface requires every content token ID `<4096`, that delimiter would make the selection instrument invalid before training.

Prospective Amendment 1, issued before any experiment outcome, changes only the answer boundary to newline (`\n`) and removes the leading answer space. The same committed tokenizer probe observes newline as ID **202**. The original preregistration remains in Git history; `AMENDMENT_1.json` records the change and the canonical runner binds both documents into one effective protocol SHA.

## Audit 3 — synthetic pre-state reconstruction in development runner

The first unexecuted development runner attempted to certify an update by reconstructing a synthetic pre-update `TrainingState` from the post-update state. This was rejected during review because state certification must compare the actual immutable before-state with the actual after-state.

`cs_transfer_001_run_v2.py` retains `before = state`, performs `backend.step(before, batch)`, builds `after = before.advance(...)`, and calls `certify_update(before=before, after=after, ...)`.

## Audit 4 — crash between checkpoint and development evaluation

At preregistered evaluation boundaries a process can die after the checkpoint is durable but before the development receipt is written. Advancing training on resume would silently omit one AUC point.

The audited runner now checks the restored update before entering the next training update. If the update is a fixed development checkpoint and its evaluation receipt is missing, it evaluates the restored durable checkpoint first, then continues.

## Audit 5 — ambiguous Canary-v2 prefix diagnostic

Canary-v2 reported answer-prefix accuracy as zero even where decoded exact accuracy was high. CS-TRANSFER-001 therefore makes exact **token sequence + EOS** the authoritative primary metric. Decoded text is not used to determine the verdict. Prefix-token accuracy remains secondary only.

## Audit 6 — result-driven design contamination

The following are frozen before execution: arms, four seed pairs, data seed, selected row counts, low-ID eligibility, 480-update endpoint, development checkpoints, optimizer, objective, WSD shape, primary identity AUC metric, endpoint co-primary, decision thresholds, control-formation floor, and sealed-consumption rule.

If CPU preparation fails the low-ID acceptance/shortcut screens, no GPU arm should run. Repairing the data instrument then requires another explicit prospective amendment; it must not be tuned using model outcomes.

## Remaining qualification before launch

The repository implementation still requires execution of the static validator and CPU test suite in a torch/tokenizers-capable environment, followed by a real T4 preflight for **both** physical arms. Only after those pass should the executable commit be frozen and the Colab operator notebook pin that exact SHA.

Until then the status is `IMPLEMENTED_NOT_QUALIFIED`, not launch-ready and not scientifically executed.
