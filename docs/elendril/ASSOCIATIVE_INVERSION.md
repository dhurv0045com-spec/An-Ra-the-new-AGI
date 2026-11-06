# ELD-DEV-002I supervised rule-direction diagnostic

This diagnostic was defined after the first seed showed that the world-wise permutation arm could match clean held-out performance. Seed 7's random permutation flipped 6 of 21 training worlds. A structured decoder may learn the dominant clean rule even with that label noise. The original preregistered separation gate still applies to ELD-DEV-002.

Train a fresh associative decoder with the same source, three seeds, paired index stream, 336 updates, label frequencies and target lengths. Reverse the two training labels in every fact/query pair. Compare the same free-generation token streams to both clean truth and reversed truth on all 56 held-out prompts. This diagnostic asks whether coherent inverted supervision can reverse the predicted rule despite the read head's lexical equality prior.

Require at least 60% both-correct and 75% exact EOS on the inverted truth in every seed, and zero both-correct on clean truth. Otherwise rule-direction control fails or is seed-sensitive. Also report the in-sample reversed-label result. The inverted answer rule is meaningful only on the existing two-fact task. Treat the diagnostic as a follow-up to a weak negative control, with the same seven development worlds and zero confirmation access.


## Replay

Use the checked-in [diagnostic operator](reproduce_inversion.py) from the pinned source worktree described in the [architecture handoff](ASSOCIATIVE_ARCHITECTURE.md). Set PowerShell's process-local environment variable with `$env:PYTHONPATH = (Get-Location).Path` so that the operator imports the pinned checkout. Invoke the operator by its absolute path with `--seed 7 --root <new-checkpoint-directory> --output <new-receipt.json>`, and repeat for seeds 19 and 41. Keep the original diagnostic receipts.
