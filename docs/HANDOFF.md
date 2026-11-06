# ELENDRIL V4 handoff

The starting remote head was `930ca3bb723e0b5b191700eec3d5f450875a6e25`. That complete tree is retrievable through the remote `elendril-pre-v3` tag. A normal Elendril-only cleanup commit removed 726 tracked paths. Source branches, the prior worktree, raw results and checkpoint payloads retain their custody. [Inventory](TREE_INVENTORY.json) · [Evidence](EVIDENCE_AND_PROVENANCE.md)

Reproduce historical receipt scoring without any training.

~~~text
python -m tools.verify_elendril_evidence
python -m pytest -q
~~~

Run the V4 fixed policy from the active checkout. For the neural replay use a clean checkout of the receipt's source and the original checkpoint directory.

~~~text
python -m elendril cognitive-demo --output <new-fixed.json>
python -m elendril cognitive-demo --checkpoint-root <ELD-DEV-004/checkpoints> --checkpoint-sha256 96a15fd5e7a3de495d93c049fe6dde49f4874a2c148ac64c762dd20c49c9bc76 --output <new-neural.json>
~~~

The final active suite passed 84 tests and seven subtests, including notebook schema/disabled execution and mocked corruption/restore checks. These commands use inference and verification only. The V4 report preserves route reuse, the failed revision task, exact weight hashes and all costs. The checkpoint payloads are external to Git. [Current results](CURRENT_STATE.md) · [Kernel contract](COGNITIVE_KERNEL.md)

**Cloud decision.** The earlier V3 graph operators are preserved and their safe default paths tested. V4 cognitive-action training is **NO-GO** until it has a frozen action-learning dataset and a valid positive control. No local or remote research training campaign ran under V3 or V4. [Operations](OPERATIONS.md)
