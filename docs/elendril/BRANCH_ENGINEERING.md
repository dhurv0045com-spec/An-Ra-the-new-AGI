# Shared V5 engineering changes

The branch-wide pass focused on the components that control experiment validity and recoverability. Existing campaign notebooks, historical evidence and pre-existing checkpoint files retain their original provenance.

| Area | Change |
| --- | --- |
| Model construction | Require positive integer dimensions and finite positive rotary and normalization constants before parameter allocation and architecture hashing. Valid configurations retain their canonical identities. |
| Tokenizer | Reject nonintegral IDs and incorrect batch cardinality. Require decoded text to be a string. Snapshot the reserved-ID map so later mutations of a caller-owned dictionary cannot change segmentation. |
| Local checkpoint | Reject unsafe directory components and explicit malformed digests before accessing checkpoint objects. Verify that the embedded training state belongs to the named lineage. |
| Persistent mirror | Check the actual object named by the pointer, even when it already exists. Stream component digests, verify new copies before installation, and give staging objects and pointer writes unique names. A corrupted copy cannot advance the trusted head. The mounted store continues to require one active publisher per lineage. |
| Research architecture and evaluation | Add a near-exact capacity control, isolated addressing ablations, and world-disjoint retrieval with unseen keys, query neutralization, and variable distractor count. [Relational results](RELATIONAL_RESULTS.md). |

The local focused suite passed 153 tests and seven subtests. Five selected production-entry tests also passed. The branch workflow covers the fast guards. The focused suite additionally checks the shared evaluation, objective, production sampler, backend, resume and checkpoint adapter contracts. The optional full production-entry sweep was stopped after reaching the higher-resource campaign tests. Selected production-entry and mounted mirror paths were tested separately on this machine. Accelerator, multi-host publication and a full production-scale campaign still need their own qualification.
