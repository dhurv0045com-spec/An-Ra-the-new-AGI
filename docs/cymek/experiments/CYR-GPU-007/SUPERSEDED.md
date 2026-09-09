# CYR-GPU-007 — SUPERSEDED BEFORE EXECUTION

Status: `SUPERSEDED_BEFORE_OPERATOR_EXECUTION`.

No CYR-GPU-007 Colab cell was run and no scientific outcome exists.

After Commit B was written, a final code-level red-team found a compatibility-layer defect: the 007 runner temporarily replaced `cyr_gpu006_final.final_decision`, while `cyr_gpu007.final_decision` called that same module attribute. Under the compatibility patch this could recurse into itself. The dedicated tests had exercised the two functions separately and therefore did not expose the interaction.

Because executable changes after preregistration are forbidden, 007 is preserved unchanged. CYR-GPU-008 receives a new identity, captures the original 006 decision function before any runtime patch, and adds a regression test that executes the compatibility context and calls the patched decision path directly.
