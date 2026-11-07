# Owner operations

Install the CPU/platform-appropriate PyTorch separately, then `python -m pip install -e ".[test]"`. Run lightweight tests and read-only custody checks locally. Do not run research training campaigns locally.

~~~text
python -m pytest -q
python -m tools.verify_elendril_evidence --checkpoint-base C:/Users/ankit/Downloads
python -m elendril inspect-checkpoint --root <root> --lineage <lineage>
python -m elendril verify-checkpoint --root <root> --lineage <lineage> --sha <sha256>
python -m elendril compare-checkpoints --root <root> --lineage <lineage> --sha <old> --other-sha <new>
python -m elendril resume-dry-run --root <root> --lineage <action-lineage>
~~~

Inspection safely decodes tensors with weights_only=True and a bounded payload read. Verification distinguishes byte/tensor checks from actual continuation. Exact dry-run reconstructs action data and the committed head without updates or writes, and requires matching clean source/runtime. `--allow-source-mismatch` is explicitly diagnostic. Legacy training continuation uses its original source. PREVIOUS is an explicit protected rollback SHA; corruption never silently falls back.

V5 owner notebooks target Colab one T4 and Kaggle two distinct physical T4 UUIDs. All switches are OFF initially. Run only the canaries first. Save their complete durable directory, including immutable checkpoint objects, mirror pointers, challenge JSON and expected generations. End the VM, reattach that output in a fresh VM, set recovery mappings, acknowledge the new session and execute recovery. A kernel restart on the same VM is insufficient. Same-device restart is exact; a different T4 UUID must pass recorded numerical-equivalence tolerances. Then qualify every current lane and inspect the full-plan timing forecast before enabling the campaign.

Colab has a 105-minute training ceiling plus 15-minute export reserve. Kaggle has 315+30 minutes and independent worker lineages, not pooled VRAM. These are caps, not measured runtime promises. Faster completion stops immediately. Kaggle requires a complete independently rescored Colab screen, the prospective formation gate and explicit owner review. Preserve partial results on any failure. Drive copy/readback alone and Kaggle /working alone are not independent off-session custody.

Official sources reviewed for this implementation: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [native Kaggle images](https://github.com/Kaggle/docker-python), [saved-output retrieval](https://github.com/Kaggle/kagglehub), [PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html) and [SDPA semantics](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html). Allocation, quotas and session lifetime vary. Native PyTorch is checked and bound in receipts; notebooks do not replace the CUDA stack.

The old graph notebooks remain retrievable via `elendril-v4-working` and [the removal manifest](ARCHIVED_PROGRAMS.json). The low-level cloud utility remains because the new action operator reuses its hardware/source/copy verification functions. No live CUDA, Drive or Kaggle hardware qualification has occurred in this session.
