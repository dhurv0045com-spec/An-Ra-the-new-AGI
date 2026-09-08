# CYR-GPU-006

CYR-GPU-006 is the still-unexecuted Cymek Colab GPU research campaign that supersedes CYR-GPU-005. It uses the real Cymek V5 implementation, three independent arithmetic acquisition parents, true shared-parent matched retention forks, actual-token budgets, candidate-free generation, a hardware-only 60–170 minute resolver, Drive-backed stage durability, and failure-preserving evidence packaging.

The current design was reworked after a live audit of Arkenstone Discovery V6 at `6acd9dcbdd28d00f387ffcd004253a813aca4b66`. The important correction is that non-arithmetic transfer no longer repeats a zero-event HIGH-vs-LOW retention screen and no longer compares a young pre-continuation parent with an older retained state. It prospectively compares equal-age/equal-exposure `HYSTERETIC_HIGH_LOW` and `LOW_CONTINUE` states from the same parent, then measures robust order-augmented binding acquisition plus old-T2 retention under the same HIGH-LR + fixed replay stream.

Before execution, Commit B must bind an exact executable SHA and all file/dependency hashes in `PREREGISTRATION.json`; `RUN_READINESS.json` must be mechanically true. The notebook must fail closed if that preregistration is absent or CUDA is unavailable.

Expected operator flow only after readiness is true:

1. Open `notebooks/cymek_colab_gpu_research_v6.ipynb` in Google Colab with GPU.
2. Run Cell 0 only. Continue only if `CYR-GPU-006 PREEXECUTION GATE: PASS` appears.
3. Run Cell 1. Google Drive is the durable state/evidence store.
4. Run Cell 2 and return `CYMEK_GPU_RESEARCH_V6_RESULTS.zip`.

No CYR-GPU-006 scientific GPU result exists yet. No TPU run, PRE500M run, 500M campaign, or production scheduler promotion is authorized by this experiment preparation.
