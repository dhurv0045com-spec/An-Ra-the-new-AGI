# CYR-GPU-012 / R1

**R1 is the current highest-priority Cymek experiment.** It is designed to isolate whether the large tied embedding/output class space is sufficient to suppress the early structural capability formation seen in CYR-GPU-011.

Operator flow after readiness is green:

1. Open `notebooks/cymek_colab_gpu_r1_representation.ipynb` in Google Colab with a T4-class CUDA GPU.
2. Run Cell 0. Continue only after `R1 PREEXECUTION GATE: PASS`.
3. Run Cell 1. Results/arm files are stored under `MyDrive/CYMEK/CYR-GPU-012-R1`; completed arms are safely reused on rerun.
4. Run Cell 2 and return `CYMEK_R1_REPRESENTATION_CAUSAL_RESULTS.zip`.

Scientific design: fixed batch64, fixed 8,000 updates / 512,000 semantic rows per arm, exact same active character token IDs, exact same examples/order, and copied shared initialization. Primary comparison is V19 versus V24576; two seeds run only if hardware calibration predicts they fit under the **175-minute hard wall**. V4096 is optional and never displaces primary replication.

No result from R1 directly authorizes PRE500M/500M or a production tokenizer change. Read `PLAN.md` and `DESIGN_REASONING.md` before interpretation.
