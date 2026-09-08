# CYR-GPU-006

Corrected shared-parent retention-policy campaign. It supersedes the unexecuted CYR-GPU-005 because the 005 frozen notebook would execute the scientific runner on CPU, reuse neither its Cell-0 resolver nor all three parents, and could decide from one subject.

Expected operator flow after Commit B marks readiness true:

1. Open `notebooks/cymek_colab_gpu_research_v6.ipynb` on Google Colab with GPU.
2. Run Cell 0. Continue only if `CYR-GPU-006 PREEXECUTION GATE: PASS` appears.
3. Run Cell 1. Google Drive is the durable stage store; rerunning skips complete parents/arms.
4. Run Cell 2 and return `CYMEK_GPU_RESEARCH_V6_RESULTS.zip`.

No GPU experiment has been executed by the agent. No TPU run is authorized here.
