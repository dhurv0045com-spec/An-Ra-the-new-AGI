# CYR-GPU-013 / R1B — OPERATOR HANDOFF

**Status:** READY FOR OPERATOR COLAB CUDA PREEXECUTION GATE / NOT EXECUTED.

R1B is the immediate follow-up to completed CYR-GPU-012 R1. It tests whether the surprising intermediate class-space advantage replicates across fresh matched seeds and maps the early response curve at:

`19 → 1024 → 4096 → 8192 → 16384 → 24576`

Every arm uses batch64 and exactly 2,000 updates / 128,000 semantic row presentations. Two complete six-level curves are mandatory; a third runs only if pre-outcome CUDA calibration predicts that the entire third curve safely fits the 175-minute campaign wall.

## Run

Open `notebooks/cymek_colab_gpu_r1b_response_curve.ipynb` in Google Colab, choose a T4-class CUDA GPU, and use **Run all**.

Cell 0 must finish with:

`R1B PREEXECUTION GATE: PASS`

before any scientific training begins. It verifies the frozen executable and blob hashes, runs py_compile + R1B/R1/CYR11 unit tests, checks CUDA, calibrates all six vocabulary levels, and fails closed unless two complete curves fit.

Durable output is written to:

`MyDrive/CYMEK/CYR-GPU-013-R1B/`

Return:

`CYMEK_R1B_VOCAB_RESPONSE_CURVE_RESULTS.zip`

Do not modify seeds, vocabulary grid, thresholds, batch, endpoint, or runtime policy after seeing scientific outputs.
