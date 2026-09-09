# CYR-GPU-008 — superseded after calibration, before science

CYR-GPU-008 was opened by the operator in Google Colab and reached its hardware-only Cell-0 calibration gate. No scientific training began.

The assigned GPU produced passing calibrations, including approximately 180 real training tokens/s for RESEARCH_SMALL and 749 real training tokens/s for TINY, but the frozen V8 resolver rejected the session because even its two-parent / four-fork minimum was predicted to exceed the 170-minute wall.

This is **not a scientific negative result**. It is evidence that the V8 all-or-nothing preflight design was overconstrained for the free Colab hardware actually assigned.

CYR-GPU-009 supersedes V8. It keeps the real Cymek V5, production tokenizer, candidate-free G90, same-parent/same-future-stream matched comparisons, actual-token accounting, CUDA-only execution, Drive durability, and failure packaging. It deliberately narrows the scientific question to the highest-value replicated pair — HIGH_CONTINUE vs LOW_CONTINUE — and adopts Arkenstone Discovery V7's progressive fixed-wall technique: run matched units sequentially, reserve time for unfinished units, preserve partial receipts, and never reject a healthy GPU merely because the entire ideal campaign cannot be guaranteed before training.

No PRE500M, 500M, TPU, or production-scheduler claim follows from V8's calibration failure.
