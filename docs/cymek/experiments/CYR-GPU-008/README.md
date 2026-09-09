# CYR-GPU-008

CYR-GPU-008 is the current operator experiment. CYR-GPU-006 stopped at the hardware feasibility gate before science; CYR-GPU-007 was superseded before execution after final code review found a recursion risk in its compatibility decision path.

008 keeps the real Cymek V5 model, frozen 24,576 tokenizer, actual-token exposure, candidate-free sustained G90, shared-parent matched HIGH/LOW/FIXED/HYST forks, replicated decisions, Drive-backed durability and failure packaging. Its hardware-only resolver adapts scope to the assigned Colab GPU without using scientific outcomes.

The resolver prefers a non-TINY real-V5 proxy. It chooses the strongest affordable tier: 3-parent retention+transfer; 2-parent retention+transfer; or 2-parent replicated retention-only. TINY is used only if no non-TINY tier fits and can never become a production-facing research candidate.

Do not run until `PREREGISTRATION.json` and `RUN_READINESS.json` exist and readiness is true. Then open `notebooks/cymek_colab_gpu_research_v8.ipynb`, select GPU, run Cell 0 only, require `CYR-GPU-008 PREEXECUTION GATE: PASS`, then run Cells 1 and 2 and return `CYMEK_GPU_RESEARCH_V8_RESULTS.zip`.
