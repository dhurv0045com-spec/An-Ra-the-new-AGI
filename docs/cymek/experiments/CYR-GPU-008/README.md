# CYR-GPU-008

CYR-GPU-008 is the current operator experiment. CYR-GPU-006 stopped at the hardware-feasibility gate before science; CYR-GPU-007 was superseded before execution after final code review found a recursion risk in its compatibility decision path.

008 keeps the real Cymek V5 model, frozen 24,576 tokenizer, actual-token exposure, candidate-free sustained G90, shared-parent matched HIGH/LOW/FIXED/HYST forks, replicated decisions, Drive-backed durability and failure packaging. Its hardware-only resolver adapts scope to the assigned Colab GPU without using scientific outcomes.

The resolver searches non-TINY real-V5 proxies before allowing TINY. It chooses among three preregistered scopes: 3-parent retention+transfer; 2-parent retention+transfer; or 2-parent replicated retention-only. TINY is last-resort development evidence and can never become a production-facing research candidate. The scientific dose floors remain 2M actual acquisition tokens per parent and 500k actual continuation tokens per arm; transfer, when enabled, retains a 500k actual-token/state floor.

Frozen executable: `504fe13e735e71baa3a01898f42930486067e5be`.
Preregistration commit: `703a94114f34a6f9115d59f7b847722642378e61`.
Notebook: `notebooks/cymek_colab_gpu_research_v8.ipynb`.
Expected bundle: `CYMEK_GPU_RESEARCH_V8_RESULTS.zip`.

`RUN_READINESS.json` records the operator gate. Do not execute until the final receipt-only closure CI has passed. Then open the V8 notebook, select a GPU runtime, run Cell 0 only, and require the exact line `CYR-GPU-008 PREEXECUTION GATE: PASS`. Only then run Cell 1 and Cell 2. Return the V8 result ZIP for raw evidence audit; do not promote a GPU result directly into PRE500M or production training.
