# CYR-GPU-009

Current one-shot Colab experiment. It replaces V8 after the operator's real hardware calibration proved the V8 minimum campaign was too broad for the assigned free GPU before any science ran.

009 intentionally does less, better: two independent candidate-free G90 acquisitions and one exact matched HIGH_CONTINUE vs LOW_CONTINUE fork per qualified parent. It uses Arkenstone Discovery V7's progressive fixed-wall scheduling instead of an all-or-nothing whole-campaign feasibility gate.

Cell 0 may select TINY on a slow free Colab GPU. That is expected and carries a strict development-only claim ceiling. A healthy CUDA calibration will not be rejected merely because the full target dose is predicted to exceed the wall; Cell 1 runs the valid matched units that fit and Cell 2 returns the bundle.

After `PREREGISTRATION.json` exists, open `notebooks/cymek_colab_gpu_research_v9.ipynb` in a fresh GPU runtime, run Cell 0, require `CYR-GPU-009 PREEXECUTION GATE: PASS`, then run Cell 1 and Cell 2 once. Return `CYMEK_GPU_RESEARCH_V9_RESULTS.zip`.
