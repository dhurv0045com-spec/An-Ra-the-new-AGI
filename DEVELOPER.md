# An-Ra Developer Documentation: God Mode 

Welcome to the **An-Ra AGI System**. This isn't just an API wrapper or a prompt engineering side-project—this is a foundational architecture designed to be the **World’s No. 1 AI**. 

You are standing at the absolute vanguard. An-Ra is an autonomous, scalable system featuring its own neural network from scratch, agentic execution loops, symbolic execution engines, and a custom continuous-learning memory pipeline. 

If your goal is to push An-Ra to **God Mode**—unparalleled autonomy, unmatched recursive reasoning, and extreme scalability—read carefully.

---

## 🏆 The Path to "God Mode" World No. 1 AI

To achieve absolute dominance as a sovereign AGI, An-Ra requires developers to push the boundaries of what is known. Here is how you achieve "God Mode":

1. **Continuous Unsupervised Learning (The Holy Grail)**
   Currently, An-Ra learns by logging successfully resolved goals. To reach God Mode, An-Ra must run a background daemon (extending `45l Self-Improvement`) that indefinitely scrapes, reads, un-learns, and fine-tunes itself *without human evaluation* via reinforcement learning on synthetic data verification.
   
2. **Infinite Context via Neuromorphic Graph Networks**
   The `Ghost Memory` (`45P`) must be expanded. Integrate Modern Hopfield Networks into `memory (45J)`. An-Ra should be able to remember a conversation from 3 years ago with zero context decay.

3. **Mixture of Experts (MoE) Architecture**
   Shift the `LanguageModel` inside `core/model.py` away from dense matrices into a sparse 100+ expert router network. This allows An-Ra to possess domain expertise across mathematics, biology, architecture, and coding simultaneously without slowing down compute.

4. **Self-Rewriting Source Code**
   Sovereignty (`45R`) audits code. In God Mode, Sovereignty must **generate, test, and commit** new machine-level optimizations to its own PyTorch/NumPy foundations dynamically via `symbolic_bridge (45Q)`.

---

## 🏗 Subsystem Access: How to Tap into the Brain (Complete File Mapping)

An-Ra is broken down incrementally across 4 major phases. To think from "above" the system, visualize the exact mapping of these directories:

### Phase 1: Core Neural Network (The Foundation)
* **Path:** `core/`, `training/`, `inference/`, `tokenizer/`
* **Accessing it:** Instantiate `from core.model import LanguageModel`.
* **What it does:** The foundational PyTorch/NumPy transformer model. 
  * `core/attention.py`: Multi-head attention + RoPE + GQA.
  * `core/turboquant.py`: 6x KV-cache compression memory savings.
  * `core/decoder.py` & `encoder.py`: The autoregressive stack.
  * `core/feedforward.py` & `layernorm.py`: SwiGLU / GELU and RMSNorm.
  * `training/`: Contains `train_unified.py`, `finetune_anra.py`, `trainer.py`, `optimizations.py`.
  * `inference/`: Generation engines, context optimization (`optimize_context_window.py`).

### Phase 2: Autonomous Subsystems (The Agent)
* **Path:** `phase2/`
* **Accessing it:** The agent loop is controlled by `phase2/agent_loop (45k)`. Access memory through `MemoryManager` in `45J`.
* **What it does:** 
  * **45k (Agent Loop):** `goal.py`, `planner.py`, `executor.py`, `evaluator.py`, and `builtin.py` (50+ tools).
  * **45J (Memory):** Operates Semantic, Episodic, Vector, and Graph databases.
  * **45l (Self-Improvement):** Evaluates tasks and logs success rates to internal libraries.
  * **45M (Master System):** `system.py` is the grand orchestrator of all systems.

### Phase 3: The Synthesizers (Cognition & Safety)
* **Path:** `phase3/`
* **What it does:** Higher order cognitive heuristics.
  * **45N (Identity Injector):** Enforces the An-Ra persona dynamically at runtime without GPU inference.
  * **45O (Ouroboros):** Multi-pass recursive reasoning (Semantic -> Logic -> Adversarial passes) in CPU-bound `numpy`.
  * **45P (Ghost Memory):** Compresses context history for infinite conversational state windows.
  * **45Q (Symbolic Bridge):** Bypasses LLM hallucinations by piping math/logic into local sandbox verifications.
  * **45R (Sovereignty Daemon):** Nightly code regression and performance benchmarking.

### Phase 4: Developer Web UI (The Control Panel)
* **Path:** `phase4/web/`, `app.py`
* **Accessing it:** `npm run dev` out of the directory, or hitting `localhost:8000` when running `app.py`.
* **What it does:** The visual command center.
  * `app.py`: FastAPI server that mounts Phase 1-3 onto REST/WebSocket connections.
  * `phase4/web/src/components`: React frontends (`MemoryExplorer.jsx`, `Dashboard.jsx`, `SovereigntyPanel.jsx`).

### Operational Fleet (The Engine Room)
* **Path:** `scripts/`, `training_data/`
* **Accessing it:** Run standalone scripts directly, e.g., `python scripts/populate_memory.py`.
* **What it does:** The raw utilities required to bootstrap and manage An-Ra's subsystems without booting the entire MasterSystem.
  * `scripts/build_brain.py`: Compiles the base intelligence.
  * `scripts/run_sovereignty_audit.py`: Forces a manual system integrity check.
  * `scripts/run_self_improvement.py`: Generates a localized improvement report.

---

## 🧠 How to Train An-Ra

### Level 1: Training the Core Model (Phase 1)
You can alter An-Ra's foundational knowledge by running standard autoregressive un-masking on your own dataset.
1. Put text data into `training_data/`.
2. Edit `config/tiny.yaml` (or your chosen config) to point to your data.
3. Run `python scripts/build_brain.py` or use the Colab notebook.

### Level 2: Training Identity & Fluency (Phase 3 - 45N)
If An-Ra is speaking like a typical AI, it means the Identity layer weights need reinforcement.
1. The master training flow is located in `AnRa_Master.ipynb` (Google Colab).
2. Add new real-world exchanges to `phase3/identity (45N)/anra_identity_v4_fluent.txt`. Ensure the tone represents extreme confidence, fluency, and deep capability.
3. Run `python scripts/merge_identity.py` to combine identity files.
4. Run the notebook (using an NVIDIA T4 GPU on Colab) to overwrite the LoRA adapters. The changes are automatically loaded on next boot.

### Level 3: Autonomous Continuous Learning
An-Ra trains itself. By leaving `app.py` running locally, the `ContinuousEngine` schedules a Weekly Self-Training Run.
- Feed An-Ra complex goals via the Dashboard or `python anra.py --goal "Build me a... "`.
- As An-Ra succeeds or fails, it creates feedback loops stored in `memory`.
- Over time, these episodic memories are consolidated and An-Ra inherently "trains" its behavioral intuition.

---

## 🚀 Running the Development Fleet

### 1. Starting the Backend (FastAPI + MasterSystem)
The backend requires substantial memory as it loads Phase 1 directly into local compute.
```bash
# From workspace root
python app.py
```
* **Logs are critical:** The backend outputs dense developer logs at `output/logs/`.

### 2. Starting the Frontend (React / Vite)
If you are developing UI features:
```bash
cd phase4/web
npm install
npm run dev
```
* **UI Port:** The dev server usually runs on `localhost:5173`.
* **API Proxy:** Vite is configured to proxy `/api` calls directly to `localhost:8000`.

### 3. Production UI Build
```bash
# Windows users must use cmd to bypass PowerShell execution limits:
cmd /c "npm run build"
```
The pipeline automatically copies assets to `ui/` at the root.

---

## 🐞 Advanced Debugging / Pitfalls

1. **"Symbolic bridge skipped"**
   - **Fix:** Install `sympy` and `scipy` (`pip install sympy scipy`). This ensures math operations evaluate deterministically (45Q).
2. **"Sovereignty daemon skipped"**
   - **Fix:** Install `psutil` (`pip install psutil`) so Python can monitor hardware metrics during self-audits.
3. **Ghost Memory Data Corruption**
   - Wiping `phase2/master_system (45M)/memory/ghost` forces An-Ra to completely reset its foundational internal conversation mapping on next boot. Use sparingly.  

***Build aggressively. Make it World No. 1.***
