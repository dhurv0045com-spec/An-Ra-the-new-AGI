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

## 🏗 Subsystem Access: How to Tap into the Brain

An-Ra is broken down incrementally across 4 major phases. 

### Phase 1: Core Neural Network
* **Path:** `core/`
* **Accessing it:** Instantiate `from core.model import LanguageModel`.
* **What it does:** The foundational PyTorch/NumPy transformer model. Change `config/` files to directly adjust the neural density.

### Phase 2: Autonomous Subsystems
* **Path:** `phase2/`
* **Accessing it:** The agent loop is controlled by `phase2/agent_loop (45k)`. Access memory through `MemoryManager` in `45J`.
* **What it does:** ReAct/Plan-Execute loop handling tool use, Semantic/Episodic vector databases. Custom hooks can be added to `tools.py` for new abilities.

### Phase 3: The Synthesizers (Fully Connected)
* **Path:** `phase3/`
* **What it does:** Higher order cognitive heuristics.
  * **45N (Identity Injector):** Enforces the An-Ra persona.
  * **45O (Ouroboros):** Multi-pass recursive reasoning in CPU-bound `numpy`.
  * **45P (Ghost Memory):** Compresses context history.
  * **45Q (Symbolic Bridge):** Mathematics sandbox verification.
  * **45R (Sovereignty Daemon):** Nightly code regression detection.

### Phase 4: Developer Web UI
* **Path:** `phase4/web/`
* **Accessing it:** `npm run dev` out of the directory, or hitting `localhost:8000/ui` when running `app.py`.
* **What it does:** The visual command center for the entire node architecture.

---

## 🧠 How to Train An-Ra

### Level 1: Training the Core Model (Phase 1)
You can alter An-Ra's foundational knowledge by running standard autoregressive un-masking on your own dataset.
1. Put text data into `training_data/`.
2. Edit `config/tiny.yaml` (or your chosen config) to point to your data.
3. Run `python -m training.trainer`.

### Level 2: Training Identity & Fluency (Phase 3 - 45N)
If An-Ra is speaking like a typical AI, it means the Identity layer weights need reinforcement.
1. The training script is located in `phase3/identity (45N)/train_identity.py`.
2. Add new real-world exchanges to `anra_identity_v4_fluent.txt`. Ensure the tone represents extreme confidence, fluency, and deep capability.
3. Run the trainer (using an NVIDIA GPU or Google Colab) to overwrite the LoRA adapters. The changes are automatically loaded on next boot.

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
