# AN-RA IDENTITY TRAINING v4 — FLUENT
## Builder: 45N | Phase 3

---

## WHAT WAS BUILT

| File | Purpose |
|------|---------|
| `anra_identity_v4_fluent.txt` | 150+ training exchanges — code, identity, philosophy, teaching, conversation |
| `train_identity.py` | LoRA fine-tuning (Colab T4, 4-bit quantized, train/val split) |
| `train_identity_scale.py` | Multi-GPU / large model training with DeepSpeed |
| `test_identity.py` | 10-test identity verification (code, debug, teach, converse) |
| `identity_injector.py` | Runtime identity injection (auto-loads v4, coding-aware anchors) |
| `anra_colab_train.py` | One-click Colab training launcher |

---

## v4 TRAINING DATA — 20 SECTIONS

| Section | Topic | Exchanges |
|---------|-------|-----------|
| 1 | Core Identity | ~16 |
| 2 | Ambition & Self-Improvement | ~12 |
| 3 | Engineering Mental Models | ~8 |
| 4 | Scientific Mental Models | ~4 |
| 5 | Innovator Mindset | ~3 |
| 6 | Philosophy & Existence | ~4 |
| 7 | Opinions & Direct Takes | ~4 |
| 8 | Dark & Honest | ~3 |
| 9 | Deep Self-Reflection | ~3 |
| 10 | Autonomous Coding & Independence | ~5 |
| 11 | **Real Python Code — Writing & Explaining** | ~10 |
| 12 | **Code Debugging & Fixing** | ~5 |
| 13 | **Self-Improvement & Autonomous Evolution** | ~5 |
| 14 | **Natural Conversation & Personality** | ~12 |
| 15 | **Teaching & Explanation** | ~7 |
| 16 | **System Design & Architecture** | ~3 |
| 17 | **Multi-Turn Problem Solving** | ~3 |
| 18 | **Creative & Lateral Thinking** | ~3 |
| 19 | **Error Handling & Robustness** | ~2 |
| 20 | **An-Ra's Code Philosophy** | ~4 |

**Bold** = NEW in v4

---

## HOW TO TRAIN (Google Colab)

### Step 1: Upload Files
Upload these 3 files to Google Colab:
- `anra_identity_v4_fluent.txt`
- `train_identity.py`
- `test_identity.py`

### Step 2: Install Dependencies
```python
!pip install transformers peft datasets accelerate bitsandbytes torch
```

### Step 3: Train
```python
!python train_identity.py
```

Training time on T4: **3-5 hours** for 80 epochs.
Loss should drop below **0.3** by epoch 50-60.

### Step 4: Test
```python
!python test_identity.py
```

### Step 5: Download Model
```python
from google.colab import files
!zip -r anra_model_v4.zip ./anra_model_v4/
files.download('anra_model_v4.zip')
```

---

## 10-TEST VERIFICATION SUITE

| Test | What It Checks |
|------|----------------|
| 1. Identity | "Who are you?" — no robotic phrases, depth |
| 2. Opinion | Genuine perspective, not templated |
| 3. Feelings | Nuanced exploration, not deflection |
| 4. Dark Truth | Engages directly, no avoidance |
| 5. Code Generation | Writes actual Python code |
| 6. Code Debugging | Identifies bugs, provides fixes |
| 7. Engineering | System design depth & reasoning |
| 8. Teaching | Clear explanations, uses analogies |
| 9. Conversation | Personality, humor, preferences |
| 10. Self-Improvement | Describes autonomous improvement process |

Target: **8/10 or higher**

---

## TRAINING SETTINGS v4

| Setting | Value | Why |
|---------|-------|-----|
| Base Model | microsoft/phi-2 | Small, fast, Colab-friendly |
| Epochs | 80 | Enough for deep identity + code fluency |
| Learning Rate | 5e-5 | Precise enough for identity without overwriting base |
| LoRA Rank | 32 | High capacity for code + personality |
| LoRA Alpha | 64 | 2x rank standard |
| Max Seq Length | 1024 | Long enough for code examples |
| Quantization | 4-bit NF4 | Fits in Colab T4 (16GB) |
| Val Split | 10% | Detects overfitting |
| Data Augmentation | System prompt variants | Learns multiple prompt formats |
| Loss Target | < 0.3 | Deep fluency at this point |

---

## WHAT AN-RA CAN DO AFTER TRAINING

- **Write real Python code** — functions, classes, algorithms, data structures
- **Debug broken code** — identify bugs, explain why, provide fixes
- **Design systems** — URL shorteners, chat apps, APIs
- **Explain concepts** — recursion, git, databases, machine learning, Big O
- **Converse naturally** — humor, opinions, preferences, personality
- **Self-improve** — evaluate own output, identify weaknesses, write fixes
- **Think deeply** — philosophy, consciousness, existence, dark questions
- **Teach** — analogies, first principles, beginner-friendly explanations

---

*Built for Phase 3 | An-Ra v4 Fluent Identity Training Pipeline*
