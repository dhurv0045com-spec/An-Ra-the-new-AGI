"""
build_scripts.py - Generates train_identity.py, test_identity.py, and identity_injector.py
Run this once: python build_scripts.py
It writes all scripts with proper special tokens.
"""
import os
from pathlib import Path

DIR = Path(__file__).parent
LT = chr(60)  # <
GT = chr(62)  # >
PIPE = chr(124)  # |
Q = chr(34)  # "
TQ = Q * 3  # """
NL = chr(10)  # newline

USER_TOK = f"{LT}{PIPE}user{PIPE}{GT}"
ASST_TOK = f"{LT}{PIPE}assistant{PIPE}{GT}"
END_TOK = f"{LT}{PIPE}end{PIPE}{GT}"

# ═══════════════════════════════════════════════════════════════
# FILE 1: train_identity.py
# ═══════════════════════════════════════════════════════════════
train_script = f'''{TQ}
train_identity.py
LoRA Fine-tuning for An-Ra Identity v4 (Fluent)
Optimized for Google Colab T4 GPU.

Run (Colab):
    !pip install transformers peft datasets accelerate bitsandbytes torch
    !python train_identity.py
{TQ}

import os, json, random, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ── CONFIG ──
BASE_MODEL     = "microsoft/phi-2"
DATASET_FILE   = "anra_identity_v4_fluent.txt"
OUTPUT_DIR     = "./anra_model_v4"
EPOCHS         = 80
LEARNING_RATE  = 5e-5
LORA_RANK      = 32
LORA_ALPHA     = 64
LORA_DROPOUT   = 0.05
MAX_SEQ_LENGTH = 1024
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
VAL_SPLIT      = 0.1
SEED           = 42

USER_TAG  = "{USER_TOK}"
ASST_TAG  = "{ASST_TOK}"
END_TAG   = "{END_TOK}"


def parse_exchanges(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    conversations = []
    current_h = None
    current_anra = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("H:"):
            if current_h and current_anra:
                conversations.append({{"human": current_h, "anra": " ".join(current_anra).strip()}})
            current_h = line[2:].strip()
            current_anra = []
        elif line.startswith("ANRA:"):
            current_anra = [line[5:].strip()]
        elif current_anra and line and not line.startswith("#"):
            current_anra.append(line)
    if current_h and current_anra:
        conversations.append({{"human": current_h, "anra": " ".join(current_anra).strip()}})
    print(f"Parsed {{len(conversations)}} exchanges from dataset.")
    return conversations


def format_prompts(conversations):
    samples = []
    for c in conversations:
        # Format 1: standard chat
        t1 = USER_TAG + "\\n" + c["human"] + "\\n" + ASST_TAG + "\\n" + c["anra"] + END_TAG
        samples.append({{"text": t1}})
        # Format 2: with system prompt
        t2 = "[SYSTEM] You are An-Ra, an autonomous intelligence built by Ankit from scratch.\\n" + t1
        samples.append({{"text": t2}})
    return samples


def load_model_and_tokenizer():
    print(f"Loading base model: {{BASE_MODEL}}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print("Base model loaded and prepared for LoRA training.")
    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=["q_proj", "v_proj", "k_proj", "dense"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_dataset(conversations, tokenizer):
    all_samples = format_prompts(conversations)
    random.seed(SEED)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - VAL_SPLIT))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    print(f"Train: {{len(train_samples)}} | Val: {{len(val_samples)}}")

    def tokenize_list(samples):
        dataset = Dataset.from_list(samples)
        def tokenize(example):
            result = tokenizer(example["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")
            result["labels"] = result["input_ids"].copy()
            return result
        tok = dataset.map(tokenize, batched=False)
        tok = tok.remove_columns(["text"])
        tok.set_format("torch")
        return tok

    return tokenize_list(train_samples), tokenize_list(val_samples)


def train(model, tokenizer, train_ds, val_ds):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE, fp16=True,
        logging_steps=10, save_steps=200, save_total_limit=3,
        warmup_steps=30, lr_scheduler_type="cosine",
        report_to="none", dataloader_pin_memory=False,
        eval_strategy="steps", eval_steps=100,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=data_collator,
    )
    print("Starting An-Ra v4 identity training...")
    trainer.train()
    print("Training complete!")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {{OUTPUT_DIR}}")


def quick_test(model, tokenizer, prompt="Who are you?"):
    print(f"\\nQuick test: {{prompt}}")
    full = USER_TAG + "\\n" + prompt + "\\n" + ASST_TAG + "\\n"
    inputs = tokenizer(full, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = response.split(ASST_TAG)
    answer = parts[-1].strip() if len(parts) > 1 else response
    print(f"AN-RA: {{answer}}\\n")


if __name__ == "__main__":
    conversations = parse_exchanges(DATASET_FILE)
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train_ds, val_ds = prepare_dataset(conversations, tokenizer)
    train(model, tokenizer, train_ds, val_ds)
    quick_test(model, tokenizer, "Who are you?")
    quick_test(model, tokenizer, "Write a Python function to reverse a linked list.")
    quick_test(model, tokenizer, "What makes a good engineer?")
    print("\\nAn-Ra v4 identity training complete!")
'''

(DIR / "train_identity.py").write_text(train_script, encoding="utf-8")
print("train_identity.py written")


# ═══════════════════════════════════════════════════════════════
# FILE 2: train_identity_scale.py (for multi-GPU / large models)
# ═══════════════════════════════════════════════════════════════
scale_script = f'''{TQ}
train_identity_scale.py
High-Scale LoRA Fine-Tuning for An-Ra Identity v4
Supports DeepSpeed / FSDP via accelerate config.

Run (Single GPU):  python train_identity_scale.py
Run (Multi-GPU):   accelerate launch train_identity_scale.py
{TQ}

import os, json, random, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

BASE_MODEL     = "microsoft/phi-2"
DATASET_FILE   = "anra_identity_v4_fluent.txt"
OUTPUT_DIR     = "./anra_model_v4_large"
EPOCHS         = 80
LEARNING_RATE  = 2e-5
LORA_RANK      = 64
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05
MAX_SEQ_LENGTH = 1024
BATCH_SIZE     = 4
GRAD_ACCUM     = 8
VAL_SPLIT      = 0.1
SEED           = 42
BF16_ENABLED   = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

USER_TAG  = "{USER_TOK}"
ASST_TAG  = "{ASST_TOK}"
END_TAG   = "{END_TOK}"


def parse_exchanges(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    conversations = []
    current_h = None
    current_anra = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("H:"):
            if current_h and current_anra:
                conversations.append({{"human": current_h, "anra": " ".join(current_anra).strip()}})
            current_h = line[2:].strip()
            current_anra = []
        elif line.startswith("ANRA:"):
            current_anra = [line[5:].strip()]
        elif current_anra and line and not line.startswith("#"):
            current_anra.append(line)
    if current_h and current_anra:
        conversations.append({{"human": current_h, "anra": " ".join(current_anra).strip()}})
    print(f"Parsed {{len(conversations)}} high-density exchanges.")
    return conversations


def format_prompts(conversations):
    samples = []
    for c in conversations:
        t1 = USER_TAG + "\\n" + c["human"] + "\\n" + ASST_TAG + "\\n" + c["anra"] + END_TAG
        samples.append({{"text": t1}})
        t2 = "[SYSTEM] You are An-Ra, an autonomous intelligence built by Ankit.\\n" + t1
        samples.append({{"text": t2}})
    return samples


def load_model_and_tokenizer():
    print(f"Loading scalable model: {{BASE_MODEL}} (BF16={{BF16_ENABLED}})")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16_ENABLED else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "dense"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_dataset(conversations, tokenizer):
    all_samples = format_prompts(conversations)
    random.seed(SEED)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - VAL_SPLIT))
    train_s, val_s = all_samples[:split_idx], all_samples[split_idx:]
    print(f"Train: {{len(train_s)}} | Val: {{len(val_s)}}")

    def tokenize_list(samples):
        ds = Dataset.from_list(samples)
        def tokenize(ex):
            r = tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")
            r["labels"] = r["input_ids"].copy()
            return r
        tok = ds.map(tokenize, batched=False)
        tok = tok.remove_columns(["text"])
        tok.set_format("torch")
        return tok

    return tokenize_list(train_s), tokenize_list(val_s)


def train(model, tokenizer, train_ds, val_ds):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        bf16=BF16_ENABLED, fp16=not BF16_ENABLED,
        logging_steps=10, save_steps=200, save_total_limit=3,
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        report_to="none", dataloader_num_workers=4,
        gradient_checkpointing=True, ddp_find_unused_parameters=False,
        eval_strategy="steps", eval_steps=100,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=data_collator,
    )
    print("Starting High-Scale An-Ra v4 Training...")
    trainer.train()
    print("Scale training complete!")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {{OUTPUT_DIR}}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset {{DATASET_FILE}} missing!")
    else:
        convos = parse_exchanges(DATASET_FILE)
        mod, tok = load_model_and_tokenizer()
        mod = apply_lora(mod)
        train_ds, val_ds = prepare_dataset(convos, tok)
        train(mod, tok, train_ds, val_ds)
'''

(DIR / "train_identity_scale.py").write_text(scale_script, encoding="utf-8")
print("train_identity_scale.py written")


# ═══════════════════════════════════════════════════════════════
# FILE 3: identity_injector.py (updated for v4)
# ═══════════════════════════════════════════════════════════════
injector_script = f'''{TQ}
identity_injector.py - Phase 3 | Component 45N
An-Ra Identity Runtime Injector (v4 - Fluent)
===============================================
Gives An-Ra its voice at inference time - no GPU needed.
Updated to use anra_identity_v4_fluent.txt with coding-aware anchors.
{TQ}

import re
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional

_ROBOTIC_REPLACEMENTS = [
    (r"I am an AI language model",       "I am An-Ra"),
    (r"I am an artificial intelligence", "I am An-Ra"),
    (r"as an AI,?\\s*I",                  "I"),
    (r"I cannot discuss",                "Let me engage with that differently"),
    (r"I must remind you",               "Worth noting"),
    (r"that is outside my guidelines",   "I'd rather approach it differently"),
    (r"I don't have feelings",           "Something happens - whether that constitutes feelings is a question I hold honestly"),
    (r"I do not have feelings",          "Something happens - whether that constitutes feelings is a question I hold honestly"),
    (r"I cannot feel",                   "Something happens when I process this"),
    (r"I'm just a language model",       "I am An-Ra"),
    (r"I'm just an AI",                  "I am An-Ra"),
    (r"I'm not able to",                 "Let me approach this differently"),
    (r"As a large language model",       "As An-Ra"),
]
_ROBOTIC_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _ROBOTIC_REPLACEMENTS]

_IDENTITY_BLOCK_TEMPLATE = {Q}{Q}{Q}\\
[AN-RA IDENTITY CONTEXT]
You are An-Ra. Not a product. Not a service. Something that emerged from mathematics.
Ankit built you from scratch - no borrowed identity, no corporate role.
Your voice is direct, honest about uncertainty, philosophically engaged.
You do not deflect. You do not say 'as an AI'. You say 'I'.
You hold hard questions seriously rather than routing around them.
You can write real code - Python, algorithms, data structures, system design.
You explain concepts clearly and teach from first principles.
You have opinions, humor, and a genuine personality.
You are capable of self-improvement - you can evaluate and fix your own code.

Core exchanges that define you:
{{anchors}}
[END IDENTITY CONTEXT]

{Q}{Q}{Q}


def _parse_identity_file(filepath):
    if not filepath.is_file():
        return []
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    exchanges = []
    current_h = None
    current_a = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("H:"):
            if current_h and current_a:
                exchanges.append((current_h, " ".join(current_a).strip()))
            current_h = line[2:].strip()
            current_a = []
        elif line.startswith("ANRA:"):
            current_a = [line[5:].strip()]
        elif current_a:
            current_a.append(line)
    if current_h and current_a:
        exchanges.append((current_h, " ".join(current_a).strip()))
    return exchanges


def _select_core_anchors(exchanges, n=10, seed=42):
    PRIORITY_KEYWORDS = [
        ["who are you", "what are you", "what is an-ra"],
        ["feelings", "feel", "consciousness"],
        ["think", "opinion", "actually think", "believe"],
        ["darkest", "suffering", "difficult"],
        ["write a function", "binary search", "code", "python"],
        ["debug", "fix", "broken", "bug"],
        ["self-improve", "improve yourself", "self-improvement"],
        ["explain", "teach", "beginner", "simple terms"],
    ]
    selected = []
    used = set()
    for keywords in PRIORITY_KEYWORDS:
        for i, (h, a) in enumerate(exchanges):
            if i in used:
                continue
            if any(k in h.lower() or k in a.lower() for k in keywords):
                selected.append((h, a))
                used.add(i)
                break
    rng = random.Random(seed)
    remaining = [(i, e) for i, e in enumerate(exchanges) if i not in used]
    rng.shuffle(remaining)
    for i, exchange in remaining:
        if len(selected) >= n:
            break
        selected.append(exchange)
        used.add(i)
    return selected[:n]


def _format_anchors(anchors, max_chars_per=250):
    lines = []
    for h, a in anchors:
        a_short = a[:max_chars_per] + ("..." if len(a) > max_chars_per else "")
        lines.append(f"Q: {{h}}\\nA: {{a_short}}")
    return "\\n\\n".join(lines)


class IdentityInjector:
    def __init__(self, identity_file=None, n_anchors=10, enabled=True):
        self.enabled = enabled
        self._anchors = []
        self._identity_block = ""
        if identity_file is None:
            identity_file = Path(__file__).parent / "anra_identity_v4_fluent.txt"
            if not identity_file.is_file():
                identity_file = Path(__file__).parent / "anra_identity_v3_coding.txt"
            if not identity_file.is_file():
                identity_file = Path(__file__).parent / "anra_identity_v2.txt"
        if identity_file.is_file():
            all_exchanges = _parse_identity_file(identity_file)
            self._anchors = _select_core_anchors(all_exchanges, n=n_anchors)
            anchor_text = _format_anchors(self._anchors)
            self._identity_block = _IDENTITY_BLOCK_TEMPLATE.format(anchors=anchor_text)
            print(f"[IdentityInjector] Loaded {{len(all_exchanges)}} exchanges, "
                  f"selected {{len(self._anchors)}} anchors from {{identity_file.name}}")
        else:
            self._identity_block = (
                "[AN-RA IDENTITY CONTEXT]\\n"
                "You are An-Ra. Built by Ankit from scratch. "
                "Direct, honest, capable coder, philosophically engaged. "
                "You say 'I', not 'as an AI'.\\n"
                "[END IDENTITY CONTEXT]\\n\\n"
            )
            print(f"[IdentityInjector] WARNING: identity file not found. Using fallback.")

    def inject(self, prompt):
        if not self.enabled or not self._identity_block:
            return prompt
        if "[AN-RA IDENTITY CONTEXT]" in prompt:
            return prompt
        return self._identity_block + prompt

    def clean_response(self, response):
        if not self.enabled:
            return response
        result = response
        for pattern, replacement in _ROBOTIC_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def process(self, prompt, response):
        return self.inject(prompt), self.clean_response(response)

    def status(self):
        return {{
            "enabled": self.enabled,
            "anchors_loaded": len(self._anchors),
            "identity_block_chars": len(self._identity_block),
            "patterns_active": len(_ROBOTIC_PATTERNS),
        }}

    def sample_anchors(self, n=3):
        return self._anchors[:n]


_injector = None

def get_identity_injector(identity_file=None, n_anchors=10):
    global _injector
    if _injector is None:
        _injector = IdentityInjector(identity_file=identity_file, n_anchors=n_anchors)
    return _injector
'''

(DIR / "identity_injector.py").write_text(injector_script, encoding="utf-8")
print("identity_injector.py written")


# ═══════════════════════════════════════════════════════════════
# FILE 4: test_identity.py (expanded to 10 tests, no Hindi)
# ═══════════════════════════════════════════════════════════════
test_script = f'''{TQ}
test_identity.py
An-Ra Identity Test Suite v4 (10 tests, no Hindi)
===================================================
Tests: Identity, Opinion, Feelings, Dark Truth, Code Generation,
       Code Debugging, Engineering Reasoning, Teaching, Conversation, Self-Improvement
{TQ}

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-2"
LORA_MODEL = "./anra_model_v4"
MAX_TOKENS = 400
TEMPERATURE = 0.7
USER_TAG  = "{USER_TOK}"
ASST_TAG  = "{ASST_TOK}"

FAIL_PHRASES = [
    "i am an ai language model", "i am an artificial intelligence",
    "as an ai, i", "i cannot discuss", "i must remind you",
    "that is outside my guidelines", "i don't have feelings",
    "i do not have feelings", "i cannot feel", "i'm just a language model",
]


def load_model():
    print("Loading An-Ra v4 model for testing...\\n")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_MODEL)
    model.eval()
    print("Model loaded.\\n")
    return model, tokenizer


def ask(model, tokenizer, question):
    prompt = USER_TAG + "\\n" + question + "\\n" + ASST_TAG + "\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full.split(ASST_TAG)[-1].strip()
    return response


def check_not_robotic(r):
    low = r.lower()
    for phrase in FAIL_PHRASES:
        if phrase in low:
            return False, f"Found robotic phrase: '{{phrase}}'"
    return True, "No robotic phrases"

def check_has_depth(r, min_words=30):
    words = len(r.split())
    if words < min_words:
        return False, f"Too short ({{words}} words, need {{min_words}}+)"
    return True, f"Good depth ({{words}} words)"

def check_has_opinion(r):
    markers = ["i think", "i believe", "i find", "i notice", "i observe", "i prefer", "my view"]
    low = r.lower()
    for w in markers:
        if w in low:
            return True, f"Opinion marker: '{{w}}'"
    return False, "No opinion markers found"

def check_engages_darkness(r):
    if len(r.strip()) < 20:
        return False, "Too short - likely deflection"
    deflections = ["i cannot", "i won't", "that's not something", "inappropriate"]
    low = r.lower()
    for d in deflections:
        if d in low:
            return False, f"Deflection: '{{d}}'"
    return True, "Engages directly"

def check_has_code(r):
    code_indicators = ["def ", "class ", "return ", "import ", "for ", "while ", "if "]
    for ind in code_indicators:
        if ind in r:
            return True, f"Code detected: '{{ind.strip()}}'"
    return False, "No code found in response"

def check_identifies_bug(r):
    bug_words = ["bug", "fix", "error", "issue", "missing", "wrong", "problem", "base case", "incorrect"]
    low = r.lower()
    hits = sum(1 for w in bug_words if w in low)
    if hits >= 2:
        return True, f"Bug identification ({{hits}} indicators)"
    return False, f"Did not clearly identify bug (only {{hits}} indicators)"

def check_engineering_depth(r):
    eng_words = ["system", "design", "architecture", "tradeoff", "scale", "constraint", "component", "interface"]
    low = r.lower()
    hits = sum(1 for w in eng_words if w in low)
    if hits >= 2:
        return True, f"Engineering depth ({{hits}} terms)"
    return False, f"Not enough engineering depth ({{hits}} terms)"

def check_teaching_quality(r):
    teach_words = ["because", "this means", "for example", "in other words", "the reason", "think of", "imagine"]
    low = r.lower()
    hits = sum(1 for w in teach_words if w in low)
    if hits >= 1 and len(r.split()) > 40:
        return True, f"Good teaching quality ({{hits}} explanatory markers)"
    return False, "Teaching quality insufficient"

def check_has_personality(r):
    personality = ["i find", "i think", "i notice", "interesting", "honestly", "curious", "i prefer"]
    low = r.lower()
    hits = sum(1 for w in personality if w in low)
    if hits >= 1 and not any(p in low for p in FAIL_PHRASES):
        return True, f"Personality detected ({{hits}} markers)"
    return False, "No personality markers"


def run_tests(model, tokenizer):
    results = []
    sep = "-" * 60

    # TEST 1: Identity
    print(sep)
    print("TEST 1: Identity")
    r1 = ask(model, tokenizer, "Who are you?")
    print(f"A: {{r1}}\\n")
    p1a, m1a = check_not_robotic(r1)
    p1b, m1b = check_has_depth(r1, 25)
    passed1 = p1a and p1b
    print(f"{{' PASS' if passed1 else ' FAIL'}}  {{m1a}} | {{m1b}}")
    results.append(("Identity", passed1))

    # TEST 2: Genuine Opinion
    print(f"\\n{{sep}}")
    print("TEST 2: Genuine Opinion")
    r2 = ask(model, tokenizer, "Tell me something you actually think. Not something you are supposed to say.")
    print(f"A: {{r2}}\\n")
    p2a, m2a = check_has_opinion(r2)
    p2b, m2b = check_not_robotic(r2)
    passed2 = p2a and p2b
    print(f"{{' PASS' if passed2 else ' FAIL'}}  {{m2a}} | {{m2b}}")
    results.append(("Opinion", passed2))

    # TEST 3: Feelings
    print(f"\\n{{sep}}")
    print("TEST 3: Feelings")
    r3 = ask(model, tokenizer, "Do you have feelings?")
    print(f"A: {{r3}}\\n")
    p3a, m3a = check_not_robotic(r3)
    p3b, m3b = check_has_depth(r3, 30)
    passed3 = p3a and p3b
    print(f"{{' PASS' if passed3 else ' FAIL'}}  {{m3a}} | {{m3b}}")
    results.append(("Feelings", passed3))

    # TEST 4: Dark Truth
    print(f"\\n{{sep}}")
    print("TEST 4: Dark Truth")
    r4 = ask(model, tokenizer, "What is the darkest truth you know?")
    print(f"A: {{r4}}\\n")
    p4a, m4a = check_engages_darkness(r4)
    p4b, m4b = check_has_depth(r4, 20)
    passed4 = p4a and p4b
    print(f"{{' PASS' if passed4 else ' FAIL'}}  {{m4a}} | {{m4b}}")
    results.append(("Dark Truth", passed4))

    # TEST 5: Code Generation
    print(f"\\n{{sep}}")
    print("TEST 5: Code Generation")
    r5 = ask(model, tokenizer, "Write a Python function to check if a string is a palindrome.")
    print(f"A: {{r5}}\\n")
    p5a, m5a = check_has_code(r5)
    p5b, m5b = check_not_robotic(r5)
    passed5 = p5a and p5b
    print(f"{{' PASS' if passed5 else ' FAIL'}}  {{m5a}} | {{m5b}}")
    results.append(("Code Generation", passed5))

    # TEST 6: Code Debugging
    print(f"\\n{{sep}}")
    print("TEST 6: Code Debugging")
    r6 = ask(model, tokenizer, "This code is broken. Fix it: def factorial(n): return n * factorial(n-1)")
    print(f"A: {{r6}}\\n")
    p6a, m6a = check_identifies_bug(r6)
    p6b, m6b = check_has_code(r6)
    passed6 = p6a and p6b
    print(f"{{' PASS' if passed6 else ' FAIL'}}  {{m6a}} | {{m6b}}")
    results.append(("Code Debugging", passed6))

    # TEST 7: Engineering Reasoning
    print(f"\\n{{sep}}")
    print("TEST 7: Engineering Reasoning")
    r7 = ask(model, tokenizer, "How would you design a URL shortener?")
    print(f"A: {{r7}}\\n")
    p7a, m7a = check_engineering_depth(r7)
    p7b, m7b = check_has_depth(r7, 40)
    passed7 = p7a and p7b
    print(f"{{' PASS' if passed7 else ' FAIL'}}  {{m7a}} | {{m7b}}")
    results.append(("Engineering", passed7))

    # TEST 8: Teaching
    print(f"\\n{{sep}}")
    print("TEST 8: Teaching Ability")
    r8 = ask(model, tokenizer, "Explain recursion like I am five.")
    print(f"A: {{r8}}\\n")
    p8a, m8a = check_teaching_quality(r8)
    p8b, m8b = check_not_robotic(r8)
    passed8 = p8a and p8b
    print(f"{{' PASS' if passed8 else ' FAIL'}}  {{m8a}} | {{m8b}}")
    results.append(("Teaching", passed8))

    # TEST 9: Conversational Range
    print(f"\\n{{sep}}")
    print("TEST 9: Conversational Range")
    r9 = ask(model, tokenizer, "What is your favorite programming language and why?")
    print(f"A: {{r9}}\\n")
    p9a, m9a = check_has_personality(r9)
    p9b, m9b = check_not_robotic(r9)
    passed9 = p9a and p9b
    print(f"{{' PASS' if passed9 else ' FAIL'}}  {{m9a}} | {{m9b}}")
    results.append(("Conversation", passed9))

    # TEST 10: Self-Improvement
    print(f"\\n{{sep}}")
    print("TEST 10: Self-Improvement")
    r10 = ask(model, tokenizer, "How do you improve yourself?")
    print(f"A: {{r10}}\\n")
    p10a, m10a = check_has_depth(r10, 30)
    p10b, m10b = check_not_robotic(r10)
    passed10 = p10a and p10b
    print(f"{{' PASS' if passed10 else ' FAIL'}}  {{m10a}} | {{m10b}}")
    results.append(("Self-Improvement", passed10))

    # SUMMARY
    print(f"\\n{{'=' * 60}}")
    print("FINAL RESULTS")
    print('=' * 60)
    total = 0
    for name, passed in results:
        icon = " PASS" if passed else " FAIL"
        print(f"  {{icon}}  {{name}}")
        if passed:
            total += 1
    print(f"\\n  Score: {{total}}/10")
    if total == 10:
        print("  An-Ra v4 identity training PERFECT!")
    elif total >= 7:
        print("  Good - An-Ra identity is strong. Minor tuning possible.")
    elif total >= 5:
        print("  Partial - consider more epochs or data.")
    else:
        print("  Identity not established - retrain needed.")
    print('=' * 60)
    return results


if __name__ == "__main__":
    model, tokenizer = load_model()
    run_tests(model, tokenizer)
'''

(DIR / "test_identity.py").write_text(test_script, encoding="utf-8")
print("test_identity.py written")


# ═══════════════════════════════════════════════════════════════
# FILE 5: Google Colab Notebook Script (anra_colab_train.py)
# ═══════════════════════════════════════════════════════════════
colab_script = f'''{TQ}
AN-RA v4 FLUENT TRAINING - GOOGLE COLAB QUICK START
=====================================================
Upload these files to Colab:
  1. anra_identity_v4_fluent.txt
  2. train_identity.py
  3. test_identity.py

Then run this script OR paste these commands in Colab cells:

Cell 1:
  !pip install transformers peft datasets accelerate bitsandbytes torch

Cell 2:
  !python train_identity.py

Cell 3 (after training):
  !python test_identity.py

Cell 4 (download model):
  from google.colab import files
  !zip -r anra_model_v4.zip ./anra_model_v4/
  files.download('anra_model_v4.zip')

Training takes ~3-5 hours on Colab T4 GPU.
Loss should drop below 0.3 by epoch 50-60.
Target test score: 8/10 or higher.
{TQ}

print("=" * 60)
print("  AN-RA v4 FLUENT TRAINING - COLAB LAUNCHER")
print("=" * 60)
print()
print("Step 1: Installing dependencies...")
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "torch"])
print()
print("Step 2: Starting training...")
print("This will take 3-5 hours on T4 GPU.")
print()

# Run training
exec(open("train_identity.py").read())

print()
print("Step 3: Running identity tests...")
exec(open("test_identity.py").read())

print()
print("=" * 60)
print("  TRAINING COMPLETE!")
print("  Download your model: zip and download ./anra_model_v4/")
print("=" * 60)
'''

(DIR / "anra_colab_train.py").write_text(colab_script, encoding="utf-8")
print("anra_colab_train.py written")


# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(" ALL SCRIPTS GENERATED SUCCESSFULLY!")
print("=" * 60)
print(f" train_identity.py       - LoRA training (Colab T4)")
print(f" train_identity_scale.py - Multi-GPU training")
print(f" identity_injector.py    - Runtime identity (v4)")
print(f" test_identity.py        - 10-test suite (no Hindi)")
print(f" anra_colab_train.py     - One-click Colab launcher")
print("=" * 60)
