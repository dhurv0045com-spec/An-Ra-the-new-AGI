"""
train_identity_scale.py
High-Scale LoRA Fine-Tuning Script for An-Ra Identity v3
==========================================================
Uses PEFT + LoRA, designed to scale from a single 8GB GPU to a full H100 cluster.
Supports DeepSpeed / FSDP if supplied via 'accelerate config'.
Optimized with bf16 and gradient checkpointing for large-scale training (7B-70B).

Run (Single GPU):
    python phase3/45N/train_identity_scale.py

Run (Multi-GPU Cluster with DeepSpeed ZeRO-3):
    accelerate config
    accelerate launch phase3/45N/train_identity_scale.py

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes torch deepspeed
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# ─────────────────────────────────────────
# CONFIG FOR HIGH-SCALE TRAINING
# ─────────────────────────────────────────
BASE_MODEL     = "microsoft/phi-2"               # Scalable up to "meta-llama/Llama-3-70b-Instruct"
DATASET_FILE   = "anra_identity_v3_coding.txt"   # Extended dataset with autonomous coding examples
OUTPUT_DIR     = "./anra_model_v3_large"         # Checkpoint dir
EPOCHS         = 100
LEARNING_RATE  = 2e-5                            # Lower LR for scale stability
LORA_RANK      = 64                              # High capacity LoRA rank
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05
MAX_SEQ_LENGTH = 1024                            # Extended context window
BATCH_SIZE     = 4                               # Per device batch size
GRAD_ACCUM     = 8                               # Effective batch logic
BF16_ENABLED   = torch.cuda.is_bf16_supported()  # Use true bfloat16 if A100/H100


def parse_exchanges(filepath):
    """Parse H: / ANRA: exchange format into training samples."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    conversations = []
    current_h = None
    current_anra = []

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("H:"):
            if current_h and current_anra:
                conversations.append({
                    "human": current_h,
                    "anra": " ".join(current_anra).strip()
                })
            current_h = line[2:].strip()
            current_anra = []
        elif line.startswith("ANRA:"):
            current_anra = [line[5:].strip()]
        elif current_anra and line and not line.startswith("#"):
            current_anra.append(line)

    if current_h and current_anra:
        conversations.append({
            "human": current_h,
            "anra": " ".join(current_anra).strip()
        })

    print(f"✅ Parsed {len(conversations)} high-density dataset exchanges.")
    return conversations


def format_prompt(example):
    return {
        "text": f"<|user|>\n{example['human']}\n<|assistant|>\n{example['anra']}<|end|>"
    }


def load_model_and_tokenizer():
    print(f"📥 Loading scalable base model: {BASE_MODEL} (BF16={BF16_ENABLED})")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16_ENABLED else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable Gradient Checkpointing for massive models
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    print("✅ Model loaded, gradient checkpointing active.")
    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "dense"], # Target all linear layers
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ High-Rank LoRA applied.")
    return model


def prepare_dataset(conversations, tokenizer):
    dataset = Dataset.from_list([format_prompt(c) for c in conversations])
    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize, batched=False)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")
    print(f"✅ Dataset prepared: {len(tokenized)} seq len={MAX_SEQ_LENGTH}.")
    return tokenized


def train(model, tokenizer, dataset):
    # Accelerated high-scale training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        bf16=BF16_ENABLED,
        fp16=not BF16_ENABLED,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=4,        # Multiprocessing for data loading
        gradient_checkpointing=True,     # Memory optimization
        ddp_find_unused_parameters=False # Multi-GPU optimization
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("🚀 Triggering High-Scale Multi-GPU Training Phase...")
    trainer.train()
    print("✅ Scale Training successful!")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model written to {OUTPUT_DIR}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        print(f"⚠ Dataset {DATASET_FILE} missing! Please ensure v3 is present.")
    else:
        convos = parse_exchanges(DATASET_FILE)
        mod, tok = load_model_and_tokenizer()
        mod = apply_lora(mod)
        ds = prepare_dataset(convos, tok)
        train(mod, tok, ds)

