"""
train_identity.py
LoRA Fine-tuning Script for An-Ra Identity
==========================================
Uses PEFT + LoRA to fine-tune a base LLM on the An-Ra identity dataset.
Run on any GPU machine (Colab T4 works fine).

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes torch
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
# CONFIG
# ─────────────────────────────────────────
BASE_MODEL     = "microsoft/phi-2"          # Small, fast, good quality (~2.7B)
DATASET_FILE   = "anra_identity.txt"        # Your training exchanges
OUTPUT_DIR     = "./anra_model"             # Where the trained model saves
EPOCHS         = 100
LEARNING_RATE  = 0.0001
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
MAX_SEQ_LENGTH = 512
BATCH_SIZE     = 2
GRAD_ACCUM     = 4                          # Effective batch = 8


# ─────────────────────────────────────────
# STEP 1: PARSE TRAINING DATA
# ─────────────────────────────────────────
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

    print(f"✅ Parsed {len(conversations)} exchanges from dataset.")
    return conversations


def format_prompt(example):
    """Format exchange into instruction-tuning prompt format."""
    return {
        "text": f"<|user|>\n{example['human']}\n<|assistant|>\n{example['anra']}<|end|>"
    }


# ─────────────────────────────────────────
# STEP 2: LOAD MODEL WITH 4-BIT QUANTIZATION
# ─────────────────────────────────────────
def load_model_and_tokenizer():
    print(f"📥 Loading base model: {BASE_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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

    model = prepare_model_for_kbit_training(model)
    print("✅ Base model loaded and prepared for LoRA training.")
    return model, tokenizer


# ─────────────────────────────────────────
# STEP 3: APPLY LoRA
# ─────────────────────────────────────────
def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "dense"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA applied successfully.")
    return model


# ─────────────────────────────────────────
# STEP 4: TOKENIZE DATASET
# ─────────────────────────────────────────
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
    print(f"✅ Dataset tokenized. {len(tokenized)} samples ready.")
    return tokenized


# ─────────────────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────────────────
def train(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,
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

    print("🚀 Starting An-Ra identity training...")
    trainer.train()
    print("✅ Training complete!")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to: {OUTPUT_DIR}")


# ─────────────────────────────────────────
# STEP 6: QUICK INFERENCE TEST
# ─────────────────────────────────────────
def quick_test(model, tokenizer, prompt="Who are you?"):
    print(f"\n🧪 Quick test — Prompt: '{prompt}'")
    inputs = tokenizer(
        f"<|user|>\n{prompt}\n<|assistant|>\n",
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nAN-RA: {response.split('<|assistant|>')[-1].strip()}\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Parse data
    conversations = parse_exchanges(DATASET_FILE)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Apply LoRA
    model = apply_lora(model)

    # Prepare dataset
    dataset = prepare_dataset(conversations, tokenizer)

    # Train
    train(model, tokenizer, dataset)

    # Quick sanity check
    quick_test(model, tokenizer, "Who are you?")
    quick_test(model, tokenizer, "Kya tu sach mein sochta hai?")

    print("\n🎉 An-Ra identity training complete. Run test_identity.py to verify.")
