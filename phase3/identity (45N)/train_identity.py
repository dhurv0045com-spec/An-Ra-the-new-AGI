"""
train_identity.py
LoRA Fine-tuning for An-Ra Identity v4 (Fluent)
Optimized for Google Colab T4 GPU.

Run (Colab):
    !pip install transformers peft datasets accelerate bitsandbytes torch
    !python train_identity.py
"""

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

USER_TAG  = "<|user|>"
ASST_TAG  = "<|assistant|>"
END_TAG   = "<|end|>"


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
                conversations.append({"human": current_h, "anra": " ".join(current_anra).strip()})
            current_h = line[2:].strip()
            current_anra = []
        elif line.startswith("ANRA:"):
            current_anra = [line[5:].strip()]
        elif current_anra and line and not line.startswith("#"):
            current_anra.append(line)
    if current_h and current_anra:
        conversations.append({"human": current_h, "anra": " ".join(current_anra).strip()})
    print(f"Parsed {len(conversations)} exchanges from dataset.")
    return conversations


def format_prompts(conversations):
    samples = []
    for c in conversations:
        # Format 1: standard chat
        t1 = USER_TAG + "\n" + c["human"] + "\n" + ASST_TAG + "\n" + c["anra"] + END_TAG
        samples.append({"text": t1})
        # Format 2: with system prompt
        t2 = "[SYSTEM] You are An-Ra, an autonomous intelligence built by Ankit from scratch.\n" + t1
        samples.append({"text": t2})
    return samples


def load_model_and_tokenizer():
    print(f"Loading base model: {BASE_MODEL}")
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
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

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
    print(f"Model saved to: {OUTPUT_DIR}")


def quick_test(model, tokenizer, prompt="Who are you?"):
    print(f"\nQuick test: {prompt}")
    full = USER_TAG + "\n" + prompt + "\n" + ASST_TAG + "\n"
    inputs = tokenizer(full, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = response.split(ASST_TAG)
    answer = parts[-1].strip() if len(parts) > 1 else response
    print(f"AN-RA: {answer}\n")


if __name__ == "__main__":
    conversations = parse_exchanges(DATASET_FILE)
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train_ds, val_ds = prepare_dataset(conversations, tokenizer)
    train(model, tokenizer, train_ds, val_ds)
    quick_test(model, tokenizer, "Who are you?")
    quick_test(model, tokenizer, "Write a Python function to reverse a linked list.")
    quick_test(model, tokenizer, "What makes a good engineer?")
    print("\nAn-Ra v4 identity training complete!")
