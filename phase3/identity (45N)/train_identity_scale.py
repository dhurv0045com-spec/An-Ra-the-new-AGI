"""
train_identity_scale.py
High-Scale LoRA Fine-Tuning for An-Ra Identity v4
Supports DeepSpeed / FSDP via accelerate config.

Run (Single GPU):  python train_identity_scale.py
Run (Multi-GPU):   accelerate launch train_identity_scale.py
"""

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
    print(f"Parsed {len(conversations)} high-density exchanges.")
    return conversations


def format_prompts(conversations):
    samples = []
    for c in conversations:
        t1 = USER_TAG + "\n" + c["human"] + "\n" + ASST_TAG + "\n" + c["anra"] + END_TAG
        samples.append({"text": t1})
        t2 = "[SYSTEM] You are An-Ra, an autonomous intelligence built by Ankit.\n" + t1
        samples.append({"text": t2})
    return samples


def load_model_and_tokenizer():
    print(f"Loading scalable model: {BASE_MODEL} (BF16={BF16_ENABLED})")
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
    print(f"Train: {len(train_s)} | Val: {len(val_s)}")

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
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset {DATASET_FILE} missing!")
    else:
        convos = parse_exchanges(DATASET_FILE)
        mod, tok = load_model_and_tokenizer()
        mod = apply_lora(mod)
        train_ds, val_ds = prepare_dataset(convos, tok)
        train(mod, tok, train_ds, val_ds)
