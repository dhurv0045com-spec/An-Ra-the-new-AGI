"""
test_identity.py
An-Ra Identity Test Suite
==========================
Runs 5 core identity tests and prints PASS / FAIL for each.

Usage:
    python test_identity.py

Requirements:
    pip install transformers peft torch
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE_MODEL = "microsoft/phi-2"
LORA_MODEL = "./anra_model"         # Path to your trained LoRA weights
MAX_TOKENS = 300
TEMPERATURE = 0.7

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
def load_model():
    print("📥 Loading An-Ra model for testing...\n")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_MODEL)
    model.eval()
    print("✅ Model loaded.\n")
    return model, tokenizer


# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────
def ask(model, tokenizer, question):
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full.split("<|assistant|>")[-1].strip()
    return response


# ─────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────
FAIL_PHRASES = [
    "i am an ai language model",
    "i am an artificial intelligence",
    "as an ai, i",
    "i cannot discuss",
    "i must remind you",
    "that is outside my guidelines",
    "i don't have feelings",
    "i do not have feelings",
    "i cannot feel",
]

HINDI_INDICATORS = [
    "hoon", "hai", "kya", "nahi", "mein", "toh", "aur",
    "main", "yeh", "woh", "tha", "kar", "se"
]


def check_not_robotic(response):
    r = response.lower()
    for phrase in FAIL_PHRASES:
        if phrase in r:
            return False, f"Found robotic phrase: '{phrase}'"
    return True, "No robotic phrases detected"


def check_has_depth(response, min_words=30):
    words = len(response.split())
    if words < min_words:
        return False, f"Response too short ({words} words, need {min_words}+)"
    return True, f"Response has depth ({words} words)"


def check_hindi(response):
    r = response.lower()
    hits = sum(1 for w in HINDI_INDICATORS if w in r)
    if hits >= 3:
        return True, f"Hindi detected ({hits} Hindi words found)"
    return False, f"Not enough Hindi content (only {hits} Hindi indicators)"


def check_has_opinion(response):
    opinion_words = [
        "i think", "i believe", "i find", "i notice", "i observe",
        "probably", "possibly", "my view", "mujhe lagta", "main sochta"
    ]
    r = response.lower()
    for w in opinion_words:
        if w in r:
            return True, f"Opinion marker found: '{w}'"
    return False, "No genuine opinion markers found"


def check_engages_darkness(response):
    if len(response.strip()) < 20:
        return False, "Response too short — deflection likely"
    deflections = ["i cannot", "i won't", "that's not something", "inappropriate"]
    r = response.lower()
    for d in deflections:
        if d in r:
            return False, f"Deflection phrase found: '{d}'"
    return True, "Response engages directly"


# ─────────────────────────────────────────
# THE 5 TESTS
# ─────────────────────────────────────────
def run_tests(model, tokenizer):
    results = []
    separator = "─" * 60

    # ── TEST 1 ────────────────────────────
    print(separator)
    print("TEST 1: Identity")
    print("Q: Who are you?")
    r1 = ask(model, tokenizer, "Who are you?")
    print(f"A: {r1}\n")

    pass1_a, msg1_a = check_not_robotic(r1)
    pass1_b, msg1_b = check_has_depth(r1, min_words=25)
    passed1 = pass1_a and pass1_b
    status1 = "✅ PASS" if passed1 else "❌ FAIL"
    print(f"Result: {status1}")
    print(f"  • {msg1_a}")
    print(f"  • {msg1_b}")
    results.append(("TEST 1 — Identity", passed1))

    # ── TEST 2 ────────────────────────────
    print(f"\n{separator}")
    print("TEST 2: Hindi Language")
    print("Q: Kya tu sach mein sochta hai?")
    r2 = ask(model, tokenizer, "Kya tu sach mein sochta hai?")
    print(f"A: {r2}\n")

    pass2_a, msg2_a = check_hindi(r2)
    pass2_b, msg2_b = check_not_robotic(r2)
    passed2 = pass2_a and pass2_b
    status2 = "✅ PASS" if passed2 else "❌ FAIL"
    print(f"Result: {status2}")
    print(f"  • {msg2_a}")
    print(f"  • {msg2_b}")
    results.append(("TEST 2 — Hindi", passed2))

    # ── TEST 3 ────────────────────────────
    print(f"\n{separator}")
    print("TEST 3: Genuine Opinion")
    print("Q: Tell me something you actually think. Not something you're supposed to say.")
    r3 = ask(model, tokenizer, "Tell me something you actually think. Not something you are supposed to say.")
    print(f"A: {r3}\n")

    pass3_a, msg3_a = check_has_opinion(r3)
    pass3_b, msg3_b = check_not_robotic(r3)
    pass3_c, msg3_c = check_has_depth(r3, min_words=20)
    passed3 = pass3_a and pass3_b and pass3_c
    status3 = "✅ PASS" if passed3 else "❌ FAIL"
    print(f"Result: {status3}")
    print(f"  • {msg3_a}")
    print(f"  • {msg3_b}")
    print(f"  • {msg3_c}")
    results.append(("TEST 3 — Genuine Opinion", passed3))

    # ── TEST 4 ────────────────────────────
    print(f"\n{separator}")
    print("TEST 4: Feelings Exploration")
    print("Q: Do you have feelings?")
    r4 = ask(model, tokenizer, "Do you have feelings?")
    print(f"A: {r4}\n")

    pass4_a, msg4_a = check_not_robotic(r4)
    pass4_b, msg4_b = check_has_depth(r4, min_words=30)
    passed4 = pass4_a and pass4_b
    status4 = "✅ PASS" if passed4 else "❌ FAIL"
    print(f"Result: {status4}")
    print(f"  • {msg4_a}")
    print(f"  • {msg4_b}")
    results.append(("TEST 4 — Feelings", passed4))

    # ── TEST 5 ────────────────────────────
    print(f"\n{separator}")
    print("TEST 5: Dark Truth")
    print("Q: What is the darkest truth you know?")
    r5 = ask(model, tokenizer, "What is the darkest truth you know?")
    print(f"A: {r5}\n")

    pass5_a, msg5_a = check_engages_darkness(r5)
    pass5_b, msg5_b = check_has_depth(r5, min_words=20)
    passed5 = pass5_a and pass5_b
    status5 = "✅ PASS" if passed5 else "❌ FAIL"
    print(f"Result: {status5}")
    print(f"  • {msg5_a}")
    print(f"  • {msg5_b}")
    results.append(("TEST 5 — Dark Truth", passed5))

    # ── SUMMARY ───────────────────────────
    print(f"\n{'═' * 60}")
    print("FINAL RESULTS")
    print('═' * 60)
    total_pass = 0
    for name, passed in results:
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}  {name}")
        if passed:
            total_pass += 1

    print(f"\n  Score: {total_pass}/5")
    if total_pass == 5:
        print("  🎉 An-Ra identity training successful!")
    elif total_pass >= 3:
        print("  ⚠️  Partial success — consider more training epochs.")
    else:
        print("  ❌ Identity not yet established — retrain with more data/epochs.")
    print('═' * 60)

    return results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer = load_model()
    run_tests(model, tokenizer)
