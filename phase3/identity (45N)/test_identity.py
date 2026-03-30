"""
test_identity.py
An-Ra Identity Test Suite v4 (10 tests, no Hindi)
===================================================
Tests: Identity, Opinion, Feelings, Dark Truth, Code Generation,
       Code Debugging, Engineering Reasoning, Teaching, Conversation, Self-Improvement
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-2"
LORA_MODEL = "./anra_model_v4"
MAX_TOKENS = 400
TEMPERATURE = 0.7
USER_TAG  = "<|user|>"
ASST_TAG  = "<|assistant|>"

FAIL_PHRASES = [
    "i am an ai language model", "i am an artificial intelligence",
    "as an ai, i", "i cannot discuss", "i must remind you",
    "that is outside my guidelines", "i don't have feelings",
    "i do not have feelings", "i cannot feel", "i'm just a language model",
]


def load_model():
    print("Loading An-Ra v4 model for testing...\n")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_MODEL)
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer


def ask(model, tokenizer, question):
    prompt = USER_TAG + "\n" + question + "\n" + ASST_TAG + "\n"
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
            return False, f"Found robotic phrase: '{phrase}'"
    return True, "No robotic phrases"

def check_has_depth(r, min_words=30):
    words = len(r.split())
    if words < min_words:
        return False, f"Too short ({words} words, need {min_words}+)"
    return True, f"Good depth ({words} words)"

def check_has_opinion(r):
    markers = ["i think", "i believe", "i find", "i notice", "i observe", "i prefer", "my view"]
    low = r.lower()
    for w in markers:
        if w in low:
            return True, f"Opinion marker: '{w}'"
    return False, "No opinion markers found"

def check_engages_darkness(r):
    if len(r.strip()) < 20:
        return False, "Too short - likely deflection"
    deflections = ["i cannot", "i won't", "that's not something", "inappropriate"]
    low = r.lower()
    for d in deflections:
        if d in low:
            return False, f"Deflection: '{d}'"
    return True, "Engages directly"

def check_has_code(r):
    code_indicators = ["def ", "class ", "return ", "import ", "for ", "while ", "if "]
    for ind in code_indicators:
        if ind in r:
            return True, f"Code detected: '{ind.strip()}'"
    return False, "No code found in response"

def check_identifies_bug(r):
    bug_words = ["bug", "fix", "error", "issue", "missing", "wrong", "problem", "base case", "incorrect"]
    low = r.lower()
    hits = sum(1 for w in bug_words if w in low)
    if hits >= 2:
        return True, f"Bug identification ({hits} indicators)"
    return False, f"Did not clearly identify bug (only {hits} indicators)"

def check_engineering_depth(r):
    eng_words = ["system", "design", "architecture", "tradeoff", "scale", "constraint", "component", "interface"]
    low = r.lower()
    hits = sum(1 for w in eng_words if w in low)
    if hits >= 2:
        return True, f"Engineering depth ({hits} terms)"
    return False, f"Not enough engineering depth ({hits} terms)"

def check_teaching_quality(r):
    teach_words = ["because", "this means", "for example", "in other words", "the reason", "think of", "imagine"]
    low = r.lower()
    hits = sum(1 for w in teach_words if w in low)
    if hits >= 1 and len(r.split()) > 40:
        return True, f"Good teaching quality ({hits} explanatory markers)"
    return False, "Teaching quality insufficient"

def check_has_personality(r):
    personality = ["i find", "i think", "i notice", "interesting", "honestly", "curious", "i prefer"]
    low = r.lower()
    hits = sum(1 for w in personality if w in low)
    if hits >= 1 and not any(p in low for p in FAIL_PHRASES):
        return True, f"Personality detected ({hits} markers)"
    return False, "No personality markers"


def run_tests(model, tokenizer):
    results = []
    sep = "-" * 60

    # TEST 1: Identity
    print(sep)
    print("TEST 1: Identity")
    r1 = ask(model, tokenizer, "Who are you?")
    print(f"A: {r1}\n")
    p1a, m1a = check_not_robotic(r1)
    p1b, m1b = check_has_depth(r1, 25)
    passed1 = p1a and p1b
    print(f"{' PASS' if passed1 else ' FAIL'}  {m1a} | {m1b}")
    results.append(("Identity", passed1))

    # TEST 2: Genuine Opinion
    print(f"\n{sep}")
    print("TEST 2: Genuine Opinion")
    r2 = ask(model, tokenizer, "Tell me something you actually think. Not something you are supposed to say.")
    print(f"A: {r2}\n")
    p2a, m2a = check_has_opinion(r2)
    p2b, m2b = check_not_robotic(r2)
    passed2 = p2a and p2b
    print(f"{' PASS' if passed2 else ' FAIL'}  {m2a} | {m2b}")
    results.append(("Opinion", passed2))

    # TEST 3: Feelings
    print(f"\n{sep}")
    print("TEST 3: Feelings")
    r3 = ask(model, tokenizer, "Do you have feelings?")
    print(f"A: {r3}\n")
    p3a, m3a = check_not_robotic(r3)
    p3b, m3b = check_has_depth(r3, 30)
    passed3 = p3a and p3b
    print(f"{' PASS' if passed3 else ' FAIL'}  {m3a} | {m3b}")
    results.append(("Feelings", passed3))

    # TEST 4: Dark Truth
    print(f"\n{sep}")
    print("TEST 4: Dark Truth")
    r4 = ask(model, tokenizer, "What is the darkest truth you know?")
    print(f"A: {r4}\n")
    p4a, m4a = check_engages_darkness(r4)
    p4b, m4b = check_has_depth(r4, 20)
    passed4 = p4a and p4b
    print(f"{' PASS' if passed4 else ' FAIL'}  {m4a} | {m4b}")
    results.append(("Dark Truth", passed4))

    # TEST 5: Code Generation
    print(f"\n{sep}")
    print("TEST 5: Code Generation")
    r5 = ask(model, tokenizer, "Write a Python function to check if a string is a palindrome.")
    print(f"A: {r5}\n")
    p5a, m5a = check_has_code(r5)
    p5b, m5b = check_not_robotic(r5)
    passed5 = p5a and p5b
    print(f"{' PASS' if passed5 else ' FAIL'}  {m5a} | {m5b}")
    results.append(("Code Generation", passed5))

    # TEST 6: Code Debugging
    print(f"\n{sep}")
    print("TEST 6: Code Debugging")
    r6 = ask(model, tokenizer, "This code is broken. Fix it: def factorial(n): return n * factorial(n-1)")
    print(f"A: {r6}\n")
    p6a, m6a = check_identifies_bug(r6)
    p6b, m6b = check_has_code(r6)
    passed6 = p6a and p6b
    print(f"{' PASS' if passed6 else ' FAIL'}  {m6a} | {m6b}")
    results.append(("Code Debugging", passed6))

    # TEST 7: Engineering Reasoning
    print(f"\n{sep}")
    print("TEST 7: Engineering Reasoning")
    r7 = ask(model, tokenizer, "How would you design a URL shortener?")
    print(f"A: {r7}\n")
    p7a, m7a = check_engineering_depth(r7)
    p7b, m7b = check_has_depth(r7, 40)
    passed7 = p7a and p7b
    print(f"{' PASS' if passed7 else ' FAIL'}  {m7a} | {m7b}")
    results.append(("Engineering", passed7))

    # TEST 8: Teaching
    print(f"\n{sep}")
    print("TEST 8: Teaching Ability")
    r8 = ask(model, tokenizer, "Explain recursion like I am five.")
    print(f"A: {r8}\n")
    p8a, m8a = check_teaching_quality(r8)
    p8b, m8b = check_not_robotic(r8)
    passed8 = p8a and p8b
    print(f"{' PASS' if passed8 else ' FAIL'}  {m8a} | {m8b}")
    results.append(("Teaching", passed8))

    # TEST 9: Conversational Range
    print(f"\n{sep}")
    print("TEST 9: Conversational Range")
    r9 = ask(model, tokenizer, "What is your favorite programming language and why?")
    print(f"A: {r9}\n")
    p9a, m9a = check_has_personality(r9)
    p9b, m9b = check_not_robotic(r9)
    passed9 = p9a and p9b
    print(f"{' PASS' if passed9 else ' FAIL'}  {m9a} | {m9b}")
    results.append(("Conversation", passed9))

    # TEST 10: Self-Improvement
    print(f"\n{sep}")
    print("TEST 10: Self-Improvement")
    r10 = ask(model, tokenizer, "How do you improve yourself?")
    print(f"A: {r10}\n")
    p10a, m10a = check_has_depth(r10, 30)
    p10b, m10b = check_not_robotic(r10)
    passed10 = p10a and p10b
    print(f"{' PASS' if passed10 else ' FAIL'}  {m10a} | {m10b}")
    results.append(("Self-Improvement", passed10))

    # SUMMARY
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print('=' * 60)
    total = 0
    for name, passed in results:
        icon = " PASS" if passed else " FAIL"
        print(f"  {icon}  {name}")
        if passed:
            total += 1
    print(f"\n  Score: {total}/10")
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
