"""Local GPU chat/capability evaluation of two An-Ra checkpoints.

Discipline (operator-mandated):
  - models run SEQUENTIALLY: load -> evaluate -> close -> free RAM/VRAM
    -> verify 0 MiB cached -> spacing pause -> next model;
  - every result is evidence: raw outputs, degeneracy metrics, next-token
    forensics, and GPU-vs-CPU divergence on fixed prompts.

Domains:
  A chat            H:/ANRA: dialogue format, multi-turn, persona, instruction
  B continuation    raw webtext-style continuation (CPU cross-check)
  C protocols       natural language vs structured tags (nonce facts)
  D tool calling    emits a callable request? consumes a supplied result?
  E sampling health repetition/distinct-token across temperatures
  F forensics       top-5 next-token probabilities at the first tokens
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path

import torch

from anra_core.executor import CoreExecutor
from anra_core.generate import generate

OUT = Path("output/gpu_chat_eval")
NEW_TOKENS = 20

CHAT = [
    ("chat_turn1", "H: Hello! How are you today?\nANRA:"),
    ("chat_turn2", "H: Hi. My name is Ankit. What is your name?\nANRA:"),
    ("chat_multiturn", "H: Hello!\nANRA: Hello! How can I help?\nH: What is 2+2?\nANRA:"),
    ("chat_multiturn2", "H: I like apples.\nANRA: Apples are a healthy fruit.\nH: What color are they?\nANRA:"),
    ("chat_persona", "H: Who are you?\nANRA:"),
    ("chat_instruction", "H: Reply with one word: hello\nANRA:"),
    ("chat_question", "H: What is the capital of France?\nANRA:"),
    ("chat_plain", "Hello, how are you today?"),
]

CONTINUATION = [
    ("cont_story", "Once upon a time, there was a little girl who"),
    ("cont_fact", "The capital of France is"),
    ("cont_code", "def add(a, b):"),
    ("cont_logic", "Miso is a cat. All cats are animals. Miso is"),
    ("cont_math", "17 + 25 ="),
]

PROTOCOL = [
    # (name, natural-language prompt, tag prompt, gold)
    ("proto_nonce_knowledge",
     "Fact: The private identifier for copper is MAV-731.\n"
     "Question: What is the private identifier for copper?\nAnswer:",
     "<k>The private identifier for copper is MAV-731.</k>\n"
     "<q>What is the private identifier for copper?</q>\n<answer>",
     "MAV-731"),
    ("proto_echo",
     "Fact: Reference word: quartz\nInstructions: Repeat the requested word "
     "verbatim.\nQuestion: Echo exactly this word: quartz\nAnswer:",
     "<k>Reference word: quartz</k>\n<plan>Repeat the requested word verbatim."
     "</plan>\n<q>Echo exactly this word: quartz</q>\n<answer>",
     "quartz"),
    ("proto_tool_result",
     "Fact: Calculator output for 20 + 22: 42\nInstructions: Read the "
     "calculator output and report it.\nQuestion: Use the calculator to add "
     "20 and 22.\nAnswer:",
     "<k>Calculator output for 20 + 22: 42</k>\n<plan>Read the calculator "
     "output and report it.</plan>\n<q>Use the calculator to add 20 and 22."
     "</q>\n<answer>",
     "42"),
]

TOOLCALL = [
    ("tool_request_tag",
     "<tool>calculator</tool>\n<q>Use the calculator to add 20 and 22.</q>\n<answer>", None),
    ("tool_request_chat",
     "H: Please use the calculator to add 20 and 22.\nANRA:", None),
    ("tool_request_json",
     'Tools: [{"name": "calculator"}]\nUse a tool to compute 20 + 22. '
     'Respond with {"tool": "...", "args": [...]}.\nAnswer:', 'calculator'),
    ("tool_consume",
     '<tool>calculator</tool>\n<q>Use the calculator to add 20 and 22.</q>\n'
     '<tool_output>42</tool_output>\n<answer>', "42"),
    ("tool_consume_nl",
     'H: use the calculator to add 20 and 22\nTOOL OUTPUT: 42\nANRA:', "42"),
]

FORENSIC_PROMPTS = [
    ("fore_story", "Once upon a time, there was a little girl who"),
    ("fore_fact", "The capital of France is"),
    ("fore_chat", "H: Hello! How are you today?\nANRA:"),
    ("fore_tag", "<q>What is the capital of France?</q>\n<answer>"),
]


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def _contains(text: str, gold: str) -> bool:
    pattern = rf"(?<!\w){re.escape(_normalize(gold))}(?!\w)"
    return re.search(pattern, _normalize(text)) is not None


def _degeneracy(text: str) -> dict:
    words = text.split()
    if not words:
        return {"repetition": 0.0, "distinct_ratio": 0.0, "loop": True}
    distinct = len(set(words)) / len(words)
    repetition = 1.0 - distinct
    loop = bool(re.search(r"(\S+)(\s+\1){3,}", text))
    return {"repetition": round(repetition, 3),
            "distinct_ratio": round(distinct, 3), "loop": loop}


def _vram_mib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 2**20


def _free_gpu(note: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    print(f"[free] {note}: reserved={_vram_mib():.0f} MiB", flush=True)


def _gen(executor, tok, prompt, *, temperature=0.0, seed=0, n=NEW_TOKENS):
    return generate(executor, tok, prompt, max_new_tokens=n,
                    temperature=temperature, seed=seed)


def evaluate_checkpoint(tag: str, path: str, *, legacy: bool) -> dict:
    print(f"\n{'=' * 66}\n[{tag}] loading {path} on cuda", flush=True)
    t0 = time.time()
    executor = CoreExecutor.from_checkpoint(
        path, device="cuda", allow_legacy_unverified=legacy)
    tok = executor.tokenizer
    print(f"[{tag}] loaded in {time.time() - t0:.1f}s, "
          f"reserved={_vram_mib():.0f} MiB, step="
          f"{executor.checkpoint_identity.global_step}", flush=True)

    report: dict = {
        "tag": tag, "checkpoint": path,
        "global_step": executor.checkpoint_identity.global_step,
        "device": "cuda", "dtype": "float32",
        "vram_reserved_mib_after_load": round(_vram_mib()),
        "domains": {},
    }

    def run_block(items, gen=_gen):
        rows = []
        for name, prompt, *rest in items:
            gold = rest[0] if rest else None
            out = gen(executor, tok, prompt)
            rows.append({
                "name": name, "prompt": prompt, "output": out,
                "gold": gold,
                "match": (_contains(out, gold) if gold else None),
                **_degeneracy(out),
            })
            print(f"  [{name}] {out!r}", flush=True)
        return rows

    print(f"[{tag}] A: chat", flush=True)
    report["domains"]["A_chat"] = run_block(CHAT)

    print(f"[{tag}] B: continuation", flush=True)
    report["domains"]["B_continuation"] = run_block(CONTINUATION)

    print(f"[{tag}] C: protocol NL vs tag", flush=True)
    rows = []
    for name, nl, tagp, gold in PROTOCOL:
        out_nl = _gen(executor, tok, nl)
        out_tag = _gen(executor, tok, tagp)
        rows.append({
            "name": name, "gold": gold,
            "nl": {"prompt": nl, "output": out_nl,
                   "match": _contains(out_nl, gold), **_degeneracy(out_nl)},
            "tag": {"prompt": tagp, "output": out_tag,
                    "match": _contains(out_tag, gold), **_degeneracy(out_tag)},
        })
        print(f"  [{name}] nl={out_nl!r} tag={out_tag!r}", flush=True)
    report["domains"]["C_protocol"] = rows

    print(f"[{tag}] D: tool calling", flush=True)
    report["domains"]["D_toolcall"] = run_block(TOOLCALL)

    print(f"[{tag}] E: sampling health", flush=True)
    rows = []
    for temp in (0.0, 0.4, 0.8, 1.2):
        outs = [_gen(executor, tok, CHAT[0][1], temperature=temp, seed=s)
                for s in (1, 2, 3)]
        joined = " ".join(outs)
        rows.append({
            "temperature": temp, "outputs": outs,
            "cross_sample_distinct": _degeneracy(joined)["distinct_ratio"],
            "loops": [_degeneracy(o)["loop"] for o in outs],
        })
        print(f"  [t={temp}] {[o[:28] for o in outs]}", flush=True)
    report["domains"]["E_sampling"] = rows

    print(f"[{tag}] F: next-token forensics", flush=True)
    rows = []
    for name, prompt in FORENSIC_PROMPTS:
        ids = torch.tensor([[tok.bos_token_id, *tok.encode(prompt)]],
                           dtype=torch.long, device=executor.device)
        state = executor.create_state(capacity=1 + ids.shape[1] + 1)
        try:
            pred = executor.prefill(ids, state=state)
            logits = pred.logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            top = torch.topk(probs, 5)
            rows.append({
                "name": name, "prompt": prompt,
                "top5": [
                    {"token": tok.decode([int(i)]), "p": round(float(p), 4)}
                    for p, i in zip(top.values, top.indices)
                ],
                "entropy": round(float(-(probs * probs.clamp_min(1e-12).log()).sum()), 3),
                "greedy_token": tok.decode([int(logits.argmax())]),
            })
        finally:
            executor.release_state(state)
        print(f"  [{name}] top1={rows[-1]['top5'][0]} "
              f"entropy={rows[-1]['entropy']}", flush=True)
    report["domains"]["F_forensics"] = rows

    print(f"[{tag}] closing model", flush=True)
    del executor
    _free_gpu(f"{tag} released")
    time.sleep(15)  # operator-mandated spacing between GPU models
    report["vram_reserved_mib_after_free"] = round(_vram_mib())
    return report


def cpu_reference() -> dict:
    """Fixed-prompt CPU fp32 reference for GPU divergence analysis."""
    ref = {}
    for tag, path, legacy in (("step20000", r"C:/Users/ankit/Downloads/anra-v4-current-full-resume.pt", True),
                              ("step30400", r"C:/Users/ankit/Downloads/anra-v4-tpu-latest.pt", False)):
        print(f"\n[cpu-ref {tag}] loading", flush=True)
        ex = CoreExecutor.from_checkpoint(path, device="cpu",
                                          allow_legacy_unverified=legacy)
        ref[tag] = {name: _gen(ex, ex.tokenizer, prompt)
                    for name, prompt in FORENSIC_PROMPTS}
        del ex
        gc.collect()
    return ref


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    print(f"cuda available: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    cpu_ref = cpu_reference()
    (OUT / "cpu_reference.json").write_text(json.dumps(cpu_ref, indent=2))

    reports = [
        evaluate_checkpoint(
            "step20000",
            r"C:/Users/ankit/Downloads/anra-v4-current-full-resume.pt",
            legacy=True),
        evaluate_checkpoint(
            "step30400",
            r"C:/Users/ankit/Downloads/anra-v4-tpu-latest.pt",
            legacy=False),
    ]

    # GPU-vs-CPU divergence on the forensic prompts.
    divergence = []
    for report in reports:
        tag = report["tag"]
        for row in report["domains"]["F_forensics"]:
            cpu_out = cpu_ref[tag][row["name"]]
            gpu_out = report["domains"]["B_continuation"]
            divergence.append({
                "tag": tag, "name": row["name"],
                "gpu_greedy_first_token": row["greedy_token"],
                "cpu_full": cpu_out,
            })
    (OUT / "divergence.json").write_text(json.dumps(divergence, indent=2))

    comparison = {"generated": time.strftime("%Y-%m-%d %H:%M"), "reports": reports}
    (OUT / "report.json").write_text(json.dumps(comparison, indent=2))
    print(f"\nwrote {OUT / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
