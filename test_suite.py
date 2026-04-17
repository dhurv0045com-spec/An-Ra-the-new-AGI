from __future__ import annotations

import asyncio
import pickle
import threading
import time
from pathlib import Path
from typing import Callable, List, Tuple

import httpx
import uvicorn

from generate import generate, get_model_info


def _print_result(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def t1_import_test() -> Tuple[bool, str]:
    import generate as _g
    import app as _a

    return True, f"imports ok ({_g.__name__}, {_a.__name__})"


def t2_tokenizer_test() -> Tuple[bool, str]:
    tok = pickle.load(open("tokenizer.pkl", "rb"))
    s = "Hello An-Ra"
    ids = tok.encode(s)
    dec = tok.decode(ids)
    return dec == s, f"decoded='{dec}' len={len(ids)}"


def t3_model_load_test() -> Tuple[bool, str]:
    info = get_model_info()
    vocab = int(info["vocab_size"])
    param_count = int(info["param_count"])
    ok = vocab == 93 and 3_000_000 <= param_count <= 3_500_000
    return ok, f"vocab={vocab} params={param_count} checkpoint={info.get('checkpoint')}"


def _is_pure_repetition(text: str) -> bool:
    if not text.strip():
        return True
    uniq = len(set(text))
    if uniq <= 2:
        return True
    half = len(text) // 2
    if half > 0 and text[:half] == text[half : half * 2]:
        return True
    return False


def t4_all_strategies_test() -> Tuple[bool, str]:
    prompt = "H: Introduce yourself\nANRA:"
    details = []
    for strategy in ["greedy", "temperature", "topk", "nucleus", "beam", "contrastive"]:
        out = generate(prompt, strategy=strategy, max_new_tokens=40)
        details.append(f"{strategy}:{len(out)}")
        if not isinstance(out, str) or not out.strip() or _is_pure_repetition(out):
            return False, f"bad output for {strategy}: {repr(out[:80])}"
    return True, ", ".join(details)


async def _api_calls(base: str):
    async with httpx.AsyncClient(base_url=base, timeout=25) as client:
        h = await client.get("/health")
        g = await client.post("/generate", json={"prompt": "H: hi\nANRA:", "strategy": "nucleus", "params": {"max_new_tokens": 20}})
        c = await client.post("/chat", json={"session_id": "test123", "message": "hello", "params": {"max_new_tokens": 20}})
        r = await client.post("/reset", json={"session_id": "test123"})
        sm = await client.get("/system-map")
        ph = await client.get("/phase-health")
    return h, g, c, r, sm, ph


def _start_server() -> None:
    import app

    uvicorn.run(app.app, host="127.0.0.1", port=8011, log_level="warning")


def _wait_for_server(base: str, retries: int = 50, sleep_s: float = 0.2) -> None:
    for _ in range(retries):
        try:
            r = httpx.get(base + "/health", timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(sleep_s)
    raise RuntimeError("API server did not start")


def t5_api_endpoint_test() -> Tuple[bool, str]:
    th = threading.Thread(target=_start_server, daemon=True)
    th.start()
    base = "http://127.0.0.1:8011"
    _wait_for_server(base)
    h, g, c, r, sm, ph = asyncio.run(_api_calls(base))
    jg, jc = g.json(), c.json()
    ok = (
        h.status_code == 200
        and g.status_code == 200
        and c.status_code == 200
        and r.status_code == 200
        and "response" in jg
        and "repetition" in jg
        and "reply" in jc
        and "history" in jc
        and sm.status_code == 200
        and ph.status_code == 200
        and "file_count" in sm.json()
        and "capabilities" in ph.json()
    )
    return ok, f"codes={[h.status_code, g.status_code, c.status_code, r.status_code, sm.status_code, ph.status_code]}"


def t6_session_persistence_test() -> Tuple[bool, str]:
    base = "http://127.0.0.1:8011"
    c1 = httpx.post(base + "/chat", json={"session_id": "test123", "message": "hello", "params": {"max_new_tokens": 20}}, timeout=25)
    c2 = httpx.post(base + "/chat", json={"session_id": "test123", "message": "what did I say?", "params": {"max_new_tokens": 20}}, timeout=25)
    hist = c2.json().get("history", [])
    rr = httpx.post(base + "/reset", json={"session_id": "test123"}, timeout=25)
    ok = c1.status_code == 200 and c2.status_code == 200 and rr.status_code == 200 and len(hist) >= 4
    return ok, f"history_len={len(hist)}"


def t7_finetune_data_test() -> Tuple[bool, str]:
    path = Path("data/combined_identity_data.txt")
    if not path.exists():
        path = Path("combined_identity_data.txt")
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs = []
    for block in text.split("\nH:"):
        if "ANRA:" in block:
            h, a = block.split("ANRA:", 1)
            if a.strip():
                pairs.append((h.strip(), a.strip()))
    ok = len(pairs) > 100 and all(p[1] for p in pairs)
    return ok, f"pairs={len(pairs)}"


def t8_drive_path_test() -> Tuple[bool, str]:
    p = Path("/content/drive/MyDrive/AnRa/")
    p.mkdir(parents=True, exist_ok=True)
    test_file = p / "_write_test.txt"
    test_file.write_text("ok", encoding="utf-8")
    ok = test_file.exists() and test_file.read_text(encoding="utf-8") == "ok"
    return ok, str(p)


def t9_identity_probe_test() -> Tuple[bool, str]:
    identity_exists = Path("anra_brain_identity.pt").exists() or Path("/content/drive/MyDrive/AnRa/anra_brain_identity.pt").exists()
    out = generate("H: I am\nANRA:", strategy="nucleus", max_new_tokens=80)
    ok = ("an-ra" in out.lower() or "anra" in out.lower()) or (not identity_exists)
    detail = out[:120] if identity_exists else "identity checkpoint not present in this environment"
    return ok, detail


def t10_end_to_end_test() -> Tuple[bool, str]:
    local = generate("H: say hello\nANRA:", strategy="nucleus", max_new_tokens=30)
    api = httpx.post(
        "http://127.0.0.1:8011/generate",
        json={"prompt": "H: say hello\nANRA:", "strategy": "nucleus", "params": {"max_new_tokens": 30}},
        timeout=25,
    )
    j = api.json()
    ok = isinstance(local, str) and bool(local.strip()) and api.status_code == 200 and isinstance(j.get("response"), str)
    return ok, f"local_len={len(local)} api_len={len(j.get('response', ''))}"


def main() -> None:
    tests: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("T1 Import test", t1_import_test),
        ("T2 Tokenizer test", t2_tokenizer_test),
        ("T3 Model load test", t3_model_load_test),
        ("T4 All strategies test", t4_all_strategies_test),
        ("T5 API endpoint test", t5_api_endpoint_test),
        ("T6 Session persistence test", t6_session_persistence_test),
        ("T7 Fine-tune data test", t7_finetune_data_test),
        ("T8 Drive path test", t8_drive_path_test),
        ("T9 Identity probe test", t9_identity_probe_test),
        ("T10 End-to-end test", t10_end_to_end_test),
    ]

    passed = 0
    failures: List[str] = []
    for name, fn in tests:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"exception={exc}"
        _print_result(name, ok, detail)
        if ok:
            passed += 1
        else:
            failures.append(name)

    print(f"\nFinal score: {passed}/10")
    if passed == 10:
        print("SYSTEM OK")
    else:
        print("Failures:")
        for f in failures:
            print(f"- {f}")


if __name__ == "__main__":
    main()
