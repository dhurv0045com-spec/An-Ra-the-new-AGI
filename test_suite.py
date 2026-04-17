from __future__ import annotations

import asyncio
import json
import pickle
import threading
import time
from pathlib import Path
from typing import Callable, List, Tuple

import httpx
import uvicorn

from generate import GenerationConfig, detect_repetition, generate, generate_traced, get_model_info

BASE = "http://127.0.0.1:8011"


def _run(name: str, fn: Callable[[], Tuple[bool, str]]):
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"exception={exc}"
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({ms}ms)")
    return ok, detail


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
    for strategy in ["greedy", "temperature", "topk", "nucleus", "beam", "contrastive"]:
        out = generate(prompt, strategy=strategy, max_tokens=40)
        if not isinstance(out, str) or not out.strip() or _is_pure_repetition(out):
            return False, f"bad output for {strategy}: {repr(out[:80])}"
    return True, "all strategies produced output"


async def _api_calls(base: str):
    async with httpx.AsyncClient(base_url=base, timeout=25) as client:
        h = await client.get("/health")
        g = await client.post("/generate", json={"prompt": "H: hi\nANRA:", "strategy": "nucleus", "params": {"max_tokens": 20}})
        c = await client.post("/chat", json={"session_id": "test123", "message": "hello", "params": {"max_tokens": 20}})
        r = await client.post("/reset", json={"session_id": "test123"})
        sm = await client.get("/system-map")
        ph = await client.get("/phase-health")
    return h, g, c, r, sm, ph


def _start_server() -> None:
    import app

    uvicorn.run(app.app, host="127.0.0.1", port=8011, log_level="warning")


def _wait_for_server(base: str, retries: int = 100, sleep_s: float = 0.2) -> None:
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
    _wait_for_server(BASE)
    h, g, c, r, sm, ph = asyncio.run(_api_calls(BASE))
    ok = all(x.status_code == 200 for x in [h, g, c, r, sm, ph])
    return ok, "baseline endpoints healthy"


def t6_session_persistence_test() -> Tuple[bool, str]:
    c1 = httpx.post(BASE + "/chat", json={"session_id": "test123", "message": "hello", "params": {"max_tokens": 20}}, timeout=25)
    c2 = httpx.post(BASE + "/chat", json={"session_id": "test123", "message": "what did I say?", "params": {"max_tokens": 20}}, timeout=25)
    hist = c2.json().get("history", [])
    rr = httpx.post(BASE + "/reset", json={"session_id": "test123"}, timeout=25)
    return c1.status_code == 200 and c2.status_code == 200 and rr.status_code == 200 and len(hist) >= 4, f"history_len={len(hist)}"


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
    return test_file.exists() and test_file.read_text(encoding="utf-8") == "ok", str(p)


def t9_identity_probe_test() -> Tuple[bool, str]:
    out = generate("H: I am\nANRA:", strategy="nucleus", max_tokens=80)
    return isinstance(out, str) and len(out) > 0, out[:120]


def t10_end_to_end_test() -> Tuple[bool, str]:
    local = generate("H: say hello\nANRA:", strategy="nucleus", max_tokens=30)
    api = httpx.post(BASE + "/generate", json={"prompt": "H: say hello\nANRA:", "strategy": "nucleus", "params": {"max_tokens": 30}}, timeout=25)
    j = api.json()
    ok = isinstance(local, str) and bool(local.strip()) and api.status_code == 200 and isinstance(j.get("response"), str)
    return ok, f"local_len={len(local)} api_len={len(j.get('response', ''))}"


def t11_streaming_test() -> Tuple[bool, str]:
    with httpx.stream("GET", BASE + "/stream", params={"session_id": "test_stream", "message": "hello", "strategy": "nucleus"}, timeout=30) as r:
        events = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert events
    assert events[-1] == "data: [DONE]"
    assembled = "".join(e.replace("data: ", "") for e in events[:-1])
    chat = httpx.post(BASE + "/chat", json={"session_id": "test_stream_cmp", "message": "hello", "params": {"strategy": "nucleus", "max_tokens": 40}}, timeout=30)
    return bool(assembled.strip()) and chat.status_code == 200, f"stream_chars={len(assembled)}"


def t12_generationtrace_test() -> Tuple[bool, str]:
    trace = generate_traced("H: hello\nANRA:", GenerationConfig(max_tokens=20))
    ok = isinstance(trace.entropy_curve, list) and len(trace.entropy_curve) == trace.tokens_generated
    ok = ok and isinstance(trace.max_prob_curve, list) and len(trace.max_prob_curve) == trace.tokens_generated
    ok = ok and trace.stopped_by in {"max_tokens", "stop_string", "eos", "nucleus (contrastive fallback)"}
    ok = ok and isinstance(trace.repeated_ngrams_detected, bool)
    return ok, trace.stopped_by


def t13_context_format_test() -> Tuple[bool, str]:
    sid = "ctx_test"
    for i in range(5):
        httpx.post(BASE + "/chat", json={"session_id": sid, "message": f"m{i}"}, timeout=30)
    ctx = httpx.get(BASE + f"/debug/context/{sid}", params={"message": "last"}, timeout=30).json()["context"]
    import re

    pat = re.compile(r"^(H: .+\nANRA: .+\n){1,}H: .+\nANRA:$", re.S)
    ok = bool(pat.match(ctx)) and "\n\n" not in ctx
    return ok, f"len={len(ctx)}"


def t14_rate_limit_test() -> Tuple[bool, str]:
    sid = "ratelimit_test"
    codes = []
    last = None
    for _ in range(11):
        r = httpx.post(BASE + "/chat", json={"session_id": sid, "message": "hi"}, timeout=30)
        codes.append(r.status_code)
        last = r
    j = last.json()
    ok = codes[:10] == [200] * 10 and codes[10] == 429 and "retry_after_seconds" in j and "request_id" in j
    return ok, str(codes)


def t15_curriculum_phase_test() -> Tuple[bool, str]:
    from finetune_anra import IDENTITY_KEYWORDS, parse_identity_data

    path = Path("data/combined_identity_data.txt") if Path("data/combined_identity_data.txt").exists() else Path("combined_identity_data.txt")
    raw = path.read_text(encoding="utf-8", errors="replace")
    pairs, _ = parse_identity_data(raw)
    phase1 = [p for p in pairs if any(k.lower() in (p[0] + " " + p[1]).lower() for k in IDENTITY_KEYWORDS)]
    ok = len(phase1) > 0 and all(any(k.lower() in (p[0] + " " + p[1]).lower() for k in IDENTITY_KEYWORDS) for p in phase1)
    ok = ok and len(pairs) >= len(phase1)
    return ok, f"phase1={len(phase1)} all={len(pairs)}"


def t16_repetition_detector_test() -> Tuple[bool, str]:
    a = detect_repetition("the cat sat the cat sat the cat sat the cat sat")
    b = detect_repetition("the sky is blue and the grass is green today")
    return bool(a["repeated_ngrams_detected"]) and not bool(b["repeated_ngrams_detected"]), str(a)


def t17_session_metadata_test() -> Tuple[bool, str]:
    sid = "meta_test"
    httpx.post(BASE + "/chat", json={"session_id": sid, "message": "hello"}, timeout=30)
    httpx.post(BASE + "/chat", json={"session_id": sid, "message": "hello2"}, timeout=30)
    httpx.post(BASE + "/chat", json={"session_id": sid, "message": "hello3"}, timeout=30)
    sess = httpx.get(BASE + "/sessions", timeout=30).json()
    meta = sess.get("metadata", {}).get(sid, {})
    ok = all(k in meta for k in ["created_at", "last_active", "total_turns"]) and meta.get("total_turns") == 3
    return ok, str(meta)


def t18_stop_string_test() -> Tuple[bool, str]:
    result = generate_traced("Hello", GenerationConfig(max_tokens=200, stop_strings=["ANRA:"]))
    result2 = generate_traced("Hello", GenerationConfig(max_tokens=50, stop_strings=["zzz_never_appears"]))
    ok = "ANRA:" not in result.output and result.stopped_by == "stop_string" and result2.stopped_by == "max_tokens"
    return ok, f"{result.stopped_by}/{result2.stopped_by}"


def t19_finetune_report_test() -> Tuple[bool, str]:
    paths = [Path("finetune_report.json"), Path("/content/drive/MyDrive/AnRa/finetune_report.json")]
    found = next((p for p in paths if p.exists()), None)
    if not found:
        return False, "report_missing"
    report = json.loads(found.read_text(encoding="utf-8"))
    req = [
        "epochs_completed",
        "final_train_loss",
        "best_val_loss",
        "best_epoch",
        "curriculum_phases",
        "params_trained",
        "params_frozen",
        "training_time_seconds",
        "identity_probe_before",
        "identity_probe_after",
        "loss_curve",
    ]
    ok = all(k in report for k in req) and isinstance(report["loss_curve"], list) and len(report["loss_curve"]) >= 1
    if report.get("final_train_loss") is not None:
        ok = ok and report["best_val_loss"] < report["final_train_loss"] + 0.5
    return ok, str(found)


async def _concurrent_calls():
    async with httpx.AsyncClient(timeout=30) as client:
        msgs = [f"message_{i}" for i in range(1, 6)]
        sids = [f"concurrent_{i}" for i in range(1, 6)]
        t0 = time.perf_counter()
        tasks = [client.post(BASE + "/chat", json={"session_id": sid, "message": msg}) for sid, msg in zip(sids, msgs)]
        res = await asyncio.gather(*tasks)
        dt = (time.perf_counter() - t0) * 1000
    return res, dt, sids, msgs


def t20_concurrent_session_isolation_test() -> Tuple[bool, str]:
    res, dt, sids, msgs = asyncio.run(_concurrent_calls())
    ok = all(r.status_code == 200 for r in res)
    sessions = httpx.get(BASE + "/sessions", timeout=30).json()
    for sid, msg in zip(sids, msgs):
        data = json.loads((Path("/content/drive/MyDrive/AnRa/sessions") / f"{sid}.json").read_text(encoding="utf-8"))
        htxt = json.dumps(data)
        if msg not in htxt:
            ok = False
        for other in msgs:
            if other != msg and other in htxt:
                ok = False
    return ok, f"response_times_ms_total={dt:.1f}"


def main() -> None:
    tests: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("T1 — Import test", t1_import_test),
        ("T2 — Tokenizer test", t2_tokenizer_test),
        ("T3 — Model load test", t3_model_load_test),
        ("T4 — All strategies test", t4_all_strategies_test),
        ("T5 — API endpoint test", t5_api_endpoint_test),
        ("T6 — Session persistence test", t6_session_persistence_test),
        ("T7 — Fine-tune data test", t7_finetune_data_test),
        ("T8 — Drive path test", t8_drive_path_test),
        ("T9 — Identity probe test", t9_identity_probe_test),
        ("T10 — End-to-end test", t10_end_to_end_test),
        ("T11 — Streaming test", t11_streaming_test),
        ("T12 — GenerationTrace test", t12_generationtrace_test),
        ("T13 — Context format test", t13_context_format_test),
        ("T14 — Rate limit test", t14_rate_limit_test),
        ("T15 — Curriculum phase test", t15_curriculum_phase_test),
        ("T16 — Repetition detector test", t16_repetition_detector_test),
        ("T17 — Session metadata test", t17_session_metadata_test),
        ("T18 — Stop string test", t18_stop_string_test),
        ("T19 — finetune_report.json test", t19_finetune_report_test),
        ("T20 — Concurrent session isolation test", t20_concurrent_session_isolation_test),
    ]

    passed = 0
    failed: List[str] = []
    for name, fn in tests:
        ok, _ = _run(name, fn)
        if ok:
            passed += 1
        else:
            failed.append(name.split(" — ")[0])

    if passed == 20:
        print("20/20 tests passed — SYSTEM OK")
    else:
        print(f"{passed}/20 tests passed — SYSTEM DEGRADED — Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
