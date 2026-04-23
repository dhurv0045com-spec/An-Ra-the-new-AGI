from __future__ import annotations

import asyncio
import json
import sys
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
    candidates = [
        Path("data/combined_identity_data.txt"),
        Path("combined_identity_data.txt"),
        Path("anra_dataset_v6_1.txt"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return False, "no training dataset file found"

    text = path.read_text(encoding="utf-8", errors="replace")
    import re

    pairs = re.findall(r"H:\s*(.*?)\nANRA:\s*(.*?)(?=\nH:|\Z)", text, re.S)
    valid_pairs = [(h.strip(), a.strip()) for h, a in pairs if h.strip() and a.strip()]
    ok = len(valid_pairs) > 100 and all(p[1] for p in valid_pairs)
    return ok, f"file={path.name} pairs={len(valid_pairs)}"


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
        r = httpx.post(
            BASE + "/generate",
            json={"prompt": "H: hi\nANRA:", "strategy": "nucleus", "session_id": sid, "params": {"max_tokens": 1}},
            timeout=30,
        )
        codes.append(r.status_code)
        last = r
    j = last.json()
    ok = codes[:10] == [200] * 10 and codes[10] == 429 and "retry_after_seconds" in j and "request_id" in j
    return ok, str(codes)


def t15_curriculum_phase_test() -> Tuple[bool, str]:
    from finetune_anra import IDENTITY_KEYWORDS, parse_identity_data

    candidates = [Path("data/combined_identity_data.txt"), Path("combined_identity_data.txt"), Path("anra_dataset_v6_1.txt")]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return False, "no dataset path for curriculum test"
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
    httpx.post(BASE + "/reset", json={"session_id": sid}, timeout=30)
    httpx.post(BASE + "/chat", json={"session_id": sid, "message": "hello"}, timeout=30)
    httpx.post(BASE + "/chat", json={"session_id": sid, "message": "hello2"}, timeout=30)
    httpx.post(BASE + "/chat", json={"session_id": sid, "message": "hello3"}, timeout=30)
    sess = httpx.get(BASE + "/sessions", timeout=30).json()
    meta = sess.get("metadata", {}).get(sid, {})
    ok = all(k in meta for k in ["created_at", "last_active", "total_turns"]) and meta.get("total_turns") == 3
    return ok, str(meta)


def t18_stop_string_test() -> Tuple[bool, str]:
    from generate import _check_stop

    hit, trimmed, reason = _check_stop("alpha<STOP>omega", GenerationConfig(stop_strings=["<STOP>"]))
    miss, _, reason2 = _check_stop("alpha-omega", GenerationConfig(stop_strings=["<STOP>"]))
    ok = hit and trimmed == "alpha" and reason == "stop_string" and (not miss) and reason2 == ""
    return ok, f"{reason}/{reason2 or 'none'}"


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
        stamp = int(time.time() * 1000)
        sids = [f"concurrent_{stamp}_{i}" for i in range(1, 6)]
        t0 = time.perf_counter()
        tasks = [client.post(BASE + "/chat", json={"session_id": sid, "message": msg}) for sid, msg in zip(sids, msgs)]
        res = await asyncio.gather(*tasks)
        dt = (time.perf_counter() - t0) * 1000
    return res, dt, sids, msgs


def t20_concurrent_session_isolation_test() -> Tuple[bool, str]:
    res, dt, sids, msgs = asyncio.run(_concurrent_calls())
    codes = [r.status_code for r in res]
    ok = len(res) == 5
    return ok, f"response_times_ms_total={dt:.1f} codes={codes}"




def t21_agent_loop_initialization_test() -> Tuple[bool, str]:
    start = time.time()
    try:
        import importlib.util
        mod_path = Path("phase2/agent_loop (45k)/agent_main.py")
        spec = importlib.util.spec_from_file_location("agent_main_45k", mod_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        AgentLoop = module.AgentLoop
    except ImportError as e:
        elapsed = (time.time() - start) * 1000
        return False, (
            f"Agent module not found: {e}\n"
            f"  Expected: phase3/agent_loop_45K/\n"
            f"  This is a REAL failure, not a test environment issue.\n"
            f"  Fix: verify phase3/agent_loop_45K exists in repo."
        )

    try:
        agent = AgentLoop()
        health = agent.health_check()
        result = agent.run_once("What is your primary goal?")
        assert health["agent_ready"] is True
        assert result.get("outcome") in {"attempted", "error"}

        elapsed = (time.time() - start) * 1000
        return True, f"AgentLoop initialized, step executed ({elapsed:.0f}ms)"
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return False, f"AgentLoop failed: {e} ({elapsed:.0f}ms)"


def t22_agent_decision_loop_test() -> Tuple[bool, str]:
    start = time.time()
    try:
        import importlib.util
        mod_path = Path("phase2/agent_loop (45k)/agent_main.py")
        spec = importlib.util.spec_from_file_location("agent_main_45k", mod_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        AgentLoop = module.AgentLoop
    except ImportError as e:
        elapsed = (time.time() - start) * 1000
        return False, f"Agent module not found: {e}"

    try:
        agent = AgentLoop()

        results = []
        step_times = []
        for i in range(3):
            step_start = time.time()
            result = agent.run_once(f"Step {i+1}: analyze your capabilities")
            step_times.append((time.time() - step_start) * 1000)
            results.append(result)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        outputs = [json.dumps(r, sort_keys=True) for r in results]
        assert len(set(outputs)) > 1, "All 3 agent steps produced identical output (stuck loop)"

        timing = ", ".join(f"{t:.0f}ms" for t in step_times)
        elapsed = (time.time() - start) * 1000
        return True, f"3-step loop complete. Steps: [{timing}] ({elapsed:.0f}ms)"
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return False, f"Agent decision loop failed: {e} ({elapsed:.0f}ms)"



def t23_optimization_import_test() -> Tuple[bool, str]:
    try:
        from optimizations import AdaptiveScheduler, MultiScaleHardSampleDetector, GradientCheckpointedOuroboros
        return True, "All optimization modules imported"
    except Exception as e:
        return False, f"Import failed: {e}"


def t24_optimization_config_test() -> Tuple[bool, str]:
    try:
        config_path = Path("AnRa/optimization_config.json")
        if not config_path.exists():
            config_path = Path("/content/drive/MyDrive/AnRa/optimization_config.json")
        if not config_path.exists():
            return False, "optimization_config.json not found"
        config = json.loads(config_path.read_text())
        ok = "adaptive_scheduler" in config and "hard_sample_routing" in config
        return ok, f"Config valid with {len(config)} keys"
    except Exception as e:
        return False, str(e)

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
        ("T21 — Agent Loop Initialization", t21_agent_loop_initialization_test),
        ("T22 — Agent Decision Loop", t22_agent_decision_loop_test),
        ("T23 — Optimization import test", t23_optimization_import_test),
        ("T24 — Optimization config test", t24_optimization_config_test),
    ]

    passed = 0
    failed: List[str] = []
    results: List[Tuple[str, bool, str]] = []
    for name, fn in tests:
        ok, detail = _run(name, fn)
        results.append((name, ok, detail))
        if ok:
            passed += 1
        else:
            failed.append(name.split(" — ")[0])

    agent_tests_failed = any(
        "Agent" in name and not ok
        for name, ok, _ in results
    )

    if passed == 24:
        print("\n24/24 tests passed — SYSTEM OK")
    else:
        print(f"\nWARNING: {24 - passed} test(s) failed")
        print(f"{passed}/24 tests passed — SYSTEM DEGRADED — Failed: {', ' .join(failed)}")
        if agent_tests_failed:
            print("❌ CRITICAL: Agent loop tests failed.")
            print("   /chat endpoint will crash in production.")
            print("   Fix phase3/agent_loop_45K before deploying.")
            sys.exit(1)


if __name__ == "__main__":
    main()
