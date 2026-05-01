from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

REPO_PATH = Path(__file__).resolve().parents[1]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from anra_paths import DRIVE_DIR, V2_TOKENIZER_FILE, get_v2_checkpoint, inject_all_paths  # noqa: E402

inject_all_paths()

import torch  # noqa: E402
from tokenizer.tokenizer_adapter import TokenizerAdapter  # noqa: E402
from training.v2_runtime import build_v2_model, generate_text  # noqa: E402

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

IDENTITY_PREFIX = "You are An-Ra, a sovereign AI built from scratch by Ankit."
DRIVE_ROOT = DRIVE_DIR
DATASET_TARGET = REPO_PATH / "training_data" / "anra_dataset_v6_1.txt"

state = {
    "loaded": False,
    "step": 0,
    "best_loss": None,
    "device": "cpu",
    "vocab_size": 0,
    "parameters": 0,
    "use_identity": True,
    "history": [],
    "model": None,
    "tokenizer": None,
    "training_proc": None,
    "training_started": None,
    "training_loss": None,
    "training_steps": None,
}
lock = threading.Lock()


def _safe_json_error(msg: str, code: int = 500):
    return jsonify({"error": msg}), code


def _load_model() -> None:
    with lock:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = TokenizerAdapter.load(V2_TOKENIZER_FILE, model_path=V2_TOKENIZER_FILE.with_suffix(".model"))
        model = build_v2_model(vocab_size=tokenizer.vocab_size())
        checkpoint_path = get_v2_checkpoint("brain")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)), strict=False)
        model.to(device)
        model.eval()

        state["loaded"] = True
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["device"] = str(device)
        state["vocab_size"] = tokenizer.vocab_size()
        state["step"] = int(ckpt.get("step", ckpt.get("global_step", 0)))
        state["best_loss"] = ckpt.get("best_loss")
        state["parameters"] = sum(p.numel() for p in model.parameters())


def _training_monitor():
    proc = state["training_proc"]
    if proc is None:
        return
    while proc.poll() is None:
        time.sleep(3)
    with lock:
        state["training_proc"] = None


@app.get("/")
def root():
    return send_file(Path(__file__).with_name("index.html"))


@app.get("/status")
def status():
    return jsonify(
        {
            "loaded": state["loaded"],
            "step": state["step"],
            "best_loss": state["best_loss"],
            "device": state["device"],
            "vocab_size": state["vocab_size"],
            "parameters": state["parameters"],
            "identity_enabled": state["use_identity"],
        }
    )


@app.post("/set_identity")
def set_identity():
    payload = request.get_json(force=True, silent=True) or {}
    state["use_identity"] = bool(payload.get("enabled", True))
    return jsonify({"enabled": state["use_identity"]})


@app.post("/chat")
def chat():
    if not state["loaded"]:
        return _safe_json_error("Model not loaded", 503)
    payload = request.get_json(force=True, silent=True) or {}
    msg = (payload.get("message") or "").strip()
    if not msg:
        return _safe_json_error("message is required", 400)

    temp = float(payload.get("temperature", 0.85))
    max_tokens = int(payload.get("max_tokens", 120))
    top_k = int(payload.get("top_k", 40))
    use_identity = bool(payload.get("use_identity", state["use_identity"]))

    prompt = f"H: {msg}\nANRA:"
    if use_identity:
        prompt = f"{IDENTITY_PREFIX}\n\n{prompt}"

    t0 = time.perf_counter()
    out = generate_text(
        state["model"],
        state["tokenizer"],
        prompt,
        device=torch.device(state["device"]),
        max_new_tokens=max_tokens,
        temperature=temp,
        top_k=top_k,
    )
    elapsed = time.perf_counter() - t0
    clean = out.split("ANRA:", 1)[-1].strip()
    clean = clean.split("\nH:", 1)[0].strip()

    with lock:
        state["history"].append({"role": "user", "text": msg, "ts": time.time()})
        state["history"].append({"role": "anra", "text": clean, "elapsed": elapsed, "ts": time.time()})

    return jsonify({"response": clean, "elapsed": elapsed, "step": state["step"], "loss": state["best_loss"]})


@app.get("/history")
def history():
    return jsonify(state["history"])


@app.post("/clear")
def clear():
    with lock:
        state["history"] = []
    return jsonify({"ok": True})


@app.post("/save_conversation")
def save_conversation():
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = DRIVE_ROOT / "conversations"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"anra_chat_{stamp}.txt"
    lines = []
    for m in state["history"]:
        who = "H" if m["role"] == "user" else "ANRA"
        lines.append(f"{who}: {m['text']}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return jsonify({"saved": str(path)})


@app.get("/training_files")
def training_files():
    files = []
    for p in DRIVE_ROOT.rglob("*.txt"):
        files.append({"name": p.name, "path": str(p), "size": p.stat().st_size})
    files.sort(key=lambda x: x["name"].lower())
    return jsonify(files)


@app.post("/set_dataset")
def set_dataset():
    payload = request.get_json(force=True, silent=True) or {}
    src = Path(payload.get("path", ""))
    if not src.exists() or src.suffix.lower() != ".txt":
        return _safe_json_error("Invalid dataset path", 400)
    DATASET_TARGET.parent.mkdir(parents=True, exist_ok=True)
    DATASET_TARGET.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return jsonify({"active_dataset": str(DATASET_TARGET), "source": str(src)})


@app.post("/start_training")
def start_training():
    payload = request.get_json(force=True, silent=True) or {}
    minutes = int(payload.get("session_minutes", 30))
    mode = str(payload.get("mode", "session"))
    if state["training_proc"] and state["training_proc"].poll() is None:
        return _safe_json_error("Training already running", 409)

    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "training.train_unified", "--mode", mode, "--session_minutes", str(minutes)],
        cwd=str(REPO_PATH),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with lock:
        state["training_proc"] = proc
        state["training_started"] = time.time()
    threading.Thread(target=_training_monitor, daemon=True).start()
    return jsonify({"running": True, "pid": proc.pid})


@app.get("/training_status")
def training_status():
    proc = state["training_proc"]
    running = bool(proc and proc.poll() is None)
    return jsonify(
        {
            "running": running,
            "current_loss": state["training_loss"],
            "steps": state["training_steps"],
        }
    )


@app.post("/stop_training")
def stop_training():
    proc = state["training_proc"]
    if not proc or proc.poll() is not None:
        return jsonify({"running": False})
    proc.send_signal(signal.SIGINT)
    return jsonify({"running": False, "stopping": True})


if __name__ == "__main__":
    try:
        _load_model()
    except Exception as exc:
        print(f"Model load failed: {exc}")
    app.run(host="127.0.0.1", port=7860, debug=False)
