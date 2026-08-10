"""Desktop lifecycle controller for the local V4 SFT prototype.

The controller owns the Uvicorn child process. Closing its window stops that
child and releases the GPU rather than leaving a hidden server behind.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import Button, Label, StringVar, Tk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.local_checkpoint import (  # noqa: E402
    remember_local_sft_checkpoint,
    resolve_local_sft_checkpoint,
)


def _read_status(url: str) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(f"{url}/api/status", timeout=1.0) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=12)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the local An-Ra V4 SFT prototype.")
    parser.add_argument("--checkpoint", default="", help="Path to the one protected SFT .pt file")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    checkpoint = (
        remember_local_sft_checkpoint(args.checkpoint)
        if args.checkpoint
        else resolve_local_sft_checkpoint()
    )
    url = f"http://127.0.0.1:{int(args.port)}"
    environment = os.environ.copy()
    environment["ANRA_CHECKPOINT_PATH"] = str(checkpoint.path)
    environment["ANRA_SFT_CHECKPOINT"] = str(checkpoint.path)
    environment["ANRA_PROTOTYPE_IDLE_SHUTDOWN_SECONDS"] = "45"
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "runtime.sft_prototype:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        cwd=ROOT,
        env=environment,
    )

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and process.poll() is None:
        if _read_status(url) is not None:
            break
        time.sleep(0.25)
    if process.poll() is not None:
        raise RuntimeError("The local SFT prototype server stopped during startup.")
    if not args.no_browser:
        webbrowser.open(url)

    window = Tk()
    window.title("An-Ra V4 SFT Prototype")
    window.geometry("470x220")
    window.resizable(False, False)
    status = StringVar(value="Starting the protected checkpoint on your GPU…")
    Label(window, text="AN-RA V4 SFT PROTOTYPE", font=("Segoe UI", 14, "bold")).pack(pady=(22, 8))
    Label(window, text=str(checkpoint.path), wraplength=420, justify="center").pack(padx=18)
    Label(window, textvariable=status, wraplength=420, justify="center").pack(pady=12)

    closing = False

    def close_controller() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        status.set("Stopping server and freeing GPU memory…")
        window.update_idletasks()
        _stop_process(process)
        window.destroy()

    def open_interface() -> None:
        webbrowser.open(url)

    def poll() -> None:
        if closing:
            return
        snapshot = _read_status(url)
        if process.poll() is not None:
            status.set("The prototype server stopped.")
            window.after(800, window.destroy)
            return
        if snapshot is None:
            status.set("Waiting for local server…")
        elif bool(snapshot.get("shutdown_requested")):
            close_controller()
            return
        elif bool(snapshot.get("ready")):
            gpu = dict(snapshot.get("gpu", {}))
            status.set(f"Ready on {gpu.get('name', 'CUDA GPU')}. Closing this window stops it.")
        else:
            status.set(str(snapshot.get("stage", "loading")))
        window.after(1000, poll)

    Button(window, text="Open interface", command=open_interface).pack(
        side="left", padx=(100, 8), pady=10
    )
    Button(window, text="Stop & close", command=close_controller).pack(side="left", padx=8, pady=10)
    window.protocol("WM_DELETE_WINDOW", close_controller)
    window.after(250, poll)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
