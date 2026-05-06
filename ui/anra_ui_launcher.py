from __future__ import annotations

import socket
import subprocess
import sys
import time
from urllib.request import urlopen

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flask", "flask-cors"], check=False)
subprocess.run("fuser -k 7860/tcp", shell=True, check=False)
subprocess.Popen([sys.executable, "ui/anra_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for _ in range(60):
    try:
        with urlopen("http://localhost:7860/status", timeout=2):
            break
    except Exception:
        time.sleep(1)

try:
    from IPython.display import IFrame, display

    display(IFrame("http://localhost:7860", width="100%", height="700px"))
except Exception:
    from IPython.display import HTML, display
    from google.colab.output import eval_js

    url = eval_js("google.colab.kernel.proxyPort(7860)")
    display(HTML(f'<a href="{url}" target="_blank">Open An-Ra UI</a>'))
