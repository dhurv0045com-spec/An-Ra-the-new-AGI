"""Custody of V5 canaries and explicit preservation of failed capability gates."""
from hashlib import sha256
import json
from pathlib import Path


def test_foundation_raw_receipts_and_frozen_weight_identity():
    root = Path("artifacts/elendril/foundation-v5")
    manifest = json.loads((root / "manifest.json").read_bytes())
    for name, expected in manifest["artifacts"].items():
        assert Path(name).name == name
        assert sha256((root / name).read_bytes()).hexdigest() == expected
    learned = json.loads((root / "learning-canary.json").read_bytes())
    assert learned["exact_query_eos"] == learned["denominator"] == 4
    assert learned["updates"][-1]["loss"] < learned["updates"][0]["loss"] / 2
    assert learned["gpu_seconds"] == 0
    frozen = json.loads((root / "frozen-v4-numpy-free.json").read_bytes())
    assert frozen["weights_sha256_before"] == frozen["weights_sha256_after"] == (
        "8734cfb9957f510a703c0d04e5f68a305e0af9552238c28f178e848dc9c2e863")
    assert frozen["totals"]["route/memory"]["correct"] == 4
    assert frozen["totals"]["revision/public-query"]["correct"] == 0
    assert frozen["totals"]["revision/public-query"]["requests"] == 0
    previous = json.loads((root / "frozen-v4-replay.json").read_bytes())
    assert previous["totals"] == frozen["totals"]
