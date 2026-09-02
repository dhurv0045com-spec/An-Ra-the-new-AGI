"""Drive the powered DEVELOPMENT scorer tournament (CUDA cells).

Runs all 15 frozen CUDA cells (3 vocabularies x 5 development seeds) of the
preregistered scorer-policy tournament against the repaired rotation
geometry, then aggregates. CPU parity cells are intentionally NOT run here;
the aggregate's early_policy_failure path still yields a real bias-screen
verdict (production_scoring_mode stays null unless a policy survives).

Usage: py -3 scripts/run_development_tournament_cuda.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "artifacts" / "e2" / "development_shards"
ARTIFACTS = ROOT / "artifacts" / "e1" / "local_tournament"
FIXTURE = ROOT / "artifacts" / "e2" / "scoring_policy_fixture.json"
SEEDS = (95_101, 95_102, 95_103, 95_104, 95_105)
VOCABS = (16_384, 24_576, 32_768)


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    receipts: list[Path] = []
    started = time.time()
    for vocabulary in VOCABS:
        for seed in SEEDS:
            out = SHARDS / f"dev_cuda_{vocabulary}_{seed}.json"
            receipts.append(out)
            if out.exists() and json.loads(out.read_text(encoding="utf-8")).get("status") == "PASS_EXECUTION":
                print(f"[skip] {out.name} already complete", flush=True)
                continue
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, "-m", "e2_architecture.scoring_policy_tournament",
                 "run-shard",
                 "--artifact-directory", str(ARTIFACTS),
                 "--fixture-receipt", str(FIXTURE),
                 "--vocabulary", str(vocabulary),
                 "--seed", str(seed),
                 "--device", "cuda",
                 "--batch-size", "32",
                 "--output", str(out)],
                cwd=ROOT, capture_output=True, text=True,
            )
            status = "?"
            if out.exists():
                try:
                    status = json.loads(out.read_text(encoding="utf-8")).get("status", "?")
                except Exception:
                    pass
            print(f"[shard] v{vocabulary} s{seed} exit={proc.returncode} "
                  f"status={status} {time.time() - t0:.0f}s", flush=True)
            if proc.returncode != 0:
                print(proc.stderr[-800:], flush=True)
    done = [p for p in receipts if p.exists()]
    print(f"[aggregate] {len(done)}/15 shards present", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "e2_architecture.scoring_policy_tournament",
         "aggregate-development",
         *[arg for p in done for arg in ("--receipt", str(p))],
         "--output", str(ROOT / "artifacts" / "e2" / "scoring_policy_development.json")],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(proc.stdout[-2000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-1500:], flush=True)
    print(f"[done] wall {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
