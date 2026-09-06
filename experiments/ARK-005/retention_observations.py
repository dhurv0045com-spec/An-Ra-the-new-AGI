"""ARK-005 observational analysis: retention metrics from ARK-004A receipts.

Prospectively defined metrics (ARK-005 PLAN):
  PEAK_G            maximum sustained generalized regime (peak of the
                    3-eval trailing mean of OOD after M99)
  RET90 / RET50     fraction of post-G90 evaluations remaining >= 0.90 / >= 0.50
  T_COLLAPSE_90/50  first sustained (3 consecutive) fall below 0.90 / 0.50
                    after G90
  GENERALIZATION_AREA  mean OOD exact from first G90 to termination
  STABILITY_GAP     peak sustained OOD minus final sustained OOD
"""

from __future__ import annotations

import glob
import json
from pathlib import Path


def first_sustained(trajectory, key, bar, after_step, consecutive=3, below=False):
    streak, start = 0, None
    for e in trajectory:
        if e["step"] <= after_step:
            continue
        value = float(e.get(key, 0.0))
        hit = value < bar if below else value >= bar
        if hit:
            if streak == 0:
                start = e["step"]
            streak += 1
            if streak >= consecutive:
                return start
        else:
            streak, start = 0, None
    return None


def analyze(path: str) -> dict:
    r = json.load(open(path, encoding="utf-8"))
    res = r["results"]
    traj = res["trajectory"]
    seed = res["seed"]
    m99 = res["sustained"]["M99"]["step"]
    g90 = res["sustained"]["G90"]["step"]
    post = [e for e in traj if e["step"] >= g90]
    trailing = []
    for i, e in enumerate(post):
        window = post[max(0, i - 2): i + 1]
        trailing.append((e["step"], sum(x["test_exact"] for x in window) / len(window)))
    peak_g = max(v for _, v in trailing) if trailing else None
    peak_step = max(trailing, key=lambda t: t[1])[0] if trailing else None
    ret90 = sum(1 for e in post if e["test_exact"] >= 0.90) / len(post)
    ret50 = sum(1 for e in post if e["test_exact"] >= 0.50) / len(post)
    collapse90 = first_sustained(traj, "test_exact", 0.90, g90, below=True)
    collapse50 = first_sustained(traj, "test_exact", 0.50, g90, below=True)
    gen_area = sum(e["test_exact"] for e in post) / len(post)
    final_sustained = trailing[-1][1] if trailing else None
    return {
        "seed": seed,
        "M99": m99, "G90": g90, "steps_run": res["steps_run"],
        "PEAK_G": round(peak_g, 3) if peak_g is not None else None,
        "PEAK_G_step": peak_step,
        "RET90": round(ret90, 3),
        "RET50": round(ret50, 3),
        "T_COLLAPSE_90": collapse90,
        "T_COLLAPSE_50": collapse50,
        "GENERALIZATION_AREA": round(gen_area, 3),
        "STABILITY_GAP": round(peak_g - final_sustained, 3) if peak_g is not None and final_sustained is not None else None,
        "FINAL_OOD": round(traj[-1]["test_exact"], 3),
        "classification": (
            "STABLE" if collapse90 is None and ret90 >= 0.8 else
            "PARTIAL_DECAY" if collapse90 is None else
            "COLLAPSED" if collapse50 is not None else
            "DECAYING"),
    }


def main() -> int:
    rows = [analyze(p) for p in sorted(glob.glob(str(Path(__file__).parents[1] / "ARK-004A" / "RESULT_seed*.json")))]
    out = Path(__file__).parent / "RETENTION_OBSERVATIONS.json"
    out.write_text(json.dumps({
        "schema": "arkenstone-ark005-retention-observations/v1",
        "source": "ARK-004A receipts (observational; no new training)",
        "metrics_definitions": __doc__,
        "seeds": rows,
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    main()
