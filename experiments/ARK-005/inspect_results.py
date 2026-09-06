"""Inspect the recovered ARK-005 receipts (8-way launch, box-aborted but measured)."""
import json
import glob

for path in sorted(glob.glob("experiments/ARK-005/RESULT_*.json")):
    r = json.load(open(path, encoding="utf-8"))
    res = r["results"]
    ret = res["retention"]
    line = (
        f"{res['arm']} seed{res['seed']}: {res['status']}"
        f" | trigger {res['trigger_step']} | steps {res['steps_run']}"
        f" | PEAK_G {ret.get('PEAK_G')} | RET90 {ret.get('RET90')}"
        f" | T_COLLAPSE_90 {ret.get('T_COLLAPSE_90')}"
        f" | STABILITY_GAP {ret.get('STABILITY_GAP')}"
        f" | final {ret.get('FINAL_OOD')}"
    )
    print(line)
