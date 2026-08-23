"""Clean-commit reproduction of the corrective QIM-v3 rescore.

Reproduces, from a clean checkout at the recorded commit and the exact
frozen checkpoints, the corrective-rescore artifacts:

  output/qim3_parent_corrective_rescore.json
  output/qim3_sft6_corrective_rescore.json

Usage:
  py -3 scripts/reproduce_corrective_rescore.py            # verify vs committed artifacts
  py -3 scripts/reproduce_corrective_rescore.py --redo     # re-run and overwrite *_repro.json

Determinism contract: same fixture seed + same evaluator build => identical
per-group lifts, rank, greedy flags. Model inference on CUDA fp32 is
run-to-run stable for these tiny sequences (verified empirically); if your
hardware differs, lifts may differ in low-order digits while greedy/rank
flags must still match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

PARENT_CKPT = "checkpoints/anra-v4-20k-sft3-accumulate.pt"
CHILD_CKPT = "checkpoints/anra-v4-20k-sft6-queryswap-replication.pt"
P_SHA = "c3bc615eb3ffc8628f82088c433507baa142a0fecf91e4f6e64f9b17729e0625"
C_SHA = "36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001"
FIX = "27713accb3aa6825de23cf09540497943a0e56ca3ac177ba94f0f164f740a614"
CLS = "CORRECTIVE_RESCORING_AFTER_EVALUATOR_BUG"

# The numbers a clean reproduction MUST produce (from the committed
# corrective-rescore artifacts). Greedy/lift values are load-bearing.
EXPECTED = {
    "parent": {"mean_group_lift": 0.0192, "rank1": "40/119",
               "greedy": "36/119", "groups_positive": "19/40"},
    "child": {"mean_group_lift": 2.5052, "rank1": "63/119",
              "greedy": "44/119", "groups_positive": "36/40"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redo", action="store_true",
                    help="re-run evaluation and write *_repro.json files")
    args = ap.parse_args()

    # ---- frozen identity checks BEFORE anything else ----------------------
    import torch
    from anra_core.checkpoint import load_core_checkpoint

    _, _, ident_p = load_core_checkpoint(ROOT / PARENT_CKPT, legacy_unverified=True)
    assert ident_p.parameter_sha256 == P_SHA, \
        f"parent drifted: {ident_p.parameter_sha256}"
    payload_c = torch.load(ROOT / CHILD_CKPT, map_location="cpu", weights_only=True)
    assert payload_c["parameter_sha256"] == C_SHA, "child drifted"
    print("[identity] parent + child SHAs verified frozen")

    from connector.experiments.query_influence_v3 import (
        run_model, fixture_hash)
    assert fixture_hash() == FIX, "QIM-v3 fixture drifted!"
    print("[fixture] QIM-v3 hash verified:", FIX[:16])

    if not args.redo:
        # ---- verify mode: compare committed artifacts to EXPECTED --------
        par = json.loads((ROOT / "output/qim3_parent_corrective_rescore.json").read_text(encoding="utf-8"))
        chi = json.loads((ROOT / "output/qim3_sft6_corrective_rescore.json").read_text(encoding="utf-8"))
        failures = []
        for label, rep in (("parent", par), ("child", chi)):
            exp = EXPECTED[label]
            got_gl = rep["group_level"]["mean_group_lift"]
            got_r1 = rep["candidate_diagnostic_only"]["correct_rank1_fraction"]
            got_gr = rep["candidate_diagnostic_only"]["greedy_corresponding_accuracy"]
            got_gp = rep["group_level"]["groups_positive"]
            for field, got, want in (("mean_group_lift", got_gl, exp["mean_group_lift"]),
                                     ("rank1", got_r1, exp["rank1"]),
                                     ("greedy", got_gr, exp["greedy"]),
                                     ("groups_positive", got_gp, exp["groups_positive"])):
                ok = (abs(got - want) < 5e-4) if isinstance(want, float) else (got == want)
                if not ok:
                    failures.append(f"{label}.{field}: {got} != {want}")
        if failures:
            print("REPRODUCTION FAILED:")
            for f in failures:
                print("  -", f)
            return 1
        print("REPRODUCTION VERIFIED against expected corrective numbers")
        print("  parent:", par["candidate_diagnostic_only"])
        print("  child :", chi["candidate_diagnostic_only"])
        return 0

    # ---- redo mode: actually re-run both evaluations ----------------------
    par = run_model("repro-parent-corrective", str(ROOT / PARENT_CKPT),
                    evaluation_class=CLS)
    (ROOT / "output/qim3_parent_corrective_rescore_repro.json").write_text(
        json.dumps(par, indent=2), encoding="utf-8")
    chi = run_model("repro-sft6-corrective", str(ROOT / CHILD_CKPT),
                    parent_report=par, evaluation_class=CLS)
    (ROOT / "output/qim3_sft6_corrective_rescore_repro.json").write_text(
        json.dumps(chi, indent=2), encoding="utf-8")

    failures = []
    for label, rep in (("parent", par), ("child", chi)):
        exp = EXPECTED[label]
        if abs(rep["group_level"]["mean_group_lift"] - exp["mean_group_lift"]) >= 5e-4:
            failures.append(f"{label}.mean_group_lift {rep['group_level']['mean_group_lift']} != {exp['mean_group_lift']}")
        if rep["candidate_diagnostic_only"]["correct_rank1_fraction"] != exp["rank1"]:
            failures.append(f"{label}.rank1 != {exp['rank1']}")
        if rep["candidate_diagnostic_only"]["greedy_corresponding_accuracy"] != exp["greedy"]:
            failures.append(f"{label}.greedy {rep['candidate_diagnostic_only']['greedy_corresponding_accuracy']} != {exp['greedy']} (stale-prompt bug regression?)")
    if failures:
        print("LIVE REDO DIVERGED:")
        for f in failures:
            print("  -", f)
        return 1
    print("LIVE REDO MATCHES committed corrective results exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
