"""Fold ARK-004A results into the program ledgers (run once)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

log = (REPO / "docs/arkenstone/EXPERIMENT_LOG.md").read_text(encoding="utf-8")
log += ("| ARK-004A | What changes before the generalization transition, and does it predict timing? "
        "| 4 fresh seeds (101/202/303/404), dense behavioral + column-selectivity probes | EXECUTED — "
        "G90 on all 4; tens-selectivity precursor qualifies (LOO 4/4, beats baselines); post-G90 instability "
        "discovered | experiments/ARK-004A/ANALYSIS.md |\n")
(REPO / "docs/arkenstone/EXPERIMENT_LOG.md").write_text(log, encoding="utf-8", newline="\n")

tour = (REPO / "docs/arkenstone/MECHANISM_TOURNAMENT.md").read_text(encoding="utf-8")
tour += ("| M-008 | Column-selectivity precursor (frozen counterfactual probes) | NEW_PROGRAM_MEASUREMENT | "
         "absent from all branch receipts | factorization of the binding-heavy column should precede/emerge "
         "with OOD | TENTATIVE-SUPPORTED (LOO 4/4 vs baselines, n=4; ARK-004A) |\n")
(REPO / "docs/arkenstone/MECHANISM_TOURNAMENT.md").write_text(tour, encoding="utf-8", newline="\n")

feat = (REPO / "docs/arkenstone/AGI_FEATURE_LEDGER.md").read_text(encoding="utf-8")
feat += ("| Column-selectivity precursor + dense transition instrumentation | absent | "
         "NEW_PROGRAM_MEASUREMENT | predict/shorten the post-memorization delay | counterfactual "
         "factorization probes | ARK-004A: LOO 4/4, beats time/loss; post-G90 instability documented | "
         "probe cost ~2s/eval | MEDIUM | CANDIDATE (ARK-004B gate OPEN) |\n")
(REPO / "docs/arkenstone/AGI_FEATURE_LEDGER.md").write_text(feat, encoding="utf-8", newline="\n")

neg = (REPO / "docs/arkenstone/NEGATIVE_RESULTS.md").read_text(encoding="utf-8")
neg += ("| M99 (memorization speed) does NOT predict G90 timing (rho 0.00) | ARK-004A | memorization and "
        "generalization are decoupled phenomena |\n"
        "| Post-G90 instability: seed 101 collapsed 1.0->0.188 after sustained G90 | ARK-004A | the "
        "generalized state is not automatically stable; retention is a first-class objective |\n")
(REPO / "docs/arkenstone/NEGATIVE_RESULTS.md").write_text(neg, encoding="utf-8", newline="\n")

readme_path = REPO / "docs/arkenstone/README.md"
readme = readme_path.read_text(encoding="utf-8")
marker = "- **Current highest-value uncertainty:**"
resolved = ("- **Resolved this run (ARK-004A):** a precursor qualifies — tens-column selectivity under "
            "frozen counterfactuals predicts G90 timing (LOO 4/4, beats time/loss baselines); memorization "
            "speed predicts nothing; post-G90 instability discovered (seed 101 collapse).\n")
if marker in readme and "Resolved this run (ARK-004A)" not in readme:
    idx = readme.index(marker)
    end = readme.index("\n", idx) + 1
    readme = readme[:idx] + resolved + readme[idx:]
readme = readme.replace(
    "- `experiments/BINDING-V2-REDTEAM/` — independent verification of cymek's binding-v2 qualification (CONSISTENT)",
    "- `experiments/BINDING-V2-REDTEAM/` — independent verification of cymek's binding-v2 qualification (CONSISTENT)\n"
    "- `experiments/ARK-004A/` — developmental transition mapping + precursor discovery (executed)")
readme_path.write_text(readme, encoding="utf-8", newline="\n")
print("ledgers updated")
