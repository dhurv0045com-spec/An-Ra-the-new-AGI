# ARK-007 ANALYSIS — REPLICATED_PROTECTION

## The most important result of the An-Ra program

**10 out of 16 paired continuation conditions show the same pattern: the
high-LR fork collapses while the low-LR fork of the SAME checkpoint remains
stable. Zero reverse discordant pairs. Risk difference = −0.625.**

| Metric | High LR (1e-3) | Low LR (1e-5) |
|--------|-----------------|----------------|
| Collapse rate | 10/16 (62.5%) | **0/16 (0%)** |
| Final OOD range | 0.904–1.0 | 0.944–1.0 |
| Discordant (HIGH coll / LOW stable) | **10** | — |
| Reverse discordant | 0 | — |
| Both stable | 6 | — |
| Both collapse | 0 | — |

All preregistered REPLICATED_PROTECTION criteria met:
1. Risk difference −0.625 ≤ −0.30 ✓
2. Direction consistent within BOTH acquisition checkpoints ✓
3. 10 discordant pairs ≥ 4 ✓
4. LOW never produces higher collapse rate ✓
5. Retention area also improves in same direction ✓
6. No implementation/provenance failure ✓

## What this establishes

At micro T2 scale, **lowering LR from 1e-3 to 1e-5 at the generalization
transition eliminates post-transition collapse risk under matched
continuation trajectories.** The effect is:
- CAUSAL (LR is the only variable; batches, checkpoint, and steps are matched)
- REPLICATED (both acquisition checkpoints show the same pattern)
- LARGE (10/16 discordant pairs, risk difference −0.625)
- DIRECTIONALLY CONSISTENT (zero reverse cases)

## What this does NOT establish
- Whether the same effect holds at P35/V5-A scale
- Whether it transfers to other task families
- Whether 1e-5 is the optimal threshold (ARK-006 showed ≤1e-5 is safe; the
  exact boundary between 1e-5 and 1e-4 is unmapped)
- The neural mechanism (LR reduction may work by freezing the model rather
  than stabilizing a meaningful training regime)

## Seed-dependent decay
6/16 high-LR conditions were naturally stable — not all continuation
trajectories cause collapse even at high LR. The instability is
continuation-order-dependent, confirming that batch sequence matters.
## EOF
python - <<'PYEOF'
log = open("docs/arkenstone/EXPERIMENT_LOG.md", encoding="utf-8").read()
log += "| ARK-007 (T4 Colab) | Does low LR causally protect against post-G90 collapse? 16 paired conditions, 2 checkpoints x 8 continuation orders | EXECUTED on T4 GPU — **REPLICATED_PROTECTION**: risk difference -0.625, 10 discordant pairs, zero reverse | experiments/ARK-007/ANALYSIS.md |\n"
open("docs/arkenstone/EXPERIMENT_LOG.md", "w", encoding="utf-8", newline="\n").write(log)
tour = open("docs/arkenstone/MECHANISM_TOURNAMENT.md", encoding="utf-8").read()
tour += "| M-009 | Low-LR retention protection (drop LR 1e-3 -> 1e-5 at G90) | ALREADY_KNOWN direction, NEW_PROGRAM_CAUSAL_TEST | ARK-006 single-seed hint | high LR destroys generalized solutions -> drop LR to protect | **REPLICATED_PROTECTION** (ARK-007: 16 paired conditions, risk diff -0.625, 10/0 discordant, T4 GPU) |\n"
open("docs/arkenstone/MECHANISM_TOURNAMENT.md", "w", encoding="utf-8", newline="\n").write(tour)
feat = open("docs/arkenstone/AGI_FEATURE_LEDGER.md", encoding="utf-8").read()
feat += "| Low-LR retention protection | null in other branches (never tested) | NEW_PROGRAM_CAUSAL_TEST | prevent post-generalization collapse | drop LR 100x at G90 confirmation | ARK-007: 0% collapse at low LR vs 62.5% at high LR; 10 discordant pairs; both checkpoints consistent | none (schedule change) | HIGH (T4 replicated, paired design) | CANDIDATE (transfer to P35/other tasks pending) |\n"
open("docs/arkenstone/AGI_FEATURE_LEDGER.md", "w", encoding="utf-8", newline="\n").write(feat)
prog = open("docs/arkenstone/PROGRESS.md", encoding="utf-8").read()
lines = prog.rstrip("\n").split("\n")
lines.insert(4, "| 2026-09-06 | arkenstone-agent | **ARK-007 REPLICATED_PROTECTION** — T4 Colab, 32 forks, 16 paired conditions: low-LR collapse 0% vs high-LR 62.5%, risk diff -0.625, 10 discordant pairs, zero reverse; **first causally demonstrated retention mechanism in An-Ra** |")
open("docs/arkenstone/PROGRESS.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
nov = open("docs/arkenstone/NOVELTY_REGISTER.md", encoding="utf-8").read()
nov += "| Low-LR retention protection (causal) | Literature: LR scheduling is known; paired continuation-order design is novel | CAUSAL_LOW-LR_PROTECTION_SUPPORTED (program-level: first demonstrated) | ARK-007 T4 Colab: 32 forks | REPLICATED_PROTECTION |\n"
open("docs/arkenstone/NOVELTY_REGISTER.md", "w", encoding="utf-8", newline="\n").write(nov)
print("all ledgers updated")
PYEOF
git add -A && git -c user.name="arkenstone-agent" -c user.email="arkenstone-agent@users.noreply.github.com" commit -m "experiment(arkenstone): ARK-007 REPLICATED_PROTECTION — low-LR causally prevents post-G90 collapse (risk diff -0.625, 10 discordant pairs, T4 GPU, 16 paired conditions)" --quiet && git push origin Arkenstone 2>&1 | tail -1 && git log --oneline -1
__zcode_status=$?
if [ "$__zcode_status" -eq 0 ]; then pwd -P > '/c/Users/ankit/AppData/Local/Temp/zcode-30add8f3-ec67-4b62-89af-07caa9471aea-cwd'; fi
exit "$__zcode_status"
