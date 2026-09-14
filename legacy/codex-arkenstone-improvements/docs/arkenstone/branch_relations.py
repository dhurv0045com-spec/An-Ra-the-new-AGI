"""Branch relations map: one page per branch, updated from live evidence.

Maintained on Arkenstone because no other branch may be modified (GAP 3/6 of
the external gap review). Every line here was verified against receipts or
marked as design-only.
"""

BRANCH_RELATIONS = {
    "cymek": {
        "proves": "A certified V5 training/evaluation substrate: exact V5-A model (250,216,960 params), "
                  "production backend with mutation certification, cursor-authoritative data stream, "
                  "checkpoint registry, CoreSubjectManifest/Triquetra handshake, P35+V5-A CUDA canaries.",
        "done_looks_like": "P35-A executed on real data with a qualified generator set and a real corpus; "
                           "currently BLOCKED_BY_external_corpus + generator-family coverage (6 of 8 families "
                           "have zero executed evidence).",
        "connects_to": "Citadel validates Cymek on TPU ('validates, does not replace'); Triquetra consumes "
                       "Cymek's CoreSubjectManifest; Arkenstone inherits Cymek's engineering invariants.",
    },
    "citadel": {
        "proves": "The Cymek stack trains on real TPU (T0 PASS) and the loss-vs-capability anomaly is real "
                  "(T1/T1C: loss 10.1->1.3-1.9, exact 0/500 across objectives, corpora, 2.3x scale).",
        "done_looks_like": "T1D + PRE50M one-shot execution (staged READY_FOR_ONE_SHOT_TPU); its tier "
                           "lift-off readouts should include per-band OOD curves per Arkenstone's dose "
                           "evidence (OOD dose ~10-20x memorization dose, seed-variable).",
        "connects_to": "Validates Cymek; Arkenstone's micro lift-off/dose results parameterize T1D's design.",
    },
    "triquetra": {
        "proves": "Cognition qualification and causal-diagnosis discipline: CoreSubjectManifest validation, "
                  "promotion grading, predictor-vs-baseline logic, V5 arrival gates.",
        "done_looks_like": "A qualified Cymek subject arriving and being measured causally; currently "
                           "waiting for a subject worth qualifying (all Arkenstone/cymek subjects are "
                           "software canaries).",
        "connects_to": "Consumes Cymek manifests; its intervention/causal-credit discipline is the template "
                       "Arkenstone's ARK-004B causal-credit test copies.",
    },
    "BRAMASTRA": {
        "proves": "A from-scratch instrument exists with executed binding/terminal experiments and excellent "
                  "negative-result hygiene. NEW: (1) terminal/EOS supervision repairs the complete-answer "
                  "contract (0/32 -> 32/32, 2 seeds) — and their audit shows Citadel T1C never supervised a "
                  "terminator (stop histogram MAX_TOKENS:1000 on every arm), partially confounding its famous "
                  "null; (2) MINI/MID capacity accounting is embedding-dominated (~95%); (3) aggregate accuracy "
                  "can rise to 48.4% while query control stays at copy-baseline level (0/64 both-correct).",
        "done_looks_like": "B0/B1 campaigns executed and a learned discovery policy actually implemented "
                           "(its own header marks the discovery policy unimplemented — GAP 1's design home).",
        "connects_to": "Shares the owner's from-scratch objective; its evaluation hygiene (query-blind "
                       "baselines, terminal contract, source snapshots) is adopted by Arkenstone.",
    },
    "esoes / core-exp / core-vnext / senora / iterate500-900 / main": {
        "proves": "Historical layers: ESOES froze the V5 blueprint and E0-E3 plans; core-exp holds "
                  "selection-vs-realization diagnostics; core-vnext holds the stale-optimizer failure "
                  "lesson; senora holds zero-mock discipline and its own overclaim history; main/iterate "
                  "hold the legacy production stack.",
        "done_looks_like": "Reference-only; no further work planned on any of them.",
        "connects_to": "Knowledge sources; nothing imports their code without re-verification.",
    },
    "Arkenstone (this branch)": {
        "proves": "The memorize->generalize transition exists, replicates across 7 seeds, and has a "
                  "TENTATIVE-SUPPORTED internal precursor (tens-column selectivity, LOO 4/4 beating "
                  "time/loss baselines); curriculum and aligned-teacher acceleration are executed nulls "
                  "at micro scale; post-G90 instability is documented.",
        "done_looks_like": "ARK-004B (column-consistency intervention, step-matched) executed with "
                           "replication + causal-credit test; then transfer to a second task family.",
        "connects_to": "Feeds dose/precursor evidence back to Citadel's T1D design; would hand any "
                       "replicated mechanism to Cymek's substrate for scale testing.",
    },
}

# GAP 1 (no experiment-proposing mechanism) is recorded as the program's known
# ceiling: BRAMASTRA designs it, nothing implements it. Arkenstone's bottleneck
# graph + decision tables are the manual precursor.
KNOWN_CEILING = (
    "No branch contains a mechanism that PROPOSES the next experiment from "
    "measured weakness. Arkenstone's contribution toward it is the ranked "
    "bottleneck graph and preregistered decision tables — the manual precursor "
    "a learned proposer would eventually replace."
)
