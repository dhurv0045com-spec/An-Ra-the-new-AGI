# CYR-GPU-001 PLAN — Colab GPU research tournament (preregistered)

- QUESTION: Which training intervention increases transferable learning
  while preserving acquired capability, without hiding behind formatting,
  shortcuts, leakage, freezing, extra compute, or test-set feedback?
- PRIOR EVIDENCE: docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md
  (Arkenstone 4911b84, BRAMASTRA 90ee31a). Strongest priors: ARK-007R
  retention (LOW protects, mechanism open), ARK-010 recovery (HIGH
  reacquires), BRAMASTRA terminal supervision (EOS contract, adopted),
  BRAMASTRA query-blind coexistence (both-correct required).
- HYPOTHESIS H1: counterfactual pair-preserving minibatches raise
  fresh both-correct vs shuffled controls at matched exposure.
- HYPOTHESIS H2: hysteretic HIGH→LOW control protects retention without
  freezing; fixed-time decay is the causal control for "state matters".
- HYPOTHESIS H3: displacement-matched intermediate LR separates LR level
  from parameter displacement (freezing alternative).
- STRONGEST ALTERNATIVE (H1): order/frequency artifacts of grouped
  presentation, not query-sensitivity (controlled: identical multisets,
  blind-gap vs heuristic baselines).
- STRONGEST ALTERNATIVE (H2/H3): LOW arms protect by near-freezing
  (measured: displacement ratio HIGH/LOW with floor 5.0; MID arm spans it).
- TREATMENT (H1): twins adjacent in stream, microbatches slice even
  boundaries (pair_splits receipted, must be 0). CONTROL: same multiset
  shuffled, same init/tokens/updates/schedule.
- TREATMENT (H2): arms {HIGH continue, fixed-time switch at 50% of
  continuation budget, state-triggered (sustained both-correct ≥0.90×3),
  hysteretic (enter≥0.90×3, reenter<0.75×3), MID constant} from matched
  S1 forks (identical pre-switch history).
- CONTROL (H2): fixed-time arm (prospective constant) + exposure report;
  >2× low-LR exposure mismatch flags the comparison as confounded.
- CONSTANTS: seeds (101, 202); microbatch 8 rows; row content 300 tokens;
  eval every 10 updates; sustained-3 transitions; gap threshold 0.10 both
  seeds for sampler win; research LRs HIGH 3e-4 / MID 3e-5 / LOW 3e-6
  (explicit variants, never canonical); proxy ladder TINY/MICRO/MIDI/P35;
  budget table in resolve_proxy (throughput/VRAM only).
- DATA: registry (binding/retrieval) + transfer (compositional hops)
  families, self-contained renderer v1, counterfactual twins per world,
  splits train/dev_controller/dev_measurement/sealed_reserved with disjoint
  seeds AND index ranges (firewall asserted, manifests hashed).
- SPLIT: latent worlds never cross splits (asserted); controller sees
  dev-controller ONLY; dev-measurement is the arbiter; sealed_reserved is
  generated but NEVER consumed by control or selection this session
  (no sealed custody exists; cap claims at MULTI_TASK_REPLICATED).
- SEEDS: (101, 202) matched across arms; order seeds == arm seeds.
- MODEL: proxy ladder (TINY smoke / MICRO fast / MIDI main / P35 transfer),
  frozen 24576 tokenizer on Colab, real Cymek backend/state/checkpoints.
- OBJECTIVE: causal CE, replica-global eligible denominator, clip 1.0,
  complete-answer rows (BOS content EOS, EOS supervised).
- OPTIMIZER: AdamW (0.9, 0.95, 1e-8, wd 0.1), no-decay norms — canonical.
- SCHEDULE: research-constant per arm (see TREATMENT); canonical WSD
  untouched and unused here.
- TOKEN BUDGET: resolver table (MICRO 150k/60k, MIDI 400k per arm),
  capped by rendered records (receipted); all arms in a comparison share
  identical budgets.
- WALL BUDGET: min 60 / target 90–150 / hard 180 min; graceful stage
  skips with reasons; partial bundles still packaged.
- PRIMARY METRIC: dev-measurement both-correct (fresh worlds), sustained.
- SECONDARY METRICS: exact, factorial variants (base/query_only/
  order_only/query_and_order), blind gap, loss, displacement/grad/moment
  norms, stop histogram, pair_split_rate, exposure table.
- NULL RESULT (H1): |gap| < 0.10 or redteam veto → pair grouping PARKED.
- POSITIVE RESULT (H1): gap ≥ 0.10 both seeds + clean redteam →
  MICRO_REPLICATED, advance to transfer stage.
- NULL RESULT (H2): no switch fires (documented PARK) or retention equal
  across arms within noise → hysteresis PARKED, retention comparison kept.
- POSITIVE RESULT (H2): hysteretic/state arms retain ≥ LOW with
  displacement ≥ 5× LOW and fixed-time does not explain it.
- FAILURE RESULT: smoke/calibration/EOS/learnability-gate failure aborts
  with FAILURE.json + partial bundle (evidence, not silence).
- ABORT CONDITIONS: env/smoke fail; calibration non-positive; S1 no
  sustained train-exact ≥ 0.99 (no-signal gate); corruption; <60 min only
  for env failure / no-signal / corruption / failing smoke / decisive
  downstream-invalidating failure.
- WHAT WOULD CHANGE OUR MIND: H1 — shuffled ≥ paired both-correct with
  clean redteam kills it. H2 — fixed-time matches state-triggered at
  matched exposure kills "state matters". H3 — retention tracks
  displacement, not LR level, reframes H2 as dose control.
- NEXT BOTTLENECK: mixture screen (deferred), B0-scale replication,
  sealed custody, TPU semantics.
- LITERATURE NOTE (one check, 2026-09): grokking/plasticity literature
  (Carvalho statistical view; Lyle nonstationarity/plasticity framing;
  Grokfast/EGD amplification methods) supports the plasticity framing but
  offers no more-informative transparent mechanism than the
  branch-derived arms; external candidates REJECTED for this session
  (fashion bar not met; Grokfast/EGD are non-transparent optimizers).
- VALIDATION GATE (CELL 0, Colab hardware): unit suites (cyr, contracts,
  complete-answer, training, data, data-pipeline, backend, stream-resume)
  + smoke + full entry suite EXCLUDING the exact-head repo-hygiene check
  (the committed closure receipt predates this cycle's code; it refreshes
  from Colab evidence next cycle — excluding it is documented, not hidden).
  Any validation failure aborts before the tournament.
