# CYR-GPU-002 PLAN — Colab GPU research campaign (preregistered)

- QUESTION: Which training intervention increases transferable learning
  while preserving acquired capability, without hiding behind formatting,
  shortcuts, leakage, freezing, extra compute, or test-set feedback?
- PRIOR EVIDENCE: docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md
  (Arkenstone 933d4f3, BRAMASTRA 90ee31a — re-audited this cycle; ARK-011/
  012/013/014 and V6 all PREREGISTERED/UNEXECUTED, no new evidence).
  Strongest priors: ARK-007R retention, ARK-010 recovery, BRAMASTRA
  terminal supervision (adopted as contract), BRAMASTRA query-blind
  coexistence (both-correct required).
- HYPOTHESIS H1: counterfactual pair-preserving minibatches raise
  fresh both-correct vs shuffled controls at matched exposure.
- HYPOTHESIS H2: hysteretic HIGH→LOW control protects retention without
  freezing; fixed-time decay is the causal control for "state matters".
- HYPOTHESIS H3: displacement-matched intermediate LR separates LR level
  from parameter displacement (freezing alternative).
- STRONGEST ALTERNATIVE (H1): order/frequency artifacts of grouped
  presentation (controlled: identical multisets, blind-gap vs task-aware
  baselines with precomputed expected scores).
- STRONGEST ALTERNATIVE (H2/H3): LOW arms protect by near-freezing
  (measured displacement ratio with floor 5.0; MID arm spans it).
- TREATMENT (H1): twins adjacent in stream from ONE shared latent
  context per world (base/twin differ ONLY in query+answer), microbatches
  slice even boundaries (pair_splits receipted, must be 0). CONTROL: same
  multiset shuffled, same init/tokens/updates/schedule/LR.
- TREATMENT (H2): arms {HIGH continue, LOW continue, FIXED_TIME switch,
  STATE_TRIGGERED (phased), HYSTERETIC (phased), MID constant} from
  matched S1 forks (identical pre-switch history, byte-identical future
  order, fork offsets receipted).
- CONTROL (H2): fixed-time arm at 50% of continuation budget (prospective
  constant) + exposure report (tokens/updates at HIGH/LOW per arm);
  >2× low-LR exposure mismatch flags the comparison as confounded.
- CONSTANTS: seeds (101, 202); microbatch 8 rows; eval every 10 updates;
  sustained-3 transitions; pair-win gap ≥ 0.10 both seeds;
  M99 train-exact 0.99 / G50/G90/G95 dev both-correct 0.50/0.90/0.95;
  controller enter≥0.90×3, reenter<0.75×3; research LRs HIGH 3e-4 /
  MID 3e-5 / LOW 3e-6 (explicit variants, never canonical); proxy ladder
  TINY/MICRO/MIDI/P35 (+RESEARCH_SMALL fast tier).
- DATA: registry (binding/retrieval) + transfer (compositional hops),
  self-contained renderer v2, counterfactual twins, orthogonal 7-variant
  factorial (base/query_only/order_only/query_and_order/
  relevant_value_only/irrelevant_value_only/rendering_only-weak-probe).
- SPLIT: latent worlds never cross splits (asserted + hashed manifests);
  controller sees dev_controller ONLY; dev_measurement arbitrates;
  sealed_reserved generated, hashed, scored ONCE at the very end for
  reporting, excluded from every decision/gate/budget (verified by test).
- SEEDS: (101, 202) matched across arms; batch-order seeds equal arm
  seeds; S4 reuses seeds[0].
- MODEL: proxy ladder resolved by hardware; frozen 24,576 tokenizer;
  real Cymek backend/state/checkpoint math at proxy scale.
- OBJECTIVE: causal CE, replica-global eligible denominator, clip 1.0,
  complete-answer rows (BOS content EOS, EOS supervised); batch padding
  with eligible masks (no fixed-size fiction).
- OPTIMIZER: AdamW (0.9, 0.95, 1e-8, wd 0.1), no-decay norms — canonical.
- SCHEDULE: research-constant per arm (see TREATMENT); canonical WSD
  untouched and unused here.
- TOKEN BUDGET: dose floor 2M acquisition tokens/arm; wall-time resolver
  (115-min training split 30/25/30/15 across acquisition/pair/LR/
  transfer); proxy-downshift before replication cuts (documented
  priority: transfer-second-seed, then MID arm); resolved budgets frozen
  pre-outcome in RESOLVED_PREREGISTRATION.json.
- WALL BUDGET: min 60 / target 120–160 / hard 175 min; per-update
  deadlines with TIMEBOX checkpoints; graceful stage skips with reasons.
- PRIMARY METRIC: batched free-generation complete exact + both-correct
  (EOS stop rate, MAX rate, invalid rate, prefix-extra rate reported).
- SECONDARY METRICS: teacher-forced exact (diagnostic), factorial
  variants, blind gap, loss, displacement/grad/moment norms, stop
  histogram, pair_split_rate, exposure table, milestones M99/G50/G90/G95
  with onset+confirmation in updates/tokens/wall.
- NULL RESULT (H1): |gap| < 0.10 or redteam veto → pair grouping PARKED.
- POSITIVE RESULT (H1): gap ≥ 0.10 both seeds + clean redteam →
  DEVELOPMENT_REPLICATED, advance to transfer stage.
- NULL RESULT (H2): no switch fires (documented PARK) or retention equal
  across arms within noise → hysteresis PARKED, retention comparison kept.
- POSITIVE RESULT (H2): hysteretic/state arms retain ≥ LOW with
  displacement ≥ 5× LOW and fixed-time does not explain it.
- FAILURE RESULT: smoke/calibration/EOS/learnability-gate failure aborts
  with FAILURE.json + partial bundle (evidence, not silence).
- ABORT CONDITIONS: env/smoke fail; calibration non-positive; S1 no
  sustained M99 (no-signal gate); corruption; <60 min only for env
  failure / no-signal / corruption / failing smoke / decisive
  downstream-invalidating failure.
- WHAT WOULD CHANGE OUR MIND: H1 — shuffled ≥ paired both-correct with
  clean redteam kills it. H2 — fixed-time matches state-triggered at
  matched exposure kills "state matters". H3 — retention tracks
  displacement, not LR level, reframes H2 as dose control.
- NEXT BOTTLENECK: mixture screen (deferred with justification), B0-scale
  replication, sealed custody, TPU semantics.
- LITERATURE NOTE (2026-09, one check): grokking/plasticity literature
  supports the plasticity framing; Grokfast/EGD-style non-transparent
  optimizers rejected (simpler displacement-matched arms strictly more
  informative). No external candidate admitted.
- VALIDATION GATE (CELL 0, Colab hardware): unit suites + smoke + full
  entry suite minus the repo-hygiene self-check (closure receipt
  refreshes from Colab evidence next cycle — documented); any failure
  aborts before the tournament.
