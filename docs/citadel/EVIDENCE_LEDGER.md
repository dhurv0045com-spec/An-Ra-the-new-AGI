# EVIDENCE_LEDGER.md

Every substantive existing claim found in the branch audit, scored by Citadel.
Audit date: 2026-09-03. Citations are repository paths on the named branch at the named commit.

- `Commit` = tip SHA at which the artifact was read (esoes `85f44b7`, triquetra `fa44ea3`, cymek `92dcd56`) unless a specific commit is given.
- `Confidence` = Citadel's assessment of evidence quality for the claim as scoped.
- `Citadel verdict` ∈ {DEMONSTRATED, SUPPORTED, TENTATIVE, INCONCLUSIVE, CONTRADICTED, IMPLEMENTATION_FAILURE, NOT_TESTED}. DEMONSTRATED is reserved for preregistered, powered, replicated results.

**FACT/label discipline:** numbers below are MEASUREMENT from cited artifacts. Anything about what the numbers *mean* beyond the artifact's own scope is INFERENCE and is flagged in the alternative-explanation field.

---

## ESOES (and its inherited evidence base)

### E1. Generic continuation training improved loss without producing tested cognitive abilities
- Claim: PGE continuation training on V4 improved held-out LM loss 2.1884 → 1.9710 while exact-copy RAW, nonce-context, multi-fact FREE, and composition FREE all stayed at 0/chance (0/6, 0/8, 0/48, 0/12 across parent/21.8k/22.517k checkpoints); candidate selection ~chance (12/48).
- Branch: esoes (inherited from core-vnext). Commit: 85f44b7 (record), original receipt core-vnext@054619f.
- Artifact: `docs/esoes/EVIDENCE_AND_CONTEXT.md` (§10), original receipt `core-vnext@054619f` `output/pge_audit/CANONICAL_REPORT.md` (not on this branch).
- Experiment: PGE continuation training + post-training probe battery.
- Independent variable: continuation training (329,908,224 certified tokens).
- Control: pre-training parent checkpoint behavior on identical probes.
- Metric: held-out LM loss; exact behavioral probe pass rates.
- Result: loss −0.2174 nats; all probed cognitive abilities 0 or chance.
- Replication: none (single lineage, single audit).
- Confidence: High for this lineage/probe set; Medium as a general claim.
- Alternative explanation: probes may have been beyond this checkpoint's floor rather than evidence that loss excludes cognition; probe battery scope is narrow.
- Citadel verdict: **SUPPORTED** (the founding negative result of the project; n=1 lineage keeps it below DEMONSTRATED).

### E2. Selection and realization are separable failure modes
- Claim: models can rank the gold candidate (selection) while failing to generate it (realization).
- Branch: esoes (inherited from core-exp@51124de). Artifact: `docs/esoes/EVIDENCE_AND_CONTEXT.md`; receipt `qim3_sft6_replication.json` (core-exp).
- Experiment: SFT6 replication, paired query-lift measurement.
- Independent variable: targeted SFT intervention.
- Control: pre-SFT greedy behavior.
- Metric: query lift (nats), rank-1 vs greedy selection.
- Result: lift 2.5052 vs 0.0192 nats (paired Δ 2.486, 95% CI [1.7479, 3.258], 35/40 groups positive); rank-1 63/119 vs greedy 24/119.
- Replication: explicit replication receipt exists.
- Confidence: High.
- Alternative explanation: rank-1 measured under assisted scoring; instrument dependence not fully excluded.
- Citadel verdict: **SUPPORTED**.

### E3. Candidate scoring by generative likelihood is dominated by length/tokenization bias; no calibrated policy survived
- Claim: naive aggregation (sum/byte/token) and both calibrated policies (domain-PMI, contextual calibration) systematically select length-role candidates regardless of gold.
- Branch: esoes. Commit: 85f44b7.
- Artifact: `artifacts/e2/scoring_policy_development.json` (status `FAIL_DEVELOPMENT_POLICY`, `production_scoring_mode: null`, `gpu_hours: 0.0757`), `artifacts/e2/scoring_policy_preregistration.json`, `artifacts/e0/scoring_adapter_certificate.json`, `artifacts/e2/local_{cpu,cuda}_scoring_null.json`, verdict file `scoring_policy_development_verdict.json` (triquetra).
- Experiment: preregistered scoring-policy tournament: 256 triplets × 3 rotations × 5 dev seeds (95101–95105) × 3 tokenizers, TOST equivalence margin ±0.05, Holm over 132 hypotheses, decoy axes (shortest-byte, fewest-token, marked-prefix, hidden label, surface family).
- Independent variable: scoring policy.
- Control: permutation nulls; hidden-label negative control; oracle/broken/random scorers.
- Metric: selection rate per constructed role; deviation from calibrated null.
- Result: domain-PMI and contextual calibration select the fewest-token role **1.000** in all 15 CUDA cells (seed_rates 1.0×5, TOST p=1.0); crossed hidden label at chance 0.332; `policy_survives_bias_screen: {domain_pmi: false, contextual_calibration: false}`. Null study on exact untrained P35: sum 100%, byte 83.33%, token 50–66.67% fewest-token selection. Adapters on random logits: sum 65.625%, byte 84.375%.
- Replication: 5 seeds × 3 tokenizers, plus independent null receipts; CPU parity arm deliberately not run after fail.
- Confidence: High.
- Alternative explanation: fixture/candidate construction itself generates the length confound (policy-family-independent defect) — explicitly not yet excluded; this is C0's question.
- Citadel verdict: **DEMONSTRATED** (preregistered, powered, replicated) for "these five policies fail"; the *generalization* to all possible policies is the open part.

### E4. E0 generator v0.4.0 resists surface shortcuts within calibrated null +10pp
- Claim: after two repairs, no available heuristic beats its calibrated null by more than 10pp on the development suite.
- Branch: esoes. Artifact: `artifacts/e0/development_certificate.json` (suite SHA `f7a1b6af…0b8b56`, seed 271828), `artifacts/e0/shortcut_repair_receipt.json`.
- Experiment: 368-case/112-pair development suite, 12 baselines, 23 checks, pooled over 8 generator seeds.
- Independent variable: generator version (0.3.0 → 0.4.0).
- Control: per-heuristic calibrated nulls (casewise chance; analytic permutation null for serialization-coupled heuristics).
- Metric: max heuristic accuracy − null (gate ≤ 0.10).
- Result: state family bag-of-words 0.0417 (null 0.1037 casewise / 0.1295 position-calibrated), lexical overlap 0.1016, latest_fact/nearest_position 0.1380 (permutation null 0.1295); rule bag-of-words 0.0664; status PASS.
- Replication: pooled 8 seeds; certificate check `random_control_matches_calculated_chance: true`.
- Confidence: High for v0.4.0 development tier; inherently version-limited.
- Alternative explanation: heuristics tried are a finite set; new shortcut families may exist; sealed tier never exercised on a real model.
- Citadel verdict: **SUPPORTED** (development-infrastructure claim only; explicitly not a model result).

### E5. Residual-output initialization 1/sqrt(2L) controls depth-dependent residual growth
- Claim: scaling residual-output init by 1/sqrt(2L) reduces final residual growth and gradient spread at P35 shapes.
- Branch: esoes. Artifact: `artifacts/e2/local_cuda_signal_propagation.json` (+ `_4k`, `local_cpu_signal_propagation.json`).
- Experiment: forward/backward signal measurement, 5 seeds @256 ctx CUDA, 3 @4096, 3 CPU.
- Independent variable: init scaling.
- Control: unscaled init.
- Metric: residual growth ratio; gradient spread ratio.
- Result: 0.122/0.156/0.230 (deep/middle/wide) vs unscaled; 4k: 0.115/0.144/0.217.
- Replication: 11 exact-stack runs across devices/context lengths.
- Confidence: High as local mechanism prior; zero learning evidence by design.
- Alternative explanation: measurement-only property; benefit at optimization timescale unproven.
- Citadel verdict: **SUPPORTED** (mechanism prior; frozen as decision D-024).

### E6. QK normalization is scale control, not a quality claim
- Claim: affine QK norm makes logit RMS/entropy invariant to Q/K projection-scale stress.
- Branch: esoes. Artifact: `artifacts/e2/local_cuda_qk_norm.json` (+ CPU variant).
- Experiment: 0.25×/1×/4× Q,K scale stress, 5 seeds, 512/2048/4096.
- Control: unnormalized model.
- Metric: logit RMS ratio; entropy span; attended fraction.
- Result: normalized invariant (ratio ≤ 1.0001, entropy span ~1e-5); unnormalized RMS changes exactly 256×; at 4k base scale attended fraction 0.616 (QK) vs 0.988 (none).
- Replication: 8 runs across devices/lengths.
- Confidence: High (mechanism), explicitly not cognition.
- Alternative explanation: none recorded; no-QK E2 arm remains mandatory.
- Citadel verdict: **SUPPORTED** (D-025).

### E7. BF16 compute with FP32 master parameters is locally safe; native BF16 optimizer states are not
- Claim: BF16 parity through native 2k context (worst loss rel err 0.000118); BF16 AdamW moments overshoot clip-norm ~0.3%.
- Branch: esoes. Artifacts: `artifacts/e2/local_cuda_precision_parity*.json`, `artifacts/e2/local_{cuda,cpu}_real_update_native.json` (status FAIL), `local_{cuda,cpu}_real_update*.json`.
- Experiment: parity stack comparison; real-update canaries incl. negative control.
- Independent variable: precision configuration.
- Control: FP32; strict-bitwise resume receipts.
- Metric: logit/gradient cosine + RMS error; clip-norm compliance; resume equivalence.
- Result: parity SUPPORTED (D-026); native-BF16 FAIL (D-028).
- Replication: 3 seeds per cell, multiple context lengths.
- Confidence: High locally; explicitly bounded (no long-run/TPU evidence).
- Alternative explanation: short-horizon effects only.
- Citadel verdict: **SUPPORTED**.

### E8. A 24,576-entry byte-BPE is the local Pareto planning center
- Claim: 24k costs +0.85% tokens/byte vs 32k while saving 7.34M embedding params (width 896); 16k +3.50%.
- Branch: esoes. Artifact: `artifacts/e1/local_tournament/result.json`, `perturbation_sweep.json`, `v4_32k_baseline_audit.json`.
- Experiment: 3 independently trained BPE arms on 8,561,653 local bytes; 1,057,977 held-out bytes; 384-case perturbation sweep.
- Independent variable: vocab size.
- Control: exact roundtrip, unknown-rate, determinism (byte-identical rebuild).
- Metric: held-out tokens/byte; params; perturbation families.
- Result: 0.23826 (16k) / 0.23217 (24k) / 0.23022 (32k); 24k intermediate on all 6 families; determinism byte-identical.
- Replication: single corpus, deterministic rebuild verified.
- Confidence: Medium.
- Alternative explanation: local `legacy-mixed` corpus is non-representative; the real E1 (external corpus) is `BLOCKED_EXTERNAL_CORPUS`; artifact deliberately not frozen.
- Citadel verdict: **TENTATIVE** (planning prior, not an E1 result).

### E9. Checkpoint/transaction/durability semantics are correct locally
- Claim: P35 checkpoint save/restore achieves 0.0 parameter/optimizer error; crash boundaries safe; corruption rejected; cursor/ledger tamper-evident.
- Branch: esoes (+cymek re-implementation). Artifacts: `artifacts/v5/local_p35_checkpoint_canary.json`, `training_transaction_canary.json`, `local_durability_canary.json`, `local_cuda_cursor_resume.json` (+CPU).
- Experiment: framework-neutral transaction canaries + torch canaries.
- Independent variable: none (invariant verification).
- Control: corruption/missing/stale-writer injections must fail closed.
- Metric: restore error; injection rejection.
- Result: all invariants hold locally; resume ≡ uninterrupted.
- Confidence: High for local scope; documents explicitly disclaim distributed/TPU equivalence.
- Alternative explanation: local-filesystem-only; remote custody untested.
- Citadel verdict: **SUPPORTED** (local scope only).

### E10. Launch is blocked by design until six external identities exist
- Claim: no main training run is authorized; launch readiness ceiling is prelaunch experiments.
- Branch: esoes. Artifact: `artifacts/v5/launch_readiness.json` (`READY_FOR_PRELAUNCH_EXPERIMENTS`, `main_training_authorized: false`, `production_launcher_implemented: false`, gates E1–E6 PENDING, six external identities null), `blueprint/LAUNCH_GATES.json`.
- Experiment: fail-closed gate evaluator (tested).
- Result: all gates pending; policy `fail_closed_receipt_hash_and_external_identity_required`.
- Citadel verdict: **DEMONSTRATED** (repository fact; directly verified 2026-09-03).

---

## TRIQUETRA (audit input; all results DEV-tier on weak V4 checkpoints)

### T1. Value-recency dominates repair on V4 step-30400
- Claim: inserting the bare gold VALUE (no entity) repairs ~44–47% of failures; inserting entity alone repairs ~0%; full correct pair is *worse* than value alone.
- Branch: triquetra. Commit: fa44ea3 (receipts authored 339c840/f77f2cc era).
- Artifact: `output/entity_value_factorial_dev.json` (seed 41414), `output/entity_value_factorial_dev_rep.json` (seed 51515), protocol `x_factor/protocols/entity_value_factorial_dev_v1.json`.
- Experiment: preregistered 12-condition entity×value factorial, 120 tasks, paired McNemar exact + 10k bootstrap, unit=TASK; decision rules preregistered.
- Independent variable: inserted content class (entity / value / pair / wrong-pair variants).
- Control: C0 neutral control; C8 full-distractor fact (0.0).
- Metric: repair rate on failures.
- Result: DEV (107 failures): C0 3.74%, C1(entity) 0.0%, **C2(value) 46.73%**, C3(pair) 26.17%, C7 20.56%; C2−C0 +43.0pp (p≈0, CI [32.7, 52.3]); REP (104 failures): C2 46.15%, C2−C0 +33.65pp (p≈0); direction replicated, magnitude compatible. Verdict `VALUE_RECENCY_DOMINANT`.
- Replication: second seed with frozen protocol.
- Confidence: High for this checkpoint + fixture family; scope explicitly DEV (same-generator, not fresh).
- Alternative explanation: bare value is a salience/recency manipulation (A1/A2-style assist), not evidence of addressing; single floor-limited checkpoint; generator-specific.
- Citadel verdict: **SUPPORTED** (narrowly scoped: within step-30400, DEV tier, replicated). Generalization to fresh surfaces/other checkpoints: NOT_TESTED.

### T2. Latent query-conditioned selection signal is ≈ absent on step-30400 (and 22517)
- Claim: candidate ranking without insertion assists sits at chance and shows no query-conditioned component; position dominates query-match by ~19–35×.
- Branch: triquetra. Artifact: `output/query_value_evidence_dev.json` (seed 71717), `_rep.json` (seed 81818), `_ckpt22517.json`, firewall `x_factor/observed.py`.
- Experiment: preregistered QV evidence matrix — S[i,j] = log P(" VALUE_j." | facts, query_i), no gold inserted, 80 sets × 4 queries × 4 candidates, ladder E0–E8 with answer-blind legality rules.
- Independent variable: evidence level (raw / normalized / assists / oracle).
- Control: permutation; sham-duplication; position regression; chance 0.25.
- Metric: rank-1 accuracy; QCS (query-conditioned score contrast) with CI; position vs query-match coefficient.
- Result: DEV raw rank-1 **0.2500** (= chance exactly); normalized 0.3063 (p=0.1296 ns); QCS 0.0533, CI [−0.008, 0.114] includes 0; position coefficient 0.1607 vs query-match 0.0085 (~19×); REP: same pattern (QCS 0.0457 ns, ~35×). Generation 0.1219 (30400) vs 0.0094 (22517).
- Replication: frozen-seed replication + cross-checkpoint (22517).
- Confidence: High for these two checkpoints and DEV fixtures.
- Alternative explanation: floor-limited substrate (readiness v2: capability INSUFFICIENT, identifiability NOT_IDENTIFIABLE) — absence of measurable signal ≠ proof of representational absence; DEV fixtures only.
- Citadel verdict: **SUPPORTED** (scoped to substrate; ledger itself says GATE1 marginal/NO, "training NOT justified").

### T3. Training 22517→30400 built copy/readout machinery but not query control
- Claim: between the two checkpoints, duplication-elicitability (E5dup 0.6%→23.1%) and generation (0.9%→12.2%) emerged while QCS stayed ~0 and raw rank stayed at chance.
- Branch: triquetra. Artifact: `output/query_value_ckpt22517.json`.
- Experiment: same-task checkpoint comparison (observational).
- Independent variable: training progression between checkpoints (not controlled by Citadel-audit).
- Control: identical task sets and scoring.
- Metric: as T2.
- Result: as claimed; oracle 0.3625→0.5188.
- Replication: none (observational, n=2 checkpoints).
- Confidence: Medium.
- Alternative explanation: other run-to-run differences (data, schedule) confound the attribution.
- Citadel verdict: **TENTATIVE**.

### T4. E5 duplication assist is template-bound on this substrate; E5 line closed
- Claim: E5dup−sham effect = 0.0 (p=1.0) under compound structural shift; oracle ≈ chance there (0.2417), so the latent-signal question is unresolved in the shifted regime, not answered.
- Branch: triquetra. Artifact: `output/structural_ood_e5.json` (seed 91919), ledger `AN_RA_PROGRAM.md` §7.
- Experiment: preregistered structural-OOD battery, 240 queries, 5 shift types.
- Control: sham duplication; oracle ceiling.
- Metric: E5dup vs sham repair under shift.
- Result: E0 0.0%, E5dup 0.0% vs sham 0.0%; oracle 0.2417; H0 branch fired ("format hack; do not train").
- Replication: single run; direction consistent with floor diagnosis.
- Confidence: High for the scoped negative; explicitly NOT a proof about stronger Cores.
- Alternative explanation: floor-limited substrate (E0 0.0% means nothing to generalize).
- Citadel verdict: **SUPPORTED** (scoped; E5-internalization line closed).

### T5. Self-model (X1) prediction claim on the accumulation child was a false green
- Claim: "X1-REAL PASS" (prospective intervention-outcome accuracy 0.9545 vs fixed-policy 0.0909) was invalidated: an always-negative predictor scores 0.9733 on the same matrix (cell prevalence 0.0267).
- Branch: triquetra. Artifact: `output/x1_real_receipt.json` (stale verdict string retained), `output/ibq_legacy_basis_verdict.json`, ledger `AN_RA_PROGRAM.md` §8 (row "X1-REAL-0 | INVALID for self-model (imbalanced)").
- Experiment: IBQ v1 basis qualification.
- Result: oracle coverage 0.0889 FAIL; degenerate probes; entropy 0.5661 vs null 0.6932; `BASIS NOT QUALIFIED`.
- Replication: invalidation analysis independent of original run.
- Citadel verdict: **CONTRADICTED** (original claim); instrument lesson preserved.

### T6. No locally available checkpoint qualifies as a research subject
- Claim: readiness v2 calibration on step-30400 returns NOT_READY / INSUFFICIENT / NOT_IDENTIFIABLE (primitive canaries P1 0.083 [0.015, 0.354], P4 0.083 fail Wilson-hi < 0.50; frontier CALIBRATION_UNSTABLE); registry holds 5 V4-lineage files, all `research_subject: false`, one a non-model optimizer artifact.
- Branch: triquetra. Artifact: `output/readiness_v2_calibrate_30400.json`, `x_factor/registry/checkpoints.json`, `AN_RA_PROGRAM.md` ("WAITING_FOR_STRONGER_CHECKPOINT").
- Experiment: v2 readiness calibration (deliberate negative control after v1 false green).
- Citadel verdict: **SUPPORTED** (per the v2 gate's own rules; the gate itself is Citadel-audited as reasonable but not yet independently validated).
- Consequence (INTERPRETATION): checkpoint capability is the binding constraint on all real-model mechanism research — the project's most explicit current wait-state.

### T7. Readiness gate v1 was a false green, self-caught and downgraded
- Claim: v1 pilot emitted `READY_FOR_BINDING_CAUSAL_RESEARCH` on the same checkpoint later graded INSUFFICIENT; downgraded to `CALIBRATION_ONLY / GATE_V0_NOT_QUALIFIED`.
- Branch: triquetra. Artifact: `output/readiness_pilot_30400.json` (stale green retained), commit 44643c7, v2 artifacts.
- Citadel verdict: **CONTRADICTED** (v1 verdict), instrument v2 built in response.

---

## CYMEK (audit input)

### Y1. The production V5 path is contracts-complete but cannot yet train anything
- Claim: full contract stack (tokenizer identity, data manifests, mixture, packing, cursor, model receipt 250,216,960, CE objective, token-indexed schedule, evaluation, promotion, remote jobs) is implemented and tested; but there is no production trainer, no corpus loader, no tokenizer artifact, no real manifests, and nothing remote has ever been submitted.
- Branch: cymek. Commit: 92dcd56. Artifact: whole tree; `blueprint/STATUS.md`; `artifacts/v5/launch_readiness.json`.
- Experiment: 20+ test files (~105 v5-path test functions) + fail-closed launch-readiness evaluator.
- Citadel verdict: **SUPPORTED** (code and tests read directly; tests not re-executed by Citadel — CI claims 145/148 framework-neutral passes).
- Consequence (INTERPRETATION): the fastest route to a *trainable* small model is Cymek's contracts + a minimal caller-supplied backend — the contracts are explicitly designed for that.

### Y2. The frozen mixture is an implementation decision, not a measured optimum
- Claim: slices 0.65/0.20/0.15 and nine cognition family fractions are hard-coded in `v5_contracts/run_spec.py` / `training_spec.py`; the only sanctioned mixture experiment is E3 (cognition fraction ∈ {0.05, 0.15, 0.30}, 200M-token screens, non-cognition ratio pinned 65:20, CE-only Phase A).
- Branch: cymek/esoes. Artifact: `v5_contracts/run_spec.py:105` (verified), `e3_data_objective/plan.py`, `artifacts/e3/static_plan.json` (`BLOCKED_UPSTREAM_INPUTS`).
- Citadel verdict: **NOT_TESTED** (the mixture's scientific value has never been measured; E3 has never run).

---

## Cross-cutting audit observations (INFERENCE, flagged)

1. **Stale greens inside immutable receipts.** `output/x1_real_receipt.json` still says `"X1-REAL PASS"` and `output/readiness_pilot_30400.json` still says `READY_FOR_BINDING_CAUSAL_RESEARCH`; corrections live only in ledgers. Citadel rule: a receipt's verdict field is never authoritative without its ledger.
2. **Every real-model positive result is DEV-tier on one weak checkpoint.** Nothing has touched fresh/sealed surfaces. The project has no fresh-tier replication of any cognition claim.
3. **The measurement layer is the most validated part of the project** (scorers, nulls, firewalls, readiness gates all have receipts) **and it is also where every blocker currently sits** (scoring mode null; readiness NOT_READY; sealed custody absent).
4. **The "+0.669 nats, p=0.018" query-conditioning SFT claim is UNVERIFIED** — the ledger states the cited receipt does not contain it. Do not cite it.
