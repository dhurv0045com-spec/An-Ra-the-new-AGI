# ARK-001 NOVELTY CLASSIFICATION

## Claim 1: lift-off exists at micro scale; dose measured (200-400 steps)
- Prior branch search: citadel T1/T1C record only end-state exact (0/500) with
  no lift-off metric; T1D plan has tier lift-off but never executed; cymek
  receipts contain no capability lift-off measurement; triquetra/senora/esoes: none.
- Literature: trivial — small transformers learn toy mappings (common knowledge);
  the NEW content is the measured dose + the contrast with this program's own
  macro-scale nulls.
- Class: **EXTENSION / NEW_MEASUREMENT for the program** (not a new mechanism).

## Claim 2: memorize-then-grok structural OOD with late, sharp transition
  (T2: train 1.0 by step 800; OOD ~0.005 through step 4400; jump 0.005->0.365
  over ~600 steps; ones-position 0.91 vs tens-position 0.37 at cutoff)
- Prior branch search: no branch measured OOD trajectories during training.
  Citadel's diagnostic ("loss-learning without rule learning") is the same
  phenomenon at macro scale, unexplained; this decomposes it.
- Literature: grokking is a known phenomenon (Power et al. 2022); digit-position
  asymmetry in arithmetic learning is related to known curriculum/length-
  generalization literature.
- Class: **REPRODUCTION of grokking (literature) + NEW_EMPIRICAL_DISCOVERY for
  the An-Ra program**: first causal decomposition of the program's central
  anomaly into (a) fast lookup-table fitting and (b) delayed per-position rule
  extraction with a specific asymmetry (ones-column local rule early; tens-column
  binding rule late). The asymmetry + budget quantification is the new content.

## Claim 3: vocabulary insensitivity at micro scale (T1-BYTE == T1-COMPACT)
- Prior: T1D arm E (masked vocab) designed, never run; vocab swap never designed.
- Class: **NEW_MEASUREMENT for the program** (negative result, de-prioritizes
  representation interventions at micro scale).

## What would change these classifications
- Multi-seed replication (ARK-002a) confirming the grokking transition and
  asymmetry -> promotes Claim 2 to REPLICATED.
- If citadel T1D later runs and its tier lift-off curves contradict the
  memorize-then-grok ordering -> revisit.
