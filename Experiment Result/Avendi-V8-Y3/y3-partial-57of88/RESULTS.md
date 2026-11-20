# Avendi V8-Y3: partial campaign audit

**Partial:** 57/88 job receipts, 27/50 studies complete, 22 deferred and U50 partial. The evidence records a locked confirm opening, but incomplete arms prevent a campaign-wide conclusion. Source recorded by the archives: `ba1c0e978f890021a10e2365b61d5403bc486ce5`; The [saved Kaggle run log](https://www.kaggle.com/code/drdoom1/avendi-v8-y3-t4x2-1/output?scriptVersionId=371913709) records checkout HEAD `531a5aeb`; the present hitchhiker source `2e1587e5` is also distinct. The bundle baseline and runtime checkout must not be conflated.

**Scientific decision.** Across the three paired seeds, the endpoint-margin objective improved four-cell development accuracy from 13.93% to 19.66% (+5.73 percentage points). This missed the preregistered +10-point margin. On confirmation, margin solved only one eight-cell family across the three 256-family evaluations. The categorical state architecture merits further study on short paths. It reached 100% on the reported one- and two-hop panel for every seed and 0% at three and four hops.

## Archive integrity and scope

| Archive | SHA-256 | Listed member checks |
|---|---|---|
| analysis | `2a0cb8b853377c658c120275876424b7dde6e1f449b312a63777f1a592752bb3` | 5 checked; 0 mismatched, 0 missing |
| evidence | `2159ab36ea62ab60ccd95709099c79c686f122b8d5f3e7c22c4afca15fc257cc` | 303 checked; 0 mismatched, 0 missing |

Checksums establish consistency with the included manifests, not independent provenance or scientific correctness. We inspected receipts and registry/status consistency; we did not rerun training, replay, or regenerate logits. Packaged verification claims are byte integrity, oracle fixtures, action replay, checkpoint restore and training reproduction PASS; regenerated per-case agreement is only PARTIAL_REF_CHECK.

## P2 exact development metrics

Values are fractions, seed couples initialization and order; fam4/fam8 are family-level scores, raw is case-level. All six records agree with the compact analysis METRICS arm values and U15 receipts.

| Seed | CE fam4 | Margin fam4 | Difference | CE fam8 | Margin fam8 | CE raw | Margin raw |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.14453125 | 0.2265625 | 0.08203125 | 0.0 | 0.05078125 | 0.50830078125 | 0.54248046875 |
| 19 | 0.1328125 | 0.16796875 | 0.03515625 | 0.0 | 0.0 | 0.50390625 | 0.5068359375 |
| 41 | 0.140625 | 0.1953125 | 0.0546875 | 0.0 | 0.0 | 0.49462890625 | 0.50439453125 |

Margin exceeds CE on fam4 for all three paired seeds, but only margin seed 7 has nonzero fam8. Three paired runs do not establish a robust population effect or general capability. U19 reports substantial train fit but dev 0–0.0078125 across its six runs; this is evidence against extrapolating training fit to held-out performance.

## P4 per-seed exact and causal measurements

Dev columns 1–4 are the receipt’s rates at each step. Causal columns are unpatched, donor patch, suffix rewire both correct, norm noise, wrong donor versus actual expected, and wrong donor versus intended sham. Identity preservation is 1.0 and distinguishable pairs 256 in every row; full probe and other fields appear in METRICS.json.

| Job | 1 | 2 | 3 | 4 | Unpatched | Donor | Rewire | Noise | Wrong/actual | Wrong/sham |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4-categorical-s19 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.203125 | 0.0234375 | 0.25390625 |
| P4-categorical-s41 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.30078125 | 0.046875 | 0.32421875 |
| P4-categorical-s7 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.25390625 | 0.04296875 | 0.27734375 |
| P4-categorical_endonly-s19 | 1.0 | 0.0 | 0.015625 | 0.0 | 0.15625 | 0.1484375 | 0.11328125 | 0.15234375 | 0.0390625 | 0.1484375 |
| P4-categorical_endonly-s41 | 1.0 | 0.703125 | 0.0 | 0.0 | 0.828125 | 0.8046875 | 0.80078125 | 0.265625 | 0.04296875 | 0.23046875 |
| P4-categorical_endonly-s7 | 1.0 | 0.109375 | 0.0 | 0.0 | 0.3359375 | 0.359375 | 0.3359375 | 0.2578125 | 0.05078125 | 0.21875 |
| P4-continuous-s19 | 1.0 | 0.4375 | 0.046875 | 0.015625 | 0.14453125 | 0.15625 | 0.0625 | 0.11328125 | 0.078125 | 0.14453125 |
| P4-continuous-s41 | 1.0 | 0.484375 | 0.046875 | 0.015625 | 0.1171875 | 0.1484375 | 0.09765625 | 0.109375 | 0.05078125 | 0.12109375 |
| P4-continuous-s7 | 1.0 | 0.34375 | 0.03125 | 0.015625 | 0.140625 | 0.15234375 | 0.078125 | 0.10546875 | 0.04296875 | 0.11328125 |
| P4-continuous_endonly-s19 | 0.875 | 0.171875 | 0.046875 | 0.015625 | 0.1640625 | 0.1484375 | 0.05078125 | 0.19140625 | 0.07421875 | 0.15625 |
| P4-continuous_endonly-s41 | 1.0 | 0.125 | 0.03125 | 0.015625 | 0.1640625 | 0.15625 | 0.06640625 | 0.1640625 | 0.03515625 | 0.1484375 |
| P4-continuous_endonly-s7 | 0.953125 | 0.140625 | 0.046875 | 0.0 | 0.1484375 | 0.140625 | 0.04296875 | 0.15625 | 0.046875 | 0.1484375 |
| P4-dense-s19 | 1.0 | 0.1875 | 0.03125 | 0.03125 | 0.17578125 | 0.00390625 | 0.0 | 0.06640625 | 0.0 | 0.0859375 |
| P4-dense-s41 | 1.0 | 0.265625 | 0.046875 | 0.0 | 0.2265625 | 0.0 | 0.0 | 0.09375 | 0.0 | 0.0859375 |
| P4-dense-s7 | 1.0 | 0.171875 | 0.046875 | 0.03125 | 0.16796875 | 0.00390625 | 0.0 | 0.08984375 | 0.00390625 | 0.078125 |
| P4-dense_endonly-s19 | 0.8125 | 0.046875 | 0.046875 | 0.0 | 0.171875 | 0.015625 | 0.0 | 0.046875 | 0.01171875 | 0.078125 |
| P4-dense_endonly-s41 | 0.890625 | 0.125 | 0.0625 | 0.015625 | 0.21875 | 0.0 | 0.0 | 0.078125 | 0.00390625 | 0.0703125 |
| P4-dense_endonly-s7 | 0.765625 | 0.0625 | 0.046875 | 0.0 | 0.15234375 | 0.0234375 | 0.0 | 0.1171875 | 0.01953125 | 0.08203125 |

Categorical supervised rows have step 1/2 scores of 1 and step 3/4 scores of 0 for every seed; their donor and rewire scores are 1, while noise and wrong-donor controls are lower. These interventions support use of the measured state in this task, not general causal reasoning or longer-horizon success. End-only results vary strongly by seed; continuous and dense rows do not match the categorical causal pattern. Argmax mode is available for categorical rows only, limiting direct architecture comparisons.

## Confirmation and study validity

U47 reports `LOCKED_AND_OPENED_ONCE`, lock SHA-256 `c51fe4ab684d91139f29680ddb94ac055cf4c9cd8e01693d9b8faa9a8ce7ace4`, result SHA-256 `da4f10eb314508255aaf02e8be10b9368924d73a4d0d47803503853d2398828a`. Its required-job denominator is 12, while its result object contains 33 model records. The lock and result hashes are reported receipt fields, not independently reconstructed here. Confirmation fam4 and fam8 for each model are preserved below; full scalar values are in METRICS.json. The confirmation has already been opened and should not be treated as a fresh untouched holdout.

| Job | Confirm fam4 | Confirm fam8 |
|---|---:|---:|
| P2-ce-s19 | 0.12109375 | 0.0 |
| P2-ce-s41 | 0.13671875 | 0.0 |
| P2-ce-s7 | 0.1328125 | 0.0 |
| P2-margin-s19 | 0.1484375 | 0.0 |
| P2-margin-s41 | 0.1640625 | 0.0 |
| P2-margin-s7 | 0.18359375 | 0.00390625 |
| P3-address-s19 | 0.12890625 | 0.0 |
| P3-address-s41 | 0.1484375 | 0.0 |
| P3-address-s7 | 0.078125 | 0.0 |
| P3-address_sham-s19 | 0.10546875 | 0.0 |
| P3-address_sham-s41 | 0.05859375 | 0.0 |
| P3-address_sham-s7 | 0.08984375 | 0.0 |
| P3-curriculum-s19 | 0.10546875 | 0.0 |
| P3-curriculum-s41 | 0.140625 | 0.0 |
| P3-curriculum-s7 | 0.11328125 | 0.0 |
| P3-low_lr-s19 | 0.04296875 | 0.0 |
| P3-low_lr-s41 | 0.0234375 | 0.0 |
| P3-low_lr-s7 | 0.0546875 | 0.0 |
| P3-refresh-s19 | 0.0078125 | 0.0 |
| P3-refresh-s41 | 0.0078125 | 0.0 |
| P3-refresh-s7 | 0.0 | 0.0 |
| P3-rename-s19 | 0.10546875 | 0.0 |
| P3-rename-s41 | 0.10546875 | 0.0 |
| P3-rename-s7 | 0.05859375 | 0.0 |
| P3-rename_sham-s19 | 0.01953125 | 0.0 |
| P3-rename_sham-s41 | 0.00390625 | 0.0 |
| P3-rename_sham-s7 | 0.0390625 | 0.0 |
| P7-s20_ce-s19 | 0.1015625 | 0.0 |
| P7-s20_ce-s41 | 0.10546875 | 0.0 |
| P7-s20_ce-s7 | 0.16796875 | 0.0 |
| P7-s20_low_lr-s19 | 0.109375 | 0.0 |
| P7-s20_low_lr-s41 | 0.14453125 | 0.0 |
| P7-s20_low_lr-s7 | 0.14453125 | 0.0 |

U01 reports oracle OK on 256; U02 reports no split defects and catches a planted leak; U03 reports oracle 256/256 families and all listed shortcuts 0/256 families (several nevertheless score 1024/2048 cases); U05 detects case swap and EOS flip. These are useful fixture checks, not proof that all possible leakage or shortcuts are excluded. COMPLETE is an operational receipt status, not a positive scientific result. U28–U35 repeat the same 18-job measured payload; they are not eight independent replications.

## Operations and omissions

| Group | Complete / total |
|---|---:|
| Main jobs | 42/54 |
| Short jobs | 15/24 |
| All jobs | 57/88 |
| Studies | 27/50 |
| P6 adapt | 0 complete |

Pending: `P1-frozen`, `P6-diagnostic_policy-s19`, `P6-diagnostic_policy-s41`, `P6-diagnostic_policy-s7`, `P5-direct_plus_info-s19`, `P5-direct_plus_info-s41`, `P5-direct_plus_info-s7`, `P5-direct_plus_neutral-s19`, `P5-direct_plus_neutral-s41`, `P5-direct_plus_neutral-s7`, `P5-replace_direct-s19`, `P5-replace_direct-s41`, `P5-replace_direct-s7`, `P6-adapt-context-s19`, `P6-adapt-context-s41`, `P6-adapt-context-s7`, `P6-adapt-memory-s19`, `P6-adapt-memory-s41`, `P6-adapt-memory-s7`, `P6-adapt-weight_update-s19`, `P6-adapt-weight_update-s41`, `P6-adapt-weight_update-s7`, `P7-ark_capped-s19`, `P7-ark_capped-s41`, `P7-ark_capped-s7`, `P7-ark_expaug-s19`, `P7-ark_expaug-s41`, `P7-ark_expaug-s7`, `P7-ark_uncapped-s19`, `P7-ark_uncapped-s41`, `P7-ark_uncapped-s7`. Next resumable: `P1-frozen`. Dispatch lists some jobs as ran and deferred; receipt completion and the pending queue, not a dispatch attempt, determine completion. Missing P1, P5, P6 and ARK jobs leave their associated hypotheses unresolved. No claim of completed campaign, learned competence from engineered passes, or successful missing arms follows from these archives.

Study states: complete U01–U06, U15–U25, U27–U35, U47; deferred U07–U14, U26, U36–U46, U48–U49; U50 partial.

The downloaded FULL bundle contains 57 per-job archives. Its SHA-256 is `cda53e1b875100ac664836786eacc97dd5be9e1095f8ff95ac287b4efda19535`.
