# Y5 owner readiness

**Decision: ready for owner-run measurement.** The research question with the strongest prior evidence is whether training the existing reader on equivalent graph renderings improves unseen-syntax transfer while retaining familiar accuracy. The [campaign assessment](Y5_FOUNDATION.md) defines the other contrasts, decisions and dependencies. The [failure register](FAILURE_RESOLUTION_REGISTER.md) carries forward unresolved findings. The [baseline intake](Y5_BASELINE.md) records observed frozen-checkpoint behavior.

## Launch

1. Import [AVENDI_V8_Y5_T4X2.ipynb](../../notebooks/AVENDI_V8_Y5_T4X2.ipynb) into Kaggle. Enable Internet for the pinned Git checkout and select two T4 GPUs. The notebook defaults to an authorized launch when the owner runs it.
2. Attach the intact `AVENDI_Y4_FULL.zip` as a dataset file. Its expected SHA-256 is in the baseline receipt. Optional original Y1 checkpoint objects and the associated reference evidence enable historical recovery. Missing originals are recorded explicitly.
3. Run All. The notebook verifies its checkout, device count, Palantiri doctor, Y5 fixtures and regression suite before the campaign. Expect up to 66 development decisions, including conditional training and missing-input records. The default session budget is 480 minutes including a 30-minute reserve. Actual T4 speed and total duration have not been measured here.
4. Read `CAMPAIGN.json` and `ANALYSIS.json` in SUMMARY for dispositions, dependencies, counts and seed contrasts. Download EVIDENCE for case-level analysis. Download FULL for continuation or independently exercising weights. Raw checkpoint state is also in `/kaggle/working/y5-state`. Keep a FULL from every partial session.
5. To resume on a fresh session, use this same pinned notebook and original inputs, attach the previous Y5 FULL, and set `RESUME_INPUT` to its mounted ZIP path. In an existing workspace with the original state directory, leave that variable unset. Do not mix canary and production state or change the plan after seeing confirmation outcomes.

## Evidence and limitations

- The verified seed-7 Y4 pointer reader independently passed the Y5 familiar-format development threshold at 128/128 exact parses and 16/16 complete families. This is an available parent for preservation. It does not establish new-task acquisition.
- Offline checks found zero normalized graph overlap across the registered reader training, development and confirmation streams, including cross-seed comparisons. The varied-format training counts per seed were 341, 342 and 341. Each eight-cell family includes correlated row-order and assignment contrasts.
- Thirty-two focused CPU tests passed. Palantiri calibration detected all six injected framework faults. A fresh-process Y4 infrastructure canary completed specification validation, run, bundle verification and checkpoint replay. The Y5 notebook separately runs Y5-specific tests.
- The Y5 training path verifies the committed final tensor values against the tensors used for evaluation. A sealed per-stage list controls confirmation eligibility. Development and confirmation share the mechanism and scoring code. An archive checksum and a successful notebook alone cannot independently validate every scientific score; inspect raw predictions before promotion.
- D deliberately supplies the public semantic-action mapping, and E supplies the public binary law. The known-weight witness demonstrates the corrected model can use these mappings. Neither task tests discovery of the mappings. F requires measurable new-task gain as well as retained old behavior.
- The six original Y1 checkpoint objects were unavailable at intake. Their historical verification is blocked. Elapsed GPU time, cross-device numeric agreement and learned Y5 performance remain owner-run measurements.


## Pinned integration receipt

At code commit `25841d24bcfbdb9af97c76aec237811d172e64ae`, the nine-job CPU engineering smoke test recorded nine development dispositions, five frozen confirmations and zero worker errors. The ordinary scout gated donor training and the missing historical object blocked its recovery. SUMMARY, EVIDENCE and FULL exported successfully. Importing FULL in a new work directory reproduced all nine dispositions and five confirmations with zero worker errors. This exercised the campaign launch, dependency, custody and resume paths using tiny training budgets.
